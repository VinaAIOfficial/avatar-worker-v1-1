"""Single short avatar job. Credentials and external file URLs are not accepted."""
import base64
import binascii
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
import unicodedata
import wave
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(os.environ.get('SADTALKER_DIR', '/opt/SadTalker'))
MAX_VIDEO_BYTES = 4_000_000
Image.MAX_IMAGE_PIXELS = 20_000_000

def decode_field(value, max_bytes, name):
    if not isinstance(value, str) or len(value) > (max_bytes + 2) // 3 * 4:
        raise ValueError(name + ': missing or too large')
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(name + ': invalid base64') from exc
    if not data or len(data) > max_bytes:
        raise ValueError(name + ': invalid size')
    return data

def validate_wav(data):
    with wave.open(io.BytesIO(data), 'rb') as wav:
        if wav.getcomptype() != 'NONE' or wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError('Audio must be uncompressed PCM WAV, 16-bit mono')
        if not 8000 <= wav.getframerate() <= 48000:
            raise ValueError('WAV sample rate must be 8–48 kHz')
        frames = wav.getnframes()
        duration = frames / wav.getframerate()
        if duration <= 0:
            raise ValueError('Audio must contain frames')
        if len(wav.readframes(frames)) != frames * 2:
            raise ValueError('Truncated WAV')
    return duration

def run(cmd, timeout, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        # Model logs contain no API keys; do not print incoming base64 payloads.
        raise RuntimeError((result.stderr or result.stdout)[-2400:])
    return result.stdout

def probe(path):
    return json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                           '-of', 'json', str(path)], 20))

def package_video(source, audio, output, duration):
    run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', str(source), '-i', str(audio),
         '-map', '0:v:0', '-map', '1:a:0',
         '-vf', 'scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x17191c,setsar=1,tpad=stop_mode=clone:stop_duration=1',
         '-r', '25', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
         '-maxrate', '1500k', '-bufsize', '3000k', '-pix_fmt', 'yuv420p',
         '-c:a', 'aac', '-b:a', '96k', '-t', str(duration), '-shortest',
         '-movflags', '+faststart', str(output)], 60)
    info = probe(output)
    videos = [s for s in info['streams'] if s['codec_type'] == 'video']
    audios = [s for s in info['streams'] if s['codec_type'] == 'audio']
    if not videos or not audios or (videos[0]['width'], videos[0]['height']) != (1280, 720):
        raise RuntimeError('Output does not contain 720p video and audio')
    actual = float(info['format']['duration'])
    if abs(actual - duration) > 0.15:
        raise RuntimeError('Output duration differs from input audio')
    if abs(float(videos[0].get('duration', actual)) - duration) > 0.15:
        raise RuntimeError('Video stream ended early')
    run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(output),
         '-f', 'null', '-'], 40)
    data = output.read_bytes()
    if len(data) > MAX_VIDEO_BYTES:
        raise RuntimeError('Output too large for this starter API response')
    return data, actual

# Moderate starting presets, to be compared on the user's portrait before production.
# All use real 512px checkpoints, crop mode, and automatic blinking (no --still).
MOTION = {'natural': (0, 1.0), 'calm': (0, 0.9), 'expressive': (1, 1.1), 'reference': (0, 1.0)}

def inference_command(source,audio,results,motion):
    pose,expression=MOTION[motion]
    cmd=['python3','inference.py','--source_image',str(source),'--driven_audio',str(audio),
         '--result_dir',str(results),'--checkpoint_dir',str(ROOT/'checkpoints'),
         '--preprocess','crop','--size','512','--batch_size','1',
         '--pose_style',str(pose),'--expression_scale',str(expression)]
    if motion=='reference':
        reference=Path('/opt/worker/references/motion.mp4')
        if not reference.is_file():raise ValueError('Reference mode requires references/motion.mp4 baked into the worker; use natural otherwise')
        cmd += ['--ref_eyeblink',str(reference),'--ref_pose',str(reference)]
    return cmd

def handler(event):
    request = event.get('input', {})
    if not isinstance(request, dict):
        raise ValueError('input must be an object')
    # For Runpod build probes and manual diagnostics. Does not generate an avatar.
    if request.get('action') == 'diagnose':
        import torch
        return {'worker': 'avatar-pipeline-v1.1', 'input_contract': '1.2', 'avatar_character_limit': 120, 'cuda_available': torch.cuda.is_available(),
                'checkpoint_present': (ROOT / 'checkpoints/SadTalker_V0.0.2_512.safetensors').is_file(),
                'message': 'Diagnostic only; run a real image/audio job to verify inference.'}
    if request.get('contract')!='1.2':raise ValueError('This worker requires input contract 1.2')
    motion=request.get('motion','natural')
    if motion not in MOTION:raise ValueError('Unknown motion preset')
    narration = request.get('narration')
    if not isinstance(narration, str) or not 1 <= len(unicodedata.normalize('NFC', narration)) <= 120:
        raise ValueError('Avatar narration must contain 1-120 characters')
    image_bytes = decode_field(request.get('image_base64'), 2_000_000, 'image_base64')
    audio_bytes = decode_field(request.get('audio_base64'), 1_000_000, 'audio_base64')
    duration = validate_wav(audio_bytes)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; select a GPU endpoint, not CPU')
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='avatar-') as folder:
        temp = Path(folder)
        source, audio = temp / 'source.jpg', temp / 'audio.wav'
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.format not in ('JPEG', 'PNG'):
                raise ValueError('Image must be JPEG or PNG')
            img = ImageOps.exif_transpose(img).convert('RGB')
            if min(img.size) < 256:
                raise ValueError('Image must be at least 256px on each side')
            img.thumbnail((1024, 1024))
            img.save(source, 'JPEG', quality=92)
        audio.write_bytes(audio_bytes)
        results = temp / 'results'
        # Start a fresh inference process per request for isolation and predictable GPU cleanup.
        run(inference_command(source,audio,results,motion),timeout=780,cwd=ROOT)
        clips = sorted(results.glob('*.mp4'))
        if len(clips) != 1:
            raise RuntimeError('SadTalker did not produce one final MP4; check face detection logs')
        video, actual = package_video(clips[0], audio, temp / 'avatar.mp4', duration)
    return {'worker':'avatar-pipeline-v1.1','motion':motion,'video_base64': base64.b64encode(video).decode('ascii'),
            'sha256': hashlib.sha256(video).hexdigest(), 'duration_seconds': actual,
            'width': 1280, 'height': 720, 'face_model_size': 512,
            'processing_seconds': round(time.monotonic() - start, 2)}

if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler': handler})
