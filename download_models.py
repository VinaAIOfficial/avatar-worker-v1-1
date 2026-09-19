"""Download the official 512px model and face detectors at build time."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path('/opt/SadTalker')
FILES = {
    'checkpoints/SadTalker_V0.0.2_512.safetensors':
        'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors',
    'checkpoints/mapping_00229-model.pth.tar':
        'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar',
    'gfpgan/weights/alignment_WFLW_4HG.pth':
        'https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth',
    'gfpgan/weights/detection_Resnet50_Final.pth':
        'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
}

def main():
    manifest = {}
    for name, url in FILES.items():
        dest = ROOT / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                digest = hashlib.sha512()
                req = urllib.request.Request(url, headers={'User-Agent': 'Avatar-Worker/1.1'})
                with urllib.request.urlopen(req, timeout=120) as src, dest.open('wb') as out:
                    while chunk := src.read(1024 * 1024):
                        out.write(chunk)
                        digest.update(chunk)
                if dest.stat().st_size < 1024 * 1024:
                    raise ValueError('Model download unexpectedly small: ' + name)
                manifest[name] = {'source': url, 'sha512': digest.hexdigest(), 'bytes': dest.stat().st_size}
                print('Downloaded:', name, flush=True)
                break
            except Exception:
                dest.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    # Recorded digests aid diagnosis; not claimed to be vendor-verified checksums.
    (ROOT / 'model-manifest.json').write_text(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
