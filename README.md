# Avatar Worker V1.1

Standalone RunPod serverless worker for Full Pipeline V1.1 / input contract 1.2.
Read **HUONG_DAN_CONNECT.md** for the Vietnamese GitHub → RunPod setup steps.

- SadTalker 512px checkpoint, crop portrait, batch size 1, no still mode.
- Natural / calm / expressive presets; optional pose and blinking reference video.
- One image/audio job per request; the desktop app submits avatar scenes sequentially.
- Up to 120 narration characters; no hard ten-second audio gate.
- Base64 JPEG/PNG + PCM WAV in, 720p H.264/AAC at 25 fps out.
- Full Pipeline places clips on its 30 fps timeline at normal playback speed.

The Dockerfile pins the inherited upstream commit and dependencies. Downloaded checkpoint digests are recorded for troubleshooting. Model license and notices remain those of the upstream project: https://github.com/OpenTalker/SadTalker .

This source folder is ready to upload as a new repository. It is not evidence that a remote repository or RunPod endpoint has already been deployed.
