# Wan2.2 Video Service

This container runs Wan2.2 locally on GPU using the official `generate.py` entry point.

## Quickstart

Build the image from the repo root:

```bash
docker build -t wan-video-service assets/video-service
```

Run with GPU access and a mounted model cache:

```bash
docker run --rm --gpus all \
  -e HF_TOKEN="<your_hf_token>" \
  -e WAN_PRECACHE=true \
  -e ENABLE_VOICE_TO_VIDEO=0 \
  -p 8081:8081 \
  -v $(pwd)/models:/models \
  wan-video-service
```

Then generate a video:

```bash
curl -X POST http://localhost:8081/generate_video \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cinematic drone flyover of snow-capped mountains at sunrise"}'
```

The response returns a Base64 MP4 data URI plus a downloadable filename.

## Notes

Voice-to-video and other audio-conditioned flows are disabled by default. Set `ENABLE_VOICE_TO_VIDEO=1` only if you provide the
required optional audio dependencies.
