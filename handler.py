import runpod
import base64
import uuid
import tempfile
import subprocess
import os

import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline

FPS = 12
DURATION_S = 8
NUM_FRAMES = FPS * DURATION_S
RESOLUTION = (576, 1024)  # 9:16

pipe = None  # lazy load


def run_ffmpeg(frames_dir, output_mp4):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", f"{frames_dir}/%03d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_mp4
    ]
    subprocess.run(cmd, check=True)


def get_pipe():
    global pipe
    if pipe is None:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16
        ).to("cuda")
    return pipe


def handler(event):
    # ✅ Healthcheck do Hub (sem GPU/modelo)
    if not event or event == {} or event.get("healthcheck") is True:
        return {"ok": True}

    # prompt vem direto no event (mais simples)
    prompt = event.get("prompt", "cinematic scene")

    # (por enquanto) frame inicial preto
    image = Image.new("RGB", RESOLUTION, color=(0, 0, 0))

    pipe = get_pipe()
    result = pipe(image=image, num_frames=NUM_FRAMES)
    frames = result.frames[0]

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = os.path.join(tmp, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for i, f in enumerate(frames):
            f.save(os.path.join(frames_dir, f"{i:03d}.png"))

        out_mp4 = os.path.join(tmp, f"video_{uuid.uuid4().hex}.mp4")
        run_ffmpeg(frames_dir, out_mp4)

        with open(out_mp4, "rb") as f:
            mp4_bytes = f.read()

    return {
        "ok": True,
        "duration_s": DURATION_S,
        "fps": FPS,
        "resolution": "576x1024",
        "mp4_base64": base64.b64encode(mp4_bytes).decode("utf-8")
    }


runpod.serverless.start({"handler": handler})
