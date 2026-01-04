import runpod
import base64
import uuid
import tempfile
import subprocess
import os

from PIL import Image
import torch
from diffusers import StableVideoDiffusionPipeline

# =========================
# CONFIG FIXA DO PROJETO
# =========================
FPS = 12
DURATION_S = 8
NUM_FRAMES = FPS * DURATION_S
RESOLUTION = (576, 1024)  # 9:16 vertical

pipe = None  # lazy load


# =========================
# PIPELINE
# =========================
def get_pipe():
    global pipe
    if pipe is None:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16
        )
        pipe.to("cuda")
    return pipe


# =========================
# FFMPEG
# =========================
def run_ffmpeg(frames_dir, output_mp4):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(FPS),
        "-i", f"{frames_dir}/%03d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_mp4
    ]
    subprocess.run(cmd, check=True)


# =========================
# HANDLER
# =========================
def handler(event):
    pipe = get_pipe()

    prompt = event.get("input", {}).get("prompt", "cinematic scene")

    result = pipe(
        prompt=prompt,
        num_frames=NUM_FRAMES,
        height=RESOLUTION[1],
        width=RESOLUTION[0]
    )

    frames = result.frames[0]

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = os.path.join(tmp, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for i, frame in enumerate(frames):
            frame.save(os.path.join(frames_dir, f"{i:03d}.png"))

        output_mp4 = os.path.join(tmp, f"video_{uuid.uuid4().hex}.mp4")
        run_ffmpeg(frames_dir, output_mp4)

        with open(output_mp4, "rb") as f:
            mp4_base64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "ok": True,
        "duration_s": DURATION_S,
        "fps": FPS,
        "resolution": "576x1024",
        "video_base64": mp4_base64
    }


runpod.serverless.start({"handler": handler})
