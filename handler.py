import runpod
import base64
import uuid
import tempfile
import subprocess
import os

from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import torch


# ===== CONFIG =====
MODEL_PATH = "/models/svd"
FPS = 12
DURATION_S = 8
NUM_FRAMES = FPS * DURATION_S
RESOLUTION = (576, 1024)  # 9:16


def run_ffmpeg(frames_dir, output_mp4):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "f_%03d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_mp4
    ]
    subprocess.run(cmd, check=True)


def handler(event):
    image_b64 = event["input"].get("image_base64")
    if not image_b64:
        return {"error": "image_base64 is required"}

    image_bytes = base64.b64decode(image_b64)
    img = Image.open(tempfile.NamedTemporaryFile(delete=False))
    img.file.write(image_bytes)
    img.file.flush()

    image = Image.open(img.name).convert("RGB").resize(RESOLUTION)

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    pipe.enable_attention_slicing()

    result = pipe(
        image=image,
        num_frames=NUM_FRAMES,
        decode_chunk_size=1,
        motion_bucket_id=90,
        noise_aug_strength=0.02,
    )

    frames = result.frames[0]

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = os.path.join(tmp, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for i, f in enumerate(frames):
            f.save(os.path.join(frames_dir, f"f_{i:03d}.png"))

        output_mp4 = os.path.join(tmp, f"video_{uuid.uuid4().hex}.mp4")
        run_ffmpeg(frames_dir, output_mp4)

        with open(output_mp4, "rb") as f:
            mp4_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "ok": True,
        "duration": DURATION_S,
        "fps": FPS,
        "resolution": "576x1024",
        "video_base64": mp4_b64
    }


runpod.serverless.start({"handler": handler})
