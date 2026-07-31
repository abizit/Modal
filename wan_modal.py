"""A single-page Modal studio for FLUX.1 Kontext and Wan2.2."""

import io
import math
import uuid
from pathlib import Path

import modal

APP_NAME = "motion-and-kontext-studio"
WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
HF_CACHE_PATH = "/root/.cache/huggingface"
OUTPUT_PATH = "/outputs"
ASSET_PATH = "/assets"

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("motion-studio-model-cache", create_if_missing=True)
outputs = modal.Volume.from_name("motion-studio-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "accelerate>=1.0",
        "fastapi>=0.115",
        "huggingface_hub[hf_xet]>=0.30",
        "imageio>=2.37",
        "imageio-ffmpeg>=0.6",
        "numpy>=1.26",
        "pillow>=10.0",
        "python-multipart>=0.0.20",
        "safetensors>=0.5",
        "torch>=2.4",
        "transformers>=4.48",
    )
    .pip_install("git+https://github.com/huggingface/diffusers.git")
    .add_local_file("web_ui.html", remote_path=f"{ASSET_PATH}/index.html")
)


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={HF_CACHE_PATH: model_cache},
    timeout=60 * 60,
)
def download_models() -> None:
    """Preload gated FLUX and Wan weights into the persistent Modal Volume."""
    from huggingface_hub import snapshot_download

    for model_id in (WAN_MODEL_ID, FLUX_MODEL_ID):
        snapshot_download(model_id, cache_dir=HF_CACHE_PATH, token=True)
    model_cache.commit()


@app.function(
    image=image,
    gpu="L40S",
    timeout=20 * 60,
    scaledown_window=10 * 60,
    max_containers=1,
    secrets=[hf_secret],
    volumes={HF_CACHE_PATH: model_cache, OUTPUT_PATH: outputs},
)
@modal.asgi_app()
def web_app():
    """Serve the UI and keep only the currently selected model in memory."""
    import torch
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from PIL import Image

    web = FastAPI(title="Motion & Kontext Studio")
    state: dict[str, object | None] = {"mode": None, "pipe": None}

    def unload_pipeline() -> None:
        if state["pipe"] is not None:
            del state["pipe"]
            state["pipe"] = None
            state["mode"] = None
            torch.cuda.empty_cache()

    def get_pipeline(mode: str):
        if state["mode"] == mode:
            return state["pipe"]
        unload_pipeline()
        if mode == "image":
            from diffusers import FluxKontextPipeline

            pipe = FluxKontextPipeline.from_pretrained(
                FLUX_MODEL_ID, torch_dtype=torch.bfloat16
            )
        elif mode == "video":
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

            vae = AutoencoderKLWan.from_pretrained(
                WAN_MODEL_ID, subfolder="vae", torch_dtype=torch.float32
            )
            pipe = WanImageToVideoPipeline.from_pretrained(
                WAN_MODEL_ID, vae=vae, torch_dtype=torch.bfloat16
            )
        else:  # guarded by the route, kept for direct-call safety
            raise ValueError(f"Unsupported mode: {mode}")
        pipe.enable_model_cpu_offload()
        state["mode"] = mode
        state["pipe"] = pipe
        model_cache.commit()
        return pipe

    def video_dimensions(source: Image.Image) -> tuple[int, int]:
        """Preserve image orientation while targeting economical 480p output."""
        aspect = source.width / source.height
        area = 832 * 480
        width = max(32, round(math.sqrt(area * aspect) / 32) * 32)
        height = max(32, round(math.sqrt(area / aspect) / 32) * 32)
        return width, height

    @web.post("/api/generate")
    async def generate(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        mode: str = Form(...),
    ):
        if mode not in {"image", "video"}:
            raise HTTPException(status_code=400, detail="Mode must be image or video.")
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Write an instruction first.")
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=400, detail="Upload a JPEG, PNG, or WebP image.")

        try:
            source = Image.open(io.BytesIO(await image.read())).convert("RGB")
        except Exception as error:
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from error

        pipe = get_pipeline(mode)
        unique_id = uuid.uuid4().hex
        if mode == "image":
            edited = pipe(
                image=source,
                prompt=prompt.strip(),
                guidance_scale=2.5,
                num_inference_steps=28,
            ).images[0]
            filename = f"kontext-{unique_id}.png"
            output = Path(OUTPUT_PATH) / filename
            edited.save(output, "PNG")
            media_type = "image/png"
        else:
            from diffusers.utils import export_to_video

            width, height = video_dimensions(source)
            frames = pipe(
                image=source,
                prompt=prompt.strip(),
                negative_prompt=(
                    "static, flicker, blurry, low quality, subtitles, watermark, "
                    "deformed, distorted limbs, extra fingers"
                ),
                width=width,
                height=height,
                num_frames=81,
                num_inference_steps=30,
                guidance_scale=5.0,
                generator=torch.Generator(device="cuda").manual_seed(torch.seed()),
            ).frames[0]
            filename = f"wan22-{unique_id}.mp4"
            output = Path(OUTPUT_PATH) / filename
            export_to_video(frames, str(output), fps=24)
            media_type = "video/mp4"

        outputs.commit()
        return FileResponse(output, media_type=media_type, filename=filename)

    # Register this catch-all mount after API routes.
    web.mount("/", StaticFiles(directory=ASSET_PATH, html=True), name="ui")
    return web
