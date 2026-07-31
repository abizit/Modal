"""A single-page Modal studio for FLUX.1 Kontext and Wan2.2."""

import io
import hmac
import math
import os
import uuid
from pathlib import Path

import modal

APP_NAME = "motion-and-kontext-studio"
WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
QWEN_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
QWEN_IDENTITY_LORA_ID = "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"
HF_CACHE_PATH = "/root/.cache/huggingface"
OUTPUT_PATH = "/outputs"
ASSET_PATH = "/assets"
ASPECT_RATIOS = {
    "source": None,
    "1:1": 1.0,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
}
IMAGE_QUALITY = {
    "draft": (768 * 768, 20),
    "standard": (1024 * 1024, 28),
    "high": (1536 * 1536, 40),
}
VIDEO_QUALITY = {
    "draft": (480 * 832, 49, 20),
    "standard": (480 * 832, 81, 30),
    "high": (704 * 1280, 121, 40),
}

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("motion-studio-model-cache", create_if_missing=True)
outputs = modal.Volume.from_name("motion-studio-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
studio_auth_secret = modal.Secret.from_name(
    "studio-auth", required_keys=["APP_USERNAME", "APP_PASSWORD", "APP_SESSION_SECRET"]
)

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
        "peft>=0.14",
        "pillow>=10.0",
        "python-multipart>=0.0.20",
        "safetensors>=0.5",
        "torch>=2.4",
        "torchvision>=0.19",
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
    """Preload all model weights into the persistent Modal Volume."""
    from huggingface_hub import snapshot_download

    for model_id in (WAN_MODEL_ID, FLUX_MODEL_ID, QWEN_MODEL_ID, QWEN_IDENTITY_LORA_ID):
        snapshot_download(model_id, cache_dir=HF_CACHE_PATH, token=True)
    model_cache.commit()


@app.function(
    image=image,
    gpu="H100",
    timeout=20 * 60,
    scaledown_window=10 * 60,
    max_containers=1,
    secrets=[hf_secret, studio_auth_secret],
    volumes={HF_CACHE_PATH: model_cache, OUTPUT_PATH: outputs},
)
@modal.asgi_app()
def web_app():
    """Serve the UI and keep only the currently selected model in memory."""
    import torch
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from PIL import Image
    from starlette.middleware.sessions import SessionMiddleware

    web = FastAPI(title="Motion & Kontext Studio")
    state: dict[str, object | None] = {"mode": None, "pipe": None}

    @web.middleware("http")
    async def require_login(request: Request, call_next):
        if request.url.path == "/login":
            return await call_next(request)
        if request.session.get("studio_user"):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Sign in before generating."}, status_code=401)
        return RedirectResponse(url="/login", status_code=303)

    # Add this after the auth middleware so session parsing happens first.
    web.add_middleware(
        SessionMiddleware,
        secret_key=os.environ["APP_SESSION_SECRET"],
        https_only=True,
        same_site="lax",
        max_age=60 * 60 * 12,
    )

    @web.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        message = "Incorrect username or password." if request.query_params.get("error") else ""
        return HTMLResponse(
            f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sign in · Motion & Kontext</title><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&family=Manrope:wght@400;500;600&display=swap" rel="stylesheet"><style>:root{{--ink:#20221e;--paper:#eeece3;--acid:#d7ff2f}}*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;background:var(--paper);color:var(--ink);display:grid;place-items:center;font-family:Manrope,sans-serif}}body:before{{content:"";position:fixed;inset:0;opacity:.25;background-image:radial-gradient(#5f625a .55px,transparent .65px);background-size:5px 5px;pointer-events:none}}main{{position:relative;width:min(460px,calc(100% - 32px));border:1px solid var(--ink);padding:30px;background:#f7f6ef;box-shadow:8px 8px 0 var(--ink)}}.mark{{display:block;font:500 11px 'DM Mono',monospace;letter-spacing:.08em;text-transform:uppercase;margin-bottom:52px}}h1{{font:normal 56px/.9 'Instrument Serif',serif;letter-spacing:-.05em;margin:0 0 14px}}p{{font-size:14px;line-height:1.6;color:#666960;margin:0 0 28px}}label{{display:block;font:500 11px 'DM Mono',monospace;letter-spacing:.06em;text-transform:uppercase;margin:18px 0 8px}}input{{width:100%;border:1px solid #c9c9bb;padding:13px;background:#fffef9;font:15px Manrope,sans-serif}}button{{width:100%;border:1px solid var(--ink);padding:14px;margin-top:26px;background:var(--acid);font:600 12px 'DM Mono',monospace;letter-spacing:.05em;text-transform:uppercase;cursor:pointer}}button:hover{{background:var(--ink);color:var(--acid)}}.error{{min-height:18px;margin-top:12px;color:#a12d22;font:11px 'DM Mono',monospace}}</style></head><body><main><span class="mark">Motion & Kontext / Private studio</span><h1>Enter the<br><i>studio.</i></h1><p>Sign in to access the generation workspace.</p><form method="post" action="/login"><label for="username">Username</label><input id="username" name="username" autocomplete="username" required autofocus><label for="password">Password</label><input id="password" name="password" type="password" autocomplete="current-password" required><button type="submit">Sign in ↗</button><div class="error">{message}</div></form></main></body></html>'''
        )

    @web.post("/login")
    async def login(request: Request, username: str = Form(...), password: str = Form(...)):
        is_valid = hmac.compare_digest(username, os.environ["APP_USERNAME"]) and hmac.compare_digest(
            password, os.environ["APP_PASSWORD"]
        )
        if not is_valid:
            return RedirectResponse(url="/login?error=1", status_code=303)
        request.session["studio_user"] = username
        return RedirectResponse(url="/", status_code=303)

    @web.post("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse(url="/login", status_code=303)

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
        if mode == "kontext":
            from diffusers import FluxKontextPipeline

            pipe = FluxKontextPipeline.from_pretrained(
                FLUX_MODEL_ID, torch_dtype=torch.bfloat16
            )
        elif mode == "qwen":
            from diffusers import QwenImageEditPlusPipeline

            pipe = QwenImageEditPlusPipeline.from_pretrained(
                QWEN_MODEL_ID, torch_dtype=torch.bfloat16
            )
            pipe.load_lora_weights(QWEN_IDENTITY_LORA_ID)
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
        # The H100 has enough VRAM for the image editors; Wan benefits from
        # offload because its video decoder has a larger temporary memory peak.
        if mode == "video":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        state["mode"] = mode
        state["pipe"] = pipe
        return pipe

    def dimensions_for_aspect(
        source: Image.Image, aspect_ratio: str, target_area: int, multiple: int
    ) -> tuple[int, int]:
        aspect = ASPECT_RATIOS[aspect_ratio] or source.width / source.height
        width = max(multiple, round(math.sqrt(target_area * aspect) / multiple) * multiple)
        height = max(multiple, round(math.sqrt(target_area / aspect) / multiple) * multiple)
        return width, height

    @web.post("/api/generate")
    async def generate(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        mode: str = Form(...),
        aspect_ratio: str = Form("source"),
        quality: str = Form("standard"),
    ):
        if mode not in {"kontext", "qwen", "video"}:
            raise HTTPException(status_code=400, detail="Choose Kontext, Qwen, or video mode.")
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Write an instruction first.")
        if aspect_ratio not in ASPECT_RATIOS:
            raise HTTPException(status_code=400, detail="Choose a supported aspect ratio.")
        if quality not in IMAGE_QUALITY:
            raise HTTPException(status_code=400, detail="Choose draft, standard, or high quality.")
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=400, detail="Upload a JPEG, PNG, or WebP image.")

        try:
            source = Image.open(io.BytesIO(await image.read())).convert("RGB")
        except Exception as error:
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from error

        pipe = get_pipeline(mode)
        unique_id = uuid.uuid4().hex
        if mode == "kontext":
            area, steps = IMAGE_QUALITY[quality]
            width, height = dimensions_for_aspect(source, aspect_ratio, area, multiple=16)
            flux_negative_prompt = negative_prompt.strip()
            edited = pipe(
                image=source,
                prompt=prompt.strip(),
                negative_prompt=flux_negative_prompt or None,
                true_cfg_scale=1.5 if flux_negative_prompt else 1.0,
                width=width,
                height=height,
                guidance_scale=2.5,
                num_inference_steps=steps,
            ).images[0]
            filename = f"kontext-{unique_id}.png"
            output = Path(OUTPUT_PATH) / filename
            edited.save(output, "PNG")
            media_type = "image/png"
        elif mode == "qwen":
            area, steps = IMAGE_QUALITY[quality]
            width, height = dimensions_for_aspect(source, aspect_ratio, area, multiple=16)
            edited = pipe(
                image=source,
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip() or " ",
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=torch.Generator(device="cuda").manual_seed(torch.seed()),
            ).images[0]
            filename = f"qwen-identity-{unique_id}.png"
            output = Path(OUTPUT_PATH) / filename
            edited.save(output, "PNG")
            media_type = "image/png"
        else:
            from diffusers.utils import export_to_video

            area, num_frames, steps = VIDEO_QUALITY[quality]
            width, height = dimensions_for_aspect(source, aspect_ratio, area, multiple=32)
            frames = pipe(
                image=source,
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip()
                or "static, flicker, blurry, low quality, subtitles, watermark, "
                "deformed, distorted limbs, extra fingers",
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=5.0,
                generator=torch.Generator(device="cuda").manual_seed(torch.seed()),
            ).frames[0]
            filename = f"wan22-{unique_id}.mp4"
            output = Path(OUTPUT_PATH) / filename
            export_to_video(frames, str(output), fps=24)
            media_type = "video/mp4"

        await outputs.commit.aio()
        return FileResponse(output, media_type=media_type, filename=filename)

    # Register this catch-all mount after API routes.
    web.mount("/", StaticFiles(directory=ASSET_PATH, html=True), name="ui")
    return web
