"""A single-page Modal studio for FLUX.1 Kontext and Wan2.2."""

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
VIDEO_MOTION_MODES = {"subtle", "normal", "major"}
VIDEO_STABILITY_NEGATIVE = (
    "static, flicker, blurry, low quality, subtitles, watermark, body morphing, body inflation, "
    "stretching body, changing body proportions, distorted anatomy, warped limbs, extra limbs, "
    "duplicate hands, malformed hands, warped face, melted face, sudden pose change, camera jump"
)
VIDEO_MOTION_DIRECTIONS = {
    "subtle": (
        "Use only subtle, realistic motion such as breathing, a gentle head movement, or a small weight "
        "shift. Keep the starting pose and body position essentially unchanged."
    ),
    "normal": (
        "Perform the requested motion naturally and continuously. Keep realistic body volume, stable anatomy, "
        "and consistent proportions throughout the entire clip."
    ),
    "major": (
        "Treat the requested action as one continuous, physically plausible human movement. Use natural joint "
        "articulation and stable body volume from start to finish. Do not inflate, stretch, melt, duplicate, "
        "or deform any part of the subject while performing the action."
    ),
}
IDENTITY_LOCKS = {
    "qwen": (
        "Identity preservation is mandatory. Treat each discernible person in the supplied references as "
        "a separate identity. Preserve every person's exact facial identity, facial structure, skin tone, "
        "and distinguishing facial features. If the requested image contains multiple people, retain each "
        "person's face separately. Do not merge, swap, substitute, redesign, or blend faces. Preserve the "
        "reference pose, body position, camera framing, composition, scene, and location by default. Change "
        "those only when the creative direction explicitly requests a change. Clothing, hairstyle, and body "
        "styling may follow the creative direction."
    ),
    "kontext": (
        "Identity preservation is mandatory. Preserve the source subject's exact facial identity, facial "
        "structure, skin tone, and distinguishing facial features. Do not create a new face. Preserve the "
        "reference pose, body position, camera framing, composition, scene, and location by default. Change "
        "those only when the creative direction explicitly requests a change. Clothing, hairstyle, and body "
        "styling may follow the creative direction."
    ),
    "video": (
        "Identity preservation is mandatory. Keep the source subject's exact facial identity, facial "
        "structure, skin tone, and distinguishing facial features stable throughout every frame. Do not "
        "morph, swap, or introduce another face. Preserve the reference pose, camera framing, composition, "
        "scene, and location by default. Change motion, pose, framing, scene, or location only when the "
        "creative direction explicitly requests a change. Clothing, hairstyle, and body styling may follow "
        "the creative direction."
    ),
}
ANATOMY_GUARDRAIL = (
    "extra limbs, extra arms, extra legs, extra hands, duplicate limbs, duplicate hands, "
    "extra fingers, fused fingers, missing fingers, malformed hands, malformed feet, "
    "deformed anatomy, distorted body, unnatural body proportions"
)

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("motion-studio-model-cache", create_if_missing=True)
outputs = modal.Volume.from_name("motion-studio-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
studio_auth_secret = modal.Secret.from_name(
    "studio-auth", required_keys=["APP_USERNAME", "APP_PASSWORD", "APP_SESSION_SECRET"]
)
mega_secret = modal.Secret.from_name(
    "mega-credentials", required_keys=["MEGA_EMAIL", "MEGA_PASSWORD"]
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("nodejs", "npm")
    .pip_install(
        "fastapi>=0.115",
        "itsdangerous>=2.2",
        "python-multipart>=0.0.20",
    )
    .run_commands("mkdir -p /opt/mega && npm install --prefix /opt/mega megajs@^1.3.0")
    .add_local_file("web_ui.html", remote_path=f"{ASSET_PATH}/index.html")
    .add_local_file("mega_upload.js", remote_path=f"{ASSET_PATH}/mega_upload.js")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "accelerate>=1.0",
        "ftfy>=6.3",
        "huggingface_hub[hf_xet]>=0.30",
        "imageio>=2.37",
        "imageio-ffmpeg>=0.6",
        "numpy>=1.26",
        "peft>=0.14",
        "pillow>=10.0",
        "safetensors>=0.5",
        "torch>=2.4",
        "torchvision>=0.19",
        "transformers>=4.48",
    )
    .pip_install("git+https://github.com/huggingface/diffusers.git")
)


@app.function(
    image=gpu_image,
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


image_worker_state: dict[str, object | None] = {"mode": None, "pipe": None}
video_worker_state: dict[str, object | None] = {"pipe": None}


def dimensions_for_aspect(source, aspect_ratio: str, target_area: int, multiple: int) -> tuple[int, int]:
    aspect = ASPECT_RATIOS[aspect_ratio] or source.width / source.height
    width = max(multiple, round(math.sqrt(target_area * aspect) / multiple) * multiple)
    height = max(multiple, round(math.sqrt(target_area / aspect) / multiple) * multiple)
    return width, height


def prompt_with_identity_lock(mode: str, creative_prompt: str) -> str:
    return f"{IDENTITY_LOCKS[mode]}\n\nCreative direction: {creative_prompt.strip()}"


def negative_prompt_with_anatomy_guardrail(creative_negative_prompt: str, enabled: bool) -> str:
    user_negative = creative_negative_prompt.strip()
    if not enabled:
        return user_negative or " "
    return f"{ANATOMY_GUARDRAIL}, {user_negative}" if user_negative else ANATOMY_GUARDRAIL


def output_metadata(
    filename, media_type, mode, prompt, negative_prompt, aspect_ratio, quality, count, anatomy_guardrail,
    motion_mode=None,
):
    import json
    from datetime import datetime, timezone

    metadata = {
        "filename": filename,
        "media_type": media_type,
        "mode": mode,
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "aspect_ratio": aspect_ratio,
        "quality": quality,
        "reference_count": count,
        "identity_lock": True,
        "anatomy_guardrail": anatomy_guardrail,
        "motion_mode": motion_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (Path(OUTPUT_PATH) / filename).with_suffix(".json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return metadata


def run_image_generation(
    image_data: list[bytes], prompt: str, negative_prompt: str, mode: str, aspect_ratio: str, quality: str,
    anatomy_guardrail: bool,
) -> dict:
    """Shared image-worker implementation; called inside the selected GPU container."""
    import io

    import torch
    from PIL import Image

    sources = [Image.open(io.BytesIO(data)).convert("RGB") for data in image_data]
    source = sources[0]
    if image_worker_state["mode"] != mode:
        if image_worker_state["pipe"] is not None:
            del image_worker_state["pipe"]
            torch.cuda.empty_cache()
        if mode == "kontext":
            from diffusers import FluxKontextPipeline

            pipe = FluxKontextPipeline.from_pretrained(FLUX_MODEL_ID, torch_dtype=torch.bfloat16)
        elif mode == "qwen":
            from diffusers import QwenImageEditPlusPipeline

            pipe = QwenImageEditPlusPipeline.from_pretrained(QWEN_MODEL_ID, torch_dtype=torch.bfloat16)
            pipe.load_lora_weights(QWEN_IDENTITY_LORA_ID)
        else:
            raise ValueError(f"Unsupported image mode: {mode}")
        if mode == "kontext":
            # Keep the 12B Kontext pipeline within the L40S's 48 GB VRAM budget.
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        image_worker_state.update(mode=mode, pipe=pipe)

    pipe = image_worker_state["pipe"]
    model_prompt = prompt_with_identity_lock(mode, prompt)
    guarded_negative_prompt = negative_prompt_with_anatomy_guardrail(
        negative_prompt, anatomy_guardrail
    )
    area, steps = IMAGE_QUALITY[quality]
    width, height = dimensions_for_aspect(source, aspect_ratio, area, multiple=16)
    unique_id = uuid.uuid4().hex
    if mode == "kontext":
        generated = pipe(
            image=source,
            prompt=model_prompt,
            negative_prompt=guarded_negative_prompt.strip() or None,
            true_cfg_scale=1.5 if guarded_negative_prompt.strip() else 1.0,
            width=width,
            height=height,
            guidance_scale=2.5,
            num_inference_steps=steps,
        ).images[0]
        filename = f"kontext-{unique_id}.png"
    else:
        generated = pipe(
            image=sources,
            prompt=model_prompt,
            negative_prompt=guarded_negative_prompt,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.Generator(device="cuda").manual_seed(torch.seed()),
        ).images[0]
        filename = f"qwen-identity-{unique_id}.png"
    generated.save(Path(OUTPUT_PATH) / filename, "PNG")
    metadata = output_metadata(
        filename, "image/png", mode, prompt, negative_prompt, aspect_ratio, quality, len(sources), anatomy_guardrail
    )
    outputs.commit()
    return metadata


@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=20 * 60,
    scaledown_window=90,
    max_containers=1,
    secrets=[hf_secret],
    volumes={HF_CACHE_PATH: model_cache, OUTPUT_PATH: outputs},
)
def generate_qwen(
    image_data: list[bytes], prompt: str, negative_prompt: str, aspect_ratio: str, quality: str,
    anatomy_guardrail: bool,
) -> dict:
    """Identity editing is the sole H100 workload."""
    return run_image_generation(
        image_data, prompt, negative_prompt, "qwen", aspect_ratio, quality, anatomy_guardrail
    )


@app.function(
    image=gpu_image,
    gpu="L40S",
    timeout=20 * 60,
    scaledown_window=90,
    max_containers=1,
    secrets=[hf_secret],
    volumes={HF_CACHE_PATH: model_cache, OUTPUT_PATH: outputs},
)
def generate_kontext(
    image_data: list[bytes], prompt: str, negative_prompt: str, aspect_ratio: str, quality: str,
    anatomy_guardrail: bool,
) -> dict:
    """Run Kontext economically on L40S with CPU offloading."""
    return run_image_generation(
        image_data, prompt, negative_prompt, "kontext", aspect_ratio, quality, anatomy_guardrail
    )


@app.function(
    image=gpu_image,
    gpu="L40S",
    timeout=20 * 60,
    scaledown_window=90,
    max_containers=1,
    secrets=[hf_secret],
    volumes={HF_CACHE_PATH: model_cache, OUTPUT_PATH: outputs},
)
def generate_video(
    image_data: bytes, prompt: str, negative_prompt: str, aspect_ratio: str, quality: str,
    anatomy_guardrail: bool, motion_mode: str, variation_count: int,
) -> dict:
    """Run Wan on the lower-cost GPU worker, isolated from the web service."""
    import io

    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    source = Image.open(io.BytesIO(image_data)).convert("RGB")
    model_prompt = (
        f"{prompt_with_identity_lock('video', prompt)}\n\n"
        f"Motion direction: {VIDEO_MOTION_DIRECTIONS[motion_mode]}"
    )
    guarded_negative_prompt = negative_prompt_with_anatomy_guardrail(
        negative_prompt, anatomy_guardrail
    )
    if video_worker_state["pipe"] is None:
        vae = AutoencoderKLWan.from_pretrained(
            WAN_MODEL_ID, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_ID, vae=vae, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        video_worker_state["pipe"] = pipe

    area, num_frames, steps = VIDEO_QUALITY[quality]
    if motion_mode == "major":
        # Major pose changes benefit from more denoising iterations; cap duration so the model has less
        # opportunity to drift after completing the action.
        num_frames = min(num_frames, 81)
        steps = max(steps, 50)
    width, height = dimensions_for_aspect(source, aspect_ratio, area, multiple=32)
    variants = []
    for _ in range(variation_count):
        frames = video_worker_state["pipe"](
            image=source,
            prompt=model_prompt,
            negative_prompt=f"{guarded_negative_prompt}, {VIDEO_STABILITY_NEGATIVE}",
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=5.0,
            generator=torch.Generator(device="cuda").manual_seed(torch.seed()),
        ).frames[0]
        filename = f"wan22-{uuid.uuid4().hex}.mp4"
        export_to_video(frames, str(Path(OUTPUT_PATH) / filename), fps=24)
        variants.append(output_metadata(
            filename, "video/mp4", "video", prompt, negative_prompt, aspect_ratio, quality, 1,
            anatomy_guardrail, motion_mode
        ))
    outputs.commit()
    return {"primary": variants[0], "variants": variants}


@app.function(
    image=web_image,
    max_containers=1,
    secrets=[studio_auth_secret, mega_secret],
    volumes={OUTPUT_PATH: outputs},
)
@modal.asgi_app()
def web_app():
    """Serve the UI and keep only the currently selected model in memory."""
    import asyncio
    import json
    import subprocess
    from datetime import datetime, timezone

    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.middleware.sessions import SessionMiddleware

    web = FastAPI(title="Motion & Kontext Studio")
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

    @web.post("/api/generate")
    async def generate(
        images: list[UploadFile] = File(...),
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        anatomy_guardrail: bool = Form(False),
        mode: str = Form(...),
        aspect_ratio: str = Form("source"),
        quality: str = Form("standard"),
        motion_mode: str = Form("normal"),
        variation_count: int = Form(1),
    ):
        if mode not in {"kontext", "qwen", "video"}:
            raise HTTPException(status_code=400, detail="Choose Kontext, Qwen, or video mode.")
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Write an instruction first.")
        if aspect_ratio not in ASPECT_RATIOS:
            raise HTTPException(status_code=400, detail="Choose a supported aspect ratio.")
        if quality not in IMAGE_QUALITY:
            raise HTTPException(status_code=400, detail="Choose draft, standard, or high quality.")
        if motion_mode not in VIDEO_MOTION_MODES:
            raise HTTPException(status_code=400, detail="Choose subtle, normal, or major motion.")
        if variation_count not in {1, 2, 3}:
            raise HTTPException(status_code=400, detail="Choose one, two, or three variations.")
        if not 1 <= len(images) <= 4:
            raise HTTPException(status_code=400, detail="Upload between one and four reference images.")
        for upload in images:
            if upload.content_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise HTTPException(status_code=400, detail="Upload JPEG, PNG, or WebP reference images.")

        image_data = [await upload.read() for upload in images]
        job_id = uuid.uuid4().hex
        job_path = Path(OUTPUT_PATH) / f"job-{job_id}.json"
        job = {
            "job_id": job_id,
            "status": "queued",
            "mode": mode,
            "anatomy_guardrail": anatomy_guardrail,
            "motion_mode": motion_mode if mode == "video" else None,
            "variation_count": variation_count if mode == "video" else 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        job_path.write_text(json.dumps(job), encoding="utf-8")
        await outputs.commit.aio()
        try:
            if mode == "video":
                call = await generate_video.spawn.aio(
                    image_data[0], prompt, negative_prompt, aspect_ratio, quality, anatomy_guardrail,
                    motion_mode, variation_count,
                )
            elif mode == "qwen":
                call = await generate_qwen.spawn.aio(
                    image_data, prompt, negative_prompt, aspect_ratio, quality, anatomy_guardrail
                )
            else:
                call = await generate_kontext.spawn.aio(
                    image_data, prompt, negative_prompt, aspect_ratio, quality, anatomy_guardrail
                )
        except Exception as error:
            job["status"] = "failed"
            job_path.write_text(json.dumps(job), encoding="utf-8")
            await outputs.commit.aio()
            raise HTTPException(status_code=502, detail="Could not start the generation worker.") from error
        job["call_id"] = call.object_id
        job_path.write_text(json.dumps(job), encoding="utf-8")
        await outputs.commit.aio()
        return job

    @web.get("/api/jobs/{job_id}")
    async def generation_job(job_id: str):
        if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
            raise HTTPException(status_code=400, detail="That is not a valid generation job.")
        job_path = Path(OUTPUT_PATH) / f"job-{job_id}.json"
        if not job_path.is_file():
            raise HTTPException(status_code=404, detail="Generation job not found.")
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job.get("status") == "failed" or not job.get("call_id"):
            return job
        try:
            metadata = await asyncio.to_thread(
                modal.FunctionCall.from_id(job["call_id"]).get, timeout=0
            )
        except TimeoutError:
            job["status"] = "running"
            return job
        except Exception as error:
            print(f"Generation job failed: {error}")
            job["status"] = "failed"
            return job
        await outputs.reload.aio()
        primary_metadata = metadata.get("primary", metadata)
        if not (Path(OUTPUT_PATH) / primary_metadata["filename"]).is_file():
            job["status"] = "running"
            return job
        job.update(status="complete", result=metadata)
        return job

    def output_path(filename: str) -> Path:
        if Path(filename).name != filename or not filename.endswith((".png", ".mp4")):
            raise HTTPException(status_code=400, detail="That is not a valid studio output.")
        output = Path(OUTPUT_PATH) / filename
        if not output.is_file():
            raise HTTPException(status_code=404, detail="The generated file is no longer available.")
        return output

    @web.get("/api/history")
    async def generation_history():
        history = []
        for metadata_file in Path(OUTPUT_PATH).glob("*.json"):
            try:
                entry = json.loads(metadata_file.read_text(encoding="utf-8"))
                filename = entry.get("filename", "")
                if output_path(filename).is_file():
                    history.append(entry)
            except (json.JSONDecodeError, OSError, HTTPException):
                continue
        history.sort(key=lambda entry: entry.get("created_at", ""), reverse=True)
        return {"items": history[:48]}

    @web.get("/api/output/{filename}")
    async def get_output(filename: str):
        output = output_path(filename)
        media_type = "video/mp4" if filename.endswith(".mp4") else "image/png"
        return FileResponse(output, media_type=media_type)

    @web.delete("/api/history/{filename}")
    async def delete_output(filename: str):
        """Permanently remove one generated render and its archive record."""
        output = output_path(filename)
        output.unlink()
        metadata = output.with_suffix(".json")
        if metadata.is_file():
            metadata.unlink()
        await outputs.commit.aio()
        return {"detail": "Generation permanently deleted."}

    @web.post("/api/upload-to-mega")
    async def upload_to_mega(filename: str = Form(...)):
        """Upload a generated output to the private AI generations MEGA folder."""
        output = output_path(filename)
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                ["node", f"{ASSET_PATH}/mega_upload.js", str(output)],
                check=True,
                capture_output=True,
                text=True,
                timeout=15 * 60,
            )
        except subprocess.TimeoutExpired as error:
            raise HTTPException(status_code=504, detail="Mega upload timed out. Please try again.") from error
        except subprocess.CalledProcessError as error:
            diagnostic = (error.stderr or error.stdout or "No diagnostic was returned.").strip()
            print(f"MEGA upload failed: {diagnostic[-1000:]}")
            raise HTTPException(
                status_code=502,
                detail="Mega upload failed. Open the Modal app logs for the MEGA diagnostic.",
            ) from error
        return {"detail": "Uploaded to Mega / AI generations.", "filename": completed.stdout.strip()}

    # Register this catch-all mount after API routes.
    web.mount("/", StaticFiles(directory=ASSET_PATH, html=True), name="ui")
    return web
