"""A single-page Modal studio for FLUX.1 Kontext and Wan2.2."""

import hashlib
import hmac
import json
import math
import os
import uuid
from pathlib import Path

import modal

APP_NAME = "motion-and-kontext-studio"
WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_HIGH_MOTION_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
QWEN_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_NSFW_LORA_ID = "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"
DEFAULT_NSFW_FINAL_MODEL_ID = "SG161222/RealVisXL_V5.0"
DEFAULT_NSFW_FINAL_MODEL_FILE = ""
DEFAULT_NSFW_FACE_ADAPTER_ID = "h94/IP-Adapter"
DEFAULT_NSFW_FACE_ADAPTER_SUBFOLDER = "sdxl_models"
DEFAULT_NSFW_FACE_ADAPTER_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
HF_CACHE_PATH = "/root/.cache/huggingface"
OUTPUT_PATH = "/outputs"
ADAPTERS_PATH = "/adapters"
FACE_MODELS_PATH = "/face-models"
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
adapters = modal.Volume.from_name("motion-studio-adapters", create_if_missing=True)
face_models = modal.Volume.from_name("motion-studio-face-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
studio_auth_secret = modal.Secret.from_name(
    "studio-auth", required_keys=["APP_USERNAME", "APP_PASSWORD", "APP_SESSION_SECRET"]
)
mega_secret = modal.Secret.from_name(
    "mega-credentials", required_keys=["MEGA_EMAIL", "MEGA_PASSWORD"]
)
adapter_config_secret = modal.Secret.from_name("qwen-adapter-config")

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "nodejs", "npm")
    .pip_install(
        "fastapi>=0.115",
        "itsdangerous>=2.2",
        "pillow>=10.0",
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
        "insightface>=0.7.3",
        "ftfy>=6.3",
        "huggingface_hub[hf_xet]>=0.30",
        "imageio>=2.37",
        "imageio-ffmpeg>=0.6",
        "numpy>=1.26",
        "onnxruntime-gpu>=1.20",
        "opencv-python-headless>=4.10",
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
    secrets=[hf_secret, adapter_config_secret],
    volumes={HF_CACHE_PATH: model_cache},
    timeout=60 * 60,
)
def download_models() -> None:
    """Preload all model weights into the persistent Modal Volume."""
    from huggingface_hub import snapshot_download

    for model_id in (
        WAN_MODEL_ID, WAN_HIGH_MOTION_MODEL_ID, FLUX_MODEL_ID, QWEN_MODEL_ID,
        configured_nsfw_final_model_id(),
    ):
        snapshot_download(model_id, cache_dir=HF_CACHE_PATH, token=True)
    snapshot_download(configured_nsfw_face_adapter_id(), cache_dir=HF_CACHE_PATH, token=True)
    model_cache.commit()


def adapter_specs_from_environment(name: str, fallback: list[dict] | None = None) -> list[dict]:
    """Read swappable LoRA settings from a JSON environment variable."""
    # Modal secrets can preserve a copied line break inside a long JSON value. It is not meaningful in
    # an adapter specification, so remove line breaks before parsing rather than failing a worker startup.
    raw_specs = os.environ.get(name, "").replace("\r", "").replace("\n", "").strip()
    if not raw_specs:
        return list(fallback or [])
    try:
        specs = json.loads(raw_specs)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must contain a JSON list of LoRA specifications.") from error
    if not isinstance(specs, list):
        raise ValueError(f"{name} must contain a JSON list of LoRA specifications.")
    normalized = []
    for spec in specs:
        if not isinstance(spec, dict) or not isinstance(spec.get("source"), str) or not spec["source"].strip():
            raise ValueError(f"Each {name} entry needs a non-empty source.")
        strength = spec.get("strength", 1.0)
        if not isinstance(strength, (int, float)) or not 0 <= strength <= 2:
            raise ValueError(f"Each {name} strength must be between 0 and 2.")
        source = spec["source"].strip()
        # A Hugging Face repo ID never contains whitespace. Removing copied indentation lets a JSON
        # secret wrap a long repo ID across lines while preserving valid spaces in local file paths.
        if not source.startswith("/"):
            source = "".join(source.split())
        normalized.append({
            "source": source,
            "weight_name": spec.get("weight_name") or None,
            "revision": spec.get("revision") or None,
            "strength": float(strength),
        })
    return normalized


def configured_nsfw_loras() -> list[dict]:
    return adapter_specs_from_environment(
        "QWEN_NSFW_LORA_SPECS", [{"source": DEFAULT_NSFW_LORA_ID, "strength": 1.0}]
    )


def configured_realism_loras() -> list[dict]:
    return adapter_specs_from_environment("QWEN_REALISM_LORA_SPECS")


def configured_nsfw_final_loras() -> list[dict]:
    """Optional SDXL LoRAs used only by the RealVisXL final pass."""
    return adapter_specs_from_environment("NSFW_FINAL_LORA_SPECS")


def configured_nsfw_final_model_id() -> str:
    model_id = os.environ.get("NSFW_FINAL_MODEL_ID", DEFAULT_NSFW_FINAL_MODEL_ID).strip()
    if not model_id or any(character.isspace() for character in model_id):
        raise ValueError("NSFW_FINAL_MODEL_ID must be a Hugging Face repository ID without whitespace.")
    return model_id


def configured_nsfw_final_model_file() -> str:
    return os.environ.get("NSFW_FINAL_MODEL_FILE", DEFAULT_NSFW_FINAL_MODEL_FILE).strip()


def configured_nsfw_face_adapter_id() -> str:
    adapter_id = os.environ.get("NSFW_FINAL_FACE_ADAPTER_ID", DEFAULT_NSFW_FACE_ADAPTER_ID).strip()
    if not adapter_id or any(character.isspace() for character in adapter_id):
        raise ValueError("NSFW_FINAL_FACE_ADAPTER_ID must be a Hugging Face repository ID without whitespace.")
    return adapter_id


def configured_nsfw_face_adapter_subfolder() -> str:
    return os.environ.get("NSFW_FINAL_FACE_ADAPTER_SUBFOLDER", DEFAULT_NSFW_FACE_ADAPTER_SUBFOLDER).strip()


def configured_nsfw_face_adapter_weight() -> str:
    return os.environ.get("NSFW_FINAL_FACE_ADAPTER_WEIGHT", DEFAULT_NSFW_FACE_ADAPTER_WEIGHT).strip()


def adapter_cache_path(source: str) -> Path:
    return Path(ADAPTERS_PATH) / hashlib.sha256(source.encode("utf-8")).hexdigest()


def resolve_adapter_source(spec: dict) -> str:
    """Resolve a mounted adapter file or cache a Hugging Face adapter repository on the adapter Volume."""
    from huggingface_hub import snapshot_download

    source = spec["source"]
    source_path = Path(source)
    if source_path.is_absolute():
        adapter_root = Path(ADAPTERS_PATH).resolve()
        try:
            source_path.resolve().relative_to(adapter_root)
        except ValueError as error:
            raise ValueError("Adapter paths must be inside the mounted /adapters volume.") from error
        if not source_path.exists():
            raise ValueError(f"Configured adapter path does not exist: {source}")
        return str(source_path)
    destination = adapter_cache_path(source)
    if not destination.exists():
        print(f"Downloading Qwen adapter: {source}")
        snapshot_download(
            source,
            local_dir=destination,
            revision=spec.get("revision"),
            token=True,
        )
        adapters.commit()
    return str(destination)


@app.function(
    image=gpu_image,
    secrets=[hf_secret, adapter_config_secret],
    volumes={ADAPTERS_PATH: adapters},
    timeout=60 * 60,
)
def download_adapters() -> None:
    """Preload configured Qwen and SDXL adapters into the persistent adapter Volume."""
    for spec in configured_nsfw_loras() + configured_realism_loras() + configured_nsfw_final_loras():
        resolve_adapter_source(spec)
    adapters.commit()


@app.function(
    image=gpu_image,
    volumes={FACE_MODELS_PATH: face_models},
    timeout=20 * 60,
)
def download_face_models() -> None:
    """Preload InsightFace detection weights used to protect every face in RealVisXL final passes."""
    from insightface.app import FaceAnalysis

    print("Downloading InsightFace buffalo_l models for multi-person identity protection.")
    face_analyser = FaceAnalysis(
        name="buffalo_l", root=FACE_MODELS_PATH, providers=["CPUExecutionProvider"]
    )
    face_analyser.prepare(ctx_id=-1, det_size=(640, 640))
    face_models.commit()


image_worker_state: dict[str, object | None] = {"mode": None, "pipe": None, "loaded_loras": {}}
nsfw_final_worker_state: dict[str, object | None] = {"pipe": None, "loaded_loras": {}}
video_worker_state: dict[str, object | None] = {"pipe": None}
high_motion_video_worker_state: dict[str, object | None] = {"pipe": None}


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
    motion_mode=None, nsfw_enabled=False, realism_pass=False, nsfw_final_engine=None, video_engine=None,
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
        "nsfw_enabled": nsfw_enabled,
        "realism_pass": realism_pass,
        "nsfw_final_engine": nsfw_final_engine,
        "video_engine": video_engine,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (Path(OUTPUT_PATH) / filename).with_suffix(".json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return metadata


def thumbnail_path_for(output: Path) -> Path:
    """Keep the lightweight Archive preview beside its original render."""
    return output.with_suffix(".thumb.jpg")


def create_image_thumbnail(source, destination: Path) -> None:
    """Save a compact JPEG preview without changing the original render."""
    from PIL import Image, ImageOps

    with Image.open(source) as image:
        preview = ImageOps.exif_transpose(image).convert("RGB")
        preview.thumbnail((480, 480), Image.Resampling.LANCZOS)
        preview.save(destination, "JPEG", quality=82, optimize=True)


def create_video_thumbnail(frames, destination: Path) -> None:
    """Use the first generated frame as the Archive preview for a video."""
    import numpy as np
    from PIL import Image

    first_frame = frames[0]
    if isinstance(first_frame, Image.Image):
        preview = first_frame.convert("RGB")
    else:
        frame_array = np.asarray(first_frame)
        if np.issubdtype(frame_array.dtype, np.floating):
            # Diffusers may return RGB frames as float32 in either [0, 1] or [0, 255].
            scale = 255.0 if frame_array.size and frame_array.max() <= 1.0 else 1.0
            frame_array = np.clip(frame_array * scale, 0, 255).astype(np.uint8)
        else:
            frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
        preview = Image.fromarray(frame_array).convert("RGB")
    preview.thumbnail((480, 480), Image.Resampling.LANCZOS)
    preview.save(destination, "JPEG", quality=82, optimize=True)


def protected_face_mask(image):
    """Detect faces in the Qwen draft and return a soft head-protection mask for the finalizer."""
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    from PIL import Image

    face_analyser = nsfw_final_worker_state.get("face_analyser")
    if face_analyser is None:
        print("Loading InsightFace detector for multi-person identity protection.")
        face_analyser = FaceAnalysis(
            name="buffalo_l",
            root=FACE_MODELS_PATH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        nsfw_final_worker_state["face_analyser"] = face_analyser
        # InsightFace downloads its detection models on first use; retain them for later L40S final passes.
        face_models.commit()

    rgb = np.asarray(image)
    faces = face_analyser.get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not faces:
        raise ValueError(
            "No face was detected in the Qwen draft, so the multi-person final pass was skipped to avoid identity drift."
        )
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for face in faces:
        left, top, right, bottom = face.bbox.astype(int)
        width, height = right - left, bottom - top
        # Cover facial features, ears, and enough hairline to avoid seams after the final body/detail pass.
        center = (int((left + right) / 2), int(top + height * 0.38))
        axes = (max(1, int(width * 0.84)), max(1, int(height * 1.02)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10, sigmaY=10)
    return Image.fromarray(mask, mode="L"), len(faces)


def configure_loras(pipe, specs: list[dict], worker_state: dict, label: str, strength_multiplier: float = 1.0) -> None:
    """Load adapters once and activate only the adapters requested for this render."""
    if not specs:
        if worker_state["loaded_loras"]:
            pipe.disable_lora()
        return
    loaded_loras = worker_state["loaded_loras"]
    adapter_names = []
    adapter_weights = []
    for spec in specs:
        key = json.dumps(spec, sort_keys=True)
        adapter_name = loaded_loras.get(key)
        if adapter_name is None:
            adapter_name = f"adapter_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"
            source = resolve_adapter_source(spec)
            load_options = {"adapter_name": adapter_name}
            if spec["weight_name"]:
                load_options["weight_name"] = spec["weight_name"]
            print(f"Loading {label} LoRA {adapter_name} from {spec['source']}")
            pipe.load_lora_weights(source, **load_options)
            loaded_loras[key] = adapter_name
        adapter_names.append(adapter_name)
        adapter_weights.append(spec["strength"] * strength_multiplier)
    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)


def configure_qwen_loras(pipe, specs: list[dict], strength_multiplier: float = 1.0) -> None:
    configure_loras(pipe, specs, image_worker_state, "Qwen", strength_multiplier)


def run_image_generation(
    image_data: list[bytes], prompt: str, negative_prompt: str, mode: str, aspect_ratio: str, quality: str,
    anatomy_guardrail: bool, nsfw_enabled: bool = False, nsfw_strength: float = 0.9,
    realism_pass: bool = False, save_intermediate: bool = False,
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
        else:
            raise ValueError(f"Unsupported image mode: {mode}")
        if mode == "kontext":
            # Keep the 12B Kontext pipeline within the L40S's 48 GB VRAM budget.
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        image_worker_state.update(mode=mode, pipe=pipe, loaded_loras={})

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
        nsfw_loras = configured_nsfw_loras() if nsfw_enabled else []
        configure_qwen_loras(pipe, nsfw_loras, nsfw_strength)
        if nsfw_enabled:
            print(
                f"Qwen NSFW mode enabled with {len(nsfw_loras)} adapter(s) at strength {nsfw_strength:.2f}."
            )
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
        if realism_pass:
            realism_loras = configured_realism_loras()
            if not realism_loras:
                raise ValueError("Realism pass was requested but QWEN_REALISM_LORA_SPECS is not configured.")
            print(f"Running Qwen realism pass with {len(realism_loras)} adapter(s).")
            configure_qwen_loras(pipe, nsfw_loras + realism_loras)
            generated = pipe(
                image=[sources[0], generated],
                prompt=(
                    f"{IDENTITY_LOCKS['qwen']}\n\nRefine the supplied draft with natural skin texture, "
                    "photorealistic detail, and restrained retouching. Preserve its identity, pose, outfit, "
                    f"composition, and scene.\n\nCreative direction: {prompt.strip()}"
                ),
                negative_prompt=guarded_negative_prompt,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                width=width,
                height=height,
                num_inference_steps=12,
                generator=torch.Generator(device="cuda").manual_seed(torch.seed()),
            ).images[0]
        elif nsfw_enabled:
            # Keep future SFW requests free of the prior request's optional adapters.
            configure_qwen_loras(pipe, nsfw_loras, nsfw_strength)
        filename = f"qwen-draft-{unique_id}.png" if save_intermediate else f"qwen-identity-{unique_id}.png"
    generated.save(Path(OUTPUT_PATH) / filename, "PNG")
    if save_intermediate:
        # The L40S finalizer reads this temporary Qwen render from the shared Volume. Do not expose it
        # as an Archive entry; only the completed final image should appear there.
        outputs.commit()
        return {"filename": filename, "width": width, "height": height}
    create_image_thumbnail(Path(OUTPUT_PATH) / filename, thumbnail_path_for(Path(OUTPUT_PATH) / filename))
    metadata = output_metadata(
        filename, "image/png", mode, prompt, negative_prompt, aspect_ratio, quality, len(sources), anatomy_guardrail,
        nsfw_enabled=nsfw_enabled if mode == "qwen" else False,
        realism_pass=realism_pass if mode == "qwen" else False,
    )
    outputs.commit()
    return metadata


@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=20 * 60,
    scaledown_window=90,
    max_containers=1,
    secrets=[hf_secret, adapter_config_secret],
    volumes={HF_CACHE_PATH: model_cache, OUTPUT_PATH: outputs, ADAPTERS_PATH: adapters},
)
def generate_qwen(
    image_data: list[bytes], prompt: str, negative_prompt: str, aspect_ratio: str, quality: str,
    anatomy_guardrail: bool, nsfw_enabled: bool = False, nsfw_strength: float = 0.9,
    realism_pass: bool = False, save_intermediate: bool = False,
) -> dict:
    """Identity editing is the sole H100 workload."""
    return run_image_generation(
        image_data, prompt, negative_prompt, "qwen", aspect_ratio, quality, anatomy_guardrail,
        nsfw_enabled, nsfw_strength, realism_pass, save_intermediate,
    )


@app.function(
    image=gpu_image,
    gpu="L40S",
    timeout=20 * 60,
    scaledown_window=90,
    max_containers=1,
    secrets=[hf_secret, adapter_config_secret],
    volumes={HF_CACHE_PATH: model_cache, OUTPUT_PATH: outputs, ADAPTERS_PATH: adapters, FACE_MODELS_PATH: face_models},
)
def generate_realvis_nsfw_final(
    qwen_draft_filename: str, identity_image_data: bytes, prompt: str, negative_prompt: str,
    aspect_ratio: str, quality: str, anatomy_guardrail: bool, nsfw_strength: float,
    denoise_strength: float, reference_count: int = 1,
) -> dict:
    """Finish a Qwen draft on the L40S with RealVisXL and face-conditioned SDXL img2img."""
    import io

    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    outputs.reload()
    draft_path = Path(OUTPUT_PATH) / qwen_draft_filename
    if not draft_path.is_file():
        raise ValueError("The temporary Qwen draft was not available to the NSFW finalizer.")
    draft = Image.open(draft_path).convert("RGB")
    identity_image = Image.open(io.BytesIO(identity_image_data)).convert("RGB")
    face_mask, protected_face_count = protected_face_mask(draft)

    if nsfw_final_worker_state["pipe"] is None:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        from transformers import CLIPVisionModelWithProjection

        model_id = configured_nsfw_final_model_id()
        model_file = configured_nsfw_final_model_file()
        face_adapter_id = configured_nsfw_face_adapter_id()
        print(f"Loading RealVisXL NSFW finalizer from {model_id}.")
        # The Plus Face SDXL checkpoint requires the ViT-H image encoder. Letting the generic loader
        # infer an encoder produces 1664-dimensional image embeddings for an adapter expecting 1280.
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            face_adapter_id,
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
            cache_dir=HF_CACHE_PATH,
        )
        if model_file:
            checkpoint_path = hf_hub_download(
                model_id, filename=model_file, cache_dir=HF_CACHE_PATH, token=True
            )
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                checkpoint_path, image_encoder=image_encoder, torch_dtype=torch.float16
            )
        else:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id, image_encoder=image_encoder, torch_dtype=torch.float16, cache_dir=HF_CACHE_PATH
            )
        pipe.load_ip_adapter(
            face_adapter_id,
            subfolder=configured_nsfw_face_adapter_subfolder(),
            weight_name=configured_nsfw_face_adapter_weight(),
            cache_dir=HF_CACHE_PATH,
        )
        pipe.enable_model_cpu_offload()
        nsfw_final_worker_state.update(pipe=pipe, loaded_loras={})

    pipe = nsfw_final_worker_state["pipe"]
    final_loras = configured_nsfw_final_loras()
    configure_loras(pipe, final_loras, nsfw_final_worker_state, "RealVisXL NSFW", nsfw_strength)
    area, steps = IMAGE_QUALITY[quality]
    width, height = dimensions_for_aspect(draft, aspect_ratio, area, multiple=8)
    guarded_negative_prompt = negative_prompt_with_anatomy_guardrail(negative_prompt, anatomy_guardrail)
    use_single_face_adapter = protected_face_count == 1
    pipe.set_ip_adapter_scale(0.85 if use_single_face_adapter else 0.0)
    face_mask = face_mask.resize((width, height), Image.Resampling.LANCZOS)
    draft_for_final = draft.resize((width, height), Image.Resampling.LANCZOS)
    print(
        "Running RealVisXL face-conditioned NSFW final pass "
        f"(denoise={denoise_strength:.2f}, adapters={len(final_loras)}, protected_faces={protected_face_count}, "
        f"single_face_adapter={use_single_face_adapter})."
    )
    generation_options = {
        "prompt": (
            "Preserve the face, facial proportions, skin tone, and distinguishing features of the supplied "
            "identity portrait. Preserve the Qwen draft's pose, camera framing, composition, and scene. "
            "Make only the requested adult styling and photorealistic detail changes.\n\n"
            f"Creative direction: {prompt.strip()}"
        ),
        "negative_prompt": guarded_negative_prompt.strip() or None,
        "image": draft_for_final,
        "strength": denoise_strength,
        "guidance_scale": 5.0,
        "num_inference_steps": max(24, min(steps, 32)),
        "generator": torch.Generator(device="cuda").manual_seed(torch.seed()),
    }
    # Diffusers still expects image embeddings once an IP-Adapter is attached. For multiple people the
    # adapter scale is zero, so reference 01 is encoded but has no influence on the final pixels.
    generation_options["ip_adapter_image"] = identity_image
    generated = pipe(**generation_options).images[0]
    # For two or more people, restore every detected Qwen face/head region after RealVisXL has refined the
    # body and scene. A single global face adapter cannot reliably assign two separate reference identities.
    generated = Image.composite(draft_for_final, generated, face_mask)
    filename = f"realvisxl-nsfw-{uuid.uuid4().hex}.png"
    output = Path(OUTPUT_PATH) / filename
    generated.save(output, "PNG")
    create_image_thumbnail(output, thumbnail_path_for(output))
    metadata = output_metadata(
        filename, "image/png", "qwen", prompt, negative_prompt, aspect_ratio, quality, reference_count,
        anatomy_guardrail, nsfw_enabled=True, nsfw_final_engine="realvisxl-ip-adapter-face",
    )
    # The intermediate is deliberately not archived once its final result exists.
    draft_path.unlink(missing_ok=True)
    thumbnail_path_for(draft_path).unlink(missing_ok=True)
    outputs.commit()
    return metadata


@app.function(timeout=45 * 60, max_containers=1)
def generate_qwen_with_realvis_final(
    image_data: list[bytes], prompt: str, negative_prompt: str, aspect_ratio: str, quality: str,
    anatomy_guardrail: bool, nsfw_strength: float, final_denoise_strength: float,
) -> dict:
    """Run Qwen first, release the H100, then finish the same draft on the L40S."""
    if len(image_data) > 1:
        print("Qwen uses every reference; RealVisXL will protect every detected face in the Qwen draft.")
    qwen_draft = generate_qwen.remote(
        image_data, prompt, negative_prompt, aspect_ratio, quality, anatomy_guardrail,
        False, 0.9, False, True,
    )
    return generate_realvis_nsfw_final.remote(
        qwen_draft["filename"], image_data[0], prompt, negative_prompt, aspect_ratio, quality,
        anatomy_guardrail, nsfw_strength, final_denoise_strength, len(image_data),
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
        create_video_thumbnail(frames, thumbnail_path_for(Path(OUTPUT_PATH) / filename))
        variants.append(output_metadata(
            filename, "video/mp4", "video", prompt, negative_prompt, aspect_ratio, quality, 1,
            anatomy_guardrail, motion_mode, video_engine="wan5b"
        ))
    outputs.commit()
    return {"primary": variants[0], "variants": variants}


@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=30 * 60,
    scaledown_window=90,
    max_containers=1,
    secrets=[hf_secret],
    volumes={HF_CACHE_PATH: model_cache, OUTPUT_PATH: outputs},
)
def generate_high_motion_video(
    image_data: bytes, prompt: str, negative_prompt: str, aspect_ratio: str, quality: str,
    anatomy_guardrail: bool, motion_mode: str, variation_count: int,
) -> dict:
    """Run the dedicated 14B Wan I2V model on an 80 GB H100 for demanding motion."""
    import io

    import torch
    from diffusers import WanImageToVideoPipeline
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
    if high_motion_video_worker_state["pipe"] is None:
        pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_HIGH_MOTION_MODEL_ID, torch_dtype=torch.bfloat16
        )
        # The upstream I2V-A14B model requires an 80 GB GPU. H100 avoids CPU offload drift and latency.
        pipe.to("cuda")
        high_motion_video_worker_state["pipe"] = pipe

    area, num_frames, steps = VIDEO_QUALITY[quality]
    # The official I2V-A14B recipe uses 81 frames / 40 steps. A shorter clip reduces temporal drift;
    # major physical actions receive the extra denoising budget.
    num_frames = min(num_frames, 81)
    steps = max(steps, 50 if motion_mode == "major" else 40)
    width, height = dimensions_for_aspect(source, aspect_ratio, area, multiple=32)
    variants = []
    for _ in range(variation_count):
        frames = high_motion_video_worker_state["pipe"](
            image=source,
            prompt=model_prompt,
            negative_prompt=f"{guarded_negative_prompt}, {VIDEO_STABILITY_NEGATIVE}",
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=3.5,
            generator=torch.Generator(device="cuda").manual_seed(torch.seed()),
        ).frames[0]
        filename = f"wan22-i2v14b-{uuid.uuid4().hex}.mp4"
        output_path = Path(OUTPUT_PATH) / filename
        export_to_video(frames, str(output_path), fps=16)
        create_video_thumbnail(frames, thumbnail_path_for(output_path))
        variants.append(output_metadata(
            filename, "video/mp4", "video", prompt, negative_prompt, aspect_ratio, quality, 1,
            anatomy_guardrail, motion_mode, video_engine="wan14b"
        ))
    outputs.commit()
    return {"primary": variants[0], "variants": variants}


@app.function(
    image=web_image,
    max_containers=1,
    secrets=[studio_auth_secret, mega_secret, adapter_config_secret],
    volumes={OUTPUT_PATH: outputs},
)
@modal.asgi_app()
def web_app():
    """Serve the UI and keep only the currently selected model in memory."""
    import asyncio
    import json
    import subprocess
    import time
    from collections import deque
    from datetime import datetime, timezone

    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.middleware.sessions import SessionMiddleware

    web = FastAPI(title="Motion & Kontext Studio")
    login_attempts: dict[str, deque[float]] = {}
    login_attempt_limit = 5
    login_attempt_window_seconds = 15 * 60

    def login_attempt_key(request: Request, username: str) -> tuple[str, str]:
        """Rate-limit both the client and the submitted account name."""
        client = request.client.host if request.client else "unknown"
        account = username.strip().casefold()[:128] or "empty"
        return f"client:{client}", f"account:{account}"

    def prune_login_attempts(now: float) -> None:
        cutoff = now - login_attempt_window_seconds
        for key, attempts in list(login_attempts.items()):
            while attempts and attempts[0] <= cutoff:
                attempts.popleft()
            if not attempts:
                login_attempts.pop(key, None)

    def login_retry_after(keys: tuple[str, str], now: float) -> int | None:
        prune_login_attempts(now)
        retry_after = 0
        for key in keys:
            attempts = login_attempts.get(key, ())
            if len(attempts) >= login_attempt_limit:
                retry_after = max(retry_after, int(login_attempt_window_seconds - (now - attempts[0])) + 1)
        return retry_after or None

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
        now = time.monotonic()
        attempt_keys = login_attempt_key(request, username)
        retry_after = login_retry_after(attempt_keys, now)
        if retry_after:
            raise HTTPException(
                status_code=429,
                detail="Too many sign-in attempts. Please try again later.",
                headers={"Retry-After": str(retry_after)},
            )
        is_valid = hmac.compare_digest(username, os.environ["APP_USERNAME"]) and hmac.compare_digest(
            password, os.environ["APP_PASSWORD"]
        )
        if not is_valid:
            for key in attempt_keys:
                login_attempts.setdefault(key, deque()).append(now)
            return RedirectResponse(url="/login?error=1", status_code=303)
        for key in attempt_keys:
            login_attempts.pop(key, None)
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
        video_engine: str = Form("wan5b"),
        variation_count: int = Form(1),
        nsfw_enabled: bool = Form(False),
        nsfw_strength: float = Form(0.9),
        nsfw_final_engine: str = Form("realvisxl"),
        nsfw_final_denoise: float = Form(0.32),
        realism_pass: bool = Form(False),
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
        if video_engine not in {"wan5b", "wan14b"}:
            raise HTTPException(status_code=400, detail="Choose a supported video engine.")
        if variation_count not in {1, 2, 3}:
            raise HTTPException(status_code=400, detail="Choose one, two, or three variations.")
        if not 0 <= nsfw_strength <= 1.5:
            raise HTTPException(status_code=400, detail="Choose an NSFW strength between 0 and 1.5.")
        if nsfw_final_engine not in {"realvisxl", "qwen"}:
            raise HTTPException(status_code=400, detail="Choose a supported NSFW rendering path.")
        if not 0.15 <= nsfw_final_denoise <= 0.55:
            raise HTTPException(status_code=400, detail="Choose a final-pass denoise strength between 0.15 and 0.55.")
        if mode != "qwen" and (nsfw_enabled or realism_pass):
            raise HTTPException(status_code=400, detail="NSFW and realism controls are available only for Qwen Identity.")
        if nsfw_enabled and nsfw_final_engine == "qwen":
            try:
                configured_nsfw_loras()
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
        if nsfw_enabled and nsfw_final_engine == "realvisxl":
            try:
                configured_nsfw_final_model_id()
                configured_nsfw_face_adapter_id()
                configured_nsfw_final_loras()
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
        if realism_pass and nsfw_final_engine == "qwen":
            try:
                if not configured_realism_loras():
                    raise HTTPException(
                        status_code=400,
                        detail="Configure QWEN_REALISM_LORA_SPECS before enabling the realism pass.",
                    )
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
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
            "video_engine": video_engine if mode == "video" else None,
            "variation_count": variation_count if mode == "video" else 1,
            "nsfw_enabled": nsfw_enabled if mode == "qwen" else False,
            "nsfw_strength": nsfw_strength if mode == "qwen" and nsfw_enabled else None,
            "nsfw_final_engine": nsfw_final_engine if mode == "qwen" and nsfw_enabled else None,
            "nsfw_final_denoise": nsfw_final_denoise if mode == "qwen" and nsfw_enabled and nsfw_final_engine == "realvisxl" else None,
            "realism_pass": realism_pass if mode == "qwen" and nsfw_final_engine == "qwen" else False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        job_path.write_text(json.dumps(job), encoding="utf-8")
        await outputs.commit.aio()
        try:
            if mode == "video":
                video_function = generate_high_motion_video if video_engine == "wan14b" else generate_video
                call = await video_function.spawn.aio(
                    image_data[0], prompt, negative_prompt, aspect_ratio, quality, anatomy_guardrail,
                    motion_mode, variation_count,
                )
            elif mode == "qwen":
                if nsfw_enabled and nsfw_final_engine == "realvisxl":
                    call = await generate_qwen_with_realvis_final.spawn.aio(
                        image_data, prompt, negative_prompt, aspect_ratio, quality, anatomy_guardrail,
                        nsfw_strength, nsfw_final_denoise,
                    )
                else:
                    call = await generate_qwen.spawn.aio(
                        image_data, prompt, negative_prompt, aspect_ratio, quality, anatomy_guardrail,
                        nsfw_enabled, nsfw_strength, realism_pass,
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

    def thumbnail_path(filename: str) -> Path:
        return thumbnail_path_for(output_path(filename))

    def create_legacy_thumbnail(output: Path) -> Path:
        """Backfill previews for renders saved before Archive thumbnails existed."""
        preview = thumbnail_path_for(output)
        if preview.is_file():
            return preview
        if output.suffix == ".png":
            create_image_thumbnail(output, preview)
        else:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", "0", "-i", str(output), "-frames:v", "1",
                    "-vf", "scale=480:480:force_original_aspect_ratio=decrease", "-q:v", "4", str(preview),
                ],
                check=True,
                capture_output=True,
            )
        return preview

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

    @web.get("/api/thumbnail/{filename}")
    async def get_thumbnail(filename: str):
        output = output_path(filename)
        preview = thumbnail_path(filename)
        if not preview.is_file():
            try:
                preview = await asyncio.to_thread(create_legacy_thumbnail, output)
                await outputs.commit.aio()
            except Exception as error:
                raise HTTPException(status_code=500, detail="Could not prepare the archive thumbnail.") from error
        return FileResponse(preview, media_type="image/jpeg")

    @web.delete("/api/history/{filename}")
    async def delete_output(filename: str):
        """Permanently remove one generated render and its archive record."""
        output = output_path(filename)
        output.unlink()
        preview = thumbnail_path_for(output)
        if preview.is_file():
            preview.unlink()
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
