# Wan2.2 on Modal

This deploys a single web studio with two workflows:

- **FLUX.1 Kontext [dev]** for instruction-guided image editing.
- **Qwen Image Edit 2511** with the configured identity-edit LoRA.
- **Wan2.2 TI2V-5B** for image-to-video generation.

The UI accepts a JPEG, PNG, or WebP, a prompt, and returns the resulting PNG or MP4 in the browser.

Choose **Match source** to preserve the source image’s aspect ratio, or select a specific aspect ratio. Quality controls both render size and inference steps; high quality uses more GPU time.

## Set up and deploy

```bash
python -m pip install -r requirements.txt
modal setup

# Create a Hugging Face token after accepting the FLUX.1 Kontext [dev] license,
# then store it in Modal. The model is gated and cannot download without it.
modal secret create huggingface-secret HF_TOKEN=your_huggingface_token

# Recommended: fetch all model weights before opening the UI.
modal run wan_modal.py::download_models

modal deploy wan_modal.py
```

`modal deploy` prints the public URL. Open it to upload an image, select **Edit image** or **Animate image**, write an instruction, and download the browser result. The initial model downloads are persisted in `motion-studio-model-cache`; generated files are stored in `motion-studio-outputs`.

## Optional Qwen adapters

Qwen Identity defaults to the base Qwen Image Edit 2511 pipeline. Enable **NSFW / Explicit** in the Qwen UI to load the configured optional adapters while keeping the existing multi-reference identity flow. The first use caches adapters in the persistent `motion-studio-adapters` Volume.

Create a Modal secret named `qwen-adapter-config` before deploying:

```bash
modal secret create qwen-adapter-config \
  'QWEN_NSFW_LORA_SPECS=[{"source":"ScottzillaSystems/qwen-image-edit-plus-nsfw-lora","strength":1.0}]'
```

`QWEN_NSFW_LORA_SPECS` is a JSON list. Each entry accepts `source` (a Hugging Face repository ID or a path under `/adapters`), optional `weight_name`, optional `revision`, and `strength` (0–2). The UI slider multiplies each configured NSFW strength. To cache configured adapters before serving requests, run:

```bash
modal run wan_modal.py::download_adapters
```

The optional **Apply realism / natural skin pass** control runs a short second Qwen edit and therefore adds GPU cost. It requires a configured realism LoRA:

```bash
modal secret create qwen-adapter-config \
  'QWEN_NSFW_LORA_SPECS=[{"source":"ScottzillaSystems/qwen-image-edit-plus-nsfw-lora","strength":1.0}]' \
  'QWEN_REALISM_LORA_SPECS=[{"source":"your-org/your-qwen-realism-lora","weight_name":"adapter.safetensors","strength":0.7}]'
```

Replace the example realism source and filename with an actual Qwen Image Edit-compatible LoRA. A merged ComfyUI AIO checkpoint is not a Diffusers LoRA and needs a separate worker; do not place it in these settings.

## RealVisXL NSFW final pass

For the strongest identity retention with explicit finishing, select **RealVisXL final pass — recommended** after enabling NSFW. The app first creates an identity-locked draft with base Qwen on the H100, then releases the H100 and sends that draft to an L40S RealVisXL img2img pass. The finalizer detects every face in the Qwen draft and restores each face/head region with a soft protection mask after it refines the body and scene. For a single detected person it also uses SDXL IP-Adapter Plus Face conditioning. It defaults to a conservative `0.32` denoise value.

The default final model is `SG161222/RealVisXL_V5.0`, which is a Diffusers-format SDXL repository. Set `NSFW_FINAL_MODEL_FILE` only when you intentionally use a single-file SDXL checkpoint; otherwise leave it unset. Override the model in the existing `qwen-adapter-config` secret if you use another compatible SDXL checkpoint:

```bash
modal secret create qwen-adapter-config \
  'QWEN_NSFW_LORA_SPECS=[{"source":"ScottzillaSystems/qwen-image-edit-plus-nsfw-lora","strength":1.0}]' \
  'NSFW_FINAL_MODEL_ID=SG161222/RealVisXL_V5.0' \
  'NSFW_FINAL_LORA_SPECS=[{"source":"your-org/your-sdxl-nsfw-lora","weight_name":"adapter.safetensors","strength":0.8}]'
```

`NSFW_FINAL_LORA_SPECS` uses the same JSON format as the Qwen list, but every listed adapter must be SDXL-compatible. Do not reuse a Qwen LoRA here. The final LoRA list is optional: without it, RealVisXL still runs as a face-conditioned, low-denoise photoreal finalizer. The default face adapter is `h94/IP-Adapter`, `sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors`; advanced replacements can be set through `NSFW_FINAL_FACE_ADAPTER_ID`, `NSFW_FINAL_FACE_ADAPTER_SUBFOLDER`, and `NSFW_FINAL_FACE_ADAPTER_WEIGHT`. Re-run `modal run wan_modal.py::download_models` after changing the final model or face adapter, and `modal run wan_modal.py::download_adapters` after changing final LoRAs.

For two-person work, upload the woman's clear portrait as reference 01 and the man's as reference 02, and identify them by number in the prompt. Qwen continues to use every supplied reference in its first step; the final pass protects every detected Qwen face instead of prioritizing reference 01. Preload its face detector once before your first final pass:

```bash
modal run wan_modal.py::download_face_models
```

Start with final denoise `0.30–0.35` and avoid values above `0.45` when identity is important. Use only lawful images of consenting adults.

## Notes

- The studio loads only the selected model into the GPU worker at a time, avoiding the cost of keeping both models resident.
- The app uses an H100 for the larger Qwen Image Edit pipeline and fast image inference.
- FLUX.1 Kontext [dev] is gated and under the FLUX.1 dev non-commercial license. Accept its Hugging Face license before creating the token.
- The code has not been deployed from this workspace: run `modal setup` with your Modal account first.
