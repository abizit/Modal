# Local Identity Draft Studio Plan

## Goal

Run inexpensive image drafts on a 32 GB Apple Silicon MacBook Pro, then use the existing hosted Qwen Image Edit 2511 studio only for selected final renders.

This is intentionally a separate local setup. It must not be merged into, deployed with, or depend on the Modal project.

## Current progress — 2026-08-02

Completed on the MacBook Pro:

- ComfyUI Desktop installed as a standalone app at `/Volumes/Abizit's SSD/AI/ComfyUI`.
- Apple Silicon was detected; the local installation uses the MPS/Metal-capable ComfyUI Desktop setup.
- Starter workflows and their model downloads were skipped intentionally.
- Installed `ComfyUI_photomakerV2_native` by `zhangp3652`; ComfyUI was restarted successfully.
- Installed `comfyui_controlnet_aux` by `fannovel16`; ComfyUI was restarted successfully.
- Confirmed that the built-in **Load PhotoMaker Model** node is available on the canvas.

Not yet completed:

- No model weights have been downloaded.
- No workflow has been assembled or tested.

### Resume here

Download the official `photomaker-v2.bin` file (about 1.8 GB) from [TencentARC/PhotoMaker-V2](https://huggingface.co/TencentARC/PhotoMaker-V2). Leave the filename unchanged and keep it in Downloads temporarily. The next session should move it into ComfyUI's `models/photomaker/` folder, restart or refresh ComfyUI, and select it in the **Load PhotoMaker Model** node.

## Target workflow

```text
Original identity photos (2–4) + prompt + optional pose reference
        ↓
Mac-local SDXL draft workflow
  PhotoMaker V2 + OpenPose ControlNet
        ↓
Choose a strong draft manually
        ↓
Original identity photos + chosen draft
        ↓
Existing Modal Qwen Image Edit 2511 final render
```

The local workflow explores clothing, setting, lighting, camera framing, and pose cheaply. Qwen is reserved for the final face and identity refinement.

## 1. Local software

Install ComfyUI Desktop for Apple Silicon and choose **MPS** during first-run setup. MPS uses Apple Metal acceleration on the M1 Pro GPU.

Keep ComfyUI and models outside the Modal repository. Recommended location:

```text
~/AI/ComfyUI/
```

Suggested folders:

```text
~/AI/ComfyUI/
├── models/
│   ├── checkpoints/
│   ├── controlnet/
│   ├── photomaker/
│   ├── instantid/
│   └── vae/
├── input/
├── output/
└── user/default/workflows/
```

Do not put model weights, generated images, or ComfyUI’s Python environment under the Modal git repository.

## 2. Required components

Use the following components in the first version.

| Purpose | Component | Why |
|---|---|---|
| Image-generation base | SDXL 1.0 or one photorealistic SDXL checkpoint | Fits the Mac workflow and has broad ComfyUI support. |
| Identity conditioning | PhotoMaker V2 | Works from multiple identity photos and has good portrait consistency. |
| Pose control | OpenPose preprocessor + SDXL OpenPose ControlNet | Lets the draft follow a supplied body pose. |
| Optional face lock | InstantID | Use only for a single adult subject when face likeness needs a stronger anchor. |
| Upscaling | None initially | Keep drafts inexpensive and send only selected images to Qwen. |

Do not start with FLUX.2, Rapid-AIO, Qwen, video models, face-swap tools, or multiple competing identity adapters. They add complexity and memory pressure before the basic workflow is proven.

## 3. ComfyUI custom nodes

Install only the nodes needed for the workflow:

1. ComfyUI Manager, to install and update nodes.
2. A maintained PhotoMaker V2 node.
3. A maintained InstantID node, but leave it disabled in the default workflow.
4. ControlNet auxiliary preprocessors for OpenPose.

Before installing any custom node, inspect its repository, recent maintenance activity, license, and model-download instructions. Avoid workflows that bundle opaque executable installers or unverified model downloads.

## 4. Model-download checklist

Download each model from its official project page or canonical Hugging Face repository, then record the source URL and model revision in a local `models-manifest.md` file.

Required downloads:

1. One SDXL base or photorealistic checkpoint.
2. PhotoMaker V2 adapter and its required image/face encoder.
3. OpenPose ControlNet compatible with the selected SDXL checkpoint.
4. OpenPose preprocessor model, if the node does not package it separately.

Optional download:

5. InstantID ControlNet, adapter, and face-analysis model.

Keep enough free disk space for model weights, caches, and outputs: reserve at least 60 GB before downloading. Do not download multiple SDXL checkpoints until the first one has been tested.

## 5. Default workflow: Identity Draft

Use this for clothes, styling, background, lighting, and camera experiments where the existing pose can vary naturally.

Inputs:

- Two to four clean portrait photographs of one consenting adult subject.
- A plain-language creative prompt.
- A negative prompt focused on visible failures, for example: `deformed hands, duplicate limbs, distorted face, text, watermark`.

Recommended first settings:

| Setting | Starting value |
|---|---|
| Resolution | 768 × 1024 portrait, or 1024 × 768 landscape |
| Batch size | 1 |
| Steps | 24–30 |
| CFG | Follow the selected SDXL checkpoint’s recommendation; start near 5–7 |
| Seed | Fixed while tuning prompt and identity strength; random only after it works |
| PhotoMaker identity strength | Start at the node default; raise gradually if the face drifts |

Prompt convention:

```text
portrait of a woman img, wearing [specific outfit], [pose or action], in [location],
[camera framing], [lighting], realistic editorial photography
```

PhotoMaker commonly uses `img` after the class word as its identity trigger; confirm the exact syntax in the chosen node’s documentation.

Acceptance criteria:

- The face is recognizably the reference subject.
- Clothing, pose, and scene follow the prompt.
- Hands and body are plausible enough to be useful as a Qwen input.
- The draft does not need to be final-quality.

## 6. Pose Draft workflow

Use this only when pose is deliberate rather than incidental.

Inputs:

- The same PhotoMaker identity references.
- A pose reference image, or an OpenPose skeleton generated from one.
- The same creative prompt, describing outfit and scene rather than repeating pose details.

Workflow order:

1. Load the identity images into PhotoMaker V2.
2. Extract an OpenPose map from the pose reference.
3. Feed that map to the SDXL OpenPose ControlNet.
4. Generate one image at a time.
5. Tune pose-control strength before increasing PhotoMaker strength. Excessive identity weight can fight a major body-pose change.

Starting tuning order:

1. Establish a believable pose without identity conditioning.
2. Enable PhotoMaker V2 at its default strength.
3. Increase PhotoMaker strength only until facial identity is usable.
4. If identity and pose conflict, simplify the pose rather than stacking more adapters.

## 7. Optional InstantID variant

Keep InstantID as a separate workflow, not an extra node in every generation.

Use it when:

- There is one subject.
- A clear front-facing portrait is available.
- Facial likeness matters more than a radical pose change.

Do not use it for group shots or as the default multi-photo workflow. PhotoMaker V2 is the primary identity method because it naturally uses several reference photos.

## 8. Mac performance guardrails

The M1 Pro has 32 GB unified memory shared by macOS and the GPU. Protect system responsiveness:

- Close memory-heavy apps before generation.
- Use batch size 1.
- Start at 768 × 1024; do not begin at 2K or 4K.
- Generate one candidate per request rather than large batches.
- Save workflows before changing nodes or model versions.
- Watch Activity Monitor’s Memory Pressure graph; stop if it turns yellow or red.
- If MPS encounters an unsupported operation, use the compatible node/model alternative rather than silently falling back to CPU for the whole workflow.

Generation can take minutes. That is acceptable because the goal is low-cost drafts, not real-time output.

## 9. Qwen finalisation recipe

When a local draft is selected, use the hosted studio with:

1. Reference 01: one original, clean face/identity photo.
2. Reference 02: the selected local draft.
3. Optional references 03–04: other clean identity photos only if they improve likeness.
4. A focused prompt, for example:

```text
Use reference 02 as the required composition, pose, outfit, and scene. Preserve the
exact facial identity, facial structure, skin tone, and distinguishing features from
reference 01. Correct only visible anatomy or rendering artifacts.
```

This avoids asking Qwen to invent the entire scene again. It uses Qwen to repair and lock identity around a composition already chosen locally.

## 10. Validation plan

Run this before relying on the workflow:

1. Pick one subject with four varied but clear photos.
2. Test three outfit changes, three scene changes, and three poses.
3. For each test, generate three local drafts with a fixed seed first.
4. Record whether the local draft preserves a recognisable identity and pose.
5. Send only the best draft from each test to Qwen.
6. Compare identity, hands, composition retention, and total hosted cost.
7. Adjust one control at a time; preserve the known-good workflow JSON.

Success means that most Qwen calls start from a good-enough composition, reducing wasted Qwen generations. It does not require local results to equal Qwen quality.

## 11. Separation and privacy

- Keep local drafts in the local ComfyUI output directory.
- Do not store Modal credentials in ComfyUI, workflow JSON, shell history, or local helper scripts.
- Upload selected files manually to the existing authenticated studio in the first version.
- Keep original references and generated images limited to people who have consented to their use.

## 12. Future enhancements

Only after the basic workflows are stable:

1. Add a local image selector that copies the chosen draft and identity photos into a `qwen-final` folder.
2. Add a simple metadata sidecar containing the prompt, seed, model, and workflow version.
3. Add a local upscaler for selected drafts only.
4. Evaluate FLUX.2 klein 4B as a separate draft engine, not as a replacement for PhotoMaker until identity quality has been tested on the Mac.
