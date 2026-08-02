# Motion & Kontext Studio — Session Changelog

## 2026-07-31

### Studio capabilities

- Added Qwen-only NSFW adapter controls, configurable LoRA specifications, and an optional realism refinement pass. SFW Qwen requests now run without optional adapters.
- Archive cards now load compact JPEG thumbnails instead of full-size renders. Archive actions still use the original PNG or MP4; older saved renders receive a thumbnail automatically on first preview.

- Built a private Modal-hosted studio for:
  - FLUX.1 Kontext image editing.
  - Qwen Image Edit Plus identity-oriented image editing with the configured LoRA.
  - Wan2.2 image-to-video generation.
- Added prompt and optional negative-prompt fields.
- Added output aspect-ratio controls: source, 1:1, 2:3, 3:2, 9:16, and 16:9.
- Added draft, standard, and high render-quality presets.
- Fixed source-image previews to use `object-fit: contain`, so the whole image is visible.
- Added up to four image references. Qwen uses all references; Kontext and Wan use reference 01.
- Added reuse actions:
  - Generated images can be reused as references.
  - Generated videos can contribute their final frame as a reference image.

### Private access and storage

- Added session-based login using the `studio-auth` Modal secret:
  - `APP_USERNAME`
  - `APP_PASSWORD`
  - `APP_SESSION_SECRET`
- Added persistent Archive history in the `motion-studio-outputs` Modal Volume.
- Archive items retain prompt, engine, output ratio, quality, reference count, timestamp, and media type.
- Added Archive actions: reuse, download, upload to Mega, and permanent delete.
- Delete removes both the generated media file and its archive metadata from the Modal Volume after confirmation.

### Mega uploads

- Added the `mega-credentials` Modal secret integration:
  - `MEGA_EMAIL`
  - `MEGA_PASSWORD`
- Added an **Upload to Mega** action for current and archived results.
- Files are uploaded server-side to `AI generations`, creating the folder if necessary.
- Added `mega_upload.js`, using MEGAJS and Node Web Crypto for client-side encryption support.
- Mega upload had a Node Web Crypto runtime error; the code was fixed, but the final end-to-end upload should be re-tested after the next deployment.

### Cost and reliability architecture

- Split the GPU-free web service from generation workers. Login, UI, Archive, downloads, and Mega uploads run on CPU only.
- Model cache remains in `motion-studio-model-cache`; outputs remain in `motion-studio-outputs`.
- GPU assignment now is:
  - Qwen Identity: H100.
  - FLUX Kontext: L40S with CPU offloading.
  - Wan2.2 video: L40S with CPU offloading.
- GPU workers use `scaledown_window=90` and `max_containers=1` to reduce idle GPU spend.
- Converted generation from a browser-bound request into a detached Modal job:
  - The browser receives a job ID immediately.
  - It polls job status.
  - The render continues and is archived even if the browser is closed.

### Important operational notes

- Deploy the current version with:

  ```bash
  modal deploy wan_modal.py
  ```

- No model re-download should be necessary because weights are already stored in the persistent model-cache volume.
- Test all three engines after deployment. Kontext on L40S is intentionally more memory-efficient but may be slower than the prior A100/H100 configuration.
- The repository commits currently on `main` are:
  - `e30f40c` — background jobs, archive deletion, lower-cost GPU workers.
  - `8d5a678` — CPU web service / GPU worker split.
  - `66e8a45` — Mega uploads and Archive history.
  - `12c984c` — multi-reference generation workflow.
  - `9d72ffb` — authenticated Modal generation studio.
- Pushing to `origin` is still pending confirmation that `github.com:abizit/Modal.git` is an authorized destination.
# Add RealVisXL face-conditioned NSFW final pass

- Keep Qwen's multi-reference identity edit on the H100, then optionally run a conservative RealVisXL SDXL img2img finalizer on the L40S.
- Add IP-Adapter Plus Face conditioning, configurable SDXL final LoRAs, final denoise controls, and job/archive metadata for the new path.
