# Wan2.2 on Modal

This deploys a single web studio with two workflows:

- **FLUX.1 Kontext [dev]** for instruction-guided image editing.
- **Wan2.2 TI2V-5B** for image-to-video generation.

The UI accepts a JPEG, PNG, or WebP, a prompt, and returns the resulting PNG or MP4 in the browser.

## Set up and deploy

```bash
python -m pip install -r requirements.txt
modal setup

# Create a Hugging Face token after accepting the FLUX.1 Kontext [dev] license,
# then store it in Modal. The model is gated and cannot download without it.
modal secret create huggingface-secret HF_TOKEN=your_huggingface_token

# Recommended: fetch both sets of model weights before opening the UI.
modal run wan_modal.py::download_models

modal deploy wan_modal.py
```

`modal deploy` prints the public URL. Open it to upload an image, select **Edit image** or **Animate image**, write an instruction, and download the browser result. The initial model downloads are persisted in `motion-studio-model-cache`; generated files are stored in `motion-studio-outputs`.

## Notes

- The studio loads only the selected model into the GPU worker at a time, avoiding the cost of keeping both models resident.
- The L40S is deliberately more conservative than Wan's official 24 GB minimum, particularly during video decoding.
- FLUX.1 Kontext [dev] is gated and under the FLUX.1 dev non-commercial license. Accept its Hugging Face license before creating the token.
- The code has not been deployed from this workspace: run `modal setup` with your Modal account first.
