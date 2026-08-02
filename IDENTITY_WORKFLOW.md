# Identity-preserved two-person workflow

Use this workflow for lawful images of consenting adults. Its purpose is to preserve the identities of two people while changing pose, wardrobe, scene, or adult styling.

## One-time setup

1. Deploy the current worker:

   ```bash
   modal run wan_modal.py::download_models
   modal run wan_modal.py::download_face_models
   modal deploy wan_modal.py
   ```

2. In the studio, use **02 / Qwen identity**. Do not use Kontext for a two-person identity edit.

3. For the most reliable result, use the **RealVisXL final pass — recommended** explicit path. It runs Qwen first, then protects every detected face while RealVisXL refines the body, clothing, scene, and detail.

## Prepare references

Use two strong face references before adding optional extras.

1. **Reference 01 — woman:** a clear, front-facing or three-quarter portrait with good light, no filters, no sunglasses, and the full face visible.
2. **Reference 02 — man:** the same quality of clear portrait.
3. Keep each reference to one person. Do not upload a couple photo as the primary identity reference.
4. If you add a third or fourth reference, use it only for a specific outfit, location, or object. State its purpose in the prompt.
5. Use images that show the real hairline, face shape, skin tone, and distinguishing features. Avoid heavily edited social-media images.

## Make the Qwen draft

This is the identity-critical stage. Give Qwen the complete creative direction here, including pose, clothing, environment, camera angle, and relationship between the people.

1. Select **Qwen identity**.
2. Add the woman as reference 01 and the man as reference 02.
3. Start with **Standard** quality. Use **High** only after the composition is correct.
4. Leave **Enable NSFW / Explicit** off for the first draft when identity is most important.
5. Turn on **Anatomy Protection** for nude or difficult full-body poses.
6. Write the prompt using explicit reference numbers and a single clear composition.

Example prompt:

```text
Reference 01 is the woman and reference 02 is the man.
Preserve both identities exactly: their faces, facial structure, skin tone, body type, and distinguishing features.

Create a full-body editorial bedroom photograph. The woman is standing on the left and the man is standing on the right, facing each other. Keep both faces clearly visible in a three-quarter view. Change their wardrobe to [describe clothing], use [describe lighting], and frame the image as [describe camera angle and aspect].
```

For adult styling, state only the final composition and pose you want, while keeping the identity instruction at the top. Do not use contradictory directions such as “preserve the face exactly” and “change the woman into a different person.”

## Review before the final pass

Do not spend time refining an incorrect draft.

Check these points:

- Both faces still resemble their references.
- The correct identity is assigned to the correct body.
- The pose and camera angle are close to the goal.
- Hands, feet, and body proportions are usable.
- Faces are not tiny, hidden, cropped, or turned fully away.

If one identity is wrong, regenerate the Qwen draft with a simpler pose and a clearer numbered instruction. Do not use RealVisXL to repair a swapped identity.

## Run the protected RealVisXL final pass

Use this only after the Qwen draft is good.

The app performs the Qwen-to-RealVisXL handoff automatically within one submission. To use an approved Qwen draft as the locked composition source, make it a third reference first:

1. Generate the Qwen draft with **NSFW / Explicit off**.
2. Review it. Continue only if both identities, the pose, framing, and basic composition are correct.
3. Click **Use as reference** on that accepted result. It is added to the current references as **reference 03**.
4. Keep the original portraits loaded: reference 01 remains the woman and reference 02 remains the man. Do not remove them.
5. Enable **NSFW / Explicit**.
6. Choose **RealVisXL final pass — recommended**.
7. Start with **Identity-safe final denoise: 0.30–0.35**.
8. Keep **Final LoRA strength** at the configured default unless you have an SDXL-compatible final LoRA installed.
9. Submit with a prompt that declares references 01 and 02 as the identity sources and reference 03 as the approved composition draft.

Use this second-pass prompt pattern:

```text
Reference 01 is the woman and reference 02 is the man. Preserve both identities exactly.
Reference 03 is the approved composition, pose, camera framing, and scene draft. Preserve it exactly.

Refine only [the requested adult styling, wardrobe change, body detail, lighting, or scene detail]. Keep both faces visible, realistic anatomy, natural skin texture, and correct hands.
```

This is intentional: the second request still runs Qwen before RealVisXL, but Qwen now has the two original identity portraits plus the approved composition image. RealVisXL then receives that new Qwen draft automatically.

What happens:

1. Qwen uses references 01, 02, and 03 to create the multi-reference identity and composition draft on the H100.
2. RealVisXL refines the draft on the L40S.
3. Every face detected in the Qwen draft is restored with a soft head/face protection mask. This prevents either person’s identity from drifting during the final pass.

For two people, the final pass intentionally does **not** let a single global face adapter favor reference 01. The protected Qwen faces are the identity source of truth.

## Settings guide

| Goal | Qwen draft | RealVisXL denoise | Notes |
| --- | --- | --- | --- |
| Maximum identity retention | Base Qwen, Anatomy Protection on | 0.25–0.30 | Best default for two people. |
| Change clothing or lighting | Base Qwen | 0.30–0.35 | Describe the wardrobe clearly in Qwen. |
| New full-body pose | Base Qwen, simple one-action pose | 0.25–0.32 | Get the pose right in Qwen before finalizing. |
| More photoreal body/skin detail | Base Qwen | 0.35–0.40 | Inspect faces closely after rendering. |
| Aggressive final restyling | Base Qwen | 0.40–0.45 maximum | Higher values increase drift and seam risk. |

Avoid denoise above `0.45` when identity matters. Values above `0.50` are for experimentation, not reliable identity preservation.

## Prompt patterns

### Change clothes

```text
Reference 01 is the woman and reference 02 is the man. Preserve both identities exactly.
Keep the current pose, camera angle, and scene. Change only the woman’s outfit to [outfit] and the man’s outfit to [outfit]. Keep realistic fabric, accurate hands, and natural body proportions.
```

### Change pose

```text
Reference 01 is the woman and reference 02 is the man. Preserve both identities exactly.
Create a full-body photo. The woman is [precise pose] and the man is [precise pose]. Their faces remain visible in three-quarter view. Keep [scene], [lighting], and [camera framing].
```

### Nude or adult styling

```text
Reference 01 is the woman and reference 02 is the man. Preserve both identities exactly.
Create a photorealistic adult image of consenting adults. Use [precise pose], [camera angle], [lighting], and [scene]. Keep both faces visible and unobstructed, realistic anatomy, natural skin texture, and correct hands.
```

Keep prompts concise. SDXL/RealVisXL has a shorter text context than Qwen, so place the essential pose, camera, and style instruction early.

## If identity drifts

1. Lower final denoise by `0.05`.
2. Use clearer, unfiltered portrait references.
3. Make the faces larger and visible in the requested Qwen composition.
4. Simplify the pose; generate the correct pose first, then make a second edit for styling.
5. Regenerate the Qwen draft rather than repeatedly finalizing an identity-swapped result.
6. For two people, confirm Qwen assigned reference 01 and reference 02 correctly before enabling the final pass.

## If the final pass fails

- Run `modal run wan_modal.py::download_face_models` once, then deploy again.
- Check Modal logs for `protected_faces=`. For a couple image it should normally be `2`.
- If it reports no detected faces, make the Qwen draft more front-facing or use Qwen direct NSFW mode for that composition.
- If face/head boundaries look too unchanged, lower the final denoise; the protection mask is intentionally conservative.
