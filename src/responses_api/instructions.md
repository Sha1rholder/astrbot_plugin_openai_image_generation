# Image generation instructions (Responses API)

Use `openai_image_generate_webp_responses_api` to generate and deliver lossless static WebP with OpenAI Responses API and `gpt-image-2.6-sunburst`.

## Parameters

- `prompt`: Object with required `text` and optional `images` reference indices
- (Optional) `previous_response_id`: The `response_id` saved in a successful earlier request's archive. Use it to continue editing those images with a new prompt. Omit or use `null` for an independent request. Never invent an ID or use a local request number or image generation call ID.
- (Optional) `size`: String in `WIDTHxHEIGHT` format such as `1536x1024`. Both dimensions must be multiples of 16 and no greater than 3840. The aspect ratio must be between 1:3 and 3:1. Total pixels must be between 655,360 and 8,294,400. Resolutions above 2560x1440 total pixels are experimental. Automatically selected if omitted.
- (Optional) `quality`: `low`, `medium`, `high`, `xhigh`, `max`. Defaults to `auto`.
- (Optional) `complexity`: `low`, `medium`, `high`. Control how much the image model plans before rendering. Lower complexity can suit focused edits, such as changing a background or a sign. Higher complexity can help with dense text, complex layouts, and relationships between objects. Higher complexity can increase latency and introduce unwanted changes, so compare results for your task. Defaults to `high`.
- (Optional) `background_transparency`: `opaque` or `transparent`. Defaults to `auto`.
- (Optional) `number_of_images`: Integer from 1 to 5. Defaults to 1; set manually only when generating multiple images at once.

In most cases you should omit optional arguments as all have reasonable defaults.

## References and delivery

The instructions tool lists available image indices below. Indices start at 1, with current message images first, followed by quoted images. Select up to 16 indices in `prompt.images`, in the order you describe them. Omit `images` for text-only generation or when continuing with `previous_response_id` alone. A continuation may also include new reference indices. If an earlier image has no usable response ID, ask the user to resend or quote it. Use the listed indices for references.

```json
{"prompt":{"text":"把图改为水彩画风","images":[1]}}
```

The tool blocks until the API request body is sent or an earlier failure is known. A sent status confirms transmission only, not server acceptance or completed generation. Acknowledge the sent request once without claiming completion; do not wait or call the tool again. On failure, explain the tool error instead of claiming submission. The Responses API describes the generated images after generation. Its returned text and all images will be delivered together when ready. Only claim completion after the results have actually been delivered.

## Previous requests

Search the archive directory shown below with `rg -n -i -F -g request.yaml -- 'keyword' 'ARCHIVE_DIR'`. Each numbered directory contains the original arguments in `request.yaml`. The submission acknowledgement gives the exact archive path for that request. Successful generations also contain `0.webp`, `1.webp`, etc. Verify that image files exist before sending them.

After a successful Responses API generation, `request.yaml` also contains a top-level `response_id` when the provider returns one. For a follow-up edit, read the archive corresponding to the user's intended image and pass that ID as `previous_response_id`. Do not reuse `request.previous_response_id`, which points to the parent response. No previous response is selected automatically. Wait for generation to finish before reading the new ID.

```json
{"prompt":{"text":"保留主体，将背景改为雪山"},"previous_response_id":"resp_..."}
```

The plugin sends `store: true` so the provider can retain response context. Continuation requires the same endpoint and account to have access to the stored response. If an ID has expired or is unavailable, report the failure and ask the user to resend or quote the image for a new request. Do not automatically retry or silently drop `previous_response_id`.
