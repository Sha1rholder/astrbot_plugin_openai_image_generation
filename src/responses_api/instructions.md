# Image generation instructions (Response API)

Use `openai_image_generate_webp` to generate and deliver lossless static WebP with OpenAI Responses API and `gpt-image-2.6-sunburst`.

## Parameters

- `prompt`: Object with required `text` and optional `images` reference indices
- (Optional) `size`: String in `WIDTHxHEIGHT` format such as `1536x1024`. Both dimensions must be multiples of 16 and no greater than 3840. The aspect ratio must be between 1:3 and 3:1. Total pixels must be between 655,360 and 8,294,400. Resolutions above 2560x1440 total pixels are experimental. Automatically selected if omitted.
- (Optional) `quality`: `low`, `medium`, `high`, `xhigh`, `max`. Defaults to `auto`.
- (Optional) `complexity`: `low`, `medium`, `high`. Control how much the image model plans before rendering. Lower complexity can suit focused edits, such as changing a background or a sign. Higher complexity can help with dense text, complex layouts, and relationships between objects. Higher complexity can increase latency and introduce unwanted changes, so compare results for your task. Defaults to `high`.
- (Optional) `background_transparency`: `opaque` or `transparent`. Defaults to `auto`.
- (Optional) `number_of_images`: Integer from 1 to 5. Defaults to 1; set manually only when generating multiple images at once.

In most cases you should omit optional arguments as all have reasonable defaults.

## References and delivery

The instructions tool lists available image indices below. Indices start at 1, with current message images first, followed by quoted images. Select up to 16 indices in `prompt.images`, in the order you describe them. Omit `images` for text-only generation. Ask the user to resend or quote images from earlier history. Use the listed indices for references.

```json
{"prompt":{"text":"把图改为水彩画风","images":[1]}}
```

A queued result means generation is running in the background. Acknowledge the request once. The Responses API describes the generated images after generation. Its returned text and all images will be delivered together when ready. Wait for that delivery before claiming completion.

## Previous requests

Search the archive directory shown below with `rg -n -i -F -g request.yaml -- 'keyword' 'ARCHIVE_DIR'`. Each numbered directory contains the original arguments in `request.yaml`. Successful generations also contain `0.webp`, `1.webp`, etc. Verify that image files exist before sending them.
