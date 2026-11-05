# Image generation instructions

Use `openai_image_api_generation_webp` to generate and deliver lossless static WebP with OpenAI's latest image generation models, currently `gpt-image-2.6-sunburst` and `gpt-image-2.6-flare`. Unless user specifies, in most cases you should omit optional arguments as all have reasonable defaults.

## Parameters

- `prompt`: Object with required `text` and optional `images` reference indices
- `description`: A short description of the images you expect. Automatically sent to the user before all generated images in the same message.
- (Optional) `model`: `sunburst` offers better quality at higher cost and latency; `flare` is faster and cheaper. Defaults to `sunburst`.
- (Optional) `size`: String in `WIDTHxHEIGHT` format such as `1536x1024`. Both dimensions must be multiples of 16 and no greater than 3840. The aspect ratio must be between 1:3 and 3:1. Total pixels must be between 655,360 and 8,294,400. Resolutions above 2560x1440 total pixels are experimental. Automatically selected if omitted.
- (Optional) `quality`: `low`, `medium`, `high`, `xhigh`, `max`. Defaults to `auto`.
- (Optional) `complexity`: `low`, `medium`, `high`. Defaults to `high`; set manually only when reducing generation latency is needed.
- (Optional) `background_transparency`: `opaque` or `transparent`. Defaults to `auto`.
- (Optional) `number_of_images`: Integer from 1 to 5. Defaults to 1; set manually only when generating multiple images at once.

## Reference images

Indices start at 1: images in the current message first, in appearance order, followed by images in quoted messages. The instructions tool lists available indices at the end. Select references explicitly in `prompt.images`, up to 16 images. For example: `{"prompt":{"text":"参考画像1の色合いを使って、夕暮れの海辺に佇む魔法少女姿の美樹さやかを描いて。足元には手放した剣、胸元にはひび割れたソウルジェム。届かなかった想いを波音に託すように、涙をこらえて微笑んでいる。青く沈む空と淡い夕日の中に、切なさと儚い希望が漂うアニメイラスト","images":[1]},"description":"海辺で泣いているさやか"}`. Omitting `images` or providing an empty array means text-only generation, even when attachments exist. If an image is only in earlier conversation history, ask the user to resend or quote it. Do not guess indices or supply filesystem paths.

## Writing image prompts

Describe the subject, composition, style, intended use, and required constraints. Specify exact lettering and placement when text is needed. Explain what each reference contributes, such as subject appearance, colors, or composition. Keep API options in their parameters; use the prompt to describe the desired image.

Image API rejects NSFW content. DO NOT INCLUDE INAPPROPRIATE CONTENT IN YOUR PROMPTS.

## Examples

Minimal generation:

```json
{"prompt":{"text":"一只橘猫在享受午后的阳光，水彩插画风格"},"description":"水彩午后橘猫"}
```

```json
{"prompt":{"text":"An angry chibi anime girl with black hair in her bedroom with pink background saying \"Why are u licking my shoes!\""},"description":"Angry chibi girl emoji","model":"flare","quality":"low","complexity":"low","size":"1024x640","number_of_images":2}
```

In most cases, your prompt should be much longer with more detailed descriptions.

When the tool returns a queued status, generation is running in the background and the images have not been delivered yet. Your description and generated images will be automatically sent together after generation completes.

## Finding previous requests

Search descriptions by keywords from the user's request in one command. Replace `ARCHIVE_DIR` with the image archive directory provided below:

```bash
rg -n -i -F -g description.txt -- 'keyword' 'ARCHIVE_DIR'
```

Each match gives the description and its request directory. That directory contains `0.webp`, `1.webp`, etc., and `request.yaml` with the original tool arguments. You can send the matching images or, if requested, `request.yaml` to user after finding them.
