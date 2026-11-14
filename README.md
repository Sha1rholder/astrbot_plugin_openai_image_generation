[简体中文](.astrbot-plugin/i18n/README.zh-CN.md)

# OpenAI Image Generator

An AstrBot plugin for generating and editing images via gpt-image-2.6 through OpenAI Responses API or Image API. Designed with minimal prompts and reasonable defaults.

Requires an `OPENAI_API_KEY` in the plugin settings or environment variable.

Generates lossless static WebP in the background. Keeps request and image archives, with quotas, proxies, and storage cleanup. Responses API can continue edits using archived response IDs.

Agents can choose the size, quality, complexity, transparency, and image count per request.
