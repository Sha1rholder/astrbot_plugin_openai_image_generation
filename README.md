[简体中文](.astrbot-plugin/i18n/README.zh-CN.md)

# OpenAI Image Generator

A super-easy-to-use AstrBot plugin. It provides your agent with latest image generation capabilities through OpenAI Image API. It has minimal system prompts, persistent storage of image generation records, and reasonable defaults that agent can override in each session. Supports gpt-image-2.6-sunburst and gpt-image-2.6-flare.

Requires an API key entered in the plugin settings or supplied through the OPENAI_API_KEY environment variable.

Accepted tool arguments are archived in `data/plugin_data/astrbot_plugin_openai_image_generation/images/<request_id>/request.yaml`. The top-level `user_id` comes directly from AstrBot's `event.get_sender_id()`; the agent does not provide it. Each parsed and validated request also appends a line to `data/plugin_data/astrbot_plugin_openai_image_generation/requests/<user_id>.log`, containing an integer Unix timestamp in seconds, a tab, and the numbered request directory name. This happens before credential checks, reference loading, or API calls, so subsequent failures remain recorded. Invalid arguments are not recorded. Special characters in IDs are percent-encoded in filenames; the YAML preserves the original ID. Logs are retained independently of image storage cleanup. IDs use the platform's sender identity and are not a cross-platform account mapping.
