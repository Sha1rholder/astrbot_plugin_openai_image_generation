# OpenAI图像生成

AstrBot插件，支持文生图和参考图生图。用户通过“是否自动优化提示词（会产生额外tokens消耗）”选择接口。默认开启，通过Responses API调用LLM优化提示词；关闭后使用Image API

用户可配置默认LLM为`terra`（默认）、`astra`、`sol`或`luna`，agent可通过`llm`参数在单次请求中覆盖。配置修改后请重载插件。两种接口都固定使用`moderation=low`，只输出`output_compression=100`的静态WebP

请在插件配置中填写API密钥，或通过环境变量OPENAI_API_KEY提供

用户配置项支持生图配额、鉴权和代理

Agent能力包括文生图和参考图生图，支持选择生图模型、尺寸、质量、复杂度、透明背景和单次生图数量
