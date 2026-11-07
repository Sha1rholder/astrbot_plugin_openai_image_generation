# OpenAI图像生成

AstrBot插件，通过OpenAI Responses API或Image API支持文生图和参考图生图，保持极简提示词和合理默认值

需要在插件配置中填或通过环境变量提供`OPENAI_API_KEY`

后台生成无损静态WebP，在agent回复结束后发送结果。保留请求和图片存档，支持配额、代理和存储清理。Responses API可通过存档中的响应ID继续编辑

Agent可在单次请求中选择模型、尺寸、质量、复杂度、透明背景和生图数量
