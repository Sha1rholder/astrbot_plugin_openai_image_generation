# OpenAI图像生成

AstrBot插件，使用gpt-image-2.6，通过OpenAI Responses API或Image API支持文生图和图片编辑，保持极简提示词和合理默认值

需要在插件配置中填或通过环境变量提供`OPENAI_API_KEY`

后台生成无损静态WebP。保留请求和图片存档，支持配额、代理和存储清理。Responses API可通过存档中的响应ID继续编辑

Agent可主动选择尺寸、质量、复杂度、透明背景和生图数量
