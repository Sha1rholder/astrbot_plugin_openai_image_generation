# OpenAI Image API生图

一个易用、极简的AstrBot插件，通过OpenAI Image API为agent提供OpenAI最新图像生成能力。系统提示词极简，图像生成记录持久保存，并提供合理且可由agent调整的默认参数。支持gpt-image-2.6-sunburst和gpt-image-2.6-flare

请在插件配置中填写API密钥，或通过环境变量OPENAI_API_KEY提供

工具参数解析和校验通过后，请求保存至`data/plugin_data/astrbot_plugin_openai_image_generation/images/<请求id>/request.yaml`，顶层`user_id`自动取自AstrBot原生的`event.get_sender_id()`，无需agent提供

同时向`data/plugin_data/astrbot_plugin_openai_image_generation/requests/<用户id>.log`追加一行：请求接收时间的Unix时间戳（整数秒）、制表符、请求id（即`request.yaml`所在的自增目录名）。记录发生在密钥检查、参考图读取和API调用之前，后续失败也保留记录；参数解析或校验失败则不记录。用户ID中的特殊字符在文件名中使用百分号编码，YAML保留原始ID。日志不随图片存储清理而删除。用户ID沿用平台提供的发送者标识，不做跨平台账号合并
