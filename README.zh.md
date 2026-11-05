# OpenAI Image API WebP Generator

AstrBot生图插件，调用`gpt-image-2.6-flare`或`gpt-image-2.6-sunburst`，将生成的WebP图片直接发送到当前会话

不提供蒙板选项

## 配置

将插件安装到AstrBot的`data/plugins/astrbot_plugin_oai_image_api_webp_generate`，在插件管理页填写`api_key`并启用插件

密钥按以下优先级读取：插件配置`api_key`、环境变量`OPENAI_API_KEY`、插件目录下的`OPENAI_API_KEY`文件、插件目录下的`.local/OPENAI_API_KEY`文件

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `api_key` | 空 | OpenAI API密钥，在管理页中遮罩显示 |
| `base_url` | `https://api.openai.com/v1` | API根地址，代理必须支持相应模型及JSON格式的Image API |
| `timeout_seconds` | 300 | 单次API请求的网络超时，不自动重试 |

在AstrBot的Agent配置中将工具调用超时设得大于API请求超时，建议至少360秒；较多参考图、Sunburst或高质量输出可能需要更长时间
修改配置后重载插件以重新创建连接

## 使用

启用聊天模型的函数调用，并向它提出生图请求，例如“画一只水彩风格的橘猫”

插件提供两个模型工具：

- `get_image_generation_instructions`：按需读取参数、提示词建议和当前消息的参考图编号
- `generate_webp_image`：生成图片并发送到会话

系统提示、工具描述、按需手册和工具返回信息均为英文，不包含本地化机制
常驻系统提示只添加一句工具入口说明，不注入完整手册；若当前人格未启用这两个工具，则不添加提示
用户也可以发送`/image_help`查看完整说明

默认使用Flare、medium质量、high规划复杂度、不透明背景、自动尺寸和1张图片
参考图需由模型在`prompt.images`中显式选择，编号先排列当前消息图片，再排列引用消息图片；历史图片需要重新发送或引用

详细参数与调用示例见[生图工具说明](image_generation_instructions.md)

## 输出和错误处理

固定使用`output_format=webp`、`output_compression=100`、`moderation=low`、`partial_images=0`、`stream=false`
质量100是API的压缩参数，不承诺编码一定无损；插件验证返回格式后保存原始字节，不重新编码

生成结果保存在AstrBot数据目录下的`plugin_data/astrbot_plugin_oai_image_api_webp_generate/images/`，卸载或重载时不删除，管理员可按需清理
平台可能对图片重新压缩，原始WebP以数据目录内的文件为准
如果图片发送失败，工具会报告已生成、已发送的数量和保存目录，避免把平台发送失败误判为生成失败
超时或网络中断时服务端可能仍已计费，插件不会自动重试

## 开发验证

插件目录中的`.venv`使用相对符号链接复用AstrBot的环境，无需重复安装依赖
首次在AstrBot的`data/plugins/`下检出插件时创建链接：

```bash
ln -s ../../../.venv .venv
```

之后直接在插件目录运行：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff format --config 'exclude=[]' main.py image_api.py tests
.venv/bin/python -m ruff check --config 'exclude=[]' main.py image_api.py tests
npx --yes pyright
```

编辑器通过`.vscode/settings.json`选择`.venv`，Pyright通过`pyrightconfig.json`继承AstrBot的规则并显式检查插件，避免被上游的`data/`排除规则跳过
测试默认使用模拟HTTP响应，不调用付费API；AstrBot自身在Python3.12导入`audioop`时仍会发出弃用警告

## 官方文档

- [Image API指南](https://developers.openai.com/api/docs/guides/image-generation)
- [图片生成接口](https://developers.openai.com/api/reference/resources/images/methods/generate)
- [参考图接口](https://developers.openai.com/api/reference/resources/images/methods/edit)
- [GPT Image 2.6提示词指南](https://developers.openai.com/api/docs/guides/image-prompting?model=gpt-image-2.6)
