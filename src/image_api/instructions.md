# Image generation instructions (Image API)

Use `openai_image_generate_webp_image_api` to generate and deliver lossless static WebP with OpenAI Image API and `gpt-image-2.6-sunburst`.

## Parameters

- `prompt`: Object with required `text` and optional `images` reference indices
- `description`: A short description of the images you expect. Automatically sent to the user with all generated images in the same message.
- (Optional) `size`: String in `WIDTHxHEIGHT` format such as `1536x1024`. Both dimensions must be multiples of 16 and no greater than 3840. The aspect ratio must be between 1:3 and 3:1. Total pixels must be between 655,360 and 8,294,400. Resolutions above 2560x1440 total pixels are experimental. Automatically selected if omitted.
- (Optional) `quality`: `low`, `medium`, `high`, `xhigh`, `max`. Defaults to `auto`.
- (Optional) `complexity`: `low`, `medium`, `high`. Control how much the image model plans before rendering. Lower complexity can suit focused edits, such as changing a background or a sign. Higher complexity can help with dense text, complex layouts, and relationships between objects. Higher complexity can increase latency and introduce unwanted changes, so compare results for your task. Defaults to `high`.
- (Optional) `background_transparency`: `opaque` or `transparent`. Defaults to `auto`.
- (Optional) `number_of_images`: Integer from 1 to 5. Defaults to 1; set manually only when generating multiple images at once.

In most cases you should omit optional arguments as all have reasonable defaults.

## Reference images

Indices start at 1: images in the current message first, in appearance order, followed by images in quoted messages. The instructions tool lists available indices at the end. Select references explicitly in `prompt.images`, up to 16 images. For example: `{"prompt":{"text":"参考画像1の色合いを使って、夕暮れの海辺に佇む魔法少女姿の美樹さやかを描いて。足元には手放した剣、胸元にはひび割れたソウルジェム。届かなかった想いを波音に託すように、涙をこらえて微笑んでいる。青く沈む空と淡い夕日の中に、切なさと儚い希望が漂うアニメイラスト","images":[1]},"description":"海辺で泣いているさやか"}`. Omitting `images` or providing an empty array means text-only generation, even when attachments exist. If an image is only in earlier conversation history, ask the user to resend or quote it. Do not guess indices or supply filesystem paths.

## Writing image prompts

Describe the subject, composition, style, intended use, and required constraints. Specify exact lettering and placement when text is needed. Explain what each reference contributes, such as subject appearance, colors, or composition. Keep API options in their parameters; use the prompt to describe the desired image.

Image API rejects NSFW content. DO NOT INCLUDE INAPPROPRIATE CONTENT IN YOUR PROMPTS.

## Examples

Everyday creation:

```json
{"prompt":{"text":"一只橘猫在享受午后的阳光，水彩插画风格"},"description":"水彩午后橘猫"}
```

Fast bulk generation:

```json
{"prompt":{"text":"An angry chibi anime girl with black hair in her bedroom with pink background saying \"Why are u licking my shoes!\""},"description":"Angry chibi girl emoji","model":"flare","quality":"low","complexity":"low","size":"1024x1240","number_of_images":5}
```

Maximum quality:

```json
{"prompt": {"text": "目标 / 用途\n商业游戏主视觉级的日系二次元角色插画，完整单幅成图。非摄影、非三维渲染、非Q版。16:9横构图。\n\n场景\n雨后初晴的欧式庭院茶室内，一位猫娘小女仆正在为来访客人送上下午茶。室内有奶油白墙面、深胡桃木落地窗框、黄铜把手、浅色石材地面。左侧露出一张小圆茶桌，铺米白桌布，桌上有插着少量白色小花的细口花瓶。窗外可见被雨水洗过的玫瑰花架、层叠绿叶和湿润石板小径，叶片带少量水珠；远景柔和概括，但仍能看出庭院结构。前景左下可有少量虚化绿叶，右后方用窗框和浅色窗帘制造空间层次。\n\n主体 / 角色\n画面中只有一位猫娘少女，约6头身，身形纤细自然。银灰色长发，发尾微卷至腰间，碎刘海轻盈整齐，两侧有贴脸细发束；背光边缘带淡淡暖金色。头顶一对与发色一致的猫耳，耳尖略深，内侧低饱和淡粉，生长位置自然；人耳被侧发遮住。只有一条蓬松银灰猫尾，从腰后自然伸出，在画面右侧轻轻弯起，不遮挡面部、双手和托盘。脸部轮廓柔和，五官精致；湖绿色眼睛有清楚瞳孔与少量顺光高光，上睫毛清晰。她微微侧头看向观众，带含蓄欢迎的微笑，亲切而略羞涩。\n\n服装\n经典黑白女仆装：黑色修身连衣裙，白色小翻领，胸前窄黑丝带，短袖略蓬，袖口洁白；及膝裙摆外覆整洁白围裙，边缘有细窄双层荷叶褶与少量镂空蕾丝，腰后蝴蝶结结构清楚。头戴小巧白色褶边女仆发饰，位于两只猫耳之间。领口中央有一枚小型祖母绿饰针。深灰不透肤连裤袜，黑色圆头低跟玛丽珍鞋。\n\n动作 / 道具\n她站在茶室通往庭院的落地窗旁，身体朝画面左侧轻转约四分之一，脸转回观众方向，双肩放松，重心自然落在后侧腿，另一只脚略向前。双手在腰前端一只小型椭圆银色托盘：左手托底，右手扶侧缘。托盘上放一只带浅金细边的白瓷茶杯与茶碟，旁边是一小碟草莓奶油蛋糕；红茶升起两三缕轻薄蒸汽。银盘反光克制，不出现杂乱镜面倒影。\n\n构图 / 光影 / 画风\n角色完整全身出镜并包含鞋子，占画面高度约85%，位置在中线稍偏右；猫耳上方、鞋底下方和尾巴周围都留出适当空间。镜头接近胸口高度，透视自然，无广角畸变。脸部是第一焦点，双手与下午茶是第二焦点。主光从左上方窗外照入，照亮面部迎光侧、猫耳绒毛边缘、肩部、围裙褶皱和茶杯口沿；暗部由室内冷灰蓝环境光柔和补亮。主色为银灰、奶油白、柔黑、鼠尾草绿，以少量祖母绿和草莓红点缀。线稿干净且有粗细变化，赛璐璐分区清晰，局部有柔和过渡；面部、双手、猫耳、蕾丝精细刻画，背景适度概括。\n\n约束\n不要出现其他人物、动物、重复面孔、多余肢体、畸形手、融合手指、额外猫耳或尾巴、悬浮道具、穿模、错误倒影。不要添加任何文字、字幕、标志、水印、签名、画框或拼图分栏。整体氛围温柔、灵动、清新、优雅，明亮通透，富有生活感与故事感"},"description": "雨后庭院茶室中的猫女仆","quality": "max","size": "2560x1440"}
```

In most cases, your prompt should be much longer with more detailed descriptions.

## Finding previous requests

Search archived requests by keywords from the user's request in one command. Descriptions are stored in `request.yaml` as `request.description`. Replace `ARCHIVE_DIR` with the image archive directory provided below:

```bash
rg -n -i -F -g request.yaml -- 'keyword' 'ARCHIVE_DIR'
```

Each match identifies a request directory. Read `request.yaml` to check the description and original tool arguments. The directory and request file are created as soon as the tool accepts the request; `0.webp`, `1.webp`, etc. are added after successful generation. Pending or failed requests may have no images. Check that images exist before sending them. You can send the matching images or, if requested, `request.yaml` to the user after finding them.
