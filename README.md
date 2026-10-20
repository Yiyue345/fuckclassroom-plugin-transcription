# FuckClassroom 本地语音转写插件

FuckClassroom 的独立本地语音转写插件，插件 ID 为 `transcription`。

## 功能

- faster-whisper 本地语音转写
- CUDA / CPU 运行环境检测与回退
- 模型下载、删除、缓存状态与下载进度
- 简体中文转换
- 独立 Worker 进程执行长时间转写任务

## 模型推荐与别名

模型下拉列表优先展示适用于中文课堂的 `large-v3-turbo`（速度与质量均衡）、`large-v3`（精度优先）、`medium`（资源折中），并提示小模型和仅英语模型的局限。推荐属于定性指引，不代表在所有录音上的精度排名。

模型选择按 faster-whisper 的 Hugging Face 仓库 ID 去重，例如 `turbo` 和 `large-v3-turbo` 只展示 `large-v3-turbo`，`large` 和 `large-v3` 只展示 `large-v3`。**历史设置中的别名仍然兼容**，打开设置会显示标准名称；不会重复下载同一仓库。

## 依赖

- FuckClassroom: `>=0.1,<0.3`
- Plugin API: `1`
- Required plugin: `processing`
- `faster-whisper>=1.1`
- `huggingface-hub>=0.32`
- `opencc-python-reimplemented>=0.1.7`

Worker 通过插件包内的 `.engine` 加载实现，不依赖主仓的 `fuckclassroom.plugins.transcription` 包。

## 开发

开发分支为 `plugin-management`。合并到 `main` 后，CI 成功会自动发布 Registry v1 beta Release。
