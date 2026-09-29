# FuckClassroom 本地语音转写插件

FuckClassroom 的独立本地语音转写插件，插件 ID 为 `transcription`。

## 功能

- faster-whisper 本地语音转写
- CUDA / CPU 运行环境检测与回退
- 模型下载、删除、缓存状态与下载进度
- 简体中文转换
- 独立 Worker 进程执行长时间转写任务

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
