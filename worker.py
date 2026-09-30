from __future__ import annotations

from pathlib import Path

from fuckclassroom.core.config import AppConfig
from .engine import LocalTranscriptionService


_service: LocalTranscriptionService | None = None


def _get_service(context) -> LocalTranscriptionService:
    global _service
    if _service is None:
        _service = LocalTranscriptionService(AppConfig(data_dir=context.data_dir))
    return _service


def handle_call(method, params, context, progress):
    service = _get_service(context)

    if method == "transcription.model_statuses":
        return service.model_statuses()
    if method == "transcription.download_model":
        return service.download_model(
            str(params.get("model") or ""),
            progress=progress,
        )
    if method == "transcription.delete_model":
        model = str(params.get("model") or "")
        progress(10, f"正在删除语音模型 {model}")
        result = service.delete_model(model)
        progress(100, f"语音模型 {model} 已删除")
        return result
    if method != "transcription.transcribe_url":
        raise ValueError(f"未知语音转写 Worker 方法：{method}")

    result = service.transcribe_url(
        str(params.get("video_url") or ""),
        Path(str(params.get("output_path") or "")),
        progress=progress,
        force=bool(params.get("force", False)),
        duration_hint_seconds=params.get("duration_hint_seconds"),
    )
    return {
        "path": str(result.path),
        "text": result.text,
        "cached": result.cached,
        "duration_seconds": result.duration_seconds,
        "export_path": str(result.export_path) if result.export_path else "",
    }
