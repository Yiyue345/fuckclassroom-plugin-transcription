from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from fuckclassroom.web.responses import task_started_response


def build_router(context) -> APIRouter:
    router = APIRouter()
    task_manager = context.services.get("task_manager")
    transcription_service = context.services.get("transcription_service")

    @router.post("/settings/transcription/models/{model_name}/download")
    def download_model(request: Request, model_name: str) -> Response:
        result_url = "/settings#transcription"

        def worker(progress):
            transcription_service.download_model(model_name, progress=progress)
            return result_url

        task = task_manager.start(f"下载语音模型 · {model_name}", worker)
        return task_started_response(request, task, result_url)

    @router.post("/settings/transcription/models/{model_name}/delete")
    def delete_model(request: Request, model_name: str) -> Response:
        result_url = "/settings#transcription"

        def worker(progress):
            transcription_service.delete_model(model_name, progress=progress)
            return result_url

        task = task_manager.start(f"删除语音模型 · {model_name}", worker)
        return task_started_response(request, task, result_url)

    return router


__all__ = ["build_router"]
