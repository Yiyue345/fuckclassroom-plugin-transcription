from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path


_COMPUTE_TYPE_ORDER = ("int8", "int8_float16", "float16", "float32", "default")
_FALLBACK_MODEL_ORDER = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
)


@dataclass(frozen=True, slots=True)
class AsrRuntimeCapabilities:
    cpu_compute_types: tuple[str, ...]
    cuda_compute_types: tuple[str, ...]
    cuda_available: bool
    cuda_reason: str
    cuda_device_count: int
    auto_device: str
    ctranslate2_version: str

    def compute_types_for(self, device: str) -> tuple[str, ...]:
        resolved = self.auto_device if device == "auto" else device
        return self.cuda_compute_types if resolved == "cuda" else self.cpu_compute_types

    def to_context(self) -> dict[str, object]:
        return {
            "cuda_available": self.cuda_available,
            "cuda_reason": self.cuda_reason,
            "cuda_device_count": self.cuda_device_count,
            "auto_device": self.auto_device,
            "ctranslate2_version": self.ctranslate2_version,
            "compute_types": {
                "cpu": list(self.cpu_compute_types),
                "cuda": list(self.cuda_compute_types),
            },
        }


def detect_asr_runtime() -> AsrRuntimeCapabilities:
    try:
        import ctranslate2
    except Exception as exc:
        return AsrRuntimeCapabilities(
            cpu_compute_types=("int8", "float32", "default"),
            cuda_compute_types=(),
            cuda_available=False,
            cuda_reason=f"CTranslate2 不可用：{exc}",
            cuda_device_count=0,
            auto_device="cpu",
            ctranslate2_version="",
        )

    version = str(getattr(ctranslate2, "__version__", "") or "")
    cpu_types = _supported_compute_types(ctranslate2, "cpu", ("int8", "float32", "default"))

    try:
        cuda_device_count = max(0, int(ctranslate2.get_cuda_device_count()))
    except Exception:
        cuda_device_count = 0

    cuda_reason = ""
    if cuda_device_count <= 0:
        cuda_reason = "未检测到可用的 NVIDIA CUDA 设备"
    else:
        missing = _missing_cuda_runtime_libraries(ctranslate2)
        if missing:
            cuda_reason = "缺少或无法加载 " + "、".join(missing)

    cuda_available = not cuda_reason
    cuda_types = (
        _supported_compute_types(
            ctranslate2,
            "cuda",
            ("int8", "int8_float16", "float16", "float32", "default"),
        )
        if cuda_available
        else ()
    )
    if cuda_available and not cuda_types:
        cuda_available = False
        cuda_reason = "CTranslate2 未报告可用的 CUDA 计算精度"

    return AsrRuntimeCapabilities(
        cpu_compute_types=cpu_types,
        cuda_compute_types=cuda_types,
        cuda_available=cuda_available,
        cuda_reason=cuda_reason,
        cuda_device_count=cuda_device_count,
        auto_device="cuda" if cuda_available else "cpu",
        ctranslate2_version=version,
    )


_MODEL_DOWNLOAD_SIZE_LABELS = {
    "tiny": "约 70 MB",
    "tiny.en": "约 70 MB",
    "base": "约 130 MB",
    "base.en": "约 130 MB",
    "small": "约 440 MB",
    "small.en": "约 440 MB",
    "medium": "约 1.4 GB",
    "medium.en": "约 1.4 GB",
    "large-v1": "约 2.8 GB",
    "large-v2": "约 2.8 GB",
    "large-v3": "约 2.8 GB",
    "large-v3-turbo": "约 1.6 GB",
    "turbo": "约 1.6 GB",
}


def build_settings_context(services, request) -> dict[str, object]:
    context = {
        "asr_runtime": detect_asr_runtime().to_context(),
        "asr_models": list(available_asr_models()),
        "asr_model_size_labels": dict(_MODEL_DOWNLOAD_SIZE_LABELS),
    }
    context.update(build_model_settings_context(services, request))
    return context


def build_model_settings_context(services, request) -> dict[str, object]:
    del request
    model_names = list(available_asr_models())
    service = services.maybe("transcription_service")
    if service is None:
        return {
            "asr_model_statuses": [
                {
                    "model": name,
                    "installed": False,
                    "size_bytes": 0,
                    "size_label": "",
                    "path": "",
                }
                for name in model_names
            ],
            "asr_model_status_error": "语音转写服务尚未启动",
        }

    try:
        statuses = service.model_statuses()
    except Exception as exc:  # Settings should remain usable when the worker is unavailable.
        return {
            "asr_model_statuses": [
                {
                    "model": name,
                    "installed": False,
                    "size_bytes": 0,
                    "size_label": "",
                    "path": "",
                }
                for name in model_names
            ],
            "asr_model_status_error": str(exc),
        }

    by_name = {
        str(item.get("model") or ""): dict(item)
        for item in statuses
        if isinstance(item, dict)
    }
    return {
        "asr_model_statuses": [
            by_name.get(
                name,
                {
                    "model": name,
                    "installed": False,
                    "size_bytes": 0,
                    "size_label": "",
                    "path": "",
                },
            )
            for name in model_names
        ],
        "asr_model_status_error": "",
    }


def available_asr_models() -> tuple[str, ...]:
    try:
        from faster_whisper.utils import _MODELS as model_repositories
    except Exception:
        return _FALLBACK_MODEL_ORDER

    names = {str(name).strip() for name in model_repositories if str(name).strip()}
    ordered = [name for name in _FALLBACK_MODEL_ORDER if name in names]
    ordered.extend(sorted(names.difference(ordered)))
    return tuple(ordered) or _FALLBACK_MODEL_ORDER


def resolve_auto_device() -> str:
    return detect_asr_runtime().auto_device


def resolve_asr_runtime_config(
    device: str,
    compute_type: str,
    *,
    capabilities: AsrRuntimeCapabilities | None = None,
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    capabilities = capabilities or detect_asr_runtime()
    requested_device = str(device or "auto").strip().lower() or "auto"
    requested_compute_type = str(compute_type or "default").strip().lower() or "default"

    resolved_device = (
        capabilities.auto_device
        if requested_device == "auto"
        else requested_device
    )
    messages: list[tuple[str, str]] = []
    if requested_device == "auto":
        messages.append(("info", f"ASR auto 已解析为 {resolved_device.upper()}"))

    supported = capabilities.compute_types_for(resolved_device)
    resolved_compute_type = requested_compute_type
    if supported and requested_compute_type not in supported:
        resolved_compute_type = next(
            (
                candidate
                for candidate in ("int8", "float32", "default")
                if candidate in supported
            ),
            supported[0],
        )
        messages.append(
            (
                "warning",
                f"{requested_compute_type} 不适用于当前 {resolved_device.upper()}，"
                f"本次使用 {resolved_compute_type}",
            )
        )

    return resolved_device, resolved_compute_type, tuple(messages)


def _supported_compute_types(module, device: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    try:
        supported = {str(item) for item in module.get_supported_compute_types(device)}
    except Exception:
        supported = set(fallback)
    supported.add("default")
    ordered = tuple(item for item in _COMPUTE_TYPE_ORDER if item in supported)
    return ordered or tuple(item for item in fallback if item in _COMPUTE_TYPE_ORDER)


def _missing_cuda_runtime_libraries(ctranslate2_module) -> list[str]:
    if sys.platform == "win32":
        package_dir = Path(ctranslate2_module.__file__).resolve().parent
        required = (
            ("cublas64_12.dll", "CUDA 12 cuBLAS"),
            ("cudnn64_9.dll", "cuDNN 9"),
        )
        missing: list[str] = []
        for filename, label in required:
            if not _loadable_windows_library(filename, package_dir):
                missing.append(label)
        return missing

    if sys.platform.startswith("linux"):
        required = (
            ("libcublas.so.12", "CUDA 12 cuBLAS"),
            ("libcudnn.so.9", "cuDNN 9"),
        )
        missing = []
        for filename, label in required:
            try:
                ctypes.CDLL(filename)
            except OSError:
                missing.append(label)
        return missing

    return ["当前操作系统不支持 CUDA 转写"]


def _loadable_windows_library(filename: str, package_dir: Path) -> bool:
    candidates: list[Path] = [package_dir / filename]

    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin" / filename)

    for raw_dir in os.environ.get("PATH", "").split(os.pathsep):
        raw_dir = raw_dir.strip().strip('"')
        if raw_dir:
            candidates.append(Path(raw_dir) / filename)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        try:
            add_directory = getattr(os, "add_dll_directory", None)
            if callable(add_directory):
                handle = add_directory(str(candidate.parent))
                try:
                    ctypes.WinDLL(str(candidate))
                finally:
                    handle.close()
            else:
                ctypes.WinDLL(str(candidate))
            return True
        except OSError:
            continue

    try:
        ctypes.WinDLL(filename)
        return True
    except OSError:
        return False
