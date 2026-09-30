from __future__ import annotations

import gc
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fuckclassroom.core.config import AppConfig
from .errors import LocalTranscriptionError


class _ChunkedDecodeError(RuntimeError):
    pass


ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class LocalTranscriptionResult:
    path: Path
    text: str
    cached: bool
    duration_seconds: float | None = None
    export_path: Path | None = None


@dataclass(frozen=True)
class _AudioChunk:
    index: int
    start: float
    end: float
    decode_start: float
    decode_end: float


_MODEL_CACHE: dict[tuple[object, ...], object] = {}
_MODEL_LOCK = threading.Lock()
_TRANSCRIPTION_CONDITION = threading.Condition()
_ACTIVE_TRANSCRIPTIONS = 0
_MODEL_MAINTENANCE = False
_OPENCC_CONVERTER = None
_OPENCC_LOCK = threading.Lock()
_PREPARATION_HEARTBEAT_SECONDS = 10.0
_REMOTE_CHUNK_SECONDS = 10 * 60.0
_REMOTE_DECODE_WORKERS = 2
_REMOTE_CHUNK_OVERLAP_SECONDS = 1.5
_RANGE_PROBE_TIMEOUT_SECONDS = 8.0
_RANGE_SUPPORT_CACHE: dict[str, bool] = {}
_RANGE_SUPPORT_LOCK = threading.Lock()
_MODEL_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


class LocalTranscriptionService:
    """Stream lesson media into faster-whisper without saving the full video locally."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()

    def transcribe_url(
        self,
        video_url: str,
        output_path: Path,
        progress: ProgressCallback | None = None,
        *,
        force: bool = False,
        duration_hint_seconds: float | None = None,
    ) -> LocalTranscriptionResult:
        if not video_url:
            raise LocalTranscriptionError("本课次没有可用的课堂录像")

        if output_path.exists() and not force:
            cached_text = output_path.read_text(encoding="utf-8").strip()
            if cached_text:
                normalized = self._normalize_text(cached_text)
                if normalized != cached_text:
                    output_path.write_text(normalized + "\n", encoding="utf-8")
                export_path = self._export_transcript(output_path, normalized)
                _clear_detail(progress)
                _report(progress, 100, "已复用本地语音转写")
                return LocalTranscriptionResult(
                    output_path,
                    normalized,
                    True,
                    export_path=export_path,
                )

        duration_hint = max(0.0, float(duration_hint_seconds or 0.0))
        with _transcription_slot(self.config.asr_max_concurrent_tasks, progress):
            _report(progress, 4, "正在准备本地语音识别模型")
            using_cpu_fallback = False
            try:
                model = self._get_model(progress)
            except LocalTranscriptionError as exc:
                if not self._should_fallback_to_cpu(exc):
                    raise
                model = self._load_cpu_fallback_model(progress, exc)
                using_cpu_fallback = True
            _clear_detail(progress)

            try:
                lines, duration = self._transcribe_with_model(
                    model,
                    video_url,
                    duration_hint,
                    progress,
                )
            except LocalTranscriptionError:
                raise
            except Exception as exc:  # faster-whisper/PyAV can raise backend-specific errors.
                _raise_if_cancelled(progress)
                if not using_cpu_fallback and self._should_fallback_to_cpu(exc):
                    cpu_model = self._load_cpu_fallback_model(progress, exc)
                    try:
                        lines, duration = self._transcribe_with_model(
                            cpu_model,
                            video_url,
                            duration_hint,
                            progress,
                        )
                    except Exception as cpu_exc:
                        _raise_if_cancelled(progress)
                        raise LocalTranscriptionError(
                            f"GPU 运行库不可用，CPU 回退转写也失败：{cpu_exc}"
                        ) from cpu_exc
                else:
                    raise LocalTranscriptionError(f"本地语音转写失败：{exc}") from exc

        if not lines:
            raise LocalTranscriptionError("录像中没有识别到可用语音")

        text = "\n".join(lines).strip()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f"{output_path.name}.tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(output_path)
        export_path = self._export_transcript(output_path, text)
        _report(progress, 100, "本地语音转写完成，结果已保存到文件管理")
        return LocalTranscriptionResult(
            path=output_path,
            text=text,
            cached=False,
            duration_seconds=duration or None,
            export_path=export_path,
        )

    def _transcribe_with_model(
        self,
        model,
        video_url: str,
        duration_hint: float,
        progress: ProgressCallback | None,
    ) -> tuple[list[str], float]:
        if self._can_pipeline_remote_audio(video_url, duration_hint, progress):
            try:
                return self._transcribe_chunked_remote(
                    model,
                    video_url,
                    duration_hint,
                    progress,
                )
            except _ChunkedDecodeError as exc:
                _raise_if_cancelled(progress)
                _log(
                    progress,
                    f"分块音频预处理失败，自动回退整段流式解码：{exc}",
                    "warning",
                )
                _report(progress, 12, "分块解码不可用，正在回退整段流式音频处理")
        return self._transcribe_single_stream(model, video_url, progress)

    def _should_fallback_to_cpu(self, exc: BaseException) -> bool:
        return self.config.asr_device in {"cuda", "auto"} and _is_cuda_runtime_error(exc)

    def _load_cpu_fallback_model(
        self,
        progress: ProgressCallback | None,
        gpu_error: BaseException,
    ):
        _raise_if_cancelled(progress)
        _log(
            progress,
            f"GPU 转写运行库不可用，自动回退 CPU int8：{gpu_error}",
            "warning",
        )
        _report(progress, 8, "GPU 运行库不可用，正在切换 CPU int8 转写")
        try:
            model = self._get_model(
                progress,
                device="cpu",
                compute_type="int8",
            )
        except Exception as exc:
            raise LocalTranscriptionError(
                f"GPU 运行库不可用，CPU 回退模型加载失败：{exc}"
            ) from exc
        _clear_detail(progress)
        return model

    def _can_pipeline_remote_audio(
        self,
        video_url: str,
        duration_hint: float,
        progress: ProgressCallback | None,
    ) -> bool:
        if duration_hint <= _REMOTE_CHUNK_SECONDS * 1.25:
            return False
        parsed = urlparse(video_url)
        if parsed.scheme not in {"http", "https"}:
            return False

        _report(progress, 11, "正在检查录像服务器是否支持分段读取")
        supported = _remote_media_supports_ranges(video_url)
        if supported:
            _log(progress, "录像服务器支持 HTTP Range，将启用并行分块音频预处理。", "info")
        else:
            _log(progress, "录像服务器未确认支持 HTTP Range，使用兼容的整段流式解码。", "debug")
        return supported

    def _transcribe_single_stream(
        self,
        model,
        video_url: str,
        progress: ProgressCallback | None,
    ) -> tuple[list[str], float]:
        _report(
            progress,
            12,
            "正在连接课堂录像并准备音频；首次输出前需要完成远程解码、VAD 和特征准备"
            f"（并发上限 {self.config.asr_max_concurrent_tasks}）",
        )
        heartbeat = _PreparationHeartbeat(progress)
        heartbeat.start()
        try:
            transcriber = self._build_transcriber(model)
            transcribe_kwargs = self._transcribe_kwargs(condition_on_previous_text=True)
            segments, info = transcriber.transcribe(video_url, **transcribe_kwargs)
            duration = float(getattr(info, "duration", 0.0) or 0.0)
            tracker = _TranscriptionProgress(progress, duration)
            lines: list[str] = []
            first_segment = True
            for segment in segments:
                _raise_if_cancelled(progress)
                if first_segment:
                    heartbeat.stop()
                    first_segment = False

                raw_text = str(getattr(segment, "text", "") or "").strip()
                text = self._normalize_text(raw_text)
                if not text:
                    continue
                start = float(getattr(segment, "start", 0.0) or 0.0)
                end = float(getattr(segment, "end", start) or start)
                lines.append(f"[{_timestamp(start)} - {_timestamp(end)}] {text}")
                tracker.report(end)
            return lines, duration
        finally:
            heartbeat.stop()

    def _transcribe_chunked_remote(
        self,
        model,
        video_url: str,
        duration: float,
        progress: ProgressCallback | None,
    ) -> tuple[list[str], float]:
        chunks = _build_audio_chunks(duration)
        if len(chunks) <= 1:
            raise _ChunkedDecodeError("录像时长不足以分块")

        workers = min(_REMOTE_DECODE_WORKERS, len(chunks))
        _report(
            progress,
            12,
            f"已启用分块音频流水线：每块约 {int(_REMOTE_CHUNK_SECONDS // 60)} 分钟，"
            f"{workers} 个解码 Worker，共 {len(chunks)} 块",
        )
        _log(
            progress,
            f"ASR 分块预处理：chunks={len(chunks)}, decode_workers={workers}, "
            f"chunk_minutes={int(_REMOTE_CHUNK_SECONDS // 60)}, overlap={_REMOTE_CHUNK_OVERLAP_SECONDS:.1f}s",
            "debug",
        )

        transcriber = self._build_transcriber(model)
        tracker = _TranscriptionProgress(progress, duration)
        lines: list[str] = []
        prompt_tail = ""
        future_by_index: dict[int, Future] = {}
        next_submit = 0
        prefetch_queue_size = self.config.audio_prefetch_queue_size
        prefetch_limit = min(len(chunks), prefetch_queue_size + 1)

        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asr-audio-decode")

        def submit_one() -> None:
            nonlocal next_submit
            if next_submit >= len(chunks):
                return
            chunk = chunks[next_submit]
            future_by_index[chunk.index] = executor.submit(
                _decode_audio_chunk,
                video_url,
                chunk.decode_start,
                chunk.decode_end,
                16000,
                _cancel_check(progress),
            )
            next_submit += 1

        for _ in range(prefetch_limit):
            submit_one()

        _log(
            progress,
            f"音频预取队列 {max(0, len(future_by_index) - 1)}/{prefetch_queue_size}"
            f"（解码 Worker {workers}）",
            "debug",
        )

        try:
            for chunk in chunks:
                _raise_if_cancelled(progress)
                future = future_by_index.pop(chunk.index)
                wait_started = time.monotonic()
                while True:
                    try:
                        audio = future.result(timeout=5.0)
                        break
                    except FutureTimeoutError:
                        _raise_if_cancelled(progress)
                        elapsed = time.monotonic() - wait_started
                        media_percent = int((chunk.start / duration) * 100)
                        _report(
                            progress,
                            12 + min(78, int(media_percent * 0.78)),
                            f"正在并行预取音频块 {chunk.index + 1}/{len(chunks)}："
                            f"{_timestamp(chunk.start)} - {_timestamp(chunk.end)}；"
                            f"已等待 {_timestamp(elapsed)}；"
                            f"预取队列 {len(future_by_index)}/{prefetch_queue_size}"
                            f"（已就绪 {sum(item.done() for item in future_by_index.values())}）",
                        )
                    except Exception as exc:
                        _raise_if_cancelled(progress)
                        raise _ChunkedDecodeError(
                            f"第 {chunk.index + 1}/{len(chunks)} 块解码失败：{exc}"
                        ) from exc

                submit_one()
                _raise_if_cancelled(progress)
                queue_ready = sum(item.done() for item in future_by_index.values())
                queue_status = (
                    f"预取队列 {len(future_by_index)}/{prefetch_queue_size}"
                    f"（已就绪 {queue_ready}）"
                )
                _report(
                    progress,
                    15 + min(78, int((chunk.start / duration) * 78)),
                    f"音频块 {chunk.index + 1}/{len(chunks)} 已就绪，正在转写："
                    f"{_timestamp(chunk.start)} - {_timestamp(chunk.end)}；{queue_status}",
                )
                _log(progress, queue_status, "debug")

                kwargs = self._transcribe_kwargs(condition_on_previous_text=False)
                if prompt_tail:
                    kwargs["initial_prompt"] = prompt_tail[-240:]
                segments, _ = transcriber.transcribe(audio, **kwargs)
                chunk_texts: list[str] = []
                for segment in segments:
                    _raise_if_cancelled(progress)
                    raw_text = str(getattr(segment, "text", "") or "").strip()
                    text = self._normalize_text(raw_text)
                    if not text:
                        continue
                    local_start = float(getattr(segment, "start", 0.0) or 0.0)
                    local_end = float(getattr(segment, "end", local_start) or local_start)
                    global_start = chunk.decode_start + local_start
                    global_end = chunk.decode_start + local_end
                    midpoint = (global_start + global_end) / 2
                    is_last = chunk.index == len(chunks) - 1
                    if midpoint < chunk.start or (not is_last and midpoint >= chunk.end):
                        continue
                    global_start = max(chunk.start, global_start)
                    global_end = min(duration, max(global_start, global_end))
                    lines.append(
                        f"[{_timestamp(global_start)} - {_timestamp(global_end)}] {text}"
                    )
                    chunk_texts.append(text)
                    tracker.report(global_end)

                if chunk_texts:
                    prompt_tail = " ".join(chunk_texts)[-400:]
                _log(
                    progress,
                    f"音频块 {chunk.index + 1}/{len(chunks)} 完成："
                    f"{_timestamp(chunk.start)} - {_timestamp(chunk.end)}",
                    "debug",
                )
        finally:
            for future in future_by_index.values():
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

        return lines, duration

    def _build_transcriber(self, model):
        if self.config.asr_use_batched and self.config.asr_batch_size > 1:
            from faster_whisper import BatchedInferencePipeline

            return BatchedInferencePipeline(model=model)
        return model

    def _transcribe_kwargs(self, *, condition_on_previous_text: bool) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "language": self.config.asr_language or None,
            "beam_size": 3,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
            "condition_on_previous_text": condition_on_previous_text,
        }
        if self.config.asr_use_batched and self.config.asr_batch_size > 1:
            kwargs["batch_size"] = self.config.asr_batch_size
        return kwargs

    def _normalize_text(self, text: str) -> str:
        if not text or not self.config.asr_simplified_chinese:
            return text
        language = (self.config.asr_language or "").lower()
        if language and not language.startswith("zh"):
            return text
        return _to_simplified_chinese(text)

    def _export_transcript(self, output_path: Path, text: str) -> Path:
        try:
            relative_parent = output_path.resolve().parent.relative_to(
                self.config.outputs_dir.resolve()
            )
        except ValueError:
            relative_parent = Path()
        export_path = self.config.downloads_dir / relative_parent / "本地转写.txt"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = export_path.with_name(f"{export_path.name}.tmp")
        temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
        temporary.replace(export_path)
        return export_path

    def model_statuses(self) -> list[dict[str, object]]:
        from .environment import available_asr_models

        return [self.model_status(model) for model in available_asr_models()]

    def model_status(self, model_name: str) -> dict[str, object]:
        configured = self._validate_managed_model(model_name)
        repo_id = self._model_repo_id(configured)
        aliases = self._model_aliases_for_repo(repo_id)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise LocalTranscriptionError(
                "缺少模型管理依赖，请重新安装 requirements.txt"
            ) from exc

        try:
            snapshot = Path(
                snapshot_download(
                    repo_id,
                    cache_dir=str(self.config.asr_models_dir),
                    allow_patterns=_MODEL_ALLOW_PATTERNS,
                    local_files_only=True,
                )
            )
        except Exception:
            return {
                "model": configured,
                "repo_id": repo_id,
                "aliases": list(aliases),
                "installed": False,
                "size_bytes": 0,
                "size_label": "",
                "path": "",
            }

        installed = self._snapshot_complete(snapshot)
        size_bytes = self._snapshot_size(snapshot) if installed else 0
        return {
            "model": configured,
            "repo_id": repo_id,
            "aliases": list(aliases),
            "installed": installed,
            "size_bytes": size_bytes,
            "size_label": _format_bytes(size_bytes) if size_bytes else "",
            "path": str(snapshot) if installed else "",
        }

    def download_model(
        self,
        model_name: str,
        progress: ProgressCallback | None = None,
    ) -> dict[str, object]:
        configured = self._validate_managed_model(model_name)
        _report(progress, 0, f"正在准备下载语音模型 {configured}")
        with _MODEL_LOCK:
            self.config.asr_models_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_model_files(
                _ModelDownloadProgress(progress),
                model_name=configured,
            )
        status = self.model_status(configured)
        if not status["installed"]:
            raise LocalTranscriptionError(f"语音模型 {configured} 下载后仍未检测到完整缓存")
        _report(
            progress,
            100,
            f"语音模型 {configured} 下载完成"
            + (f"（{status['size_label']}）" if status["size_label"] else ""),
        )
        return status

    def delete_model(self, model_name: str) -> dict[str, object]:
        configured = self._validate_managed_model(model_name)
        repo_id = self._model_repo_id(configured)

        cache_name = "models--" + repo_id.replace("/", "--")
        repo_cache = self.config.asr_models_dir / cache_name
        lock_cache = self.config.asr_models_dir / ".locks" / cache_name

        aliases = self._model_aliases_for_repo(repo_id)
        cache_identity = f"repo:{repo_id}"
        with _model_maintenance_window():
            with _MODEL_LOCK:
                for key in [
                    key
                    for key in _MODEL_CACHE
                    if key and (key[0] == cache_identity or key[0] in aliases)
                ]:
                    _MODEL_CACHE.pop(key, None)
                gc.collect()
                existed = repo_cache.exists()
                try:
                    if repo_cache.exists():
                        shutil.rmtree(repo_cache)
                    if lock_cache.exists():
                        shutil.rmtree(lock_cache)
                except OSError as exc:
                    raise LocalTranscriptionError(
                        f"删除语音模型 {configured} 失败：{exc}"
                    ) from exc

        return {
            "model": configured,
            "repo_id": repo_id,
            "aliases": list(aliases),
            "deleted": existed,
        }

    @staticmethod
    def _snapshot_complete(snapshot: Path) -> bool:
        required = ("config.json", "model.bin", "tokenizer.json")
        if not all((snapshot / name).is_file() for name in required):
            return False
        return any(snapshot.glob("vocabulary.*"))

    @staticmethod
    def _snapshot_size(snapshot: Path) -> int:
        total = 0
        seen: set[str] = set()
        for item in snapshot.rglob("*"):
            if not item.is_file():
                continue
            try:
                resolved = str(item.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                total += item.stat().st_size
            except OSError:
                continue
        return total

    @staticmethod
    def _model_repo_id(model_name: str) -> str:
        try:
            from faster_whisper.utils import _MODELS as model_repositories
        except ImportError as exc:
            raise LocalTranscriptionError(
                "缺少 faster-whisper，请先重新安装 requirements.txt 中的依赖"
            ) from exc
        repo_id = model_repositories.get(model_name)
        if not repo_id:
            raise LocalTranscriptionError(f"不支持的 faster-whisper 模型：{model_name}")
        return str(repo_id)

    def _model_aliases_for_repo(self, repo_id: str) -> tuple[str, ...]:
        from .environment import available_asr_models

        aliases: list[str] = []
        for candidate in available_asr_models():
            try:
                if self._model_repo_id(candidate) == repo_id:
                    aliases.append(candidate)
            except LocalTranscriptionError:
                continue
        return tuple(aliases)

    def _model_cache_identity(self, model_name: str) -> str:
        local_candidate = Path(model_name).expanduser()
        if local_candidate.is_dir():
            try:
                return f"path:{local_candidate.resolve()}"
            except OSError:
                return f"path:{local_candidate}"
        return f"repo:{self._model_repo_id(model_name)}"

    @staticmethod
    def _validate_managed_model(model_name: str) -> str:
        configured = str(model_name or "").strip()
        from .environment import available_asr_models

        if configured not in set(available_asr_models()):
            raise LocalTranscriptionError(f"模型管理不支持该 faster-whisper 模型：{configured}")
        return configured

    def _get_model(
        self,
        progress: ProgressCallback | None = None,
        *,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise LocalTranscriptionError(
                "缺少 faster-whisper，请先重新安装 requirements.txt 中的依赖"
            ) from exc

        from .environment import resolve_asr_runtime_config

        requested_device = device or self.config.asr_device
        requested_compute_type = compute_type or self.config.asr_compute_type
        (
            effective_device,
            effective_compute_type,
            runtime_messages,
        ) = resolve_asr_runtime_config(
            requested_device,
            requested_compute_type,
        )
        for level, message in runtime_messages:
            _log(progress, message, level)
        _log(
            progress,
            f"本次 ASR 运行配置：device={effective_device}, "
            f"compute_type={effective_compute_type}",
            "info",
        )
        key = (
            self._model_cache_identity(self.config.asr_model),
            effective_device,
            effective_compute_type,
            self.config.asr_cpu_threads,
            self.config.asr_num_workers,
            str(self.config.asr_models_dir),
        )
        with _MODEL_LOCK:
            model = _MODEL_CACHE.get(key)
            if model is not None:
                return model
            try:
                self.config.asr_models_dir.mkdir(parents=True, exist_ok=True)
                model_path = self._ensure_model_files(progress)
                _report(progress, 8, "正在加载本地语音识别模型")
                model = WhisperModel(
                    model_path,
                    device=effective_device,
                    compute_type=effective_compute_type,
                    cpu_threads=self.config.asr_cpu_threads,
                    num_workers=self.config.asr_num_workers,
                    local_files_only=True,
                )
            except LocalTranscriptionError:
                raise
            except Exception as exc:
                raise LocalTranscriptionError(f"语音识别模型加载失败：{exc}") from exc
            _MODEL_CACHE[key] = model
            return model

    def _ensure_model_files(
        self,
        progress: ProgressCallback | None = None,
        *,
        model_name: str | None = None,
    ) -> str:
        configured = model_name or self.config.asr_model
        local_candidate = Path(configured).expanduser()
        if local_candidate.is_dir():
            return str(local_candidate)

        try:
            from faster_whisper.utils import _MODELS as model_repositories
            from huggingface_hub import hf_hub_download, snapshot_download
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise LocalTranscriptionError(
                "缺少模型下载依赖，请重新安装 requirements.txt"
            ) from exc

        repo_id = configured if "/" in configured else model_repositories.get(configured)
        if not repo_id:
            raise LocalTranscriptionError(f"不支持的 faster-whisper 模型：{configured}")

        cache_dir = str(self.config.asr_models_dir)
        _report_detail(progress, 0, f"正在检查语音模型 {configured} 的本地缓存")
        try:
            files = snapshot_download(
                repo_id,
                cache_dir=cache_dir,
                allow_patterns=_MODEL_ALLOW_PATTERNS,
                dry_run=True,
            )
        except TypeError as exc:
            raise LocalTranscriptionError(
                "huggingface-hub 版本过旧，无法显示模型下载进度；请重新安装 requirements.txt"
            ) from exc
        except Exception as exc:
            raise LocalTranscriptionError(f"读取语音模型信息失败：{exc}") from exc

        pending = [item for item in files if getattr(item, "will_download", False)]
        model_total = sum(int(getattr(item, "file_size", 0) or 0) for item in files)
        download_total = sum(int(getattr(item, "file_size", 0) or 0) for item in pending)
        revision = next((str(getattr(item, "commit_hash", "") or "") for item in files), "") or None

        if not pending:
            if model_total:
                _report_detail(
                    progress,
                    100,
                    f"语音模型已缓存（{_format_bytes(model_total)}）",
                )
            try:
                return str(
                    snapshot_download(
                        repo_id,
                        cache_dir=cache_dir,
                        allow_patterns=_MODEL_ALLOW_PATTERNS,
                        revision=revision,
                        local_files_only=True,
                    )
                )
            except Exception as exc:
                raise LocalTranscriptionError(f"读取本地语音模型失败：{exc}") from exc

        size_label = _format_bytes(download_total) if download_total else f"{len(pending)} 个文件"
        _report_detail(progress, 0, f"首次使用：需要下载语音模型 {configured}，共 {size_label}")

        completed_bytes = 0
        completed_files = 0
        total_files = len(pending)

        for item in pending:
            filename = str(getattr(item, "filename", "") or "")
            expected_size = int(getattr(item, "file_size", 0) or 0)
            if not filename:
                continue

            base_bytes = completed_bytes
            file_index = completed_files

            class DownloadProgress(tqdm):
                def display(self, *args, **kwargs):  # noqa: ANN002, ANN003 - tqdm API compatibility.
                    return None

                def update(self, n=1):
                    result = super().update(n)
                    current = max(0, int(getattr(self, "n", 0) or 0))
                    if expected_size:
                        current = min(current, expected_size)
                    if download_total:
                        overall = min(download_total, base_bytes + current)
                        percent = int((overall / download_total) * 100)
                        message = (
                            f"正在下载语音模型 {configured}："
                            f"{_format_bytes(overall)} / {_format_bytes(download_total)}（{percent}%）"
                        )
                    else:
                        local_total = max(1, int(getattr(self, "total", 0) or 1))
                        fraction = min(1.0, current / local_total)
                        percent = int(((file_index + fraction) / max(1, total_files)) * 100)
                        message = f"正在下载语音模型 {configured}：{percent}%"
                    _report_detail(progress, percent, message)
                    return result

            try:
                hf_hub_download(
                    repo_id,
                    filename=filename,
                    cache_dir=cache_dir,
                    revision=revision,
                    tqdm_class=DownloadProgress,
                )
            except Exception as exc:
                raise LocalTranscriptionError(f"语音模型下载失败（{filename}）：{exc}") from exc

            completed_files += 1
            if expected_size:
                completed_bytes += expected_size
            if download_total:
                percent = int((min(download_total, completed_bytes) / download_total) * 100)
                _report_detail(
                    progress,
                    percent,
                    f"正在下载语音模型 {configured}："
                    f"{_format_bytes(min(download_total, completed_bytes))} / {_format_bytes(download_total)}（{percent}%）",
                )
            else:
                percent = int((completed_files / max(1, total_files)) * 100)
                _report_detail(progress, percent, f"正在下载语音模型 {configured}：{percent}%")

        _report_detail(
            progress,
            100,
            f"语音模型 {configured} 下载完成（{_format_bytes(model_total or download_total)}）",
        )
        try:
            return str(
                snapshot_download(
                    repo_id,
                    cache_dir=cache_dir,
                    allow_patterns=_MODEL_ALLOW_PATTERNS,
                    revision=revision,
                    local_files_only=True,
                )
            )
        except Exception as exc:
            raise LocalTranscriptionError(f"语音模型下载完成但本地快照不可用：{exc}") from exc


class _ModelDownloadProgress:
    """Map model-detail progress to the standalone model task's main progress bar."""

    def __init__(self, progress: ProgressCallback | None) -> None:
        self.progress = progress

    def __call__(self, percent: int, message: str) -> None:
        _report(self.progress, percent, message)

    def detail(self, percent: int, message: str) -> None:
        _report(self.progress, percent, message)

    def clear_detail(self) -> None:
        clear = getattr(self.progress, "clear_detail", None)
        if callable(clear):
            clear()

    def log(self, message: str, level: str = "info") -> None:
        log = getattr(self.progress, "log", None)
        if callable(log):
            log(message, level)

    @property
    def cancel_requested(self) -> bool:
        return bool(getattr(self.progress, "cancel_requested", False))

    def raise_if_cancelled(self) -> None:
        raise_if_cancelled = getattr(self.progress, "raise_if_cancelled", None)
        if callable(raise_if_cancelled):
            raise_if_cancelled()


class _TranscriptionProgress:
    """Keep the UI fresh without writing one task log entry for every Whisper segment."""

    def __init__(self, progress: ProgressCallback | None, duration: float) -> None:
        self.progress = progress
        self.duration = max(0.0, duration)
        self.last_media_percent = -1
        self.last_report_at = 0.0

    def report(self, media_end: float) -> None:
        if self.progress is None:
            return
        if self.duration <= 0:
            now = time.monotonic()
            if now - self.last_report_at >= 1.0:
                _report(self.progress, 55, f"正在本地转写：已处理到 {_timestamp(media_end)}")
                self.last_report_at = now
            return

        media_percent = max(0, min(100, int((media_end / self.duration) * 100)))
        now = time.monotonic()
        percent_jump = media_percent - self.last_media_percent
        if self.last_media_percent >= 0 and percent_jump < 5:
            if media_percent <= self.last_media_percent or now - self.last_report_at < 1.0:
                return
        task_percent = 15 + min(80, int((media_end / self.duration) * 80))
        _report(
            self.progress,
            task_percent,
            f"正在本地转写：{_timestamp(media_end)} / {_timestamp(self.duration)}（{media_percent}%）",
        )
        self.last_media_percent = media_percent
        self.last_report_at = now


class _PreparationHeartbeat:
    """Keep long media preparation visibly alive until faster-whisper yields its first segment."""

    def __init__(self, progress: ProgressCallback | None) -> None:
        self.progress = progress
        self.started_at = time.monotonic()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.progress is None:
            return
        self.thread = threading.Thread(
            target=self._run,
            name="asr-preparation-heartbeat",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.25)

    def _run(self) -> None:
        while not self.stop_event.wait(_PREPARATION_HEARTBEAT_SECONDS):
            elapsed = time.monotonic() - self.started_at
            try:
                _report(
                    self.progress,
                    12,
                    "正在读取并预处理课堂录像音频："
                    f"已等待 {_timestamp(elapsed)}；正在进行远程解码 / VAD / 特征准备，"
                    "完成后会开始显示媒体转写百分比",
                )
            except Exception:
                return


def _build_audio_chunks(duration: float) -> list[_AudioChunk]:
    chunks: list[_AudioChunk] = []
    start = 0.0
    index = 0
    while start < duration:
        end = min(duration, start + _REMOTE_CHUNK_SECONDS)
        decode_start = max(0.0, start - (_REMOTE_CHUNK_OVERLAP_SECONDS if index else 0.0))
        decode_end = min(
            duration,
            end + (_REMOTE_CHUNK_OVERLAP_SECONDS if end < duration else 0.0),
        )
        chunks.append(_AudioChunk(index, start, end, decode_start, decode_end))
        index += 1
        start = end
    return chunks


def _remote_media_supports_ranges(url: str) -> bool:
    parsed = urlparse(url)
    cache_key = f"{parsed.scheme}://{parsed.netloc}"
    with _RANGE_SUPPORT_LOCK:
        cached = _RANGE_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    supported = False
    request = Request(
        url,
        headers={
            "Range": "bytes=0-0",
            "Accept": "*/*",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
        },
    )
    try:
        with urlopen(request, timeout=_RANGE_PROBE_TIMEOUT_SECONDS) as response:
            status = int(getattr(response, "status", 0) or 0)
            content_range = str(response.headers.get("Content-Range", "") or "")
            accept_ranges = str(response.headers.get("Accept-Ranges", "") or "").lower()
            supported = status == 206 or content_range.lower().startswith("bytes ") or "bytes" in accept_ranges
    except Exception:
        supported = False

    with _RANGE_SUPPORT_LOCK:
        _RANGE_SUPPORT_CACHE[cache_key] = supported
    return supported


def _decode_audio_chunk(
    video_url: str,
    start: float,
    end: float,
    sampling_rate: int,
    cancel_check: Callable[[], None] | None = None,
):
    """Decode a seekable remote media slice into the float32 waveform Whisper expects."""
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise _ChunkedDecodeError("缺少 PyAV / NumPy 音频解码依赖") from exc

    if end <= start:
        return np.zeros((0,), dtype=np.float32)

    arrays: list[object] = []
    first_frame_start: float | None = None
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sampling_rate,
    )

    try:
        with av.open(video_url, mode="r", metadata_errors="ignore") as container:
            if not container.streams.audio:
                raise _ChunkedDecodeError("录像中没有音频轨")
            stream = container.streams.audio[0]
            if start > 0:
                if stream.time_base:
                    seek_offset = int(start / float(stream.time_base))
                    container.seek(seek_offset, stream=stream, backward=True, any_frame=False)
                else:
                    container.seek(int(start * 1_000_000), backward=True, any_frame=False)

            frame_counter = 0
            for frame in container.decode(stream):
                frame_counter += 1
                if cancel_check is not None and frame_counter % 32 == 0:
                    cancel_check()

                frame_start = _audio_frame_time(frame, stream)
                frame_duration = float(getattr(frame, "samples", 0) or 0) / float(
                    getattr(frame, "sample_rate", 0) or sampling_rate
                )
                frame_end = frame_start + frame_duration
                if frame_end <= start:
                    continue
                if frame_start >= end:
                    break
                if first_frame_start is None:
                    first_frame_start = frame_start
                for converted in resampler.resample(frame):
                    arrays.append(converted.to_ndarray().reshape(-1))

            for converted in resampler.resample(None):
                arrays.append(converted.to_ndarray().reshape(-1))
    finally:
        del resampler

    if not arrays:
        raise _ChunkedDecodeError(
            f"{_timestamp(start)} - {_timestamp(end)} 没有解码到音频"
        )

    pcm = np.concatenate(arrays)
    if pcm.dtype != np.int16:
        pcm = pcm.astype(np.int16, copy=False)
    audio = pcm.astype(np.float32) / 32768.0
    first_time = start if first_frame_start is None else first_frame_start
    trim_left = max(0, int(round((start - first_time) * sampling_rate)))
    target_samples = max(1, int(round((end - start) * sampling_rate)))
    audio = audio[trim_left : trim_left + target_samples]
    if audio.size == 0:
        raise _ChunkedDecodeError(
            f"{_timestamp(start)} - {_timestamp(end)} 音频切片为空"
        )
    return audio


def _audio_frame_time(frame, stream) -> float:
    frame_time = getattr(frame, "time", None)
    if frame_time is not None:
        return max(0.0, float(frame_time))
    pts = getattr(frame, "pts", None)
    time_base = getattr(stream, "time_base", None)
    if pts is not None and time_base is not None:
        return max(0.0, float(pts * time_base))
    return 0.0


@contextmanager
def _transcription_slot(limit: int, progress: ProgressCallback | None = None):
    global _ACTIVE_TRANSCRIPTIONS
    limit = max(1, int(limit))
    with _TRANSCRIPTION_CONDITION:
        while _MODEL_MAINTENANCE or _ACTIVE_TRANSCRIPTIONS >= limit:
            _raise_if_cancelled(progress)
            if _MODEL_MAINTENANCE:
                _report(
                    progress,
                    10,
                    "正在等待语音模型维护完成",
                )
            else:
                _report(
                    progress,
                    10,
                    f"正在等待本地转写槽位：{_ACTIVE_TRANSCRIPTIONS}/{limit} 个任务运行中",
                )
            _TRANSCRIPTION_CONDITION.wait(timeout=1.0)
        _ACTIVE_TRANSCRIPTIONS += 1
        active = _ACTIVE_TRANSCRIPTIONS
    _report(progress, 11, f"已进入本地转写队列：{active}/{limit} 个任务运行中")
    try:
        yield
    finally:
        with _TRANSCRIPTION_CONDITION:
            _ACTIVE_TRANSCRIPTIONS = max(0, _ACTIVE_TRANSCRIPTIONS - 1)
            _TRANSCRIPTION_CONDITION.notify_all()


@contextmanager
def _model_maintenance_window():
    global _MODEL_MAINTENANCE
    with _TRANSCRIPTION_CONDITION:
        while _MODEL_MAINTENANCE:
            _TRANSCRIPTION_CONDITION.wait(timeout=1.0)
        if _ACTIVE_TRANSCRIPTIONS > 0:
            raise LocalTranscriptionError("当前有本地转写任务运行，完成或取消后再删除模型")
        _MODEL_MAINTENANCE = True
    try:
        yield
    finally:
        with _TRANSCRIPTION_CONDITION:
            _MODEL_MAINTENANCE = False
            _TRANSCRIPTION_CONDITION.notify_all()


_CUDA_RUNTIME_ERROR_MARKERS = (
    "cublas",
    "cudnn",
    "cuda runtime",
    "cuda driver",
    "cuda error",
    "libcuda",
    "nvcuda",
)


def _is_cuda_runtime_error(exc: BaseException) -> bool:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        message = str(current).lower()
        if any(marker in message for marker in _CUDA_RUNTIME_ERROR_MARKERS):
            return True
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if isinstance(cause, BaseException):
            pending.append(cause)
        if isinstance(context, BaseException):
            pending.append(context)
    return False


def _to_simplified_chinese(text: str) -> str:
    global _OPENCC_CONVERTER
    with _OPENCC_LOCK:
        if _OPENCC_CONVERTER is None:
            try:
                from opencc import OpenCC
            except ImportError as exc:
                raise LocalTranscriptionError(
                    "缺少 OpenCC 简繁转换依赖，请重新安装 requirements.txt"
                ) from exc
            _OPENCC_CONVERTER = OpenCC("t2s")
        converter = _OPENCC_CONVERTER
    return converter.convert(text)


def _timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_bytes(value: int) -> str:
    size = max(0, int(value))
    units = ("B", "KB", "MB", "GB")
    amount = float(size)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{size} B"


def _report(progress: ProgressCallback | None, percent: int, message: str) -> None:
    if progress:
        progress(max(0, min(100, int(percent))), message)


def _report_detail(progress: ProgressCallback | None, percent: int, message: str) -> None:
    if progress is None:
        return
    detail = getattr(progress, "detail", None)
    if callable(detail):
        detail(max(0, min(100, int(percent))), message)
    else:
        progress(4, message)


def _clear_detail(progress: ProgressCallback | None) -> None:
    if progress is None:
        return
    clear_detail = getattr(progress, "clear_detail", None)
    if callable(clear_detail):
        clear_detail()


def _log(progress: ProgressCallback | None, message: str, level: str = "info") -> None:
    if progress is None:
        return
    logger = getattr(progress, "log", None)
    if callable(logger):
        logger(message, level)


def _cancel_check(progress: ProgressCallback | None) -> Callable[[], None] | None:
    if progress is None:
        return None
    check = getattr(progress, "raise_if_cancelled", None)
    return check if callable(check) else None


def _raise_if_cancelled(progress: ProgressCallback | None) -> None:
    check = _cancel_check(progress)
    if check is not None:
        check()
