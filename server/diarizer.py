from __future__ import annotations

import inspect
import logging
import os
import shutil
import subprocess
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioChunk:
    seq: int
    path: Path
    offset_ms: int
    duration_ms: int


@dataclass(slots=True)
class SpeakerTurn:
    start_ms: int
    end_ms: int
    speaker: str


class PyannoteSpeakerDiarizer:
    def __init__(
        self,
        *,
        hf_token: str,
        model: str,
        ffmpeg_bin: str,
        device: str = "auto",
        sample_rate: int = 16_000,
        num_speakers: int = 0,
        min_speakers: int = 0,
        max_speakers: int = 0,
    ):
        if not hf_token:
            raise RuntimeError("DIARIZATION_HF_TOKEN is not set")
        if shutil.which(ffmpeg_bin) is None:
            raise RuntimeError(f"{ffmpeg_bin} is not installed")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        self.hf_token = hf_token
        self.model = model
        self.ffmpeg_bin = ffmpeg_bin
        self.device = device
        self.sample_rate = max(8_000, int(sample_rate))
        self.num_speakers = max(0, int(num_speakers))
        self.min_speakers = max(0, int(min_speakers))
        self.max_speakers = max(0, int(max_speakers))

        self._pipeline: Any = None
        self._resolved_device = "cpu"

    def preflight(self) -> None:
        self._load_pipeline_class()

    def diarize(
        self,
        *,
        session_id: str,
        chunks: list[AudioChunk],
        work_dir: Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerTurn]:
        if not chunks:
            return []

        self._ensure_pipeline()
        work_dir.mkdir(parents=True, exist_ok=True)
        wav_path = work_dir / f"{session_id}.timeline.wav"

        self._build_timeline_wav(chunks=chunks, output_path=wav_path)
        if not wav_path.exists() or wav_path.stat().st_size <= 44:
            return []

        resolved_num = self.num_speakers if num_speakers is None else max(0, int(num_speakers))
        resolved_min = self.min_speakers if min_speakers is None else max(0, int(min_speakers))
        resolved_max = self.max_speakers if max_speakers is None else max(0, int(max_speakers))

        if resolved_min > 0 and resolved_max > 0 and resolved_min > resolved_max:
            resolved_min, resolved_max = resolved_max, resolved_min

        kwargs: dict[str, Any] = {}
        if resolved_num > 0:
            kwargs["num_speakers"] = resolved_num
        else:
            if resolved_min > 0:
                kwargs["min_speakers"] = resolved_min
            if resolved_max > 0:
                kwargs["max_speakers"] = resolved_max

        try:
            diarization = self._pipeline(str(wav_path), **kwargs)
            turns: list[SpeakerTurn] = []

            for turn, _, label in diarization.itertracks(yield_label=True):
                start_ms = max(0, int(round(float(turn.start) * 1000)))
                end_ms = max(start_ms, int(round(float(turn.end) * 1000)))
                if end_ms <= start_ms:
                    continue
                turns.append(SpeakerTurn(start_ms=start_ms, end_ms=end_ms, speaker=str(label)))

            return _normalize_speaker_labels(turns)
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        Pipeline = self._load_pipeline_class()

        torch = None
        try:
            import torch as _torch

            torch = _torch
        except Exception:  # noqa: BLE001
            torch = None

        try:
            pipeline = Pipeline.from_pretrained(self.model, use_auth_token=self.hf_token)
        except AttributeError as exc:
            message = str(exc)
            if "NoneType" in message and "eval" in message:
                raise RuntimeError(self._hf_access_help_message()) from exc
            raise
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "Weights only load failed" in message or "Unsupported global" in message:
                raise RuntimeError(
                    "Failed to load pyannote checkpoint with current torch serialization defaults. "
                    "Restart after upgrading this repository and reinstalling requirements. "
                    "If it still fails, clear Hugging Face cache for pyannote models and retry."
                ) from exc
            raise

        if pipeline is None:
            raise RuntimeError(self._hf_access_help_message())

        self._pipeline = pipeline

        resolved = self.device.strip().lower() or "auto"
        if resolved == "auto":
            resolved = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

        if torch is not None:
            self._pipeline.to(torch.device(resolved))
        self._resolved_device = resolved
        logger.info("diarizer ready: model=%s device=%s", self.model, self._resolved_device)

    def _hf_access_help_message(self) -> str:
        return (
            "Failed to load pyannote diarization pipeline. "
            "DIARIZATION_HF_TOKEN may not have access to required gated models. "
            "Accept terms on Hugging Face for "
            f"'{self.model}' and 'pyannote/segmentation-3.0', "
            "and use a token with read access."
        )

    def _load_pipeline_class(self) -> Any:
        try:
            import huggingface_hub
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"huggingface_hub import failed: {type(exc).__name__}: {exc}") from exc

        hf_version = getattr(huggingface_hub, "__version__", "unknown")
        hf_params = inspect.signature(hf_hub_download).parameters
        compat_hf_hub_download = huggingface_hub.hf_hub_download
        if "use_auth_token" not in hf_params:
            if "token" not in hf_params:
                raise RuntimeError(
                    "Incompatible huggingface_hub version detected "
                    f"({hf_version}). hf_hub_download has neither use_auth_token nor token."
                )

            already_patched = getattr(huggingface_hub.hf_hub_download, "_whistx_use_auth_token_compat", False)
            if not already_patched:
                original_hf_hub_download = hf_hub_download

                def _hf_hub_download_compat(*args: Any, use_auth_token: Any = None, **kwargs: Any) -> Any:
                    if use_auth_token is not None and "token" not in kwargs:
                        kwargs["token"] = use_auth_token
                    return original_hf_hub_download(*args, **kwargs)

                setattr(_hf_hub_download_compat, "_whistx_use_auth_token_compat", True)
                huggingface_hub.hf_hub_download = _hf_hub_download_compat
                try:
                    import huggingface_hub.file_download as _hf_file_download

                    _hf_file_download.hf_hub_download = _hf_hub_download_compat
                except Exception:  # noqa: BLE001
                    pass
                logger.info(
                    "Patched huggingface_hub compatibility for use_auth_token (version=%s)",
                    hf_version,
                )
            compat_hf_hub_download = huggingface_hub.hf_hub_download

        try:
            import torchaudio
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"torchaudio import failed: {type(exc).__name__}: {exc}") from exc

        if not hasattr(torchaudio, "AudioMetaData"):
            version = getattr(torchaudio, "__version__", "unknown")
            raise RuntimeError(
                "Incompatible torchaudio version detected "
                f"({version}). pyannote.audio currently requires torchaudio with AudioMetaData. "
                "Use torch/torchaudio < 2.9."
            )

        try:
            from pyannote.audio import Pipeline
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"pyannote.audio import failed: {detail}") from exc

        self._patch_torch_checkpoint_loader()

        # pyannote may cache hf_hub_download as a module-level symbol.
        # Keep it aligned with our compatibility wrapper when needed.
        if getattr(compat_hf_hub_download, "_whistx_use_auth_token_compat", False):
            try:
                import pyannote.audio.core.pipeline as _pyannote_pipeline

                py_hf = getattr(_pyannote_pipeline, "hf_hub_download", None)
                if py_hf is not compat_hf_hub_download:
                    _pyannote_pipeline.hf_hub_download = compat_hf_hub_download
            except Exception:  # noqa: BLE001
                pass

        return Pipeline

    def _patch_torch_checkpoint_loader(self) -> None:
        """Patch pyannote checkpoint loading for torch>=2.6 default weights_only behavior."""
        try:
            import torch
        except Exception:  # noqa: BLE001
            return

        # Keep safe-global allowlist broad enough for common pyannote checkpoints.
        safe_globals: list[Any] = [torch.torch_version.TorchVersion]
        try:
            from pyannote.audio.core.task import Specifications

            safe_globals.append(Specifications)
        except Exception:  # noqa: BLE001
            pass
        try:
            torch.serialization.add_safe_globals(safe_globals)
        except Exception:  # noqa: BLE001
            pass

        def _wrap_loader(loader: Any) -> Any:
            if not callable(loader):
                return loader
            if getattr(loader, "_whistx_weights_only_compat", False):
                return loader

            params = inspect.signature(loader).parameters
            if "weights_only" not in params:
                return loader

            def _loader_compat(
                path_or_url: Any,
                map_location: Any = None,
                weights_only: bool | None = None,
                **kwargs: Any,
            ) -> Any:
                # torch>=2.6 changed default to weights_only=True.
                # pyannote checkpoints require full object load from trusted source.
                if weights_only is not False:
                    weights_only = False
                return loader(path_or_url, map_location=map_location, weights_only=weights_only, **kwargs)

            setattr(_loader_compat, "_whistx_weights_only_compat", True)
            return _loader_compat

        try:
            from lightning_fabric.utilities import cloud_io as _lf_cloud_io

            patched_cloud_load = _wrap_loader(getattr(_lf_cloud_io, "_load", None))
            if patched_cloud_load is not getattr(_lf_cloud_io, "_load", None):
                _lf_cloud_io._load = patched_cloud_load
                logger.info("Patched lightning_fabric _load for torch weights_only compatibility")
        except Exception:  # noqa: BLE001
            pass

        try:
            import pytorch_lightning.core.saving as _pl_saving

            patched_pl_load = _wrap_loader(getattr(_pl_saving, "pl_load", None))
            if patched_pl_load is not getattr(_pl_saving, "pl_load", None):
                _pl_saving.pl_load = patched_pl_load
                logger.info("Patched pytorch_lightning pl_load for torch weights_only compatibility")
        except Exception:  # noqa: BLE001
            pass

        try:
            import lightning.pytorch.core.saving as _lightning_saving

            patched_lightning_load = _wrap_loader(getattr(_lightning_saving, "pl_load", None))
            if patched_lightning_load is not getattr(_lightning_saving, "pl_load", None):
                _lightning_saving.pl_load = patched_lightning_load
                logger.info("Patched lightning.pytorch pl_load for torch weights_only compatibility")
        except Exception:  # noqa: BLE001
            pass

        try:
            import pyannote.audio.core.model as _pyannote_model

            patched_pl_load = _wrap_loader(getattr(_pyannote_model, "pl_load", None))
            if patched_pl_load is not getattr(_pyannote_model, "pl_load", None):
                _pyannote_model.pl_load = patched_pl_load
                logger.info("Patched pyannote model loader for torch weights_only compatibility")
        except Exception:  # noqa: BLE001
            pass

    def _build_timeline_wav(self, *, chunks: list[AudioChunk], output_path: Path) -> None:
        timeline = array("h")
        sorted_chunks = sorted(chunks, key=lambda c: (c.offset_ms, c.seq))

        for chunk in sorted_chunks:
            pcm = self._decode_to_pcm(chunk.path)
            if not pcm:
                continue

            start_idx = max(0, int(round(chunk.offset_ms * self.sample_rate / 1000)))
            end_idx = start_idx + len(pcm)

            if end_idx > len(timeline):
                timeline.extend([0] * (end_idx - len(timeline)))

            for idx, sample in enumerate(pcm):
                target = start_idx + idx
                merged = int(timeline[target]) + int(sample)
                if merged > 32_767:
                    merged = 32_767
                elif merged < -32_768:
                    merged = -32_768
                timeline[target] = merged

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(self.sample_rate)
            wav_out.writeframes(timeline.tobytes())

    def _decode_to_pcm(self, path: Path) -> array:
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "-",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            logger.warning("ffmpeg decode failed: path=%s err=%s", path, stderr or "unknown")
            return array("h")
        if not proc.stdout:
            return array("h")

        pcm = array("h")
        pcm.frombytes(proc.stdout)
        if sys.byteorder != "little":
            pcm.byteswap()
        return pcm


def _normalize_speaker_labels(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    mapping: dict[str, str] = {}
    ordered: list[SpeakerTurn] = []
    next_idx = 0

    for turn in sorted(turns, key=lambda t: (t.start_ms, t.end_ms)):
        if turn.speaker not in mapping:
            mapping[turn.speaker] = f"SPK_{next_idx:02d}"
            next_idx += 1
        ordered.append(
            SpeakerTurn(
                start_ms=turn.start_ms,
                end_ms=turn.end_ms,
                speaker=mapping[turn.speaker],
            )
        )
    return ordered
