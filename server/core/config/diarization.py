from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import env_first_non_empty, ensure_dir, to_bool, to_int


@dataclass(frozen=True)
class DiarizationConfig:
    diarization_enabled: bool
    diarization_hf_token: str
    diarization_model: str
    diarization_device: str
    diarization_sample_rate: int
    diarization_num_speakers: int
    diarization_min_speakers: int
    diarization_max_speakers: int
    diarization_work_dir: Path
    diarization_keep_chunks: bool
    ffmpeg_bin: str


def load_diarization_config(app_data_dir: Path) -> DiarizationConfig:
    return DiarizationConfig(
        diarization_enabled=to_bool("DIARIZATION_ENABLED", False),
        diarization_hf_token=env_first_non_empty("DIARIZATION_HF_TOKEN") or "",
        diarization_model=env_first_non_empty("DIARIZATION_MODEL") or "pyannote/speaker-diarization-3.1",
        diarization_device=env_first_non_empty("DIARIZATION_DEVICE") or "auto",
        diarization_sample_rate=max(8_000, to_int("DIARIZATION_SAMPLE_RATE", 16_000)),
        diarization_num_speakers=max(0, to_int("DIARIZATION_NUM_SPEAKERS", 0)),
        diarization_min_speakers=max(0, to_int("DIARIZATION_MIN_SPEAKERS", 0)),
        diarization_max_speakers=max(0, to_int("DIARIZATION_MAX_SPEAKERS", 0)),
        diarization_work_dir=ensure_dir(Path(env_first_non_empty("DIARIZATION_WORK_DIR") or str(app_data_dir / "diarization"))),
        diarization_keep_chunks=to_bool("DIARIZATION_KEEP_CHUNKS", False),
        ffmpeg_bin=env_first_non_empty("DIARIZATION_FFMPEG_BIN", "FFMPEG_BIN") or "ffmpeg",
    )
