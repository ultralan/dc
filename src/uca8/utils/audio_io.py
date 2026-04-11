from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

CACHE_VERSION = 1


def _normalize_pcm(waveform: np.ndarray) -> np.ndarray:
    if np.issubdtype(waveform.dtype, np.floating):
        return waveform.astype(np.float32, copy=False)
    max_value = np.iinfo(waveform.dtype).max
    return waveform.astype(np.float32) / max_value


def _load_with_ffmpeg(source: str) -> tuple[torch.Tensor, int]:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        raise RuntimeError("ffmpeg/ffprobe not available.")
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            source,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    sample_rate = int(stream["sample_rate"])
    channels = int(stream["channels"])
    decoded = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            source,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    waveform = np.frombuffer(decoded.stdout, dtype=np.float32)
    waveform = waveform.reshape(-1, channels).T
    return torch.from_numpy(waveform.copy()), sample_rate


def _decode_audio_file(source: str) -> tuple[torch.Tensor, int]:
    try:
        from torchcodec.decoders import AudioDecoder  # type: ignore

        samples = AudioDecoder(source).get_all_samples()
        return samples.data.to(dtype=torch.float32), int(samples.sample_rate)
    except Exception:
        pass
    try:
        import torchaudio  # type: ignore

        waveform, sample_rate = torchaudio.load(source)
        return waveform.to(dtype=torch.float32), int(sample_rate)
    except Exception:
        pass
    try:
        return _load_with_ffmpeg(source)
    except Exception:
        pass
    from scipy.io import wavfile

    sample_rate, waveform = wavfile.read(source)
    waveform = np.asarray(waveform)
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    waveform = _normalize_pcm(waveform).T
    return torch.from_numpy(waveform), int(sample_rate)


def _resample_if_needed(
    waveform: torch.Tensor,
    source_rate: int,
    target_rate: int | None,
) -> tuple[torch.Tensor, int]:
    if target_rate is None or source_rate == target_rate:
        return waveform, int(source_rate)
    resampled = resample_poly(waveform.numpy(), up=int(target_rate), down=int(source_rate), axis=-1)
    return torch.from_numpy(np.asarray(resampled, dtype=np.float32)), int(target_rate)


def _source_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _cache_file_path(
    source: str,
    *,
    source_size: int,
    source_mtime_ns: int,
    target_sample_rate: int | None,
    cache_dir: str,
) -> Path:
    digest = hashlib.sha1(
        "::".join(
            [
                str(CACHE_VERSION),
                source,
                str(source_size),
                str(source_mtime_ns),
                str(target_sample_rate or 0),
            ]
        ).encode()
    ).hexdigest()
    return Path(cache_dir) / f"{digest}.pt"


def _write_cache_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_from_disk_cache(
    source: str,
    *,
    source_size: int,
    source_mtime_ns: int,
    target_sample_rate: int | None,
    cache_dir: str | None,
) -> tuple[torch.Tensor, int]:
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = _cache_file_path(
            source,
            source_size=source_size,
            source_mtime_ns=source_mtime_ns,
            target_sample_rate=target_sample_rate,
            cache_dir=cache_dir,
        )
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            waveform = payload["waveform"]
            sample_rate = int(payload["sample_rate"])
            if not isinstance(waveform, torch.Tensor):
                raise TypeError("Cached waveform payload must be a tensor.")
            return waveform.to(dtype=torch.float32), sample_rate
    waveform, sample_rate = _decode_audio_file(source)
    waveform, sample_rate = _resample_if_needed(waveform, sample_rate, target_sample_rate)
    waveform = waveform.to(dtype=torch.float32).contiguous()
    if cache_path is not None:
        _write_cache_atomically(
            cache_path,
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            },
        )
    return waveform, sample_rate


@lru_cache(maxsize=32)
def _load_audio_file_cached(
    source: str,
    *,
    source_size: int,
    source_mtime_ns: int,
    target_sample_rate: int | None,
    cache_dir: str | None,
) -> tuple[torch.Tensor, int]:
    return _load_from_disk_cache(
        source,
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
        target_sample_rate=target_sample_rate,
        cache_dir=cache_dir,
    )


def load_audio_file(
    path: str | Path,
    *,
    target_sample_rate: int | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[torch.Tensor, int]:
    """Load audio as [channels, samples] with optional decoded/resampled disk cache."""
    resolved_path = Path(path).resolve()
    source_size, source_mtime_ns = _source_signature(resolved_path)
    cache_dir_str = str(Path(cache_dir).resolve()) if cache_dir is not None else None
    return _load_audio_file_cached(
        str(resolved_path),
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
        target_sample_rate=target_sample_rate,
        cache_dir=cache_dir_str,
    )
