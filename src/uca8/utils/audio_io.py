from __future__ import annotations

"""音频读取、重采样和缓存工具.

训练数据里可能同时有 wav/flac 等格式, 不同机器上可用的音频后端也不一定相同.
本文件把音频读取封装成一个稳定接口: ``load_audio_file`` 始终返回
``[channels, samples]`` 的 float32 tensor, 并可选使用磁盘缓存减少重复解码开销.
"""

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
    """把整数 PCM 或浮点音频统一转换成 float32."""
    if np.issubdtype(waveform.dtype, np.floating):
        return waveform.astype(np.float32, copy=False)
    max_value = np.iinfo(waveform.dtype).max
    return waveform.astype(np.float32) / max_value


def _load_with_ffmpeg(source: str) -> tuple[torch.Tensor, int]:
    """使用 ffmpeg/ffprobe 解码音频.

    这是 torchcodec/torchaudio 不可用时的通用 fallback.
    ffmpeg 输出 f32le 裸流, 再 reshape 成 ``[channels, samples]``.
    """
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
    """按可用后端逐级尝试解码音频文件.

    优先级:
    1. torchcodec;
    2. torchaudio;
    3. ffmpeg;
    4. scipy wavfile.

    这样可以兼容不同开发环境, 同时把失败 fallback 封装在一个地方.
    """
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
    """必要时用 polyphase resampling 重采样到目标采样率."""
    if target_rate is None or source_rate == target_rate:
        return waveform, int(source_rate)
    resampled = resample_poly(waveform.numpy(), up=int(target_rate), down=int(source_rate), axis=-1)
    return torch.from_numpy(np.asarray(resampled, dtype=np.float32)), int(target_rate)


def _source_signature(path: Path) -> tuple[int, int]:
    """返回源文件大小和修改时间, 用于缓存失效判断."""
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
    """根据源文件和目标采样率生成唯一缓存路径.

    hash 输入包含 cache version、绝对路径、文件大小、mtime 和目标采样率.
    任意一个变化都会生成新缓存文件, 避免读到旧音频.
    """
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
    """原子写缓存文件.

    先写到进程专属临时文件, 再 replace 到目标路径. 这样中断或多进程写入时,
    不容易留下半截缓存.
    """
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
    """读取磁盘缓存; 缓存不存在时解码并写回."""
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
        # 缓存中保存的是重采样后的 waveform, 下次可直接用于训练.
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
    """进程内 LRU 缓存包装.

    同一个训练进程内多次访问同一文件时, 可以绕过磁盘反序列化.
    """
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
    """读取音频文件.

    参数:
        path: 音频路径.
        target_sample_rate: 目标采样率; 为 ``None`` 时保持原采样率.
        cache_dir: 可选磁盘缓存目录.

    返回:
        ``(waveform, sample_rate)``. ``waveform`` 形状为 ``[channels, samples]``,
        dtype 为 float32.
    """
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
