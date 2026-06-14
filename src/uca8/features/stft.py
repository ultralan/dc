from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from uca8.features.ipd import compute_ipd_features
from uca8.features.srp_phat import SRPPHAT
from uca8.geometry.uca8 import azimuth_grid, default_mic_pairs


@dataclass(slots=True)
class STFTSpec:
    """STFT/frontend 共享参数, 用来保证各特征分支的时间维对齐."""

    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    spec_bins: int = 64
    ipd_bins: int = 32
    azimuth_bins: int = 72
    target_frames: int = 128


class MultiChannelSTFT(nn.Module):
    """计算多通道波形的复数 STFT.

    输入形状为 ``[B, C, samples]``, 输出形状为 ``[B, C, frames, freq_bins]``.
    时间维放在频率维前面, 后续 log-mel/IPD/SRP 分支都沿用
    ``[B, channels, T, bins]`` 这个布局.
    """

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
    ) -> None:
        """初始化 STFT 参数和 Hann 窗."""
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """执行多通道 STFT."""
        if waveform.ndim != 3:
            raise ValueError("Expected waveform shape [batch, channels, samples].")
        batch, channels, samples = waveform.shape
        flattened = waveform.reshape(batch * channels, samples)
        stft = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=True,
            return_complex=True,
        )
        return stft.transpose(1, 2).reshape(batch, channels, -1, stft.shape[1])


def build_mel_filterbank(
    *,
    num_freqs: int,
    sample_rate: int,
    num_mels: int,
    device: torch.device,
    dtype: torch.dtype,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """构造轻量 mel 滤波器组, 避免额外依赖 torchaudio.

    这里需要的是稳定可复现的 log-mel-like 特征, 不是完整音频库能力.
    自己构造 ``freq -> mel`` 矩阵可以减少环境差异.
    """
    upper_hz = float(sample_rate) / 2.0 if f_max is None else f_max
    fft_freqs = torch.linspace(0.0, upper_hz, steps=num_freqs, device=device, dtype=dtype)
    mel_min = 2595.0 * torch.log10(torch.tensor(1.0 + f_min / 700.0, device=device, dtype=dtype))
    mel_max = 2595.0 * torch.log10(torch.tensor(1.0 + upper_hz / 700.0, device=device, dtype=dtype))
    mel_points = torch.linspace(mel_min, mel_max, steps=num_mels + 2, device=device, dtype=dtype)
    hz_points = 700.0 * (torch.pow(10.0, mel_points / 2595.0) - 1.0)
    filterbank = torch.zeros(num_freqs, num_mels, device=device, dtype=dtype)
    for idx in range(num_mels):
        left = hz_points[idx]
        center = hz_points[idx + 1]
        right = hz_points[idx + 2]
        up_slope = (fft_freqs - left) / (center - left).clamp_min(1e-6)
        down_slope = (right - fft_freqs) / (right - center).clamp_min(1e-6)
        filterbank[:, idx] = torch.minimum(up_slope, down_slope).clamp_min(0.0)
    return filterbank / filterbank.sum(dim=0, keepdim=True).clamp_min(1e-6)


def fit_time_dim(x: torch.Tensor, target_frames: int) -> torch.Tensor:
    """把特征张量裁剪/左侧补零到模型历史窗口长度.

    如果帧数过长, 保留最后 ``target_frames`` 帧, 因为当前定位标签对应历史窗口末端.
    """
    current_frames = x.shape[-2]
    if current_frames == target_frames:
        return x
    if current_frames > target_frames:
        return x[..., -target_frames:, :]
    pad = target_frames - current_frames
    return F.pad(x, (0, 0, pad, 0))


def build_logmel_like_features(
    stft_complex: torch.Tensor,
    *,
    mel_filterbank: torch.Tensor,
    out_bins: int = 64,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回参考麦克风和多通道平均能量的 log-mel 特征.

    ``logmel_ref`` 保留第一个麦克风的频谱视角.
    ``logmel_rms`` 先对所有麦克风功率做平均, 再投影到 mel 频带,
    表示一个紧凑的多通道能量摘要. 二者都是从原始波形派生出的输入特征,
    不是标签.
    """
    power = stft_complex.abs().pow(2.0)
    mel_filterbank = mel_filterbank.to(device=power.device, dtype=power.dtype)
    ref = torch.einsum("bctf,fm->bctm", power[:, :1], mel_filterbank)
    rms = torch.einsum("bctf,fm->bctm", power.mean(dim=1, keepdim=True), mel_filterbank)
    return torch.log(ref.clamp_min(eps)), torch.log(rms.clamp_min(eps))


class UCAFeatureFrontend(nn.Module):
    """UCA/RealMAN 模型的特征前端.

    这个模块集中负责所有信号处理特征提取. 模型输入原始波形, frontend 输出固定 key:

    - ``logmel_ref``: 参考麦克风 log-mel, ``[B, 1, T, mel_bins]``.
    - ``logmel_rms``: 多通道平均功率 log-mel, ``[B, 1, T, mel_bins]``.
    - ``ipd_feat``: 麦克风对 IPD 特征, ``[B, pair_features, T, ipd_bins]``.
    - ``srp_map``: SRP-PHAT 空间谱, ``[B, 1, T, azimuth_bins]``.
    - ``vad_ratio``: 帧级活动提示, ``[B, T, 1]``.

    消融开关会在特征构造后把某个分支置零, 而不是删除分支.
    这样下游网络结构不变, 实验比较的是信息来源, 不是参数规模.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        spec_bins: int,
        ipd_bins: int,
        azimuth_bins: int,
        target_frames: int,
        mic_positions: torch.Tensor,
        sound_speed: float = 343.0,
        use_logmel_ref: bool = True,
        use_logmel_rms: bool = True,
        use_ipd: bool = True,
        use_srp: bool = True,
        use_vad: bool = True,
    ) -> None:
        """初始化特征前端.

        mic_positions 决定 SRP-PHAT 的 steering delays;
        use_* 开关用于 ablation, 不改变输出字典的 key 和 shape.
        """
        super().__init__()
        self.spec = STFTSpec(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            spec_bins=spec_bins,
            ipd_bins=ipd_bins,
            azimuth_bins=azimuth_bins,
            target_frames=target_frames,
        )
        self.use_logmel_ref = use_logmel_ref
        self.use_logmel_rms = use_logmel_rms
        self.use_ipd = use_ipd
        self.use_srp = use_srp
        self.use_vad = use_vad
        self.stft = MultiChannelSTFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        self.register_buffer(
            "mel_filterbank",
            build_mel_filterbank(
                num_freqs=n_fft // 2 + 1,
                sample_rate=sample_rate,
                num_mels=spec_bins,
                device=mic_positions.device,
                dtype=mic_positions.dtype,
            ),
            persistent=False,
        )
        self.mic_pairs = default_mic_pairs(int(mic_positions.shape[0]))
        self.srp = SRPPHAT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            mic_positions=mic_positions,
            mic_pairs=self.mic_pairs,
            azimuths=azimuth_grid(
                azimuth_bins,
                device=mic_positions.device,
                dtype=mic_positions.dtype,
            ),
            sound_speed=sound_speed,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        vad_history: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """从多通道音频中提取帧对齐特征."""
        stft_complex = self.stft(waveform)
        logmel_ref, logmel_rms = build_logmel_like_features(
            stft_complex,
            mel_filterbank=self.mel_filterbank,
            out_bins=self.spec.spec_bins,
        )
        ipd_feat = compute_ipd_features(stft_complex, self.mic_pairs, num_bins=self.spec.ipd_bins)
        srp_map = self.srp(stft_complex).unsqueeze(1)
        if not self.use_logmel_ref:
            logmel_ref = torch.zeros_like(logmel_ref)
        if not self.use_logmel_rms:
            logmel_rms = torch.zeros_like(logmel_rms)
        if not self.use_ipd:
            ipd_feat = torch.zeros_like(ipd_feat)
        if not self.use_srp:
            srp_map = torch.zeros_like(srp_map)

        # 所有分支离开 frontend 前都被对齐到同一个历史长度.
        # 下游模块不需要再各自处理时间维裁剪/补齐.
        logmel_ref = fit_time_dim(logmel_ref, self.spec.target_frames)
        logmel_rms = fit_time_dim(logmel_rms, self.spec.target_frames)
        ipd_feat = fit_time_dim(ipd_feat, self.spec.target_frames)
        srp_map = fit_time_dim(srp_map, self.spec.target_frames)
        if vad_history is None or not self.use_vad:
            vad_history = torch.zeros(
                waveform.shape[0],
                self.spec.target_frames,
                1,
                device=waveform.device,
                dtype=waveform.dtype,
            )
        elif vad_history.ndim == 2:
            vad_history = vad_history.unsqueeze(-1)
        elif vad_history.ndim != 3:
            raise ValueError("Expected vad_history shape [batch, frames] or [batch, frames, 1].")
        if vad_history.shape[1] != self.spec.target_frames:
            vad_history = fit_time_dim(vad_history.unsqueeze(1), self.spec.target_frames).squeeze(1)

        return {
            "logmel_ref": logmel_ref,
            "logmel_rms": logmel_rms,
            "ipd_feat": ipd_feat,
            "srp_map": srp_map,
            "vad_ratio": vad_history,
        }
