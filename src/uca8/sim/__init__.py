"""解析仿真工具入口.

本包用于生成可控 probe/curriculum 场景:
- ``farfield`` 负责按远场延迟模型渲染多通道音频;
- ``probe_scenarios`` 负责构造静态、移动、交叉、进入/离开等测试场景.
这些工具主要服务调试、可视化和扩展实验, 不直接代表 RealMAN 官方数据分布.
"""

from .farfield import render_farfield_history_waveform
from .probe_scenarios import (
    SCENARIO_CHOICES,
    ProbeSample,
    build_probe_rollout_samples,
    build_probe_sample,
    build_scenario,
    default_source_audio,
    evaluate_probe_suite,
    infer_transition_start_index,
    load_probe_mono_audio,
    render_history_waveform,
    summarize_probe_suite,
    transition_start_index,
)

__all__ = [
    "ProbeSample",
    "SCENARIO_CHOICES",
    "build_probe_rollout_samples",
    "build_probe_sample",
    "build_scenario",
    "default_source_audio",
    "evaluate_probe_suite",
    "infer_transition_start_index",
    "load_probe_mono_audio",
    "render_farfield_history_waveform",
    "render_history_waveform",
    "summarize_probe_suite",
    "transition_start_index",
]
