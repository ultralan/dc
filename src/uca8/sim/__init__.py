"""Analytic array simulation helpers."""

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
