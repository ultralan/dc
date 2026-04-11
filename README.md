# UCA8 TrackTrend

面向 `8` 麦克风均匀圆阵的音频侧项目骨架，目标是把两类任务放进同一个工程里：

- 多源数量 / 方位 / 轨迹建模
- 基于短窗历史的未来 `320 ms` 趋势预测

当前仓库提供的是第一版可训练骨架：

- `configs/`：Hydra 配置
- `src/uca8/`：几何、特征、数据、模型、损失、后处理
- `scripts/train.py`：基于 `lightning.fabric` 的最小训练脚本
- `tests/`：几何、标签、模型 shape 测试

## 快速开始

```powershell
uv sync --group dev
uv run python -m pytest
uv run python scripts/train.py train.limit_train_steps=2
```

如果你先只想做前向和 shape 联调，建议先把数据切到 synthetic：

```powershell
uv run python scripts/train.py data.dataset_kind=synthetic train.limit_train_steps=2
```

如果你要跑当前的 RealMAN `8ch` 训练型 MVP，建议直接使用：

```powershell
uv run python scripts/train.py --config-name realman_ring2_mvp
uv run python scripts/visualize_realman_run.py --run-dir runs/realman_ring2_mvp/<timestamp>
```

## 当前边界

- 模型已经接好 `waveform -> frontend -> encoder -> TCN -> heads` 主链路。
- `label_builder` 已经能产出 `count / heatmap / slot_state / future target`。
- `LocataLikeTrackTrendDataset` 已经能扫描本地 `dev/` 目录并构造窗口样本。
- 音频 I/O 优先尝试 `torchcodec`，其次回退 `torchaudio`，最后回退 `scipy`。

## 下一步建议

1. 把 current/future 的 slot loss 升级为更强的 Hungarian-aware/PIT 版本。
2. 在验证和可视化里加入更多轨迹连续性指标。
3. 将多子阵列训练和通道 dropout 纳入设备泛化训练策略。
4. 把 Kalman 后处理正式接到推理输出链路。
