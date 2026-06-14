# RealMAN 数据集特征与 Baseline

## Q1：RealMAN 是什么数据集？

RealMAN 的全称是：

```text
Real-recorded and annotated Microphone Array speech&Noise
```

它是 NeurIPS 2024 Datasets and Benchmarks Track 收录的数据集，目标是给多通道语音增强和声源定位提供真实录制、真实标注的数据。

它解决的核心问题是：

```text
很多多通道语音增强/定位模型依赖仿真 RIR 和仿真噪声；
但仿真声学和真实环境之间有 mismatch；
模型在真实场景落地时性能会下降。
```

RealMAN 的价值就是提供真实阵列录音、真实噪声、真实源位置标注和 direct-path clean target。

## Q2：数据集有哪些核心特征？

RealMAN 的主要特征如下：

| 特征 | 内容 |
| --- | --- |
| 麦克风阵列 | `32` 通道高保真麦克风阵列 |
| 采样率 | 原始麦克风录音 `48 kHz` |
| 语音时长 | 约 `83.7 h`，其中静态说话源约 `48.3 h`，移动说话源约 `35.4 h` |
| 噪声时长 | 约 `144.5 h` |
| 语音场景 | `32` 个场景 |
| 噪声场景 | `31` 个场景 |
| 场景类型 | 室内、室外、半室外、交通环境 |
| 声源形式 | 扬声器播放普通话语音，模拟人站立/移动说话 |
| 标注方式 | 鱼眼相机检测扬声器上的 LED，得到声源方位 |
| 定位标注 | 主要提供 azimuth angle，更新版还补充 elevation / distance 标注 |
| 增强目标 | 提供 direct-path speech 作为 clean target |
| 文本信息 | 提供 speech transcription |

## Q3：它和我们项目最相关的地方是什么？

最相关的是四点：

1. 它是真实录制，不是纯仿真。
2. 它有移动声源。
3. 它有声源方位标注。
4. 它是 `32ch` 阵列，可以抽取不同子阵列训练。

我们当前项目使用的是：

```text
RealMAN 32ch -> ring2 8ch 子阵列 -> 8 麦圆阵模型
```

本地配置中使用的通道是：

```text
channel_ids: [9, 10, 11, 12, 13, 14, 15, 16]
```

这对应当前项目里的 `ring2_8ch` 子集。

## Q4：RealMAN 的文件结构大概是什么？

官方数据包含：

```text
RealMAN
├── transcriptions.trn
├── dataset_info
│   ├── scene_images
│   ├── scene_info.json
│   └── speaker_info.csv
└── train | val | test | val_raw | test_raw
    ├── *_moving_source_location.csv
    ├── *_static_source_location.csv
    ├── dp_speech
    ├── ma_speech 或 ma_noisy_speech
    └── ma_noise
```

几个关键目录含义：

| 名称 | 含义 |
| --- | --- |
| `ma_speech` | 多通道干净语音阵列录音 |
| `ma_noisy_speech` | 多通道带噪语音阵列录音 |
| `ma_noise` | 多通道真实环境噪声 |
| `dp_speech` | direct-path speech，语音增强训练目标 |
| `*_source_location.csv` | 声源位置标注 |
| `transcriptions.trn` | 语音转写文本 |

## Q5：我们代码里从 RealMAN 读了哪些东西？

当前 [realman_ring2_dataset.py](</c:/Users/haoming lan/Desktop/dc/src/uca8/data/realman_ring2_dataset.py>) 主要读取：

- `dp_speech`：用于计算 VAD/activity
- `CH9-CH16` 多通道音频：作为模型输入
- `moving_source_location.csv`
- `static_source_location.csv`

然后构造训练标签：

- 当前声源数量 `count`
- 当前方位热力图 `heatmap`
- 当前 slot 状态 `slot_state`
- 未来声源数量 `future_count`
- 未来方位热力图 `future_heatmap`
- 未来 slot 状态 `future_slot_state`
- 运动趋势类别 `trend_class`

也就是说，我们不是直接照搬 RealMAN 官方任务，而是在它的真实阵列和位置标注基础上，构造了面向“短窗定位跟踪与未来趋势预测”的任务。

## Q6：RealMAN 官方 baseline 做了哪些任务？

官方 baseline 主要有两个任务：

1. Speech enhancement
2. Sound source localization

### Speech enhancement baseline

官方用了两个增强模型：

| 方法 | 类型 | 说明 |
| --- | --- | --- |
| `FaSNet-TAC` | 时域多通道语音分离/增强网络 | 强调麦克风数量和排列不变性 |
| `SpatialNet` | 频域多通道语音增强网络 | 强调空间信息建模 |

### Sound source localization baseline

官方用了两个定位模型：

| 方法 | 类型 | 说明 |
| --- | --- | --- |
| `CRNN` | CNN + GRU | 10 层 CNN + 1 层 GRU，预测 azimuth spatial spectrum |
| `IPDnet` | direct-path IPD 定位网络 | 学习 direct-path IPD，支持 fixed-array / variable-array |

这里很关键：RealMAN 的定位 baseline 也不是纯 raw waveform 端到端，而是显式利用空间谱或 IPD 类空间信息。

## Q7：官方用了哪些评价指标？

### 定位指标

RealMAN 的 source localization 主要评估 azimuth angle。

| 指标 | 含义 | 越大/越小越好 |
| --- | --- | --- |
| `MAE [°]` | 预测方位角和真实方位角的平均绝对误差 | 越小越好 |
| `ACC(5°) [%]` | 方位误差小于 `5°` 的帧比例 | 越大越好 |

### 语音增强指标

| 指标 | 含义 | 越大/越小越好 |
| --- | --- | --- |
| `WB-PESQ` | 宽带语音质量感知指标 | 越大越好 |
| `SI-SDR` | 尺度不变信号失真比 | 越大越好 |
| `MOS-SIG` | DNSMOS 中语音信号质量分 | 越大越好 |
| `MOS-BAK` | DNSMOS 中背景噪声质量分 | 越大越好 |
| `MOS-OVR` | DNSMOS 总体质量分 | 越大越好 |
| `CER` | 中文 ASR 字错误率 | 越小越好 |

## Q8：官方 baseline 里定位效果最好的是哪个？

要分两组看，因为官方表格不是完全同一个实验设置。

### 1. CRNN 的 sim-vs-real 对比实验

CRNN 使用 `9-channel` 子阵列做定位 baseline。训练数据分成：

- sim speech + sim noise
- sim speech + real noise
- real speech + sim noise
- real speech + real noise

结果显示，**real speech + real noise 最好**：

| 训练数据 | 静态 ACC(5°) | 静态 MAE | 移动 ACC(5°) | 移动 MAE |
| --- | ---: | ---: | ---: | ---: |
| sim + sim | 71.9 | 10.2° | 68.8 | 9.6° |
| sim + real | 76.7 | 9.9° | 70.3 | 11.1° |
| real + sim | 82.1 | 8.1° | 75.9 | 8.2° |
| real + real | **88.4** | **4.6°** | **83.9** | **4.3°** |

这个表最重要的结论是：

```text
真实语音 + 真实噪声训练，明显优于仿真数据训练。
```

### 2. IPDnet 的 fixed-array / variable-array 实验

官方还用 IPDnet 做阵列泛化实验：

| 方法 | 静态 ACC(5°) | 静态 MAE | 移动 ACC(5°) | 移动 MAE |
| --- | ---: | ---: | ---: | ---: |
| Fixed-Array IPDnet | 86.1 | 3.6° | **88.9** | **2.7°** |
| Variable-Array IPDnet | 86.1 | **3.5°** | 80.4 | 3.6° |

如果只看定位误差，`Fixed-Array IPDnet` 在移动源上表现最好：

```text
Moving speaker: ACC(5°) = 88.9%, MAE = 2.7°
```

但它是 fixed-array 设置，对指定测试阵列训练。  
`Variable-Array IPDnet` 的意义是可以训练在多种子阵列上，然后泛化到未见阵列，虽然移动源指标有一定下降。

## Q9：官方 baseline 里增强效果最好的是哪个？

增强任务中，整体看 `SpatialNet` 在 `real speech + real noise` 训练下表现最好，尤其是 `WB-PESQ`、`SI-SDR`、`MOS-OVR` 和 `CER`。

代表性结果：

| 方法 | 训练数据 | 静态 WB-PESQ | 静态 SI-SDR | 静态 MOS-OVR | 静态 CER | 移动 WB-PESQ | 移动 SI-SDR | 移动 MOS-OVR | 移动 CER |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unprocessed | - | 1.15 | -9.8 | 1.51 | 19.9 | 1.11 | -9.1 | 1.36 | 23.8 |
| FaSNet-TAC | real + real | 1.51 | 1.3 | 2.35 | 22.4 | 1.43 | 1.1 | 2.27 | 26.3 |
| SpatialNet | real + real | **2.10** | **6.1** | **2.62** | **16.0** | **1.90** | **3.8** | **2.52** | **21.5** |

所以增强任务可总结为：

```text
SpatialNet + real speech + real noise 整体最好。
```

## Q10：这些 baseline 对我们有什么启发？

### 1. 真实数据很关键

CRNN 定位实验里，`real + real` 明显优于 `sim + sim`。

这直接支持我们使用 RealMAN，而不是只靠仿真数据。

### 2. 空间特征不是过时路线

官方定位 baseline 包含：

- spatial spectrum target
- CRNN
- IPDnet
- direct-path IPD

这说明 RealMAN 作者自己的 baseline 也在利用空间谱/IPD 这类声学先验，而不是完全依赖 raw waveform 端到端。

### 3. 子阵列训练是正当路线

RealMAN 明确验证了 variable-array 网络，可以用 `32ch` 阵列的多种子阵列训练，再泛化到未见阵列。

这支持我们的：

```text
32ch RealMAN -> ring2 8ch -> 目标 8 麦设备
```

### 4. 我们和官方 baseline 的差异在哪里？

官方 source localization baseline 主要做：

```text
当前帧 azimuth localization
```

我们现在做的是：

```text
当前 count + heatmap + slot
+ 未来 count + heatmap + slot
+ motion trend
```

所以我们不是简单复现 RealMAN baseline，而是在它的数据和定位任务基础上，增加了多源槽位和短窗未来趋势预测。

## Q11：参考来源

- RealMAN arXiv：<https://arxiv.org/abs/2406.19959>
- RealMAN GitHub：<https://github.com/Audio-WestlakeU/RealMAN>
- RealMAN Hugging Face：<https://huggingface.co/datasets/AISHELL/RealMAN>
- FaSNet-TAC：<https://doi.org/10.1109/ICASSP40776.2020.9054177>
- SpatialNet：<https://doi.org/10.1109/TASLP.2024.3357036>
- IPDnet：<https://arxiv.org/abs/2405.07021>

## Q12：官方 baseline 分别对应哪些参考文献？

RealMAN 论文里的官方 baseline 可以分成两类。

### 1. RealMAN 论文内定义的 baseline

`CRNN` 定位 baseline 是 RealMAN 论文里用于 source localization 实验的基线模型。论文描述它由 `10` 层 CNN 和 `1` 层 GRU 组成，输入多通道语音，输出 azimuth spatial spectrum。

这部分引用 RealMAN 论文即可：

```text
Yang, B., Quan, C., Wang, Y., Wang, P., Yang, Y., Fang, Y.,
Shao, N., Bu, H., Xu, X., Li, X.
RealMAN: A Real-Recorded and Annotated Microphone Array Dataset
for Dynamic Speech Enhancement and Localization.
NeurIPS 2024 Datasets and Benchmarks Track.
arXiv:2406.19959.
```

### 2. 有独立原论文的 baseline

`FaSNet-TAC`：

```text
Luo, Y., Chen, Z., Mesgarani, N., Yoshioka, T.
End-to-end Microphone Permutation and Number Invariant
Multi-channel Speech Separation.
ICASSP 2020.
DOI: 10.1109/ICASSP40776.2020.9054177.
```

`SpatialNet`：

```text
Quan, C., Li, X.
SpatialNet: Extensively Learning Spatial Information for Multichannel
Joint Speech Separation, Denoising and Dereverberation.
IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.
DOI: 10.1109/TASLP.2024.3357036.
```

`IPDnet`：

```text
Wang, Y., Yang, B., Li, X.
IPDnet: A Universal Direct-Path IPD Estimation Network
for Sound Source Localization.
arXiv:2405.07021.
```

因此写论文或申报书时可以这样处理：

- 讲 RealMAN 数据集和官方 baseline 结果：引用 RealMAN
- 讲增强 baseline 的模型来源：引用 FaSNet-TAC 和 SpatialNet
- 讲定位 baseline 的 direct-path IPD 方法：引用 IPDnet
- 讲 CRNN baseline：引用 RealMAN 本文即可
