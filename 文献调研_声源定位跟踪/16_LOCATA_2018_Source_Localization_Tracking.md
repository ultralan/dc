# Lollmann et al. 2018：LOCATA 真实声源定位与跟踪数据集

## 1. 论文信息

- 标题：`The LOCATA Challenge Data Corpus for Acoustic Source Localization and Tracking`
- 作者：Heinrich W. Lollmann, Christine Evers, Alexander Schmidt, Heinrich Mellmann, Hendrik Barfuss, Patrick A. Naylor, Walter Kellermann
- 年份：`2018`
- 来源：`IEEE Sensor Array and Multichannel Signal Processing Workshop (SAM 2018)`, pp.410-414
- DOI：`10.1109/SAM.2018.8448644`
- 类型：真实声源定位与跟踪挑战数据集

## 2. 为什么这篇论文重要

RealMAN 在相关工作中把 LOCATA 作为少数直接提供声源位置标注的真实多通道数据集之一。

它的重要性在于：

- 真实录音
- 真实阵列
- 包含定位和跟踪任务
- 提供多种阵列配置
- 有公开 challenge 和评测框架

这使它成为 RealMAN 之前真实声源定位数据集的代表。

## 3. 研究问题

LOCATA 要解决的是：

如何让不同声源定位与跟踪算法在统一真实数据上进行客观比较。

它覆盖的任务包括：

- 静态单源定位
- 移动单源跟踪
- 多源定位
- 不同麦克风阵列下的算法评测

## 4. 核心内容

LOCATA 提供真实声学环境下的多通道录音，并配套 ground truth 位置信息。

它使用的阵列类型包括：

- planar array
- spherical array
- robot array
- hearing-aid array

这些设计使它适合评估算法对阵列几何和真实环境的适应能力。

## 5. 对当前项目的直接借鉴

### 5.1 说明“真实跟踪数据”是领域标准需求

LOCATA 和 RealMAN 都强调真实录音的重要性。

这支持当前项目优先使用 RealMAN，而不是只依赖仿真 RIR 数据。

### 5.2 支持 tracking 后处理的重要性

LOCATA 的任务设置明确包含 source tracking，而不仅是单帧 DOA。

这支持当前项目保留：

- 轨迹平滑
- slot 延续
- Kalman/Hungarian 类后处理
- 跨帧身份一致性约束

### 5.3 作为 RealMAN 的前置对比

LOCATA 很适合在文献综述中用来说明：

真实定位数据集早已有代表，但规模、场景数量和训练用途有限，因此 RealMAN 的大规模真实训练数据更适合当前项目。

## 6. 局限与不适用点

- 数据规模明显小于 RealMAN
- 场景多样性不如 RealMAN
- 更偏 challenge/evaluation，不是为大规模神经网络训练专门设计
- 和当前 `8` 麦圆阵设备不是同一阵列

## 7. 对本项目的使用建议

建议把 LOCATA 作为“真实声源定位与跟踪数据集先例”引用。

它适合支撑：

- 真实录音评测比纯仿真更有说服力
- source localization 和 tracking 应该一起考虑
- RealMAN 是在 LOCATA 这类真实数据路线上的规模化推进

## 8. 参考链接

- DOI：<https://doi.org/10.1109/SAM.2018.8448644>
- LOCATA 数据集页面：<https://www.locata.lms.tf.fau.de/datasets/>
- LOCATA Challenge 综述：<https://arxiv.org/abs/1909.01008>

