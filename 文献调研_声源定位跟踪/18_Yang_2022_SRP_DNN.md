# Yang et al. 2022：SRP-DNN 多移动声源定位

## 1. 论文信息

- 标题：`SRP-DNN: Learning Direct-Path Phase Difference for Multiple Moving Sound Source Localization`
- 作者：Bing Yang, Hong Liu, Xiaofei Li
- 年份：`2022`
- 来源：`ICASSP 2022`, pp.721-725
- DOI：`10.1109/ICASSP43922.2022.9746624`
- arXiv：`2202.07859`
- 类型：直接路径相位差学习 + SRP 多移动声源定位论文

## 2. 为什么这篇论文重要

RealMAN 在 sound source localization baseline 讨论中引用了这条路线。

它和当前项目非常贴，因为它同时包含：

- 多声源
- moving sources
- direct-path phase difference
- SRP spatial spectrum
- assignment ambiguity 处理

这比一般的静态 DOA 分类论文更接近我们的任务。

## 3. 研究问题

论文要解决的是：

在噪声、混响、多源交互和轨迹变化条件下，如何定位多个移动声源。

传统 SRP 类方法容易受到混响和多源干扰影响，因此作者尝试用神经网络学习更干净的 direct-path phase difference。

## 4. 核心方法

论文设计了一个 causal convolutional recurrent neural network，用来从每个麦克风对中提取 direct-path phase difference 序列。

关键设计包括：

- 用 DNN 学习 direct-path phase difference
- 用 weighted sum target 同时编码 source activity 和 phase difference
- 避免多目标预测中的 assignment ambiguity 和输出维度不确定问题
- 将学习到的相位差重新送入 SRP 公式生成 spatial spectrum
- 通过迭代检测和移除 dominant source 来降低多源交互

## 5. 对当前项目的直接借鉴

### 5.1 支持 `IPD + SRP + DNN` 融合路线

当前项目已经使用 `IPD` 和 `SRP` 类空间特征。

SRP-DNN 说明，经典 SRP 不一定要被深度模型替代，也可以作为深度模型输出后的可解释空间聚合层。

### 5.2 支持多源 assignment 需要显式处理

论文明确处理多目标输出中的 assignment ambiguity。

这和当前项目的 slot 输出、Hungarian/PIT 训练、同类多说话人建模是同一个问题。

### 5.3 支持移动声源短时序建模

论文使用 causal 时序网络处理 time-varying direct-path phase difference。

这支持当前项目用 TCN/时序 backbone 处理短窗历史和未来趋势。

## 6. 局限与不适用点

- 重点是相位差和 SRP 空间谱，不是完整 SELD 多任务框架
- 网络形式是 CRNN，不是当前项目使用的 TCN
- 没有直接设计 future head
- 依赖阵列几何和相位建模，对设备标定敏感

## 7. 对本项目的使用建议

建议把这篇列为 RealMAN 相关文献中的高优先级引用。

它可以支撑：

- 为什么保留 SRP/IPD 这类经典空间先验
- 为什么多移动声源要处理 assignment ambiguity
- 为什么声源定位不应只做单帧角度分类

## 8. 参考链接

- DOI：<https://doi.org/10.1109/ICASSP43922.2022.9746624>
- arXiv：<https://arxiv.org/abs/2202.07859>
- IEEE 资源页：<https://resourcecenter.ieee.org/conferences/icassp-2022/spsicassp22vid1365>

