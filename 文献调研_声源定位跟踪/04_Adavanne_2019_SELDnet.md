# Adavanne et al. 2019：SELDnet 联合检测定位基线

## 1. 论文信息

- 标题：`Sound Event Localization and Detection of Overlapping Sources Using Convolutional Recurrent Neural Networks`
- 作者：Sharath Adavanne, Archontis Politis, Joonas Nikunen, Tuomas Virtanen
- 年份：`2019`
- 来源：`IEEE Journal of Selected Topics in Signal Processing`, 13(1):34-48
- DOI：`10.1109/JSTSP.2018.2885636`
- 类型：深度 SELD 代表性论文

## 2. 为什么这篇论文重要

如果说当前很多 SELD 系统的“共同祖先”是哪篇论文，答案通常就是这篇。  
它把：

- 声事件检测
- 声源定位
- 时序关联

统一到一个 `CRNN` 框架里，是后续 ACCDOA、Multi-ACCDOA 等工作的直接前序。

## 3. 研究问题

论文要解决的是：  
对于同时重叠的多个声音事件，能不能在一个网络中同时完成：

- 每类事件是否发生
- 每类事件来自哪个方向

并且在时间上保持关联。

## 4. 核心方法

SELDnet 的核心结构是：

`CNN + RNN + FC`

输入使用多通道音频的：

- 幅度谱
- 相位谱

输出分成两支：

- `SED`：逐帧多标签分类
- `DOA`：逐帧三维 Cartesian 坐标回归

其关键优点是：

- 不依赖手工设计的阵列专属特征
- 可以处理重叠源
- 能在网络内部建立时间上的关联

## 5. 数据与实验设置

论文在多个数据集上验证方法，包括：

- `5` 个 Ambisonic 数据集
- `2` 个 circular array 数据集
- 覆盖无混响、混响、真实场景和低信噪比条件

因此它不是只在单一场景上有效，而是强调跨阵列格式和跨条件泛化。

## 6. 主要结果

论文的主要发现包括：

- 联合 `SELD` 网络优于分拆式基线
- 对未见过的 DOA、混响和低 SNR 有较好鲁棒性
- 在重叠源较多时，估计源数量的 recall 仍然优于传统基线

## 7. 对当前项目的直接借鉴

### 7.1 共享编码器 + 多头输出是正确方向

当前项目使用：

- 共享前端
- 共享时序主干
- 多个任务头

这和 SELDnet 的思想高度一致。  
它说明“共享表征 + 多任务输出”比为每个子任务完全单独建模更合理。

### 7.2 多通道谱特征仍然有效

SELDnet 没有依赖特别重的手工定位特征，而是直接吃多通道谱特征。  
这对当前项目的启发是：

- `log-mel` 特征是有价值的
- `IPD/SRP` 是增强项，而不是唯一输入

### 7.3 方向回归优于纯离散分类

SELDnet 用 Cartesian 回归而不是只做离散角度分类。  
当前项目虽然保留 heatmap，但槽位输出已经用到：

- `sin(theta)`
- `cos(theta)`
- `omega`

这和 SELDnet 的连续表示思路是一致的。

## 8. 局限与不适用点

- 这是“通用多类环境声”任务，不是“单类 speech 多实例”
- 采用双分支输出，检测与定位之间仍然需要做损失平衡
- 没有显式的未来趋势预测
- 轨迹更偏隐式时序建模，没有显式 slot/identity 机制

## 9. 对本项目的使用建议

建议把这篇论文当作“结构基线”，但不要直接照搬输出形式。  
更适合当前项目的改造方式是：

- 保留共享主干
- 把类别维改成 speech source state
- 把输出重心从 event class 转到 `count + heatmap + slot`
- 在此基础上再增加 future head

## 10. 参考链接

- 论文 DOI：<https://doi.org/10.1109/JSTSP.2018.2885636>
- Aalto 文献信息页：<https://research.aalto.fi/en/publications/sound-event-localization-and-detection-of-overlapping-sources-usi/>
- arXiv 预印本：<https://arxiv.org/abs/1807.00129>
