# Quan & Li 2024：SpatialNet 多通道空间信息学习

## 1. 论文信息

- 标题：`SpatialNet: Extensively Learning Spatial Information for Multichannel Joint Speech Separation, Denoising and Dereverberation`
- 作者：Changsheng Quan, Xiaofei Li
- 年份：`2024`
- 来源：`IEEE/ACM Transactions on Audio, Speech, and Language Processing`, 32:1310-1323
- DOI：`10.1109/TASLP.2024.3357036`
- 类型：多通道语音增强/分离/去混响空间建模论文

## 2. 为什么这篇论文重要

RealMAN 使用 SpatialNet 作为 speech enhancement baseline。

虽然它不是声源定位论文，但它和当前项目仍有两点强相关：

- 它处理多通道语音
- 它强调在 STFT 域显式学习空间信息

RealMAN 的实验还显示，SpatialNet 对真实阵列和仿真阵列之间的 mismatch 很敏感，这对我们使用真实 RealMAN 数据非常有提醒意义。

## 3. 研究问题

论文要解决的是：

如何在多通道语音分离、去噪和去混响中充分利用空间信息。

传统方法往往依赖固定阵列或有限空间特征，SpatialNet 试图通过网络结构系统学习频带内和频带间的空间关系。

## 4. 核心方法

SpatialNet 工作在 STFT 域，主要由交替堆叠的模块构成：

- narrow-band blocks：按频带处理空间信息
- cross-band blocks：建模频率之间的相关性
- self-attention：用于空间特征上的 speaker clustering
- temporal convolution：用于时间平滑和滤波

它说明多通道语音任务中，空间信息不是附属特征，而是模型设计的核心对象。

## 5. 对当前项目的直接借鉴

### 5.1 支持多通道空间信息显式建模

当前项目不应只把 8 路音频当成普通多通道输入。

SpatialNet 支持一个判断：多通道语音模型需要主动利用通道间空间关系。

### 5.2 支持频带维度和空间维度联合建模

当前项目的 `logmel/IPD/SRP` 特征也可以理解为空间-频率信息的组合。

后续如果升级 backbone，可以考虑更细的：

- narrow-band spatial block
- cross-band fusion
- temporal smoothing block

### 5.3 提醒真实阵列数据的重要性

RealMAN 的实验中，SpatialNet 在仿真训练和真实测试之间会出现明显 mismatch。

这说明阵列几何、麦克风位置误差、真实噪声空间相关性都会影响模型。

因此当前项目使用 RealMAN 真实录音和目标 `8ch` 子阵列训练是必要的。

## 6. 局限与不适用点

- 主任务是语音增强/分离/去混响，不是 DOA 定位
- 不能直接替代当前定位头
- 结构可能比当前 TCN backbone 更重
- 不直接处理多源定位 slot assignment

## 7. 对本项目的使用建议

建议把 SpatialNet 作为 RealMAN baseline 和多通道空间建模参考引用。

它适合支撑：

- 多通道语音任务必须显式学习空间信息
- 真实阵列和仿真阵列存在可观 mismatch
- 当前项目后续可从 TCN backbone 升级到更强的频带-空间融合结构

## 8. 参考链接

- DOI：<https://doi.org/10.1109/TASLP.2024.3357036>
- arXiv：<https://arxiv.org/abs/2307.16516>
- IEEE 页面：<https://ieeexplore.ieee.org/document/10423815/>

