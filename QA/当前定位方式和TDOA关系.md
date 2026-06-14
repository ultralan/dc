# 当前定位方式和 TDOA 的关系

## Q1：我们现在的定位方式是基于 TDOA 吗？

准确说：**不是纯 TDOA 定位，但底层用了 TDOA/相位差思想。**

当前项目不是这样做的：

```text
先显式估计每对麦克风的 TDOA
  -> 再用几何方程反解声源方向
  -> 最后得到 DOA
```

当前项目实际更接近：

```text
多通道音频
  -> STFT
  -> logmel + IPD + SRP-PHAT + VAD
  -> 空间特征编码器
  -> TCN 时序主干
  -> count / heatmap / slot / future trend
```

其中 `IPD` 和 `SRP-PHAT` 都和 TDOA 密切相关。

所以一句话总结：

```text
当前方案不是“传统 TDOA 三角定位”，而是“深度模型 + IPD/SRP-PHAT 空间先验”。
TDOA 是它的底层物理基础之一。
```

## Q2：TDOA 是什么？

TDOA 的全称是 `Time Difference of Arrival`，中文是“到达时间差”。

声音从某个方向传到多个麦克风时，因为麦克风位置不同，到达时间会有微小差异。

例如声源更靠近麦克风 1：

```text
声源 -> mic1：先到
声源 -> mic2：后到
```

那么两个麦克风之间会有一个时间差：

```text
TDOA = t_mic2 - t_mic1
```

这个时间差和声源方向有关。  
只要知道麦克风位置、声速和时间差，就可以反推出声源大概来自哪个方向。

## Q3：为什么 TDOA 能定位？

以两个麦克风为例。

假设两个麦克风间距为 `d`，声速为 `c`，声源方向和麦克风连线的夹角为 `theta`。

远场近似下，两个麦克风的到达时间差大致满足：

```text
tau = d * cos(theta) / c
```

其中：

- `tau` 是 TDOA
- `d` 是麦克风间距
- `c` 是声速，通常约 `343 m/s`
- `theta` 是声源方向相关角度

多个麦克风对会产生多个 TDOA。  
把这些时间差综合起来，就能估计 DOA，也就是声源方向。

## Q4：GCC-PHAT 和 TDOA 是什么关系？

`GCC-PHAT` 是一种经典的 TDOA 估计方法。

它的核心思路是：

```text
两个麦克风信号
  -> 计算互相关
  -> 找到相关峰值
  -> 峰值位置对应时间延迟
```

普通互相关容易受混响、音量和频谱形状影响。  
`PHAT` 会做相位变换，大致等价于弱化幅度，只强调相位差。

所以：

```text
GCC-PHAT = 更稳健地估计 TDOA 的经典方法
```

## Q5：SRP-PHAT 和 TDOA 是什么关系？

`SRP-PHAT` 的全称是：

```text
Steered Response Power with Phase Transform
```

它不是先估一个单独的 TDOA，再反解方向。  
它更像是“扫描所有可能方向”：

```text
假设声源来自 -180 度
  -> 根据麦克风几何计算理论时延
  -> 看多麦信号是否支持这个方向

假设声源来自 -175 度
  -> 再算一次

...

哪个方向得分最高，就认为声源更可能来自哪里
```

在当前代码里，[srp_phat.py](</c:/Users/haoming lan/Desktop/dc/src/uca8/features/srp_phat.py>) 会根据候选方位和麦克风几何生成 `pair_delays`，再用相位归一化后的麦对互谱计算每个方位的得分。

因此：

```text
SRP-PHAT = 用 TDOA/相位差理论做方位扫描
```

它比“先估 TDOA 再解几何方程”更适合多麦克风阵列和多方向打分。

## Q6：IPD 和 TDOA 是什么关系？

`IPD` 的全称是：

```text
Inter-channel Phase Difference
```

中文是“通道间相位差”。

对于某个频率 `f`，时间延迟 `tau` 会表现成相位差：

```text
phase_difference = 2 * pi * f * tau
```

也就是说：

```text
TDOA 是时间差；
IPD 是这个时间差在频域里的相位表现。
```

当前代码里的 [ipd.py](</c:/Users/haoming lan/Desktop/dc/src/uca8/features/ipd.py>) 会对选定麦克风对计算交叉谱相位，并用：

```text
cos(IPD), sin(IPD)
```

来编码相位差，避免角度跳变问题。

## Q7：当前项目具体用了哪些空间定位特征？

当前模型前端主要提取：

- `logmel_ref`：参考麦克风 log-mel
- `logmel_rms`：多通道 RMS log-mel
- `ipd_feat`：麦对间相位差特征
- `srp_map`：SRP-PHAT 方位扫描图
- `vad_ratio`：语音活动比例

模型结构在 [tracktrend_net.py](</c:/Users/haoming lan/Desktop/dc/src/uca8/models/tracktrend_net.py>) 中体现为：

```text
logmel -> SpectrogramEncoder
IPD    -> IPDEncoder
SRP    -> SRPEncoder

三路特征 + VAD
  -> 融合
  -> Causal TCN
  -> 多任务输出
```

因此定位不是单一公式算出来的，而是：

```text
经典空间先验 + 深度时序模型
```

## Q8：为什么不直接用纯 TDOA？

纯 TDOA 方法优点是可解释、轻量。  
但当前任务比单源静态定位复杂得多：

- 多声源可能同时存在
- 声源可能移动
- 房间有混响
- 真实噪声会干扰相关峰
- 麦克风阵列和训练数据之间可能有 mismatch
- 还要预测未来短窗趋势

纯 TDOA 通常只能解决：

```text
当前帧或短窗内，声源大概来自哪个方向
```

而当前项目要输出：

- 当前声源数量
- 当前方位热力图
- 多源槽位状态
- 未来声源数量变化
- 未来方位热力图
- 未来槽位趋势
- 运动方向分类

所以更合理的方案是：

```text
用 TDOA/IPD/SRP-PHAT 提供物理空间先验；
用深度模型学习多源、动态和真实数据中的复杂模式。
```

## Q9：这套方案的优点是什么？

当前方案的优点是兼顾可解释性和学习能力。

可解释性来自：

- 麦克风几何
- TDOA
- IPD
- SRP-PHAT
- 方位热力图

学习能力来自：

- CNN 空间特征编码器
- TCN 时序主干
- 多任务输出头
- RealMAN 真实数据训练

这比纯 TDOA 更适合真实多源语音定位和跟踪。

## Q10：一句话总结

当前项目**不是传统纯 TDOA 定位**。  
它是以 `IPD` 和 `SRP-PHAT` 作为空间前端、以 `TCN` 作为时序主干的深度多任务定位跟踪方案。

但 TDOA 仍然是它的重要物理基础：

```text
TDOA -> IPD / SRP-PHAT -> 深度模型定位与跟踪
```

