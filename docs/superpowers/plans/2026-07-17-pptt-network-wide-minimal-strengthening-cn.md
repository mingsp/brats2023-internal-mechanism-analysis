# PPTT 全网络过程解释最小补强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不增加方法模块、不改写既有预注册结论的前提下，将 PPTT 收敛为覆盖全网络有序节点、具有严格数学性质并接受统一干预检验的过程性可解释分析方法。

**Architecture:** 方法核心只保留“输出约束的节点状态、同一像素的联合转移、全路径持续性谱系”三个计算部件。验证核心只新增一个全网络过程—干预一致性扫描：对共同根节点执行一次固定空间破坏，在所有后续节点逐一恢复 PPTT 所定位的过程像素区域，形成“过程转移区间 × 恢复节点”矩阵，并直接读取原模型最终输出。现有 V6 作为单一路径的结构来源实例进入补充材料，不承担全网络因果主张。

**Tech Stack:** Python 3.10、PyTorch 2.1、NumPy、SciPy、pandas、PyArrow、pytest、Hypothesis、Matplotlib、Seaborn、YAML、JSON。

---

## 1. 文档地位与执行约束

本文件同时承担正式设计锁定和实施顺序约束。代码实现、短流程检查、服务器正式运行、统计汇总与图件生成均按本文顺序推进。若实现发现定义矛盾，应先停止并修订本文；不得在正式结果产生后根据结果修改阈值、节点、患者集合、干预算子或成功标准。

以下历史状态保持不变：

1. V3 已完成 8 个节点、7 个相邻转移、6 个 U-Net 作业和 2,250 条正式病例轨迹，继续作为全路径观察性结果来源。
2. V4 的固定候选未触发，状态必须保持 `NOT_TRIGGERED`。
3. V5 只证明 PPTT 接口和过程输出可迁移至 TransUNet，不证明不同架构具有相同内部机制。
4. V6 已验证 `up1->up2` 对应的指定 `down2` 跳接变量，但它只是一项候选特异性结构来源检验。V6 的锁文件、结果和结论不得被覆盖、重命名或解释为全网络证明。
5. CAM 与 LayerCAM 只保留为外部对比，不进入 PPTT 的状态定义、转移定义、干预区域或成功门控。

## 2. 第一性原理：需要解释的最小对象

### 2.1 为什么终点结果不够

分割模型的最终预测只给出每个像素的终点类别。即使两个模型具有相同终点混淆矩阵和相同 Dice，中间过程仍可能分别经历稳定保持、错误纠正、正确破坏或错误类别间重编码。过程解释必须保留同一像素在相邻节点之间的联合身份，不能只比较节点均值、独立热图或终点指标。

### 2.2 为什么节点序列不是物理边序列

设冻结分割网络为有向无环计算图

$$
\mathcal N=(\mathcal V,\mathcal E),
$$

在拓扑顺序上预先声明可读节点

$$
\pi=(v_1,v_2,\ldots,v_K).
$$

`down1` 至 `up4` 是观测检查点。相邻检查点区间 $(v_k,v_{k+1})$ 表示一段计算过程，不自动等价于网络中的单一物理边。PPTT 的全网络主张是覆盖全部预声明检查点与全部相邻检查点区间，不是声称遍历了每个神经元、参数或物理边。结构来源只能由单独的干预检验给出。

### 2.3 三个且仅三个方法部件

1. **输出约束的节点状态。** 将异构中间张量映射到统一的像素类别空间，并通过观察器重启一致性、置信间隔和参数随机化控制约束读出。
2. **同一像素的联合转移。** 对每个真实类别、前一节点状态和后一节点状态联合计数，保留被端点边缘化丢失的配对信息。
3. **全路径持续性谱系。** 保存每个像素贯穿全部节点的状态身份，从而区分瞬时翻转与一直保留到最终输出的纠正或破坏。

不再引入加权总分、拟合函数、响应熵、特征范数集合或新的解释模块。新增工作只增强这三个部件的数学论证与全网络干预检验。

## 3. PPTT 的正式定义

### 3.1 节点状态与可靠性

对输入图像 $x$、像素 $p$、节点激活 $h_k(x)$ 和类别数 $C$，冻结的受控观察器 $q_k$ 输出统一分辨率的类别概率：

$$
\mathbf p_k(p)=q_k(h_k(x))(p)\in[0,1]^C,
\qquad
Z_k(p)=\arg\max_c\mathbf p_k^{(c)}(p).
$$

相邻节点的可靠性掩膜记为 $R_k(p)\in\{0,1\}$。它只表示该像素在观察器重启一致性和预注册置信间隔下是否可进入正式转移统计，不把观察器输出等同于网络唯一真实语义。

### 3.2 同一像素的全路径轨迹

$$
\mathcal Z(p)=\big[Z_1(p),Z_2(p),\ldots,Z_K(p)\big].
$$

该轨迹是 PPTT 的基本解释对象。节点均值和相邻转移张量均由它导出，但只有保留像素身份的完整轨迹能够判断三节点以上的持续性。

### 3.3 真实类别条件转移张量

对相邻检查点区间 $k$，定义

$$
T_k(g,a,b)=
\sum_p
\mathbf 1\!\left[
Y(p)=g,\,Z_k(p)=a,\,Z_{k+1}(p)=b,\,R_k(p)=1
\right].
$$

全网络过程对象为

$$
\mathcal T=\big[T_1,T_2,\ldots,T_{K-1}\big],
$$

并与逐像素轨迹 $\{\mathcal Z(p)\}_p$ 同时保存。$\mathcal T$ 负责相邻事件的精确计数，逐像素轨迹负责跨多个节点的持续性判断；两者不能互相替代。

### 3.4 事件分解

在真实类别 $g$ 下，单步转移 $(a,b)$ 被唯一划分为：

| 事件 | 条件 |
|---|---|
| 稳定正确 | $a=g,\ b=g$ |
| 错误纠正 | $a\ne g,\ b=g$ |
| 正确破坏 | $a=g,\ b\ne g$ |
| 稳定错误 | $a=b,\ a\ne g$ |
| 错误重编码 | $a\ne g,\ b\ne g,\ a\ne b$ |

这些事件是转移张量单元的互斥并集，不需要额外训练分类器，也不依赖热图阈值。

### 3.5 输出锚定的持续纠正群

对肿瘤真实类别集合 $\mathcal C_T$，定义第 $k$ 个检查点区间产生并保持到输出的纠正像素集合：

$$
\begin{aligned}
P_k=\{p\mid
&Y(p)\in\mathcal C_T,\ Z_k(p)\ne Y(p),\ Z_{k+1}(p)=Y(p),\\
&Z_\ell(p)=Y(p)\ \forall\ell\ge k+1,\\
&R_\ell(p)=1\ \forall\ell\ge k,\ \widehat Y(p)=Y(p)
\}.
\end{aligned}
$$

其中 $\widehat Y$ 是原分割模型最终预测。该定义把“观察器在某一步读到纠正”与“纠正一直保留并进入原模型输出”连接起来。它不把所有正确像素纳入目标，只追踪形成时点可以明确定位的最终正确像素。

所有 $P_k$ 两两不相交，因此联合集合

$$
P_{\mathrm{all}}=\bigcup_{k=1}^{K-1}P_k
$$

不会重复计算同一像素。该性质用于全路径统计和一次性联合区域恢复。

### 3.6 全路径过程距离

对患者 $u$，定义实际存在的肿瘤类别集合

$$
\mathcal C_T(u)=\{g\in\mathcal C_T\mid N_{u,g}>0\}.
$$

跨模型比较时只在两个模型共同可靠的像素支持

$$
R^{A\cap B}_k(p)=R^A_k(p)\land R^B_k(p)
$$

上重新计算转移张量，再按该真实类别的共同可靠像素数归一化：

$$
\widetilde T_{u,k,g}(a,b)=
\frac{T_{u,k}(g,a,b)}{\sum_{a',b'}T_{u,k}(g,a',b')}.
$$

两个模型 $A,B$ 的患者级全路径差异定义为条件转移分布的平均总变差：

$$
D_{\mathrm{proc}}(u)=
\frac{1}{2(K-1)|\mathcal C_T(u)|}
\sum_{k=1}^{K-1}\sum_{g\in\mathcal C_T(u)}
\sum_{a,b}
\left|
\widetilde T^A_{u,k,g}(a,b)-
\widetilde T^B_{u,k,g}(a,b)
\right|.
$$

$D_{\mathrm{proc}}\in[0,1]$，不拟合 Dice，不设置人工权重。共同可靠支持避免把两个模型的观察器覆盖差异误当成过程差异；患者缺失的真实类别不以零分布补齐。

## 4. 必须实现的数学性质

### 4.1 相邻事件函数的完备表示

对任意真实类别条件的相邻事件权重函数 $\phi_g(a,b)$，所有像素上的置换不变可加函数满足

$$
\sum_p\phi_{Y(p)}\big(Z_k(p),Z_{k+1}(p)\big)
=
\sum_{g,a,b}\phi_g(a,b)T_k(g,a,b).
$$

因此 $T_k$ 能精确计算任意此类相邻事件统计。若两个转移张量不同，则至少存在一个单元指标函数 $\phi_g(a,b)=\mathbf 1[(g,a,b)=(g_0,a_0,b_0)]$ 将它们区分。这里的结论限定为“对置换不变可加相邻事件信息完备”，不使用统计学中更强的参数充分性表述。

### 4.2 端点不可识别的自由度

固定转移前后混淆矩阵后，对每个真实类别 $g$，$C\times C$ 联合转移表属于具有固定行、列边缘的运输多面体。在所有边缘为正且存在内部点时，其维数为

$$
(C-1)^2.
$$

若 $C$ 个真实类别均满足该条件，总隐藏过程自由度为

$$
C(C-1)^2.
$$

BraTS 四分类设置下为 $4\times3^2=36$。边缘位于边界时维数可能降低，因此论文必须写明该维数结论的内部点条件。

### 4.3 相邻张量不能唯一决定跨节点持续性

构造二分类三节点轨迹多重集合：

$$
\mathcal A=\{000,011,101,110\},
\qquad
\mathcal B=\{001,010,100,111\}.
$$

$\mathcal A$ 与 $\mathcal B$ 在 $(Z_1,Z_2)$ 和 $(Z_2,Z_3)$ 上具有完全相同的相邻状态对计数，但在真实类别为 1 时，$\mathcal A$ 包含轨迹 011 所表示的持续纠正，$\mathcal B$ 不包含。该反例证明，跨三节点及以上的持续性必须保留同一像素的完整轨迹身份。

### 4.4 持续纠正群两两不相交

若 $p\in P_k$，则从 $k+1$ 到终点全部正确。对任何 $k'>k$，$p$ 不可能再次满足 $Z_{k'}\ne Y$；对任何 $k'<k$，$p\in P_{k'}$ 会要求 $Z_k=Y$，又与 $p\in P_k$ 要求的 $Z_k\ne Y$ 矛盾。因此 $P_k\cap P_{k'}=\varnothing$。全路径微平均可以对各 $P_k$ 求和而不重复计数，事件特异性恢复也能按区间分批执行而不混合像素身份。

### 4.5 指标重构与路径望远镜恒等式

转移张量的行边缘和列边缘分别精确恢复相邻节点的混淆矩阵：

$$
C_k(g,a)=\sum_bT_k(g,a,b),
\qquad
C_{k+1}(g,b)=\sum_aT_k(g,a,b).
$$

对任何由混淆矩阵确定的指标 $M$：

$$
\sum_{k=1}^{K-1}\left[M(C_{k+1})-M(C_k)\right]
=M(C_K)-M(C_1).
$$

正式门控要求计数重构误差为 0，浮点指标路径误差不超过 `1e-12`。

### 4.6 干预算子的可证明性质

根节点空间循环平移 $\rho_\delta$ 是空间索引上的双射，因此对每个通道保留激活值多重集合：

$$
\operatorname{multiset}(h_r[c,:,:])
=
\operatorname{multiset}(\rho_\delta(h_r)[c,:,:]).
$$

它破坏空间对应关系，但不删除或合成激活向量。恢复操作在目标位置使用同一输入的干净激活：

$$
\widetilde h_j(u)=
\begin{cases}
h_j^{\mathrm{clean}}(u), & u\in\Omega_j,\\
h_j^{\mathrm{corrupt}}(u), & u\notin\Omega_j.
\end{cases}
$$

当 $\Omega_r$ 覆盖根节点全部位置时，确定性前向应恢复干净输出；最大 logit 误差必须不超过 `1e-6`。该性质证明实现正确性，不证明网络存在唯一因果机制。

## 5. 唯一新增正式实验：V7 全网络过程—干预一致性扫描

### 5.1 核心问题

V7 只回答一个问题：

> PPTT 在全部相邻检查点区间识别出的输出锚定持续纠正像素，能否预测原模型在预声明内部干预下的最终输出变化，并且该关系是否分布在全路径而非集中于一个候选区间？

它不是七个独立跳接实验，也不是对所有物理边逐一消融。所有区间使用同一患者规则、同一破坏算子、同一恢复算子、同一对照规则和同一统计门控，最终只形成一个矩阵和一个全局结论。

### 5.2 固定实验矩阵

| 项目 | 锁定设置 |
|---|---|
| 模型 | `unet_baseline`、`unet_noskip` |
| 模型种子 | 42、123、3407 |
| 数据 | BraTS2023 test 全部 250 名患者 |
| 节点 | `down1, down2, down3, down4, up1, up2, up3, up4` |
| 过程区间 | 全部 7 个相邻检查点区间 |
| 真实类别 | 1、2、3 |
| 目标过程 | 输出锚定的持续纠正群 $P_k$ |
| 破坏位置 | 共同根检查点 `down1` 的模块输出 |
| 破坏方式 | 输入等效位移 8 像素的二维空间循环平移 |
| 恢复位置 | `down2` 至 `up4` 七个接收检查点逐一恢复 |
| 主结局 | 原模型最终预测在 $P_k$ 上的真实类别保持率 |
| 外部热图 | 不使用 |

输入等效位移由输入和根激活空间尺寸确定：

$$
\delta_r=
\left(
\max\{1,\operatorname{round}(8H_r/H)\},
\max\{1,\operatorname{round}(8W_r/W)\}
\right).
$$

该位移在协议锁定后不得根据输出影响大小修改。

### 5.3 每名患者只选择一个全路径切片

对每个模型、种子和患者，在全部切片上计算 $P_1,\ldots,P_{K-1}$，选择联合集合像素数最大的切片：

$$
s^*=\arg\max_s\left|\bigcup_kP_k^{(s)}\right|.
$$

并列时选择切片编号最小者。该规则避免为每个区间分别挑选最显著切片，也允许同一张图展示一个像素群贯穿全部节点的状态变化。

病例进入干预运行需要联合集合至少 32 个像素。单个区间进入矩阵统计需要在所选切片上至少 8 个 $P_k$ 像素。所有未满足条件的病例和区间显式记录，不替换患者、不改选第二张切片。

### 5.4 事件特异性区域、逐节点批量恢复

联合集合

$$
P_{\mathrm{all}}=\bigcup_kP_k
$$

只用于全路径切片选择和不重复的微平均。正式恢复保持区间身份：对恢复节点 $v_j$，分别用自适应最大池化把每个 $P_k$ 映射到该节点空间，得到 $\Omega_{k,j}$。同一模型—种子—患者执行以下条件：

1. `clean`：原始前向，并捕获所有节点干净激活。
2. `corrupt`：对 `down1` 输出执行固定空间平移。
3. `root_full_restore`：在破坏后恢复全部 `down1` 干净激活，只用于实现审计。
4. `restore_target_k_j`：保持根破坏，只在节点 $j$ 的 $\Omega_{k,j}$ 位置恢复干净激活，覆盖全部可评估的 $k$ 和 $j=2,\ldots,K$。
5. `restore_control_k_j`：保持根破坏，只恢复该 $(k,j)$ 单元的等量匹配对照位置。

为兼顾区间特异性与运行效率，同一恢复节点的最多 7 个区间条件沿 batch 维一次前向计算：每个 batch 样本使用同一图像和同一根破坏，但使用不同的 $\Omega_{k,j}$。因此每名可评估患者最多仍为 17 次前向调用，即 clean、corrupt、root full restore、7 次节点目标恢复 batch 和 7 次节点对照恢复 batch；每个矩阵单元保持独立目标区域，不由其他区间的恢复区域影响。显存不足时只允许把 batch 分块，不允许改变条件或统计对象。

### 5.5 匹配对照

对每个过程区间，先定义输出锚定的稳定正确对照池：

$$
\begin{aligned}
C_k=\{p\mid
&Y(p)\in\mathcal C_T,\ Z_\ell(p)=Y(p)\ \forall\ell\ge k,\\
&R_\ell(p)=1\ \forall\ell\ge k,\ \widehat Y(p)=Y(p)
\}\setminus P_{\mathrm{all}}.
\end{aligned}
$$

对每个 $(k,j)$ 单元，把 $C_k$ 映射到节点 $j$ 作为对照候选。最终对照位置必须位于 $\Omega_{k,j}$ 外，并按以下固定变量进行不放回匹配：

1. 主导真实肿瘤类别；
2. 肿瘤边界距离分层，边界为 `[1.5, 3.5, 7.5]`；
3. 干净节点激活范数四分位分层；
4. 与目标特征位置数量相同。

同一 batch 中不同 $k$ 的对照匹配相互独立，因为它们属于不同前向样本。若某个单元完整等量匹配失败，该单元的目标恢复结果仍保留，区域特异性记为不可评估。不得用降低匹配条件的方式补齐。

### 5.6 原模型输出结局

对条件 $c$、过程区间 $k$ 和恢复节点 $j$：

$$
Q_{k,j}^{(c)}=
\frac{1}{|P_k|}
\sum_{p\in P_k}
\mathbf 1\!\left[\widehat Y^{(c)}(p)=Y(p)\right].
$$

主要效应为：

$$
\mathrm{TE}_k=Q_k^{(\mathrm{clean})}-Q_k^{(\mathrm{corrupt})},
$$

$$
\mathrm{IE}^{\mathrm{target}}_{k,j}
=Q_{k,j}^{(\mathrm{restore\ target})}-Q_k^{(\mathrm{corrupt})},
$$

$$
\mathrm{IE}^{\mathrm{specific}}_{k,j}
=Q_{k,j}^{(\mathrm{restore\ target})}
-Q_{k,j}^{(\mathrm{restore\ control})}.
$$

矩阵

$$
\mathbf I=\left[\mathrm{IE}^{\mathrm{specific}}_{k,j}\right]_{
k=1,\ldots,K-1;\ j=2,\ldots,K}
$$

是唯一新增的全网络干预结果对象。主对齐项为每个过程区间与其接收检查点的单元 $\mathrm{IE}^{\mathrm{specific}}_{k,k+1}$；其余单元用于显示恢复作用沿网络深度的分布，不逐单元提出独立因果结论。

### 5.7 为什么只干预验证持续纠正

PPTT 仍完整报告稳定、纠正、破坏和错误重编码事件。V7 的干预端点只选择输出锚定的持续纠正，是因为从同一输入恢复干净激活能够定义“恢复最终正确输出”的明确方向。对持续破坏像素，干净轨迹本身以错误输出结束，直接恢复干净激活不能构成充分性检验；若强行使用其他病例或人工理想激活作为供体，会引入第二套干预算子和额外因果假设。

因此，本轮用一个定义清楚的事件族验证 PPTT 定位与反事实输出的一致性，而不把不适用的算子复制到所有事件。破坏和错误重编码继续由转移张量精确计数，并作为全网络过程结果，不升级为干预因果结论。

### 5.8 全路径汇总不使用拟合权重

对患者 $u$ 的可评估区间集合 $\mathcal K_e(u)$，先定义患者内按过程像素数加权的微平均：

$$
A_{\mathrm{micro}}(u)=
\frac{
\sum_{k\in\mathcal K_e(u)}|P_{u,k}|\,
\mathrm{IE}^{\mathrm{specific}}_{u,k,k+1}
}{
\sum_{k\in\mathcal K_e(u)}|P_{u,k}|
},
$$

以及每个区间等权的宏平均：

$$
A_{\mathrm{macro}}(u)=
\frac{1}{|\mathcal K_e(u)|}
\sum_{k\in\mathcal K_e(u)}
\mathrm{IE}^{\mathrm{specific}}_{u,k,k+1}.
$$

种子级估计再对患者等权平均，置信区间通过整名患者重采样获得。微平均回答“单名患者全部可定位持续纠正像素中有多少可被相应节点的目标恢复特异性挽回”，宏平均防止该患者像素量最大的单一区间支配结论；患者等权又防止大肿瘤病例支配群体结论。两者都是定义性汇总，不训练权重、不拟合 Dice、不合并成单个分数。

## 6. 统计规则与成功标准

### 6.1 患者级统计

所有推断以患者为独立单位。每个模型种子分别报告：

- 患者配对均值差；
- 10,000 次患者级自助法 95% 置信区间；
- Wilcoxon 符号秩检验；
- 配对 Cohen's $d$；
- 秩二列相关。

主要检验族只包含每个模型—种子的 $A_{\mathrm{micro}}$ 和 $A_{\mathrm{macro}}$，共 12 项，使用 Holm 校正。矩阵单元只报告效应量、置信区间和样本量，不对 49 个单元逐一进行显著性筛选。

### 6.2 可评估覆盖

一个模型—种子—过程区间满足以下条件时记为可评估：

1. 至少 30 名患者具有该区间的有效 $P_k$；
2. 每名进入患者至少有 8 个目标像素；
3. 汇总目标像素数至少为 2,048；
4. 至少 30 名患者在接收检查点完成等量目标—对照匹配；
5. 目标恢复、对照恢复、钩子和原模型输出均通过有限值检查。

全网络覆盖成立需要每个模型至少两个种子有不低于 5/7 的可评估区间。该三分之二覆盖规则在正式干预前锁定，用于排除“只有一个区间有效却声称全网络”的情况。

### 6.3 全网络干预一致性门控

`PASS_NETWORK_WIDE` 必须同时满足：

1. 数学性质、协议哈希、数据清单、权重哈希和观察器准入均通过。
2. 两个模型均满足上述三分之二可评估覆盖。
3. 每个模型至少两个种子的 $A_{\mathrm{micro}}$ 均值和 95% 置信区间下界大于 0，Holm 校正后 `p<0.05`。
4. 每个模型至少两个种子的 $A_{\mathrm{macro}}$ 均值和 95% 置信区间下界大于 0，Holm 校正后 `p<0.05`。
5. 在通过的种子中，至少 4/7 个接收检查点单元具有正的患者均值，避免全局结果由单一区间产生。
6. 空间平移逐通道多重集合保持、全根恢复、钩子清除、匹配等量和有限值审计全部通过。

若 1、2 通过但 3 至 5 只在一个模型成立，状态为 `PARTIAL_NETWORK_ALIGNMENT`，允许表述为该模型上的全路径干预一致性，不允许扩展到架构对照。若覆盖不足，状态为 `INSUFFICIENT_NETWORK_COVERAGE`。若全局效应不通过，状态为 `OBSERVATIONAL_PROCESS_ONLY`。任何失败状态均不得通过改变位移、像素阈值、节点或患者集合重新计算。

## 7. 对抗式审查与设计回应

| 潜在质疑 | 本方案的处理 | 仍保留的边界 |
|---|---|---|
| 仍然只是在看一个跳接 | V7 同时覆盖两种模型、8 个节点、7 个过程区间；破坏作用于共同根节点，恢复作用于全部接收检查点 | V6 的跳接来源结论仍只是补充实例 |
| 相邻节点不是物理边 | 明确称为检查点区间；结构来源另由干预定义 | 不声称遍历所有计算图边 |
| 观察器可能制造状态 | 保留重启、置信间隔、参数随机化、患者置换、空间平移和指标重构门控 | 节点状态是受控可读状态，不是唯一真实语义 |
| 终点 Dice 已经包含这些信息 | 给出运输多面体自由度与构造性反例，证明端点不能识别联合转移 | 不声称 PPTT 替代所有性能指标 |
| 相邻张量足以描述过程 | 三节点二分类反例证明持续性需要像素身份 | 轨迹存储是持续性结论的必要组成 |
| 干预区域由结果选择，存在循环论证 | 区域由干净 PPTT 过程定义；每个区间使用独立区域，结局使用原模型最终输出；规则在 test 干预前锁定 | 这是构念与干预行为一致性，不是随机对照临床因果 |
| 恢复干净激活是离流形操作 | 恢复向量来自同一输入的真实干净前向；空间平移保持值多重集合；使用等量匹配区域 | 混合上下文仍可能偏离自然联合分布，不能声称唯一机制 |
| 49 个矩阵单元造成多重检验和挑结果 | 只对预声明的宏、微全局端点做主要检验；矩阵单元全部显示且不逐项筛选 | 局部单元仅用于定位，不单独升级为普适规律 |
| 结果仍可能由一个区间支配 | 同时使用宏平均、微平均、三分之二覆盖和四区间正向规则 | 不要求所有区间方向完全相同 |
| 方法模块仍然过多 | 方法只保留状态、转移、谱系；V7 是一个统一验证模块 | V1/V2/V5 属于准入和外部验证，不写成方法部件 |
| 是否证明真实因果推理机制 | 干预量是已定义内部变量和算子下的确定性反事实效应 | 不证明唯一、完整或人类语义等价的真实机制 |

## 8. 主文图件收敛

主文保持三张图，不增加普通曲线或热图数量。

### 图 1：PPTT 方法总图

冻结分割网络与有序检查点 → 输出约束节点状态 → 同一像素轨迹 → 条件转移张量 → 持续性谱系与过程输出。图中不出现 CAM、V6 跳接、实验统计和大段文字。

### 图 2：终点相近，过程不同

1. 患者—种子散点：横轴为两个模型最终宏平均肿瘤 Dice 绝对差，纵轴为 $D_{\mathrm{proc}}$。
2. 固定病例、固定切片、固定肿瘤像素的全节点命运图：列为全部 8 个节点，行保持同一像素身份；有跳接与无跳接上下对齐。
3. 仅保留三个跨种子总体效应及 95% 置信区间：中部破坏/重编码、后期持续净恢复、终点差异。

代表病例不选择最大效果病例。候选患者需在两个模型均满足肿瘤像素不少于 500、至少 4 个区间存在过程事件；对三种子平均过程向量取队列中位数，选择与中位数 L1 距离最小的患者，并列时取患者编号最小者。切片按联合过程像素最多规则确定。

### 图 3：全网络过程—干预一致性

1. 左侧为两个模型并排的 $7\times7$ $\mathbf I$ 矩阵，行是过程区间，列是恢复节点，色值为目标减匹配对照的最终输出恢复率。
2. 对角接收节点用细边框标记；不可评估单元显示为灰色，不插值补值。
3. 右侧只放一个按预声明中位规则选择的同病例同切片 `clean / corrupt / restore target / restore control` 放大图，并用轮廓标出 PPTT 过程像素，不叠加 CAM。
4. 下方报告 $A_{\mathrm{micro}}$、$A_{\mathrm{macro}}$ 和可评估区间覆盖，均带 95% 置信区间。

V6 现有路径特异性图转入补充材料。所有正式图提供中英文 PNG、PDF 和数据来源 JSON，PNG 为 600 dpi。

## 9. 正式输出契约

结果根目录固定为：

`results/v7_network_process_intervention_alignment/`

| 文件 | 一行或一个对象的含义 |
|---|---|
| `v7_protocol_lock.json` | 配置哈希、代码提交、数据清单、模型权重、节点、患者、阈值和运行授权 |
| `patient_slice_audit.parquet` | 模型—种子—患者的切片选择、联合像素数、各 $P_k$ 数量和资格状态 |
| `patient_node_audit.parquet` | 每个患者—区间—恢复节点的目标位置、匹配位置、空间位移、钩子和全根恢复审计 |
| `patient_intervention_cells.parquet` | 患者—过程区间—恢复节点的 clean、corrupt、target、control 结局和三个效应 |
| `network_alignment_summary.parquet` | 模型—种子—过程区间—恢复节点的样本量、均值、置信区间和效应量 |
| `global_alignment_statistics.parquet` | 模型—种子的宏平均、微平均、Holm 校正结果和覆盖 |
| `v7_conclusion_gate.json` | 每项门控、最终状态、允许主张和禁止主张 |
| `v7_status.json` | 6 个正式作业的完成数、失败数、患者数和文件校验 |

`patient_intervention_cells.parquet` 固定字段为：

```text
model, model_seed, patient_id, slice_id, transition_index, transition,
receiving_node, restore_node, target_pixel_count, union_pixel_count,
target_feature_count, control_feature_count, q_clean, q_corrupt,
q_restore_target, q_restore_control, total_effect, target_recovery,
control_recovery, specific_recovery, eligible, ineligible_reason
```

任何定义性缺失使用空值并同时写入 `ineligible_reason`；不得写入无穷值，不得用 0 代替不可评估结果。

## 10. 文件结构锁定

### 新建

- `configs/experiments/v7_network_process_intervention_alignment.yaml`：全部预注册设置。
- `src/pptt/transitions/theory.py`：相邻事件完备表示、运输多面体维数和反例构造。
- `src/pptt/lineage/cohorts.py`：输出锚定持续纠正群、联合群和切片选择。
- `src/pptt/interventions/network_alignment.py`：空间破坏、区间特异性区域映射、匹配和恢复算子。
- `src/pptt/interventions/network_runtime.py`：一次 clean/corrupt、逐节点 target/control 恢复的运行时。
- `src/pptt/statistics/network_alignment.py`：矩阵、宏微统计、覆盖和结论门控。
- `src/pptt/visualization/pixel_fate.py`：同一像素全节点命运图。
- `src/pptt/visualization/network_alignment.py`：全网络干预矩阵。
- `scripts/lock_v7_network_alignment_protocol.py`：协议锁定。
- `scripts/run_v7_network_alignment.py`：正式和 smoke 运行器。
- `scripts/summarize_v7_network_alignment.py`：表格、统计和门控。
- `scripts/make_network_pixel_fate_figure.py`：新版图 2。
- `scripts/make_network_alignment_figure.py`：新版图 3。
- `tests/unit/test_transition_theory.py`
- `tests/unit/test_lineage_cohorts.py`
- `tests/unit/test_network_alignment_protocol.py`
- `tests/unit/test_network_alignment_operator.py`
- `tests/unit/test_network_alignment_statistics.py`
- `tests/integration/test_network_alignment_smoke.py`

### 修改

- `scripts/run_v0_math.py`：加入新的数学性质 JSON 输出。
- `src/pptt/interventions/__init__.py`
- `src/pptt/lineage/__init__.py`
- `src/pptt/statistics/__init__.py`
- `src/pptt/transitions/__init__.py`
- `tests/visual/test_figure_contracts.py`
- `manifests/run_commands.yaml`
- `manifests/result_inventory.json`
- `docs/reproducibility_cn.md`
- `docs/final_experiment_audit_cn.md`

现有 `configs/experiments/v6_pixel_transition_causal_tracing.yaml`、V6 脚本、V6 锁和 V6 正式结果只读保留。

## 11. 分任务实施计划

### Task 1: 数学性质与反例

**Files:**
- Create: `src/pptt/transitions/theory.py`
- Create: `tests/unit/test_transition_theory.py`
- Modify: `scripts/run_v0_math.py`
- Modify: `src/pptt/transitions/__init__.py`

- [x] **Step 1: 先写数学性质失败测试**

```python
def test_four_class_endpoint_hidden_dimension_is_36():
    assert endpoint_hidden_dimension(4, truth_class_count=4) == 36


def test_equal_pairwise_transitions_can_hide_different_persistence():
    process_a, process_b = binary_three_node_counterexample()
    assert adjacent_pair_counts(process_a) == adjacent_pair_counts(process_b)
    assert persistent_correction_count(process_a, truth_class=1) == 1
    assert persistent_correction_count(process_b, truth_class=1) == 0


def test_transition_tensor_recovers_every_additive_event_functional():
    direct = float(weights[truth, before, after].sum())
    tensor_value = additive_event_from_tensor(tensor, weights)
    assert direct == tensor_value
```

- [x] **Step 2: 运行并确认测试因缺少新接口失败**

Run: `.venv/bin/python -m pytest tests/unit/test_transition_theory.py -v`

Expected: FAIL，错误指向 `pptt.transitions.theory` 或新函数尚不存在。

- [x] **Step 3: 实现固定接口**

```python
def endpoint_hidden_dimension(
    num_classes: int,
    *,
    truth_class_count: int | None = None,
) -> int:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    groups = num_classes if truth_class_count is None else truth_class_count
    if groups < 1:
        raise ValueError("truth_class_count must be positive")
    return int(groups * (num_classes - 1) ** 2)

def additive_event_from_tensor(
    tensor: np.ndarray,
    weights: np.ndarray,
) -> float:
    counts = np.asarray(tensor)
    coefficients = np.asarray(weights)
    if counts.shape != coefficients.shape or counts.ndim != 3:
        raise ValueError("tensor and weights must be aligned CxCxC arrays")
    if not np.isfinite(coefficients).all():
        raise ValueError("weights must be finite")
    return float(np.sum(counts * coefficients))

def binary_three_node_counterexample() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.uint8),
        np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=np.uint8),
    )

def adjacent_pair_counts(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    paths = np.asarray(trajectories)
    if paths.shape != (4, 3):
        raise ValueError("counterexample trajectories must have shape 4x3")
    return tuple(
        np.bincount(paths[:, k] * 2 + paths[:, k + 1], minlength=4).reshape(2, 2)
        for k in range(2)
    )

def persistent_correction_count(
    trajectories: np.ndarray,
    *,
    truth_class: int,
) -> int:
    paths = np.asarray(trajectories)
    return int(
        np.sum(
            (paths[:, 0] != truth_class)
            & np.all(paths[:, 1:] == truth_class, axis=1)
        )
    )
```

实现必须验证整数标签、类别范围、形状和有限值；四分类维数返回 36，反例轨迹严格使用本文第 4.3 节的两个多重集合。

- [x] **Step 4: 把数学结果写入 V0 正式 JSON**

`scripts/run_v0_math.py` 新增 `additive_functional_exact`、`endpoint_hidden_dimension`、`pairwise_not_lineage_sufficient` 三项；任一失败使 V0 返回非零退出码。

- [x] **Step 5: 运行数学与既有转移测试**

Run: `.venv/bin/python -m pytest tests/unit/test_transition_theory.py tests/unit/test_transition_nonidentifiability.py tests/property/test_transition_properties.py -v`

Expected: PASS。

### Task 2: 输出锚定持续纠正群与全路径切片

**Files:**
- Create: `src/pptt/lineage/cohorts.py`
- Create: `tests/unit/test_lineage_cohorts.py`
- Modify: `src/pptt/lineage/__init__.py`

- [x] **Step 1: 写入群定义、后缀可靠性和不相交测试**

```python
def test_output_anchored_cohorts_require_suffix_reliability_and_native_output():
    states = np.asarray(
        [
            [[[0, 0, 0, 0]]],
            [[[1, 0, 0, 1]]],
            [[[1, 1, 1, 1]]],
            [[[1, 1, 1, 1]]],
        ],
        dtype=np.uint8,
    )
    truth = np.ones((1, 1, 4), dtype=np.uint8)
    reliable = np.ones((3, 1, 1, 4), dtype=bool)
    reliable[2, 0, 0, 2] = False
    final_model_state = np.asarray([[[1, 1, 1, 0]]], dtype=np.uint8)
    cohorts = output_anchored_persistent_correction_cohorts(
        states,
        truth,
        reliable,
        final_model_state,
        truth_classes=(1, 2, 3),
    )
    assert cohorts.shape[0] == states.shape[0] - 1
    assert cohorts[0, 0, 0, 0]
    assert cohorts[1, 0, 0, 1]
    assert not cohorts[:, 0, 0, 2].any()
    assert not cohorts[:, 0, 0, 3].any()


def test_persistent_correction_cohorts_are_pairwise_disjoint():
    states = np.asarray(
        [[[[0, 0]]], [[[1, 0]]], [[[1, 1]]]],
        dtype=np.uint8,
    )
    truth = np.ones((1, 1, 2), dtype=np.uint8)
    reliable = np.ones((2, 1, 1, 2), dtype=bool)
    final_model_state = truth.copy()
    cohorts = output_anchored_persistent_correction_cohorts(
        states,
        truth,
        reliable,
        final_model_state,
        truth_classes=(1,),
    )
    assert np.all(cohorts.sum(axis=0) <= 1)


def test_process_slice_uses_union_count_and_smallest_slice_tie_break():
    cohorts = np.zeros((2, 2, 1, 3), dtype=bool)
    cohorts[0, 0, 0, :2] = True
    cohorts[1, 1, 0, 1:] = True
    assert select_process_slice(cohorts).slice_index == 0
```

- [x] **Step 2: 运行并确认失败**

Run: `.venv/bin/python -m pytest tests/unit/test_lineage_cohorts.py -v`

Expected: FAIL，新模块尚不存在。

- [x] **Step 3: 实现固定接口**

```python
@dataclass(frozen=True)
class ProcessSliceSelection:
    slice_index: int
    union_pixel_count: int
    transition_pixel_counts: Sequence[int]


def output_anchored_persistent_correction_cohorts(
    states: np.ndarray,
    truth: np.ndarray,
    reliable: np.ndarray,
    final_model_state: np.ndarray,
    *,
    truth_classes: Sequence[int],
) -> np.ndarray:
    """Return a boolean array with shape (K-1, S, H, W)."""


def select_process_slice(cohorts: np.ndarray) -> ProcessSliceSelection:
    """Select the smallest-index slice maximizing the disjoint cohort union."""
```

`cohorts` 输出形状固定为 `(K-1, S, H, W)`，布尔类型；函数内部若发现同一像素进入两个群，直接抛出 `RuntimeError`。

- [x] **Step 4: 运行群定义测试和既有谱系测试**

Run: `.venv/bin/python -m pytest tests/unit/test_lineage_cohorts.py tests/unit/test_lineage_depths.py -v`

Expected: PASS。

### Task 3: V7 配置与协议锁

**Files:**
- Create: `configs/experiments/v7_network_process_intervention_alignment.yaml`
- Create: `scripts/lock_v7_network_alignment_protocol.py`
- Create: `tests/unit/test_network_alignment_protocol.py`
- Modify: `manifests/run_commands.yaml`

- [x] **Step 1: 写协议拒绝行为测试**

测试必须覆盖：配置哈希变化、节点顺序变化、正式患者截断、输出目录污染、正式锁缺失、V4 状态被改写、V6 路径被当作 V7 候选时全部拒绝。

- [x] **Step 2: 写入锁定配置**

```yaml
name: v7_network_process_intervention_alignment
model_matrix: manifests/model_matrix.yaml
observer_root: results/v1_observers
v3_root: results/v3_unet_pair
output_root: results/v7_network_process_intervention_alignment
models: [unet_baseline, unet_noskip]
model_seeds: [42, 123, 3407]
observer_seeds: [17, 29, 43]
nodes: [down1, down2, down3, down4, up1, up2, up3, up4]
primary_split: test
truth_classes: [1, 2, 3]
process_event: output_anchored_persistent_correction
eligibility:
  minimum_union_pixels_on_selected_slice: 32
  minimum_transition_pixels: 8
  minimum_target_feature_positions: 4
  minimum_control_feature_positions: 4
  minimum_patients_per_transition: 30
  minimum_aggregate_pixels_per_transition: 2048
intervention:
  root_node: down1
  operator: spatial_roll_module_output
  input_equivalent_shift_yx: [8, 8]
  restore_nodes: [down2, down3, down4, up1, up2, up3, up4]
  logit_tolerance: 1.0e-6
matching:
  same_dominant_truth_class: true
  boundary_bin_edges: [1.5, 3.5, 7.5]
  activation_norm_quantile_bins: 4
  without_replacement: true
statistics:
  bootstrap_iterations: 10000
  bootstrap_seed: 20260717
  alpha: 0.05
  multiplicity: holm_global_macro_micro
  minimum_supported_seeds: 2
  minimum_evaluable_transitions: 5
  minimum_positive_receiving_transitions: 4
protocol:
  lock_before_test_intervention: true
  thresholds_may_not_change_after_lock: true
  v4_must_remain_not_triggered: true
  v6_must_remain_candidate_specific: true
```

- [x] **Step 3: 生成协议锁内容**

锁文件必须登记配置 SHA-256、当前代码提交、源码树 SHA-256、`code_dirty`、数据清单 SHA-256、6 个模型权重 SHA-256、6 个观察器状态、250 名 test 患者有序清单、V4 `triggered=false` 和 V6 限定性状态。正式授权要求 `code_dirty=false`；任何一项缺失或源码树与提交不一致时不授权正式干预。

- [x] **Step 4: 运行协议测试**

Run: `.venv/bin/python -m pytest tests/unit/test_network_alignment_protocol.py -v`

Expected: PASS。

### Task 4: 全网络破坏与逐节点恢复算子

**Files:**
- Create: `src/pptt/interventions/network_alignment.py`
- Create: `src/pptt/interventions/network_runtime.py`
- Create: `tests/unit/test_network_alignment_operator.py`
- Modify: `src/pptt/interventions/__init__.py`

- [x] **Step 1: 写算子性质失败测试**

```python
TEST_NAMES = (
    "test_spatial_roll_preserves_each_channel_multiset_exactly",
    "test_full_root_restore_reproduces_clean_logits_within_tolerance",
    "test_target_restore_changes_only_registered_spatial_positions",
    "test_hooks_are_removed_after_success_and_exception",
    "test_each_transition_mask_is_mapped_without_interpolation",
    "test_control_positions_are_disjoint_equal_count_and_stratum_matched",
)
```

- [x] **Step 2: 运行并确认失败**

Run: `.venv/bin/python -m pytest tests/unit/test_network_alignment_operator.py -v`

Expected: FAIL，新运行时尚不存在。

- [x] **Step 3: 实现数据结构与运行接口**

```python
@dataclass(frozen=True)
class NodeRestoreMasks:
    target: torch.Tensor
    control: torch.Tensor | None
    target_count: int
    control_count: int


@dataclass(frozen=True)
class NetworkAlignmentForwardSet:
    clean_logits: np.ndarray
    corrupt_logits: np.ndarray
    root_restored_logits: np.ndarray
    target_restored_logits: dict[str, np.ndarray]
    control_restored_logits: dict[str, np.ndarray | None]
    audits: dict[str, object]


def run_network_alignment_forwards(
    adapter: ModelAdapter,
    image: torch.Tensor,
    *,
    root_node: str,
    restore_nodes: Sequence[str],
    masks: Mapping[str, Mapping[int, NodeRestoreMasks]],
    input_equivalent_shift_yx: tuple[int, int],
) -> NetworkAlignmentForwardSet:
    """Batch event-specific masks per node and return every registered output."""
```

运行时必须在一次干净前向中捕获全部节点激活；根破坏和节点恢复使用组合输出钩子；同一节点的区间特异性掩膜沿 batch 维执行；每次前向结束后检查钩子数量恢复。正式主结局只从 `argmax(final logits)` 计算，不使用干预后的观察器状态。

- [x] **Step 4: 运行算子与 V6 回归测试**

Run: `.venv/bin/python -m pytest tests/unit/test_network_alignment_operator.py tests/unit/test_causal_tracing.py tests/integration/test_causal_trace_smoke.py -v`

Expected: PASS，V6 行为不变。

### Task 5: V7 病例运行器与断点续跑

**Files:**
- Create: `scripts/run_v7_network_alignment.py`
- Create: `tests/integration/test_network_alignment_smoke.py`

- [x] **Step 1: 写 synthetic adapter 集成测试**

测试模型必须包含 8 个有序节点、已知空间移位敏感区域和确定性最终输出。测试验证：7 个过程区间均出现在行记录中、7 个恢复节点均出现在列记录中、最终最多 49 个单元、不可评估区间显式记录、重复 `--resume` 不产生重复行、第二个同身份运行器被 `.run.lock` 拒绝。

- [x] **Step 2: 实现 CLI**

```text
--workspace-root
--config
--protocol-lock
--asset-root
--model
--model-seed
--device
--resume
--execution-mode {formal,smoke}
--max-patients
```

正式模式拒绝 `--max-patients`，且 `--model`、`--model-seed`、输出目录必须与锁一致。smoke 模式必须写入 `results/v7_smoke/`，不得写入正式目录。

- [x] **Step 3: 实现患者级原子输出**

每名患者先写临时 JSON，再使用 `os.replace` 原子移动为正式文件。患者 JSON 包含切片选择、各 $P_k$ 数量、全部条件最终状态摘要、49 个矩阵单元和所有审计。恢复运行只接受与作业清单完全一致的既有患者文件。

每个模型—种子作业在输出目录持有非阻塞进程锁 `.run.lock`。第二个同身份进程必须立即退出并报告 `JOB_ALREADY_RUNNING`；完整状态为 `COMPLETE` 的作业在 `--resume` 下只执行身份核验，不重复前向。

- [x] **Step 4: 运行集成 smoke**

Run: `.venv/bin/python -m pytest tests/integration/test_network_alignment_smoke.py -v`

Expected: PASS。

- [x] **Step 5: 在服务器运行一个非正式真实病例 smoke**

Run:

```bash
.venv/bin/python scripts/run_v7_network_alignment.py \
  --workspace-root . \
  --config configs/experiments/v7_network_process_intervention_alignment.yaml \
  --protocol-lock results/v7_network_process_intervention_alignment/v7_protocol_lock.json \
  --asset-root /root/autodl-tmp/A_scheme_workspace/brats2023_data \
  --model unet_baseline --model-seed 42 \
  --execution-mode smoke --max-patients 1 --device cuda
```

Expected: `SMOKE_COMPLETE`，正式结果目录未新增患者结果。

### Task 6: 矩阵统计与结论门控

**Files:**
- Create: `src/pptt/statistics/network_alignment.py`
- Create: `scripts/summarize_v7_network_alignment.py`
- Create: `tests/unit/test_network_alignment_statistics.py`
- Modify: `src/pptt/statistics/__init__.py`

- [x] **Step 1: 写统计和失败门控测试**

```python
TEST_NAMES = (
    "test_macro_and_micro_use_only_receiving_node_cells",
    "test_micro_average_does_not_double_count_disjoint_cohorts",
    "test_gate_rejects_only_one_evaluable_transition",
    "test_gate_rejects_positive_micro_but_nonpositive_macro",
    "test_gate_rejects_effect_concentrated_in_fewer_than_four_transitions",
    "test_gate_passes_two_models_two_seeds_and_five_transitions",
    "test_undefined_specificity_remains_null_not_zero",
)
```

- [x] **Step 2: 运行并确认失败**

Run: `.venv/bin/python -m pytest tests/unit/test_network_alignment_statistics.py -v`

Expected: FAIL，新统计模块尚不存在。

- [x] **Step 3: 实现固定接口**

```python
def summarize_alignment_cells(
    rows: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Return patient-bootstrap cell estimates without selecting matrix cells."""


def compute_global_alignment_statistics(
    rows: pd.DataFrame,
    *,
    receiving_node_by_transition: Mapping[str, str],
) -> pd.DataFrame:
    """Return macro and micro receiving-node statistics by model and seed."""


def evaluate_network_alignment_gate(
    global_rows: pd.DataFrame,
    cell_rows: pd.DataFrame,
    audit: Mapping[str, object],
    *,
    minimum_supported_seeds: int,
    minimum_evaluable_transitions: int,
    minimum_positive_receiving_transitions: int,
    alpha: float,
) -> dict[str, object]:
    """Return one registered status and the complete gate audit."""
```

患者 bootstrap 必须整名患者重采样，并在每次重采样内同时重算所有区间，保留患者内相关性。

- [x] **Step 4: 运行统计测试和完整测试集**

Run: `.venv/bin/python -m pytest tests/unit/test_network_alignment_statistics.py -v`

Expected: PASS。

Run: `.venv/bin/python -m pytest -v`

Expected: 所有核心测试 PASS；仅 Linux 不适用的 Windows junction 用例允许条件跳过。

### Task 7: 两张核心结果图

**Files:**
- Create: `src/pptt/visualization/pixel_fate.py`
- Create: `src/pptt/visualization/network_alignment.py`
- Create: `scripts/make_network_pixel_fate_figure.py`
- Create: `scripts/make_network_alignment_figure.py`
- Modify: `tests/visual/test_figure_contracts.py`

- [x] **Step 1: 写图件契约测试**

测试必须检查：中英文 PNG/PDF/JSON 同时存在、PNG 不低于 600 dpi、图 2 包含全部 8 个节点、图 3 矩阵为 7 行 7 列、无双纵轴、不可评估单元不填 0、同一病例和切片 ID 在所有视觉面板一致、数据 JSON 与图中数值一致。

- [x] **Step 2: 实现代表病例中位规则**

选择函数固定返回过程向量最接近队列中位数的患者；测试需验证极端最大效应病例不会被选择，编号并列规则稳定。

- [x] **Step 3: 生成新版图 2 和图 3**

Run:

```bash
.venv/bin/python scripts/make_network_pixel_fate_figure.py --workspace-root . --language en --dpi 600
.venv/bin/python scripts/make_network_pixel_fate_figure.py --workspace-root . --language zh --dpi 600
.venv/bin/python scripts/make_network_alignment_figure.py --workspace-root . --language en --dpi 600
.venv/bin/python scripts/make_network_alignment_figure.py --workspace-root . --language zh --dpi 600
```

Expected: 输出到 `results/paper_outputs/{en,zh}/`，每张图均有 PNG、PDF、JSON。

- [x] **Step 4: 执行视觉测试和像素非空检查**

Run: `.venv/bin/python -m pytest tests/visual/test_figure_contracts.py -v`

Expected: PASS。

### Task 8: 正式服务器运行

**Files:**
- Modify after completion: `docs/final_experiment_audit_cn.md`
- Modify after completion: `docs/reproducibility_cn.md`
- Modify after completion: `manifests/result_inventory.json`

- [x] **Step 1: 锁定正式协议**

Run:

```bash
.venv/bin/python scripts/lock_v7_network_alignment_protocol.py \
  --workspace-root . \
  --config configs/experiments/v7_network_process_intervention_alignment.yaml \
  --asset-root /root/autodl-tmp/A_scheme_workspace/brats2023_data
```

Expected: `v7_protocol_lock.json` 状态为 `LOCKED_BEFORE_TEST_INTERVENTION`，`formal_intervention_authorized=true`。

- [x] **Step 2: 并行启动 6 个独立作业**

该运行预计超过 2 小时，实施时把以下完整命令交由用户在服务器控制台启动；不得在未确认前擅自长跑。

```bash
mkdir -p logs/v7_formal
for model in unet_baseline unet_noskip; do
  for seed in 42 123 3407; do
    nohup .venv/bin/python scripts/run_v7_network_alignment.py \
      --workspace-root . \
      --config configs/experiments/v7_network_process_intervention_alignment.yaml \
      --protocol-lock results/v7_network_process_intervention_alignment/v7_protocol_lock.json \
      --asset-root /root/autodl-tmp/A_scheme_workspace/brats2023_data \
      --model "$model" --model-seed "$seed" \
      --execution-mode formal --resume --device cuda \
      > "logs/v7_formal/${model}_seed_${seed}.log" 2>&1 &
    echo "$! $model $seed" >> logs/v7_formal/pids.txt
  done
done
```

六作业并行前必须通过显存 smoke；若 24 GB 无法稳定容纳六进程，保持协议不变，仅将调度器并发数降为 2 或 3。并发数不是方法参数，不改变正式结果。运行器的 `.run.lock` 必须阻止重复启动同一模型—种子作业。

- [x] **Step 3: 汇总与门控**

Run:

```bash
.venv/bin/python scripts/summarize_v7_network_alignment.py \
  --workspace-root . \
  --config configs/experiments/v7_network_process_intervention_alignment.yaml \
  --protocol-lock results/v7_network_process_intervention_alignment/v7_protocol_lock.json
```

Expected: 6 个作业各 250 名患者；状态文件明确给出 `PASS_NETWORK_WIDE`、`PARTIAL_NETWORK_ALIGNMENT`、`INSUFFICIENT_NETWORK_COVERAGE` 或 `OBSERVATIONAL_PROCESS_ONLY` 之一。

- [x] **Step 4: 生成图件、清单和审计**

Run:

```bash
.venv/bin/python scripts/make_network_pixel_fate_figure.py --workspace-root . --language en --dpi 600
.venv/bin/python scripts/make_network_pixel_fate_figure.py --workspace-root . --language zh --dpi 600
.venv/bin/python scripts/make_network_alignment_figure.py --workspace-root . --language en --dpi 600
.venv/bin/python scripts/make_network_alignment_figure.py --workspace-root . --language zh --dpi 600
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
  .venv/bin/python scripts/build_result_inventory.py
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
  .venv/bin/python scripts/build_result_inventory.py --verify
```

Expected: 清单内容寻址校验 PASS，正式图件状态 PASS。

### Task 9: 最终主张审计

**Files:**
- Modify: `docs/final_experiment_audit_cn.md`
- Modify: `docs/reproducibility_cn.md`

- [x] **Step 1: 按实际门控状态写结论**

只有 `PASS_NETWORK_WIDE` 才允许写：

> PPTT 在两种 U-Net 变体、三个模型种子和多数预声明过程区间上，将输出锚定的持续纠正轨迹与原模型内部干预后的最终像素行为对应起来，获得全路径干预一致性支持。

任何状态都禁止写：

> PPTT 完全证明了网络真实且唯一的因果推理机制。

- [x] **Step 2: 保留失败与不可评估信息**

审计必须列出每个模型—种子的可评估区间、患者数、像素数、匹配失败数、根恢复误差和全局门控。V4 继续显示 `NOT_TRIGGERED`，V6 继续显示候选特异性，不允许用 V7 覆盖两者。

- [x] **Step 3: 最终回归检查**

Run: `.venv/bin/python -m pytest -v`

Expected: 全部核心测试 PASS。

Run: `git diff --check`

Expected: 无空白错误。

## 12. 执行停止条件

出现以下任一情况立即停止正式推进并只读诊断：

1. 协议锁与活动配置哈希不一致；
2. 正式锁定时 Git 工作树不干净或源码树哈希不一致；
3. V4 状态不再是 `NOT_TRIGGERED`；
4. V6 锁或结果被改写；
5. clean 前向无法复现 V3 的原模型最终状态；
6. 根全恢复最大 logit 误差超过 `1e-6`；
7. 空间平移未逐通道保持激活值多重集合；
8. 正式患者集合不是每作业 250 人；
9. 不可评估结果被写成 0 或被替换患者；
10. 统计汇总对矩阵单元进行结果导向筛选；
11. 图件使用不同病例或不同切片拼成连续轨迹。

## 13. 最终方法价值与边界

若 V7 通过，PPTT 的最强方法价值不是“解释某一条跳接为什么有效”，而是提供一种可复用的内部过程分析能力：

1. 在统一像素决策空间中定位网络何时纠正、破坏或重编码某类组织；
2. 精确说明终点指标变化由哪些像素状态转移组成；
3. 区分瞬时变化与持续进入最终输出的变化；
4. 用全网络统一干预检验过程定位是否能预测原模型反事实输出；
5. 在终点性能相近时揭示形成路径的显著差异，为网络结构诊断、模块比较和失效审查提供依据。

方法的理论结论是过程表示的精确性、信息不可约性和干预算子的操作性质；经验结论是指定模型、数据、检查点、观察器与干预集合下的过程—干预一致性。二者共同支持“过程性可解释分析方法”，但不升级为网络所有内部变量的完整因果证明。

## 14. 完成判据

只有以下事项全部完成，本轮补强才算收口：

- [x] 新数学性质均有构造测试并进入 V0 门控；
- [x] $P_k$ 输出锚定、后缀可靠且两两不相交；
- [x] V7 配置和患者清单在 test 干预前锁定；
- [x] 两种模型、三个种子、全部 250 名患者均完成或明确失败；
- [x] 全部 7 个过程区间进入覆盖审计；
- [x] 原模型最终输出而非观察器输出作为干预主结局；
- [x] 宏平均、微平均、覆盖和四区间规则共同执行；
- [x] 图 2 使用同一像素贯穿全部节点；
- [x] 图 3 展示完整过程—节点矩阵；
- [x] V4、V6 历史状态未被改写；
- [x] 主张严格服从 `v7_conclusion_gate.json`；
- [x] 完整测试、图件契约和结果清单校验全部通过。

## 15. 本文档自查结果

| 自查问题 | 对应设计 | 结果 |
|---|---|---|
| 是否仍由单一 `down2->up2` 路径支撑全方法 | V7 覆盖 8 个节点、7 个过程区间和 7 个恢复节点；V6 降为补充实例 | 通过 |
| 是否把检查点区间误写成物理边 | 第 2.2 节区分拓扑检查点和计算图边 | 通过 |
| 是否仍在堆 CAM、特征响应或加权指标 | 方法只保留状态、转移、谱系；CAM 不进入主方法，宏微统计不拟合权重 | 通过 |
| 是否有可严格核验的数学内容 | 第 4 节给出相邻事件完备表示、端点自由度、持续性反例、不相交性和精确重构 | 通过 |
| 是否用观察器输出自证观察器 | 干预目标由 PPTT 定位，主结局读取原模型最终输出 | 通过 |
| 是否因联合掩膜混淆不同区间 | 每个 $(k,j)$ 使用独立区域，沿 batch 维加速，不混合区间条件 | 通过 |
| 是否通过大量单元检验挑选结果 | 主要检验只包含预声明宏、微端点，49 个单元完整展示 | 通过 |
| 是否能阻止单一区间伪装成全网络 | 5/7 可评估覆盖、4/7 正向区间、宏微双端点和两种子复现共同门控 | 通过 |
| 是否会改写旧预注册状态 | V4 必须保持未触发，V6 文件和结论只读保留 | 通过 |
| 是否明确失败后如何表述 | 第 6.3 节定义四种互斥状态及其主张边界 | 通过 |
| 是否包含可直接执行的文件、测试和命令 | 第 10、11 节给出精确路径、接口、测试、服务器命令和预期结果 | 通过 |
| 是否遗留占位符或未闭合 Markdown 块 | 占位词扫描为空；代码围栏和公式定界符均为偶数 | 通过 |

当前剩余风险不是设计遗漏，而是正式数据可能无法通过预注册覆盖或全局效应门控。该结果必须作为方法适用范围的真实检验，不得通过追加候选、降低阈值或重选患者消除。
