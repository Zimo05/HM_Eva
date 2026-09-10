# Hawkes Memory Tree：统一实验设计

> 实验代码已经按可分工方式组织在 [`Evaluation/`](Evaluation/README.md)。
> 每个 `evaluate_<dataset>_<model>.py` 只运行一个模型—数据集单元；不存在
> `evaluate_all.py`。结果集中后仅由 `aggregate_results.py` 汇总，不会隐式启动训练。

> 核心问题：能否把复杂、变化中的点过程组织成一棵可生长的 Hawkes Tree，并通过 working memory、episodic memory 与 wake/sleep，在较小参数和计算开销下完成快速适应、长期召回与结构压缩？

---

## 1. 需要验证的核心假设

模型包含三个记忆时间尺度和一个动态结构层：

\[
\theta_t^{\mathrm{eff}}
=\theta^{\mathrm{sem}}
+\Delta\theta_t^{\mathrm{epi}}
+\Delta\theta_t^{\mathrm{wm}}.
\]

需要分别证明：

1. **Working Memory（WM）** 吸收序列内部的短期变化，使模型用更少事件进入正确 event-law 区域；
2. **Episodic Memory**保存局部 Hawkes residual，在旧规律重现时实现零更新召回；
3. **Light Sleep**把稳定、重复的 residual 从 episodic memory 压缩进 semantic Hawkes law；
4. **Deep Sleep / Dynamic Topology**为持久的新规律产生分支，并通过 merge/prune 删除冗余结构；
5. **Controller**只在具有正 counterfactual utility 时适应、检索、写入或请求结构变化；
6. **Active Frontier**在接近 all-leaf 预测效果时访问更少节点；
7. 模型在性能、常驻参数、显存、延迟和持续增长成本之间形成优于大模型 baseline 的 Pareto 前沿。

实验不应只证明“总体 Accuracy 较高”，而应形成如下机制链：

> 新动态出现 → WM 快速适应 → episodic residual 写入 → Sleep 形成或更新 semantic branch → 动态重现时直接召回 → 冗余分支被合并或裁剪。

---

## 2. Benchmark 选择

### 2.1 推荐组合

| 数据集 | 当前本地规模与动态特征 | 主要用途 | 结论 |
|---|---|---|---|
| DWS | 8 种事件；8–20 个生成簇；每簇 100 条；序列长 80 或 200 | Hawkes law、路由与结构恢复 | **必选** |
| HM-Continual | 多个 Hawkes regime 按阶段出现 | 漂移、复现、瞬态、树生长 | **必选** |
| Retweet | 24,000 条、约 261 万事件、3 类；平均长 108.8；间隔 CV≈6.18 | burst、cascade、self-excitation | **真实数据首选** |
| Taobao | 2,000 条、约 11.3 万事件、17 类；间隔 CV≈2.68 | 用户异质性、兴趣切换、非平稳行为 | **真实数据首选** |
| StackOverflow | 2,203 条、约 14.3 万事件、22 类；平均长 64.8 | 长期阶段性、稀有事件、层次动态 | **真实数据首选** |
| Amazon | 9,227 条、约 41.3 万事件、16 类；类型相对均衡 | LLM/事件语义对比 | 可替换或补充 |
| Taxi | 平均序列长约 37，类别偏斜 | 空间周期模式 | 不进入主实验 |
| Mobike | 10,556 个合格用户，但平均仅约 8.1 次骑行，且只有一个月 | 用户移动 | 当前历史过短 |
| MIMIC | 2,023 位病人、124 marks；71.86% 相邻间隔来自并发事件的人工排序 | 临床状态 | 不宜放入统一 TPP 主表 |
| COVID Policy | 只有 7 条国家序列、904 个事件，同日事件与类别缺失严重 | 小样本政策案例 | 排除 |

主实验使用三个真实数据集：

> **Retweet + Taobao + StackOverflow**

它们分别代表强 burst dynamics、用户兴趣变化和长时间尺度阶段性动态。若特别需要突出自然语言语义，可用 Amazon 替换 StackOverflow，但应将其解释为 LLM semantic-label case，而不是 Hawkes Tree 生长的主要证据。

必须报告本地数据快照统计和文件哈希。例如当前本地 Taobao 为 17 类、2,000 条序列，与部分公开版本的统计不同，不能直接复制外部论文的数据规模。

### 2.2 DWS 的公平性边界

当前 DWS pipeline 可利用 cluster membership 和真实 Hawkes 参数构造层次树与 `sequence_summary`，再通过 `--h-tree`、`--sequence-summary` 同步拓扑和执行 membership alignment。如果这些信息来自完整数据或生成真值，Hawkes Memory 就获得了 baseline 没有的 oracle structure。

正式主实验必须采用 **discovery protocol**：

- 所有模型只读取 train split 的 `event_times` 与 `event_types`；
- 真实 cluster、Hawkes 参数和完整 dendrogram 仅用于训练完成后的评价；
- Hawkes Memory 从 root-only tree 开始，或只利用 train split 做无监督初始化；
- validation/test 不参与 H-tree、encoder、normalization、residual prototype 或 threshold 构造；
- learned leaf 与 true regime 的比较在评价阶段进行，不能把匹配结果反馈给模型。

可在附录增加 `Oracle-tree upper bound`，但必须明确标注为上界，不能进入公平主表。

DWS 实验规模控制为：

- **DWS-13**：主预测、law recovery 和消融；
- **DWS-8/13/20**：仅做 regime 数量增加时的 scaling curve。

### 2.3 HM-Continual：统一持续学习 synthetic benchmark

使用固定事件词表和 decay basis；每个阶段分别生成 train/validation/test，并为每个真实 law 生成独立 frozen anchor bank。oracle 参数和 regime 标签只用于评价。

推荐 curriculum：

| Task | Regime | 机制目的 |
|---:|---|---|
| 0 | A1 | root-only 初始化 |
| 1 | B1 | 全新规律 |
| 2 | C1 | 第二个新规律 |
| 3 | A1 | exact recurrence |
| 4 | B'1 | near recurrence |
| 5 | A2 | A1 的细粒度 specialization |
| 6 | 0.9 C1 + 0.1 X | 一次性 transient contamination |
| 7 | A-merge | 检查冗余分支合并 |
| 8 | A1 | long-gap recurrence |
| 9 | 0.7 E1 + 0.3 B1 | mixture 与共享表示 |

每个 checkpoint 都在所有已出现 law 的 frozen anchors 上评价；下一任务还应在任何更新前评价一次，用于 few-shot、FWT 和 zero-update recurrence。

瞬态阶段额外生成一条同随机种子、同非瞬态事件但移除 `X` 的 counterfactual control stream，用于可靠归因 TSR，而不是把时间上紧邻 X 的任意 split 当作由 X 导致。

---

## 3. Baseline 与公平比较

### 3.1 传统 TPP

主表保留：

- **RMTPP**：经典 LSTM-based TPP；
- **FullyNN**：LSTM encoder + flexible cumulative hazard；
- **THP**：Transformer Hawkes，也是 attention/Hawkes 表示的必要控制组。

持续学习版本至少包括：

- RMTPP sequential fine-tuning；
- RMTPP + memory-budget-matched replay；
- THP sequential fine-tuning；
- offline joint training，作为非在线 upper bound。

Replay baseline 的 buffer 必须按实际字节数与 Hawkes Memory 的 persistent episodic/tree state 对齐，不能只按“样本条数”声称预算相同。

### 3.2 LLM-based TPP

主 LLM baseline 使用 **TPP-LLM**：当前本地默认 TinyLlama-1.1B，LoRA rank 16，作用于 Q/K/V/O projection。持续学习时使用同一个 adapter 顺序更新，并保存每阶段 checkpoint。

LAMP 不进入全数据集主表。它依赖外部 ChatGPT 生成 causal events 和额外 ranking model，本地流程主要面向 Amazon/GDELT。若使用，仅作为 Amazon appendix case，并单独报告 API 调用量、生成时间、外部模型版本和额外存储。

### 3.3 文本信息公平性

当前标准数据适配通常只产生 `Retweet type 0`、`Amazon category type 7` 等泛化文本，而非真实类别名称。因此分为两个条件：

1. **Anonymized dynamics-only（主条件）**
   - 统一使用 `event_0, event_1, ...`；
   - 用于判断纯 temporal dynamics 和参数/计算效率；
   - 主要结论限定为“在无额外语义信息时”。

2. **Semantic-label condition（补充条件）**
   - 仅在具有可靠、可追溯事件名称的数据集运行；
   - 用于检验 LLM 是否从语言语义获益；
   - 不与匿名条件混合平均。

### 3.4 统一训练规则

- 相同 train/validation/test 数据和相同事件定义；
- 所有预处理统计仅由 train 拟合；
- 用预先指定的 validation NLL 选择唯一 checkpoint，再从该 checkpoint 报告全部 test 指标；
- 各模型获得相同的 validation 调参次数或相同 GPU-hour 调参预算；
- synthetic/DWS 至少 5 个种子，真实数据至少 3 个种子；
- 报告均值、标准差和基于独立 sequence 的 95% bootstrap CI；
- 主要比较使用配对种子，并对核心结论做 paired permutation test 或 Wilcoxon test；
- 记录硬件、精度、batch size、训练步数和 early-stopping 规则。

---

## 4. 实验一：标准预测与 Hawkes law recovery

### 4.1 数据与模型

数据：DWS-13、Retweet、Taobao、StackOverflow。

模型：RMTPP、FullyNN、THP、TPP-LLM、Hawkes Memory。

### 4.2 基础指标

- **NLL/event ↓**：主指标，stationary 与 continual 共用；
- **Event Type Accuracy ↑**：辅助指标；
- **Macro-F1 ↑**：应对类型不平衡；
- **Time MAE ↓**；
- **Time RMSE ↓**；
- 每类 support 和多数类基线。

不同数据集的原始 NLL、MAE、RMSE 不直接求算术平均；跨数据集总结使用平均 rank 或相对 improvement。

### 4.3 Few-shot adaptation 与 Adaptation Regret

在新 law 到来后，记录观测前 `K∈{0,1,4,16,64}` 个在线事件后的 NLL：

\[
AR@K=\frac{1}{K}\sum_{k=1}^{K}
\left(\ell_k-\ell_k^{\mathrm{oracle}}\right)\downarrow .
\]

其中 `oracle` 是只使用该 law 的训练数据、拥有相同模型容量与调参预算的 task-specific offline model，不能使用当前预测位置之后的数据。

同时报告离散梯形积分得到的适应曲线面积：

\[
AUC_{\mathrm{adapt}}
=\frac{1}{K_{\max}}
\int_0^{K_{\max}}L(K)\,dK\downarrow .
\]

为了避免只比较一个任意 K，主文报告 `AUC_adapt`，正文图展示完整 NLL–K 曲线。

### 4.4 DWS Event-law Recovery

对 exponential-basis Hawkes law，定义 integrated excitation：

\[
A_{ij}=\sum_m\frac{W_{ijm}}{\beta_m}.
\]

报告：

\[
E_A=\frac{\|\hat A-A^*\|_F}{\|A^*\|_F},
\qquad
E_\mu=\frac{\|\hat\mu-\mu^*\|_2}{\|\mu^*\|_2}.
\]

以及：

- spectral-radius error
  \[
  E_\rho=|\rho(\hat A)-\rho(A^*)|;
  \]
- edge recovery F1；edge threshold 必须由 validation 固定，另报告 threshold-free AUPRC；
- latent regime ARI / NMI；
- leaf purity；
- node count、leaf count、最大/平均深度。

在计算参数误差前，必须先利用 validation/anchor likelihood 或参数距离对 learned leaves 与 true regimes 做 Hungarian matching。匹配仅用于评价。对于一个 regime 由多个叶子共同解释的情况，同时报告 mass-weighted law error，避免强迫一对一匹配掩盖 mixture 行为。

### 4.5 Intensity NISE

\[
NISE=
\frac{\int\|\hat\lambda(t)-\lambda^*(t)\|_2^2dt}
{\int\|\lambda^*(t)\|_2^2dt}\downarrow .
\]

预测与真值必须使用同一条观测历史、同一积分区间和同一时间网格；不能分别用两个模型自己生成的历史，否则误差同时混入 trajectory mismatch。

### 4.6 DWS scaling curve

仅在 DWS-8/13/20 上画一张综合曲线：

- x：真实 regime 数；
- y1：NLL/event 或 NISE；
- y2：learned leaves / nodes；
- y3：平均 visited frontier nodes；
- y4：每事件延迟。

目标是展示模型结构是否随动态复杂度按需增长，而不是盲目扩张。

---

## 5. 实验二：性能—资源与记忆压缩 Pareto

### 5.1 与 TPP-LLM 的系统效率比较

同一次训练记录：

- total resident parameters；
- trainable parameters；
- checkpoint bytes；
- optimizer-state bytes；
- persistent tree bytes 与 episodic-memory bytes；
- 每阶段新增参数/字节；
- peak GPU memory；
- GPU-hours；
- batch-1 latency；
- batch-32 throughput；
- 每事件 online update latency；
- amortized Light/Deep Sleep cost。

必须同时报告 total 和 trainable parameters。LoRA 的 trainable parameters 较少不等于推理时无需加载 LLM backbone。

定义 iso-performance 区域：validation NLL 与最优模型相差不超过 1%，或 Accuracy 相差不超过 0.5 个百分点。在该区域比较最小常驻内存、显存和推理时间。

主 Pareto 图：

- x：总常驻内存或每事件延迟；
- y：NLL/event；
- 点大小：trainable parameters；
- 颜色：模型家族。

### 5.2 Low-rank Hawkes residual 压缩

当前实现保留 `Δmu`，并对每个 Hawkes basis 的 `ΔW(:,:,m)` 独立做 truncated SVD。系统运行：

\[
r\in\{0,1,2,4,8,D\},
\]

其中大于 `D` 的 rank 自动省略，`r=D` 为 full-rank residual。

完整 residual 参数规模：

\[
P_{\mathrm{full}}=D+D^2M.
\]

若真正以 factorized 形式持久化 rank-r residual，其理论规模为：

\[
P_r\approx D+M(2Dr+r),
\qquad
CR_r=\frac{P_{\mathrm{full}}}{P_r}.
\]

**实现注意：**当前代码在 SVD 后仍重建并存储 dense `D×D×M` residual。因此主结果必须同时区分：

- `effective rank / information compression`；
- `actual serialized bytes`。

只有在改为持久化 `U,S,V` 后，才能把上述 `P_r` 当作真实存储参数量。

性能保持率：

\[
PR_r=
\frac{L_{\mathrm{base}}-L_r}
{L_{\mathrm{base}}-L_{\mathrm{full}}}.
\]

其中 `base` 为无 residual，`full` 为 full-rank residual。画两类 Pareto：

- rank/理论参数量 vs NLL、adaptation gain、forgetting；
- 实际 checkpoint bytes vs 同一组性能指标。

---

## 6. 实验三：持续学习与树生长

### 6.1 基础 continual 指标

令 `L_{t,j}` 表示完成 task `t` 后，在 frozen law/task `j` 上的 NLL。

**CL-NLL ↓**：

\[
CLNLL_t=\frac{1}{|S_t|}\sum_{j\in S_t}L_{t,j}.
\]

**Average Forgetting ↓**：

\[
F_{t,j}=L_{t,j}-\min_{u\in[t_j,t]}L_{u,j},
\qquad
AF_t=\frac{1}{|S_t|}\sum_{j\in S_t}F_{t,j}.
\]

**Backward Transfer ↑**（loss-based）：

\[
BWT_j=L_{t_j,j}-L_{T,j}.
\]

正值表示后续学习改善了旧 law，负值表示遗忘。

**Forward Transfer ↑**：

\[
FWT_j=L_{\mathrm{scratch},j}-L_{j-1,j}^{\mathrm{pre}}.
\]

`pre` 必须在学习 task `j` 前、零梯度更新状态下测量；`scratch` 是相同初始化与容量下的未迁移模型。

辅助报告 Accuracy、Macro-F1 和 adaptation AUC，但持续学习主结论以 frozen-anchor NLL 为准。

### 6.2 Recurrence Retention Ratio（RRR）

对首次出现并在长间隔后重现的 law A，记录：

- `L_A^(first,pre)`：首次训练前；
- `L_A^(first,post)`：首次训练后；
- `L_A^(return,pre)`：重现时、任何新更新前。

定义：

\[
RRR_A=
\frac{L_A^{\mathrm{first,pre}}-L_A^{\mathrm{return,pre}}}
{L_A^{\mathrm{first,pre}}-L_A^{\mathrm{first,post}}}.
\]

- `RRR≈1`：首次学到的收益基本保留；
- `RRR≈0`：完全遗忘；
- `RRR<0`：重现前比首次遇到还差；
- `RRR>1`：后续任务产生正迁移。

当分母接近 0 时 RRR 不稳定，应同时报告原始三个 NLL，并只在首次学习收益超过预先固定的最小阈值时汇总 RRR。

### 6.3 Transient Structuralization Rate（TSR）

对一次性 transient law X：

\[
TSR=
\frac{\#\text{由 X 引发的持久 semantic/topology changes}}
{\#\text{transient episodes}}\downarrow .
\]

X 消失 H 个阶段后的保留假阳性率：

\[
FPR_{\mathrm{retain}}(H)
=P(\text{X 仍被表示为 semantic leaf})\downarrow .
\]

“由 X 引发”通过有/无 X 的 paired counterfactual streams 判定；persistent change 至少跨过预先规定的 H 个阶段或 sleep cycles。理想行为是短期由 WM/Episodic 处理 X，但不永久创建 semantic branch。

### 6.4 Consolidation Precision / Recall

\[
P_{\mathrm{con}}=
\frac{\#\text{正确 consolidation 的 persistent laws}}
{\#\text{全部 consolidated laws}},
\]

\[
R_{\mathrm{con}}=
\frac{\#\text{正确 consolidation 的 persistent laws}}
{\#\text{真实 persistent laws}},
\]

并报告 `F1_con`。

“正确 consolidation”定义为：semantic leaf 在禁用 WM 与 Episodic 后，经过 Hungarian/mass-weighted matching 能以预先固定的 NISE 或参数误差阈值解释一个 persistent ground-truth law。Transient X 不计入真实 persistent laws，但若形成持久叶子则计为 false positive。

### 6.5 树生长可视化

在 task 0/3/5/7/9 保存统一布局的树快照，并与以下曲线共享横向时间轴：

- CL-NLL 与 RRR；
- node/leaf count；
- episodic rows 与 semantic bytes；
- split/merge/prune/absorb 操作；
- 各真实 law 对 learned leaf 的 matching；
- transient 阶段的 TSR/FPR；
- 每阶段训练和 Sleep 时间。

该图应成为论文的核心 storytelling figure。

---

## 7. 实验四：有针对性的消融与时间尺度分解

消融只在 **DWS-13 + HM-Continual** 上运行，不扩展到全部真实数据。

### 7.1 核心消融

| 配置 | 检验内容 | 最敏感指标 |
|---|---|---|
| Full model | 完整机制 | 全部指标 |
| w/o Working Memory | 快速局部适应 | `AR@K`、`AUC_adapt`、shift 后前 1/4/16 事件 NLL |
| w/o Episodic Memory | 中期记忆与复现召回 | RRR、AF、Episodic Recall Gain |
| Fixed Topology | 结构 consolidation | DWS law recovery、ARI、specialization NLL、leaf count |
| No Sleep Consolidation | episodic → semantic | `G_SEM`、memory rows、consolidation F1、长期 NLL |
| Heuristic Controller | learned counterfactual control | harmful action rate、写入量、TSR、NLL |

为控制实验数量，前四项进入主消融表；`No Sleep` 与 `Heuristic Controller` 可放机制表或附录，但至少应各运行 HM-Continual 主种子。

### 7.2 压缩消融

- full-rank residual vs rank `{0,1,2,4,8,D}`：检验 low-rank efficiency；
- no merge/prune：检验长期结构压缩；
- active-frontier vs all-leaf：仅做冻结 checkpoint 的推理诊断，比较 visited nodes、latency、NLL，不必全部重训。

### 7.3 Memory Timescale Decomposition

所有增益都在同一 checkpoint、同一序列和相同随机状态上做 paired counterfactual evaluation。

**Working Memory Gain**：

\[
G_{WM}(k)=
L_{\mathrm{before\ WM}}
-L_{\mathrm{after\ }k\mathrm{\ online\ events}}.
\]

写入与 Sleep 在测量窗口内关闭，防止把 episodic/semantic 收益算进 WM。

**Episodic Recall Gain**：

在 WM 关闭、禁止新写入且 memory usage 不更新时：

\[
G_{EPI}=
L_{\mathrm{semantic}}
-L_{\mathrm{semantic+episodic}}.
\]

重点在 exact/long-gap recurrence 的任何新更新前测量。这直接回答旧 law 返回时，性能改善是否来自 retrieval。

**Semantic Consolidation Gain**：

在 WM 与 Episodic retrieval 都关闭时，对同一 frozen replay/anchor set 比较 Sleep 前后：

\[
G_{SEM}=
L_{\mathrm{semantic,before\ sleep}}
-L_{\mathrm{semantic,after\ sleep}}.
\]

`G_SEM>0` 才能证明 episodic residual 被转化为 semantic Hawkes law，而不只是 memory buffer 在工作。还应同时报告 Sleep 前后 residual energy、absorbed rows、exact-rebasing 数值误差和未参与 Sleep 的 control anchors，防止把一般参数更新误认为 consolidation。

---

## 8. 结果组织

主文控制为两张表、三张核心图：

1. **Table 1：标准预测性能**  
   DWS-13 + Retweet + Taobao + StackOverflow；报告 NLL、Accuracy、Macro-F1、MAE、RMSE。

2. **Table 2：机制消融**  
   DWS-13 + HM-Continual；报告 AR/AUC、RRR、AF、NISE、consolidation F1、TSR 和 memory bytes。

3. **Figure 1：性能—系统资源 Pareto**  
   重点比较 Hawkes Memory 与 TPP-LLM，同时标出 trainable 与 total parameters。

4. **Figure 2：持续学习时间轴与树生长**  
   NLL heatmap、树快照、memory/node growth、split/merge/prune 和 recurrence 对齐展示。

5. **Figure 3：Residual compression Pareto**  
   rank、理论 factorized 参数量、实际 serialized bytes 与性能保持率。

附录放置：

- DWS-8/13/20 scaling；
- 完整 per-type 指标；
- Semantic-label TPP-LLM；
- LAMP Amazon case（若执行）；
- Oracle-tree upper bound；
- 全部种子、置信区间和超参数。
# HM_Eva
