# HawkesMemory 实验运行与分工说明

本文档供负责运行实验的同事使用。它说明每类实验想验证什么、应该运行哪个脚本、结果保存在哪里，以及完成后如何检查和汇总。

## 1. 实验要回答什么问题

HawkesMemory 不只是一个用于下一事件预测的模型。我们希望通过实验讲清楚以下四层故事：

1. **预测能力**：HawkesMemory 在 synthetic data 和真实数据上的 NLL、事件类型预测与时间预测，能否达到或超过传统 TPP 和 LLM-based TPP baseline。
2. **资源效率**：在取得相近预测效果时，HawkesMemory 是否能用更少的模型参数、checkpoint 字节、显存或推理访问量完成预测。
3. **持续学习能力**：面对新规律、旧规律回归、渐变、层级分化和短暂噪声时，HawkesMemory 能否快速适应，同时保留以前学到的 Hawkes law。
4. **机制解释**：working memory、episodic memory、sleep consolidation、动态树结构和 low-rank residual 分别贡献了什么；Hawkes Tree 是否真的随着动态规律生长、合并和压缩。

因此，实验被拆成四组：标准预测与 baseline 对比、continual learning、HawkesMemory 消融，以及 law recovery/frontier/rank/时间尺度/树生长诊断。

## 2. 为什么每个实验使用独立脚本

本目录没有 `evaluate_all.py`。每个脚本只运行一个固定的“模型 × 数据集 × 条件”，例如：

```text
evaluate_dws_HM.py           = DWS 上运行 HawkesMemory
evaluate_taobao_THP.py       = Taobao 上运行 THP
evaluate_retweet_TPP_LLM.py  = Retweet 上运行 TPP-LLM
```

这样不同同事可以在不同机器上并行运行。一个脚本不会启动其他模型，也不依赖其他标准预测实验已经完成。

每次运行会独立执行：

```text
检查并转换数据
→ 训练当前模型
→ 根据 validation 指标选择 best checkpoint
→ 在 test set 上评估
→ 导出事件级预测
→ 记录资源信息
→ 生成本次运行的结果包
```

模型和数据集身份固定在脚本内部，不能通过命令行把 `evaluate_taobao_THP.py` 改成其他模型，从而避免脚本名称与真实实验不一致。

## 3. 环境准备

### 3.1 Python 与依赖

请使用 **Python 3.10 或更高版本**，然后进入实验目录：

```bash
cd D:/Files/School/lishuang/HawkesMemory_wfy/Evaluation
```

运行只读环境检查：

```bash
python preflight.py
```

如果本次任务必须使用 GPU：

```bash
python preflight.py --require-cuda
```

`preflight.py` 会检查 Python、NumPy、Pandas、PyTorch、CUDA 以及项目目录，但不会启动训练或修改数据。

> 注意：`python` 必须本身就指向 Python 3.10+。`--python` 只用于指定后续训练子进程的解释器，不能把一个由 Python 3.7 启动的 Evaluation 入口升级成 Python 3.10。

Windows/conda 环境建议先确认：

```bash
where python
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如果 `torch.cuda.is_available()` 为 `False`，可以先使用 CPU 验证完整流程：

```bash
python evaluate_dws_HM.py --variant 13 --seed 42 --device cpu --smoke --run-id smoke-cpu
```

指定 `--device cuda:0` 时，入口现在会在创建结果目录前检查当前实验环境是否安装了 CUDA 版 PyTorch、是否能访问 GPU 0，并给出直接可读的错误，而不会等到训练阶段才失败。

不同 baseline 还需要各自模型目录中声明的依赖。TPP-LLM 首次运行时还需要能够访问其配置的基础语言模型，或提前准备好本地模型缓存。

### 3.2 先做 dry-run 和 smoke test

正式训练前建议依次执行：

```bash
python evaluate_dws_HM.py --variant 13 --seed 42 --device cuda:0 --dry-run --run-id dryrun
python evaluate_dws_HM.py --variant 13 --seed 42 --device cuda:0 --smoke --run-id smoke
```

- `--dry-run`：检查数据、实验身份和输出路径，只打印命令，不训练。
- `--smoke`：用很少的数据和 epoch 运行完整链路，检查 train → checkpoint → reload → predict 是否连通。
- 不带这两个参数时才是正式实验。

建议为 dry-run 和 smoke 分别添加 `--run-id dryrun`、`--run-id smoke`，不要与正式结果使用同一目录。

## 4. 通用参数

所有公开入口都支持以下参数：

| 参数 | 含义 | 建议 |
|---|---|---|
| `--seed` | 随机种子，必填 | 必须按 registry 分配值运行 |
| `--device` | `auto`、`cpu`、`cuda` 或 `cuda:N` | 多卡机器建议明确写 `cuda:N` |
| `--output-root` | 结果根目录 | 默认是 `Evaluation/results` |
| `--epochs` | 覆盖默认训练 epoch | 正式主结果不要随意修改 |
| `--batch-size` | 覆盖默认 batch size | 显存不足时可以调整，但需保留 manifest |
| `--resume` | 复用或继续相同实验 | 只有身份、配置和数据哈希一致才允许 |
| `--smoke` | 最小规模完整链路测试 | 正式运行前推荐使用 |
| `--dry-run` | 只检查并打印命令 | 不产生正式训练结果 |
| `--run-id` | 在结果路径末尾增加标识 | 用于 smoke、分阶段运行或特殊配置 |
| `--python` | 指定底层模型使用的 Python | 多环境机器可传 Python 可执行文件路径 |
| `--checkpoint` | 指定已有 checkpoint | 主要用于 continual 后半段续跑 |
| `--max-trials` | 实验预算元数据 | 当前入口不会自动发起超参数搜索 |
| `--max-gpu-hours` | GPU 预算元数据 | 当前不会在超时后自动终止进程 |

DWS 入口额外支持 `--variant 8|13|20`；continual 入口支持 `--task-start 0..9`、`--task-end 0..9` 和 `--data-root`；rank 入口支持 `--rank 0|1|2|4|8|D`，其中 `D` 表示 full rank。

## 5. 标准预测实验

### 5.1 实验目的与模型

标准实验用于回答：在固定 train/validation/test split 下，HawkesMemory 与传统 TPP、Transformer TPP 和 LLM-based TPP 相比，预测质量与资源开销如何。

| 模型 | 类型 | 对比意义 |
|---|---|---|
| `HM` | HawkesMemory | 我们的动态 Hawkes Tree 模型 |
| `RMTPP` | RNN-based TPP | 传统 recurrent baseline |
| `FullyNN` | Neural TPP | 传统神经点过程 baseline |
| `THP` | Transformer Hawkes Process | attention-based TPP baseline |
| `TPP_LLM` | LLM-based TPP | 大模型方向 baseline |

主指标是 test NLL/event、Accuracy、Macro-F1、time MAE/RMSE，以及 checkpoint bytes、参数规模、训练时间和可获得的 GPU 信息。

### 5.2 DWS-13：主 synthetic 实验

DWS 带有真实 Hawkes law、cluster 和树结构。DWS-13 用于主预测表、law recovery 和主要消融。

```bash
python evaluate_dws_HM.py --variant 13 --seed 42 --device cuda:0
python evaluate_dws_RMTPP.py --variant 13 --seed 42 --device cuda:0
python evaluate_dws_FullyNN.py --variant 13 --seed 42 --device cuda:0
python evaluate_dws_THP.py --variant 13 --seed 42 --device cuda:0
python evaluate_dws_TPP_LLM.py --variant 13 --seed 42 --device cuda:0
```

正式种子为 `42, 43, 44, 45, 46`。每位同事只需将命令中的 seed 换成自己领取的值。

DWS 的 `cluster` 和 ground-truth Hawkes 参数只允许用于 test 后诊断，不能进入训练 adapter。HM 使用 root-only、无 oracle 初始化。

### 5.3 DWS-8/13/20：HM scaling

Scaling 实验观察真实 regime 数量增加时，HM 的预测、树规模、visited frontier nodes 和资源开销如何变化。

当前 registry 中 DWS-8 和 DWS-20 只登记 HM；DWS-13 直接复用主实验。

```bash
python evaluate_dws_HM.py --variant 8 --seed 42 --device cuda:0
python evaluate_dws_HM.py --variant 13 --seed 42 --device cuda:0
python evaluate_dws_HM.py --variant 20 --seed 42 --device cuda:0
```

Scaling 同样使用种子 `42–46`。

### 5.4 Retweet

Retweet 用于测试真实传播序列中的非平稳、长尾和复杂时间动态。

```bash
python evaluate_retweet_HM.py --seed 2024 --device cuda:0
python evaluate_retweet_RMTPP.py --seed 2024 --device cuda:0
python evaluate_retweet_FullyNN.py --seed 2024 --device cuda:0
python evaluate_retweet_THP.py --seed 2024 --device cuda:0
python evaluate_retweet_TPP_LLM.py --seed 2024 --device cuda:0
```

正式种子为 `2024, 2025, 2026`。

### 5.5 Taobao

Taobao 用于测试用户行为事件中多类型事件、不同时间尺度和行为转移动态。

```bash
python evaluate_taobao_HM.py --seed 2024 --device cuda:0
python evaluate_taobao_RMTPP.py --seed 2024 --device cuda:0
python evaluate_taobao_FullyNN.py --seed 2024 --device cuda:0
python evaluate_taobao_THP.py --seed 2024 --device cuda:0
python evaluate_taobao_TPP_LLM.py --seed 2024 --device cuda:0
```

正式种子为 `2024, 2025, 2026`。

### 5.6 StackOverflow

StackOverflow 用于测试真实用户活动中不规则时间间隔、类型转换和群体异质性。

```bash
python evaluate_stackoverflow_HM.py --seed 2024 --device cuda:0
python evaluate_stackoverflow_RMTPP.py --seed 2024 --device cuda:0
python evaluate_stackoverflow_FullyNN.py --seed 2024 --device cuda:0
python evaluate_stackoverflow_THP.py --seed 2024 --device cuda:0
python evaluate_stackoverflow_TPP_LLM.py --seed 2024 --device cuda:0
```

正式种子为 `2024, 2025, 2026`。

### 5.7 TPP-LLM 标签协议

主 TPP-LLM 实验统一使用匿名 `event_N` 文本，防止语义先验造成不公平优势。当前数据没有经过审核的真实语义名称，因此 semantic-label 实验标记为 `not_available`。

LAMP 依赖外部 causal-event 生成和额外 ranking model，不进入主 registry 和主表。

## 6. Continual learning 实验

### 6.1 数据与实验故事

持续学习数据默认位于 `Datasets/Data/CL/hm_continual_v1`。十个阶段依次为：

| Task | 动态 | 作用 |
|---|---|---|
| 0 | `A_1` 初次出现 | 初始学习 |
| 1 | `B_1` | 新规律 |
| 2 | `C_1` | 新规律 |
| 3 | `A_1` exact recurrence | 测试旧规律快速召回 |
| 4 | `B_prime_1` | 测试近似 recurrence / drift |
| 5 | `A_2` | 测试层级 specialization |
| 6 | `C_1 + X_transient` | 测试短暂异常是否被永久结构化 |
| 7 | `A_merge` | 测试相近规律的合并与压缩 |
| 8 | `A_1` long-gap recurrence | 测试长期保持与 RRR |
| 9 | `E_1 + B_1` | 测试混合规律和组合泛化 |

Task 6 同时包含 paired no-X control，用于区分正常结构变化和 transient X 引发的持久变化。

如果数据缺失，可重新生成：

```bash
python ../Datasets/Data/CL/generate_continual_hawkes.py --benchmark unified --output ../Datasets/Data/CL/hm_continual_v1 --seed 7
```

除非要创建新版本数据，否则不要覆盖统一数据目录。所有模型和训练种子应使用同一份 `hm_continual_v1`。

### 6.2 每个阶段做什么

每个 continual 脚本会在当前 task 学习前做 pre-update，训练并保存独立 checkpoint，随后做 post-update，并在所有 frozen anchors 上评估。由此构建 checkpoint × law 矩阵。

主要指标包括 CL-NLL、Average Forgetting、BWT、FWT、adaptation gain/AUC 和 RRR；HM 还报告 TSR、树规模、episodic rows、semantic bytes 与 NISE。

### 6.3 HawkesMemory continual

```bash
python evaluate_continual_HM.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
```

正式种子为 `7, 17, 27, 37, 47`。

该实验还会生成 `resource_manifest.json`，记录每阶段序列化后的 tree + episodic state 字节，供 replay baseline 做公平预算匹配。

### 6.4 Sequential baselines

Sequential 表示模型按 task 顺序更新，只保留上一阶段模型状态，不访问旧 task 训练数据。

```bash
python evaluate_continual_RMTPP_sequential.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
python evaluate_continual_THP_sequential.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
python evaluate_continual_TPP_LLM.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
```

TPP-LLM 会顺序更新同一个 LoRA adapter，并为每阶段保存 checkpoint。

### 6.5 Joint upper bound

Joint 在 task `t` 训练时可以访问 task `0..t` 的全部训练数据。它不是严格 continual learner，而是“保留全部历史数据”的上界。

```bash
python evaluate_continual_RMTPP_joint.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
python evaluate_continual_THP_joint.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
```

### 6.6 Byte-matched replay

Replay 不能随意按样本条数设置 buffer。它必须读取同种子 HM 的资源清单，并确保实际 replay CSV 字节不超过 HM tree + episodic state 字节。

先完成 HM continual，再运行：

```bash
python evaluate_continual_RMTPP_replay.py --seed 7 --task-start 0 --task-end 9 --device cuda:0 --hm-resource-root results/continual/HM/full/seed_7
```

结束后检查 `replay_manifest.csv`，其中 `actual_bytes` 必须小于等于 `budget_bytes`。缺少 HM `resource_manifest.json` 时脚本会拒绝运行。

### 6.7 从中间 checkpoint 继续

如果 task 0–4 已在另一台机器完成，可以把 `task_04.pt` 复制过来，再将 task 5–9 作为独立结果包运行：

```bash
python evaluate_continual_HM.py --seed 7 --task-start 5 --task-end 9 --checkpoint D:/shared/task_04.pt --run-id tasks_05_09 --device cuda:0
```

RMTPP、THP 和 TPP-LLM continual 入口同样支持这种方式。请保证 checkpoint 的模型、策略、seed 和超参数与前半段一致。

## 7. HawkesMemory 消融实验

消融只在 DWS-13 和 HM-Continual 上做。Full model 直接复用 `evaluate_dws_HM.py` 和 `evaluate_continual_HM.py`。

| 条件 | 移除或固定的模块 | 主要观察 |
|---|---|---|
| `no_working` | 关闭快速 working-memory adaptation | 短期适应、AR@K、NLL |
| `no_episodic` | 禁止 episodic retrieval 和新写入 | recurrence、RRR、长期 NLL |
| `fixed_topology` | 禁止树结构变化 | law recovery、ARI、leaf 数量 |
| `no_sleep` | 关闭 sleep consolidation | semantic gain、memory rows |
| `heuristic_controller` | 使用冻结启发式 controller | 写入与召回决策质量 |
| `no_merge_prune` | 禁止合并与裁剪 | 树膨胀和压缩效率 |

DWS-13 命令：

```bash
python evaluate_dws_HM_no_working.py --variant 13 --seed 42 --device cuda:0
python evaluate_dws_HM_no_episodic.py --variant 13 --seed 42 --device cuda:0
python evaluate_dws_HM_fixed_topology.py --variant 13 --seed 42 --device cuda:0
python evaluate_dws_HM_no_sleep.py --variant 13 --seed 42 --device cuda:0
python evaluate_dws_HM_heuristic_controller.py --variant 13 --seed 42 --device cuda:0
python evaluate_dws_HM_no_merge_prune.py --variant 13 --seed 42 --device cuda:0
```

HM-Continual 命令：

```bash
python evaluate_continual_HM_no_working.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
python evaluate_continual_HM_no_episodic.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
python evaluate_continual_HM_fixed_topology.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
python evaluate_continual_HM_no_sleep.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
python evaluate_continual_HM_heuristic_controller.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
python evaluate_continual_HM_no_merge_prune.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
```

主消融优先运行 `no_working`、`no_episodic`、`fixed_topology` 和 `no_sleep`；后两项可放机制表或附录，但至少完成主种子 7。

## 8. 诊断与压缩实验

### 8.1 DWS law recovery

检查 learned leaves 是否恢复 synthetic data 的真实 Hawkes laws。训练阶段看不到 oracle；训练后才做 Hungarian matching 和参数误差诊断。

```bash
python evaluate_dws_HM_law_recovery.py --variant 13 --seed 42 --device cuda:0
```

重点查看 `law_recovery.json`、`native/node_metrics.csv` 和 `metrics.json`。Continual evaluator 另外负责 intensity-curve NISE。

### 8.2 Active-frontier vs all-leaf

验证 active frontier 能否在接近 all-leaf 预测质量时减少访问节点和推理成本。两种模式使用同一冻结 checkpoint，不重新训练。

```bash
python evaluate_dws_HM_frontier.py --variant 13 --seed 42 --device cuda:0
```

重点比较 `frontier_comparison.json`、`native/` 与 `native_all_leaf/` 中的 NLL、visited nodes、吞吐和资源指标。

### 8.3 Low-rank residual compression

每次只运行一个 rank，便于分给不同同事：

```bash
python evaluate_dws_HM_residual_rank.py --variant 13 --rank 0 --seed 42 --device cuda:0
python evaluate_dws_HM_residual_rank.py --variant 13 --rank 1 --seed 42 --device cuda:0
python evaluate_dws_HM_residual_rank.py --variant 13 --rank 2 --seed 42 --device cuda:0
python evaluate_dws_HM_residual_rank.py --variant 13 --rank 4 --seed 42 --device cuda:0
python evaluate_dws_HM_residual_rank.py --variant 13 --rank 8 --seed 42 --device cuda:0
python evaluate_dws_HM_residual_rank.py --variant 13 --rank D --seed 42 --device cuda:0
```

`rank=0` 表示无 residual，`rank=D` 表示 full rank。汇总时比较 NLL、checkpoint bytes、参数规模和性能保持率。

### 8.4 Continual 机制诊断

时间尺度分解的目标是分析 Working Memory Gain、Episodic Recall Gain 和 Semantic Consolidation Gain：

```bash
python evaluate_continual_HM_timescales.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
```

Consolidation 诊断检查 episodic residual 是否真正通过 sleep 转化为 semantic Hawkes law：

```bash
python evaluate_continual_HM_consolidation.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
```

Tree growth 诊断展示树在新规律、specialization、merge 和 transient 阶段的生长与压缩：

```bash
python evaluate_continual_HM_tree_growth.py --seed 7 --task-start 0 --task-end 9 --device cuda:0
```

树生长实验重点查看 task 0/3/5/7/9 附近的 node/leaf count、split/merge/prune/absorb、episodic rows、semantic bytes、RRR、TSR 与 NLL 时间线。

这三个入口目前复用 HM continual 的训练与 `EvaluateCL` 输出，并分别保存独立结果包。运行者应以实际生成的字段为准：当前 evaluator 已直接产出的指标可以汇总；若某个 paired counterfactual、sleep 前后快照或结构归因字段没有生成，应标记为 `not_available`，不能从普通 post-update NLL 推断或人工补值。

## 9. 输出目录与结果包

默认结果根目录为 `Evaluation/results/`，例如：

```text
results/dws/HM/full/variant_13/seed_42/
results/taobao/THP/full/seed_2024/
results/continual/HM/full/seed_7/
results/continual/RMTPP/replay/seed_7/
```

每个叶子目录都是可单独复制和提交的结果包：

| 文件或目录 | 内容 |
|---|---|
| `manifest.json` | 脚本身份、seed、参数、git、环境和输入哈希 |
| `status.json` | `running`、`complete`、`failed` 或 `dry_run` |
| `checkpoint/` | validation-best 或逐 task checkpoint |
| `metrics.json` | 核心指标 |
| `sequence_metrics.csv` | test、阶段和 anchor 细粒度指标 |
| `predictions.jsonl.gz` | 统一事件级预测 |
| `resources.json` | 时间、checkpoint bytes、参数值数量和硬件信息 |
| `logs/` | 训练与评估日志 |
| `tree_events.jsonl` | 树事件预留接口；没有结构事件时为空文件 |
| `plots/` | 本次运行产生的图 |
| `report.md` | 单次摘要与复现命令 |

事件级预测统一包含 `sequence_id`、`event_index`、`true_type`、`predicted_type`、`type_probabilities`、真实/预测时间间隔和 `event_nll`。

RMTPP 与 FullyNN 的上游 EasyTPP API 当前不暴露逐事件 probability 和 likelihood term，因此这两个字段可能为 `null`，但 aggregate test NLL 是真实模型 NLL。THP、TPP-LLM 和 HM 会导出 probability 与 event NLL。

`resources.json` 不会伪造无法读取的资源数据。若父进程无法取得子训练进程的 CUDA peak allocation，`peak_gpu_memory_bytes` 会是 `null` 并附带说明；需要严格峰值显存时，应由运行同事同时使用统一的外部 GPU 监控方案。

## 10. Resume 与并行安全

相同实验续跑示例：

```bash
python evaluate_dws_HM.py --variant 13 --seed 42 --device cuda:0 --resume
```

只有实验身份、seed、variant/rank/task 范围、训练配置、数据和外部 checkpoint 哈希一致时才会复用。若结果已是 `complete`，会直接返回；配置不同则拒绝覆盖。

不同 job 的路径由 dataset、model、condition、variant/rank 和 seed 共同决定，因此可并行运行。不要让两台机器同时写完全相同的 job；特殊重复运行应增加不同的 `--run-id`。

## 11. 分工与提交

完整任务表位于 `experiment_registry.csv`。查看待运行任务：

```bash
python list_experiments.py --status pending
```

输出形式如下：

```text
J001: python evaluate_dws_HM.py --seed 42 --variant 13
```

负责人可以按 `job_id` 分工。同事应严格使用该行的 script、seed、arguments 和 required input。正式完成后提交整个叶子结果目录，不要只提交 `metrics.json`。

## 12. 提交前验证

```bash
python validate_result.py --result-dir results/dws/HM/full/variant_13/seed_42
```

验证器会检查结果包是否齐全、状态是否为 `complete`、可访问的原始输入是否发生哈希变化，以及 DWS/HM 是否记录 oracle isolation。

验证失败时不要手工修改 status，应检查 `logs/` 和 `status.json` 中的错误。

## 13. 集中汇总

所有结果复制到同一个 `results/` 后运行：

```bash
python aggregate_results.py --results-root results
```

汇总器只读取完成结果，绝不会启动训练。默认输出到 `Evaluation/aggregate/`：

```text
all_results.csv
missing_jobs.csv
conflicts.csv
table1_prediction.csv
table2_ablation.csv
summary_with_bootstrap_ci.csv
paired_permutation_tests.csv
figures/figure1_resource_pareto.png
figures/figure2_continual_timeline.png
figures/figure3_residual_compression.png
```

`missing_jobs.csv` 列出未收到的任务；`conflicts.csv` 报告同一 job 的不同哈希；完全相同的重复结果只合并一次。统计检验只使用能够按 seed 配对的完成结果。

## 14. 推荐的个人执行流程

同事拿到 job 后，可按以下顺序操作：

```bash
cd D:/Files/School/lishuang/HawkesMemory_wfy/Evaluation
python preflight.py --require-cuda
python <脚本> --seed <种子> <其他参数> --device cuda:0 --dry-run --run-id dryrun
python <脚本> --seed <种子> <其他参数> --device cuda:0 --smoke --run-id smoke
python <脚本> --seed <种子> <其他参数> --device cuda:0
python validate_result.py --result-dir <正式结果目录>
```

如果正式任务失败，请保留结果目录和日志，修复环境后使用相同命令加 `--resume`。完成后将整个正式结果目录交给汇总负责人。
