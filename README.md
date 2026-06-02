# robocasa-training

RoboCasa 任务的策略训练。评估 + 演示流水线见 [`mujoco-experience`](https://github.com/vitorcen/mujoco-experience) 的 `scripts/robocasa_eval_*.py`（ZMQ server-client，本仓的推理 server 复用同一协议）。

*Policy training for RoboCasa tasks. Eval + demo pipeline lives in [`mujoco-experience`](https://github.com/vitorcen/mujoco-experience) (`scripts/robocasa_eval_*.py`, ZMQ server-client; inference server here reuses the same protocol).*

## 文档 (Docs)

- [`doc/training_act.html`](doc/training_act.html) —— ACT 训练 `OpenCabinet` 的设计与流水线 / *ACT training design & pipeline for OpenCabinet*
- [`doc/training_gr00t_n17.html`](doc/training_gr00t_n17.html) —— **GR00T-N1.7-3B 微调** `OpenCabinet`：bf16 选择性全参（冻 VLM、训 action head ~560M，非 8bit/非 LoRA），本机 4090 单 GPU 串行边训边 eval（watchdog）。配方移植自 LeIsaac N1.7 / *GR00T-N1.7-3B fine-tune on OpenCabinet: bf16 selective full-param (freeze VLM, train action head ~560M); recipe ported from LeIsaac N1.7*
- [`doc/act_opencabinet_results.html`](doc/act_opencabinet_results.html) —— **实验结论（负面结果）**：ACT-80M from-scratch 在 OpenCabinet 上 closed-loop SR 全程 **0%**（human / mimicgen / temporal-ensembling / pretrain-then-finetune 四个杠杆全试尽），对照 GR00T-2B VLA 同管线 **70%** / *Negative result: ACT-80M from-scratch = 0% SR on OpenCabinet (all 4 levers exhausted); GR00T-2B VLA = 70% on same pipeline*
- [`doc/mimicgen_data_strategy.html`](doc/mimicgen_data_strategy.html) —— MimicGen 数据扩增策略：从 8644 ep 池随机采样 + human 混合训练，预期缩 14 点差距 / *MimicGen augmentation strategy: sample from 8644-ep pool + human mix, expected to close ~14pt gap*

## 基准榜单 (Benchmark)

- [`benchmark/leaderboard.md`](benchmark/leaderboard.md) —— 3-policy 最终榜单 + N1.7 checkpoint sweep 过拟合曲线 / *3-policy final leaderboard + overfitting curve*
- [`benchmark/sweep_n17.py`](benchmark/sweep_n17.py) —— checkpoint 精扫脚本，seed-locked 公平基准 / *checkpoint sweep runner, seed-locked fair benchmark*

**最终榜单速览 (Quick leaderboard) —— 公平 30/30,1200 步,seed_base=0,DNF 重试凑满零剔除 / fair 30/30, 1200 steps, retried DNFs, zero exclusion:**

| Policy | SR | 成功 / Succ |
|---|---|---|
| GR00T-N1.5-multitask (下载 / downloaded) | **70.0%** | 21/30 |
| **GR00T-N1.7-MG2stage (自训, ckpt-14000)** | **53.3%** | 16/30 |
| pi0.5-pretrain-human300 (下载 / downloaded) | **23.3%** | 7/30 |

> N1.7-MG2stage = MimicGen 两阶段（8644 MG + 500 human 原生混合预训 ~34k → 纯 human 微调,峰值 step 14000）；较 human-only ckpt-11000(同口径 ~43%)真实提升 ~10 点,但未超多任务 N1.5。早先"N1.7 69% > N1.5 64%"是 exclude-DNF 不均偏置的假象,公平重测翻盘。
> *N1.7-MG2stage = MimicGen two-stage (native-mix pretrain ~34k → pure-human finetune, peak step 14000); a real ~10pt gain over human-only ckpt-11000 (~43% same basis) but does not overtake multi-task N1.5. The earlier "N1.7 69% > N1.5 64%" was DNF-exclusion bias — the fair re-test reverses it.*

## 目录 (Layout)

```
robocasa-training/
├── doc/             # 文档 (HTML + 内嵌 SVG) / Docs
├── scripts/         # 采数据 / 转格式 / 训练 / 推理 server / Data, train, inference
├── benchmark/       # 评估基准 / Eval benchmark
├── dependencies/
│   └── Isaac-GR00T  # submodule: NVIDIA 上游 pin n1.7-release (GR00T-N1.7 训练用)
└── checkpoints/     # 训练产物 (.gitignore'd) / Training outputs
```
