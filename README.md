# robocasa-training

RoboCasa 任务的策略训练。评估 + 演示流水线见 [`mujoco-experience`](https://github.com/vitorcen/mujoco-experience) 的 `scripts/robocasa_eval_*.py`（ZMQ server-client，本仓的推理 server 复用同一协议）。

## 文档 (Docs)

- [`doc/training_act.html`](doc/training_act.html) —— ACT 训练 `OpenCabinet` 的设计与流水线。
- [`doc/training_gr00t_n17.html`](doc/training_gr00t_n17.html) —— **GR00T-N1.7-3B 微调** `OpenCabinet`：bf16 选择性全参（冻 VLM、训 action head ~560M，非 8bit/非 LoRA），本机 4090 单 GPU 串行边训边 eval（watchdog）。配方移植自 LeIsaac N1.7。
- [`doc/act_opencabinet_results.html`](doc/act_opencabinet_results.html) —— **实验结论（负面结果）**：ACT-80M from-scratch 在 OpenCabinet 上 closed-loop SR 全程 **0%**（human / mimicgen / temporal-ensembling / pretrain-then-finetune 四个杠杆全试尽），对照 GR00T-2B VLA 同管线 **70%**。含完整对照矩阵、方法论、工程踩坑。**结论：小模型从头训不适合 RoboCasa atomic 操作任务，需走预训练 VLA 路线。**

## 目录 (Layout)

```
robocasa-training/
├── doc/             # 文档 (HTML + 内嵌 SVG)
├── scripts/         # 采数据 / 转格式 / 训练 / 推理 server
├── dependencies/
│   └── Isaac-GR00T  # submodule: NVIDIA 上游 pin n1.7-release (GR00T-N1.7 训练用)
└── checkpoints/     # 训练产物 (.gitignore'd)
```
