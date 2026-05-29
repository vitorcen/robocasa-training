# robocasa-training

本仓 = `mujoco-experience` 的姊妹仓，专门做 **RoboCasa 任务的策略训练**（对比 `mujoco-experience` 偏评估 + 演示）。

## 当前状态 (Status)

- **v0.1** —— ACT (Action Chunking Transformer) 训 `OpenCabinet` 单任务。设计文档：[`doc/training_act.html`](doc/training_act.html)。

## 目录约定 (Layout)

```
robocasa-training/
├── doc/                            # 训练设计文档 (HTML + 内嵌 SVG)
├── configs/                        # LeRobot / ACT 训练 yaml
├── scripts/                        # 采数据 / 转格式 / 训练 / 推理 server
└── checkpoints/                    # 训练产物 (.gitignore'd)
```

## 与父仓的关系 (Relation to parent repo)

- 父仓 [`mujoco-experience`](https://github.com/vitorcen/mujoco-experience) 把本仓作为 submodule 引入，路径 `robocasa-training/`。
- 父仓里 `scripts/robocasa_eval_*.py` 提供评估流水线（ZMQ server-client 架构）；本仓的 ACT inference server 复用同一 wire 协议 — 评估端代码不需要改。
- 数据缓存目录 `~/.cache/robocasa/`（HDF5 演示）和 `~/.cache/huggingface/lerobot/`（LeRobotDataset）跨仓共享。

## Roadmap

- [ ] v0.1 OpenCabinet · ACT 单任务跑通
- [ ] v0.2 4 个 cabinet/drawer 相关 atomic 任务
- [ ] v0.3 18 atomic 联合训练，对比 GR00T N1.5
- [ ] v0.4 Diffusion Policy 对照组
- [ ] v0.5 后训练 SmolVLA / pi0.5

详见 [`doc/training_act.html`](doc/training_act.html) §9。
