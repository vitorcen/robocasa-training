# RoboCasa OpenCabinet 最终榜单 (Final Leaderboard)

- task `OpenCabinet` (split `target`) · 30 rounds · max_steps 1200 · n_action_steps 16 · seed_base=0 (同30厨房)
- SR 排除 sim-DNF (steps==0 且未成功) ; N1.7 取 11k/13k/15k 精扫峰值 ; 基线为 authoritative 备份未重跑

| # | Policy | Success rate | Successes | Mean steps (success) | sim-DNF |
|---|--------|-------------|-----------|----------------------|---------|
| 1 | GR00T-N1.5-multitask (下载) | **64.0%** | 16/25 | 410 | 5 |
| 2 | GR00T-N1.7-OpenCabinet (自训, peak ckpt-11000) | **50.0%** | 13/26 | 627 | 4 |
| 3 | pi0.5-pretrain-human300 (下载) | **17.4%** | 4/23 | 495 | 7 |

## N1.7 checkpoint sweep (过拟合曲线)

| step | SR | successes | mean succ steps | sim-DNF |
|------|----|-----------|-----------------|---------|
| 11000 | 50.0% | 13/26 | 627 | 4 |
| 13000 | 42.3% | 11/26 | 464 | 4 |
| 15000 | 44.0% | 11/25 | 451 | 5 |
