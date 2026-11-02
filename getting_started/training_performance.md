# 训练计算优化与计时

两种 picknplace 配置共用相同的训练代码，无需分别修改启动脚本。新启动的训练默认启用以下优化：

- `adamw_torch_fused`：使用融合 AdamW；保留现有 decay/no-decay 分组、LLM 独立学习率和全局梯度裁剪。
- Qwen backbone 不生成未使用的词表 logits，不维护 KV cache；保留原始 hidden-state 捕获和第三分支的反向梯度路径。
- BF16 训练时，在加载模型、设置 `requires_grad` 后，将冻结的 `nn.Linear` FP32 权重和偏置改为 BF16 存储。跳过可训练参数、embedding 及其共享权重，不改变归一化层。DeepSpeed 仍使用原有混合精度与 optimizer master weights 机制。

启动时 `tune_llm=True` 或仅解冻部分层会自动跳过这些可训练权重。训练中途动态解冻不在此机制内，需要自行恢复权重 dtype 并更新优化器参数组。未启用实验性的 block 编译、共享 GEMM 或融合裁剪。

已选择 fused AdamW 时，加载旧 AdamW checkpoint 会保留当前的 fused 执行方式，同时恢复原有动量、学习率等状态。若恢复工具沿用旧配置中的 `adamw_torch`，则仍按该配置执行，不会强制更改用户选择的优化器。

## 记录频率与位置

默认设置：

```bash
--timing-log-steps 100 --timing-sample-steps 5
```

- 首个优化器 step 单独记录，用于观察冷启动；之后每 100 个优化器 step 汇总一次，训练结束时补记不足一个窗口的剩余步骤。
- 数据等待主机计时覆盖每一步；CUDA Events 和其余阶段的主机计时每个 100 步窗口采样 5 步，默认偏移为 1、21、41、61、81，避开默认 loss 日志的采样位置。首次启动额外采样第一个 step。
- 不在每个阶段或每步执行 CUDA synchronize；仅在汇总时等待最后一个已采样 step 的结束 event，再进行一次小张量的跨 rank 汇总。
- checkpoint 耗时在实际保存时额外记录一次。普通 loss 的日志频率仍由原来的 `logging_steps` 控制。
- `--timing-log-steps 0` 关闭新增计时，计算优化仍生效。
- 所有指标按**一个优化器更新**计量；使用梯度累积时，前向、反向、输入准备和取 batch 时间累加该更新的所有 microbatch。

启用 W&B 时，计时指标单独放在 `timing/` 分类，普通训练指标继续使用 `train/` 分类，例如：

```text
timing/rank0/data_wait_cpu_ms
timing/rank1/data_wait_cpu_ms
timing/max_rank/data_wait_cpu_ms
timing/rank0/forward_cuda_ms
timing/rank0/backward_cuda_ms
timing/rank0/optimizer_cuda_ms
```

计时通过现有 `Trainer.log()` → W&B 回调记录，**不另建 JSONL 或其他计时日志文件**。下次直接读取原有 `wandb/run-*/run-*.wandb` 本地记录即可；所有训练 rank 的指标放在同一个 W&B run 中。根据实际 world size 动态生成 `rank0` 到 `rankN-1`，单卡、双卡、四卡使用同一份实现；没有写死卡数。

`timing/world_size` 记录本次训练实际参与的进程数，汇总频率不随卡数增加；每次汇总携带所有 rank 的指标。计时曲线仍使用 `train/global_step` 作为横轴，与 loss 对齐。旧 run 已写入的 `train/timing/*` 历史记录不会自动改名；新启动任务使用上述分类。

## 指标含义

下表省略 `timing/rankN/` 前缀，时间单位均为毫秒：

| 指标 | 含义 |
|---|---|
| `updates` / `sampled_updates` | 此窗口的更新数 / 详细计时采样更新数 |
| `wall_ms_per_update` | 窗口实际墙钟时间/更新数，包含普通日志、保存、前一份计时报告等开销；首份还包含 iterator 启动 |
| `step_cpu_ms` | 从取 batch 前到更新结束的主机经过时间，覆盖全部更新；不额外等待 CUDA，不能当成纯 GPU 时间 |
| `between_updates_cpu_ms` | 更新结束到下一次取 batch 之间的间隔，包含日志、checkpoint、计时汇总等 |
| `data_wait_cpu_ms` | 真正调用 `next(dataloader)` 的主机等待均值，覆盖所有更新；不把优化器更新算成数据等待 |
| `data_wait_cpu_p95_ms` / `data_wait_cpu_max_ms` | 数据等待的窗口 P95 / 最大值，用于发现 shard 切换等长尾 |
| `data_wait_cuda_ms` | 采样的取 batch 区间在当前 CUDA stream 上的 event 间隔，可辅助判断主机等待是否暴露到 GPU 时间线 |
| `input_prepare_cpu_ms` / `input_prepare_cuda_ms` | Trainer 输入准备区间，包含 CPU→GPU 搬运和嵌套输入处理，不是纯 memcpy 内核计时 |
| `forward_cpu_ms` / `forward_cuda_ms` | 模型前向和 loss；不含之后的辅助指标聚合 |
| `backward_cpu_ms` / `backward_cuda_ms` | 反向传播及其依赖的同步/通信 |
| `clip_cpu_ms` / `clip_cuda_ms` | 普通训练路径的梯度裁剪；DeepSpeed 路径请见下一行 |
| `optimizer_cpu_ms` / `optimizer_cuda_ms` | 普通路径为实际 optimizer.step；DeepSpeed 路径为 engine.step，包含其内部裁剪、参数更新、通信、zero-grad 等 |
| `optimizer_includes_clip` | DeepSpeed 为 1；此时单独 `clip_*` 为 0，不代表没有执行梯度裁剪 |
| `step_cuda_ms` | 从取 batch 前到该更新结束的 CUDA event 区间 |
| `checkpoint_cpu_ms` | 此次 Trainer checkpoint 保存耗时，保存时记录；不包含随后 on_save 回调的额外处理 |
| `checkpoint_cpu_ms_total` | 此窗口内累计的 Trainer checkpoint 保存耗时 |

`max_rank/*` 是各 rank 对应统计量的最大值，用于发现较慢的一侧；不是逐步对齐后再求最大值的平均。

除数据等待和主机 step/gap 外，阶段均值仅来自采样步骤。尾部窗口若 `sampled_updates=0`，阶段值为 0 表示未采样，应忽略。

CUDA event 区间包含主机发射空隙、stream 依赖和部分通信等待，**不能称为纯 kernel 活跃时间**。CPU 数据等待可能与上一批 GPU 工作重叠，不能直接把 `data_wait_cpu_ms / wall_ms_per_update` 当成训练损失比例。不同统计口径和采样范围也不能机械相加。

## 如何定位瓶颈

先比较各 rank 的 `data_wait_cpu_ms`、P95/最大值和 `data_wait_cuda_ms`。若取 batch 的长等待同时出现在 CUDA 区间，且 `wall_ms_per_update` 随之升高，数据供给是优先排查方向。若数据等待很小而 forward/backward 或 DeepSpeed optimizer 区间偏长，则优先分析计算、CPU 发射及通信。

checkpoint 造成的长停顿应结合 `checkpoint_cpu_ms` 单独分析，避免误诊成数据加载。首次窗口包含冷启动，比较稳态时应排除。
