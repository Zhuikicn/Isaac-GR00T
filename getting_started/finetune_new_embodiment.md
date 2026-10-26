# Fine-tune on Custom Embodiments ("NEW_EMBODIMENT")

This guide demonstrates how to finetune GR00T on your own robot data and configuration. We provide a complete example for the Huggingface [SO-100](https://github.com/TheRobotStudio/SO-ARM100) robot under `examples/SO100`, which uses `demo_data/cube_to_bowl_5` as the demo dataset.

## Step 1: Prepare Your Data

Prepare your data in **GR00T-flavored LeRobot v2 format** by following the [data preparation guide](data_preparation.md). 

## Step 2: Prepare Your Modality Configuration

Define your own modality configuration by following the [modality config guide](data_config.md). Below is an example configuration that corresponds to the demo data:
```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


so100_config = {
    # Video: use current frame only ([0]); list camera view names matching modality.json
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "front",
            "wrist",
        ],
    ),
    # State: current proprioceptive reading; keys must match modality.json "state" entries
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",
            "gripper",
        ],
    ),
    # Action: 16-step prediction horizon; each key needs an ActionConfig
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # predict 16 future steps
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            # single_arm: RELATIVE = delta from current state (better generalization)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,       # joint-space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
            # gripper: ABSOLUTE = target position (binary open/close works better absolute)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task-instruction annotation. The key is dataset-specific and must
    # match meta/modality.json (see data_preparation.md#annotation-column-naming).
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Important: always register under EmbodimentTag.NEW_EMBODIMENT for custom embodiments
register_modality_config(so100_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

## Step 3: Run Fine-tuning


> ⚠️ **aarch64 users (Spark / Thor / Orin):** After running `install_deps.sh`, always
> activate the venv with `source .venv/bin/activate && source scripts/activate_<platform>.sh`
> (e.g. `activate_spark.sh`, `activate_thor.sh`, or `activate_orin.sh`)
> and run the example commands in this guide with **plain `python`** / `torchrun`, not `uv run python` / `uv run torchrun`. The latter will
> re-sync against the root `pyproject.toml` (which targets x86_64 Python 3.12) and
> destroy the platform-specific environment.

We'll use `gr00t/experiment/launch_finetune.py` as the entry point. Ensure that the uv environment is enabled before launching. You can do this by running the command `uv run bash <example_script_name>`.

### View Available Arguments
```bash
# Display all available arguments
uv run python gr00t/experiment/launch_finetune.py --help
```

### Execute Fine-tuning
```bash
# Configure for single GPU
export NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path ./demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus $NUM_GPUS \
    --output-dir /tmp/so100 \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 2000 \
    --use-wandb \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--base-model-path` | Path to the pre-trained base model checkpoint |
| `--dataset-path` | Path to your training dataset |
| `--embodiment-tag` | Tag to identify your robot embodiment |
| `--modality-config-path` | Path to user-specified modality config (required only for `NEW_EMBODIMENT` tag) |
| `--action-horizon` | Model prediction length and processor padded action length in frames (default: `40`) |
| `--output-dir` | Directory where checkpoints will be saved |
| `--save-steps` | Save checkpoint every N steps |
| `--max-steps` | Total number of training steps |
| `--use-wandb` | Enable Weights & Biases logging for experiment tracking |
| `--tune-llm` | Enable full-parameter tuning of the retained LLM backbone |
| `--learning-rate` | Peak learning rate for the action head and other trainable non-LLM modules (default: `1e-4`) |
| `--llm-learning-rate` | Peak learning rate for trainable LLM parameters (default: `1e-6`; unused when the LLM is frozen) |
| `--swap-hand-state` | Independent sign-flip probability per `MY_G1_BINARY` hand state during training (default: `0.05`; `0` disables it) |

### Action Prediction Horizon

The model's `--action-horizon` and the modality config's action `delta_indices`
control the prediction and supervision lengths. For 80 contiguous supervised
frames, set `action.delta_indices=list(range(80))` in your modality config and run:

```bash
source .venv/bin/activate
bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /path/to/dataset \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /path/to/modality_config.py \
    --action-horizon 80 \
    --output-dir /path/to/checkpoint_h80
```

The same `--action-horizon 80` option works directly with `launch_finetune.py`.
It sets the action head's generation length when loading pretrained weights and
the processor's `max_action_horizon`; both settings are saved with the checkpoint.
The modality config determines the returned action length at inference. Use
matching lengths to train and return all 80 frames. Shorter modality action
sequences are padded with masked supervision; longer sequences require a larger
model horizon. Existing per-frame LeRobot datasets can be reused with the chosen
modality config. At 50 Hz, 80 frames cover 1.6 seconds. Deployment's
`--execution-horizon` separately selects how many frames to execute per policy call.

### Binary Hand State Augmentation

For `MY_G1_BINARY`, the `left_hand`/`right_hand` state groups and
`left_hand_close`/`right_hand_close` action groups pass through normalization and
denormalization unchanged. Hand states also skip sin/cos encoding and global
`--state-noise-std`; other state groups retain their existing noise and clipping behavior.
Hand action predictions pass through without thresholding or clipping.

During training, `--swap-hand-state` independently samples each hand and multiplies its
state by `-1`, flipping `+1` and `-1` together across all history frames of that hand.
Action targets remain unchanged. This augmentation does not run during evaluation or
inference, and other embodiment tags are unaffected. Global `--state-dropout-prob` and
excluded states still take precedence: zeroed states remain zero. The existing global
dropout after the state encoder also remains active.

The probability must be finite and in `[0, 1]` and is saved in checkpoints. Older
checkpoints without this field use `0.05`; their removed hand noise/dropout settings
are ignored when loading.

Append these arguments to `launch_finetune.py` or training scripts that forward `"$@"`:

```bash
--swap-hand-state 0.05
```

With `examples/finetune.sh`, pass them after its `--` separator.

With `--tune-llm`, the LLM defaults to `1e-6` while the action head stays at `1e-4`.
For example, `--tune-llm --llm-learning-rate 3e-6` changes only the LLM rate.
Both rates follow the same warmup/cosine schedule, and optimizer groups retain the
existing weight decay exclusions. Logs report `learning_rate` for non-LLM modules
and `llm_learning_rate` for the LLM. Gradient clipping remains global
(`max_grad_norm=1.0` by default), computed from raw gradients before the optimizer
update; learning rates do not rescale the gradients used for clipping.

Full-LLM runs now have separate optimizer groups. To switch an old full-LLM run
to these rates, start a new run/output directory (optionally loading its model
weights) instead of resuming its old optimizer/scheduler state.

> **Note:** Validation during fine-tuning is disabled by default (`eval_strategy="no"` in the training config). To enable periodic validation, pass `--eval-strategy steps --eval-steps 500` (runs validation every 500 steps) or `--eval-strategy epoch` (runs validation every epoch). You can also adjust `--eval-batch-size` (default: 2).

## Step 4: Open Loop Evaluation

After finetuning, evaluate the model's performance using open loop evaluation:
```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path ./demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path /tmp/so100/checkpoint-2000 \
    --traj-ids 0 \
    --execution-horizon 16 \
    --steps 400 \
    --modality-keys single_arm gripper
```

### `open_loop_eval.py` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset-path` | `demo_data/cube_to_bowl_5/` | Path to LeRobot-format dataset |
| `--embodiment-tag` | `new_embodiment` | Robot embodiment tag (case-insensitive) |
| `--model-path` | `None` | Path to checkpoint. If omitted, connects to a running server via `--host`/`--port` |
| `--traj-ids` | `[0]` | Episode indices to evaluate (space-separated, e.g., `0 1 2`) |
| `--execution-horizon` | `16` | Steps of each predicted chunk to execute per inference (old `--action-horizon` deprecated) |
| `--steps` | `200` | Max steps per trajectory (capped by actual trajectory length) |
| `--denoising-steps` | `4` | Diffusion denoising iterations |
| `--save-plot-path` | `None` | Directory to save GT-vs-predicted comparison plots |
| `--modality-keys` | `None` | Action keys to plot. If omitted, plots all action dimensions |
| `--host` / `--port` | `127.0.0.1` / `5555` | Server address when `--model-path` is omitted |

### Example Evaluation Result

The evaluation generates visualizations comparing predicted actions against ground truth trajectories:

<img src="../media/open_loop_eval_so100.jpg" width="800" alt="Open loop evaluation results showing predicted vs ground truth trajectories" />

## Interpreting the Result: Is My Fine-tune Working?

`open_loop_eval.py` logs `Average MSE across all trajs` and `Average MAE across all trajs` (unnormalized action error; referred to below as `Average MSE`/`MAE`) and saves a ground-truth-vs-predicted plot. Unlike the simulation benchmarks (LIBERO, SimplerEnv, DROID), this tutorial intentionally does **not** publish a single target MSE: the demo set is only 5 episodes and your own dataset and task will differ, so a fixed number would not transfer and could mislead. Use these checks instead, in order:

1. **Plot overlap on a training trajectory (primary).** `--traj-ids 0` is part of the training set, so a model that has fit the data should track the ground-truth curves closely. Flat or constant predictions, or curves that ignore the GT shape, point to a setup problem rather than merely an under-trained model.
2. **Error decreases with training (reproducible trend).** Save intermediate checkpoints (e.g. add `--save-steps 500`) and run the eval on each. `Average MSE`/`MAE` on `traj 0` should fall steadily as steps increase. A flat or rising curve means training is not learning from your data.
3. **Record your own baseline.** Note the `Average MSE`/`MAE` from your first clean run on the unmodified `cube_to_bowl_5` command above, and treat it as your reference point. After you change dataset size, modality config, or hyper-parameters, re-run and compare against it.

For reference, one run of the command above — with `--save-steps` lowered to `500` so intermediate checkpoints are kept (1× H100, `--max-steps 2000`, eval on `--traj-ids 0`) — produced the trend below. The **shape** — error falling steadily as steps increase — is the signal; the absolute values are not a target and will differ with GPU, seed, and data.

| Checkpoint | `Average MSE` (traj 0) | `Average MAE` (traj 0) |
|-----------:|-----------------------:|-----------------------:|
| 500        | 87.5                   | 5.63                   |
| 1000       | 25.4                   | 3.30                   |
| 1500       | 13.2                   | 2.18                   |
| 2000       | 10.0                   | 1.76                   |

Averaged over all 5 training episodes, the final checkpoint scored `Average MSE` ≈ 7.5 / `Average MAE` ≈ 1.5.

When a run looks off, this table maps the common symptoms to causes that are operational (your setup) rather than a model bug:

| Symptom | Likely cause (operational, not a model bug) |
|---------|---------------------------------------------|
| MSE flat or rising across checkpoints | Learning rate too low, or data not loading — check `--dataset-path` and dataloader workers |
| Prediction curve is flat/constant | `modality.json` keys or `--modality-config-path` mismatch — action keys are not mapped |
| MSE extremely large or `NaN` (or `NaN` loss during training) | Action/state normalization — verify `meta/stats` and that action ranges are sane |
| Good on `traj 0` but poor on held-out episodes | Expected with only 5 demo episodes — this is data scarcity, not a bug |

> Establish this baseline on the **as-shipped** command before changing anything. Reproducing a known-good run first is the fastest way to separate setup mistakes from genuine issues when you scale to a larger dataset.

## Optional: Midpoint Third-View Supervision and Action–Query Gating

Third-view supervision uses 8 learnable DiT queries by default and directly
supervises the last 4. The auxiliary branch reads the complete output of the
middle cross-attention block (including its FFN and residuals): the
`ceil(number_of_cross_attention_blocks / 2)`-th cross-attention block. With the
official 32-block DiT, this is the 8th cross-attention block, overall block 15
(counting from 1). All 8 queries continue through the remaining DiT blocks.

The shared auxiliary head is `RMSNorm(1536, eps=1e-6) -> Linear(1536, 4096) ->
GELU -> Linear(4096, 2048)` for the official model. RMSNorm has a learnable scale
initialized to 1 and no bias. Its output is used only for supervision and is
not written back into the DiT sequence. Input/output widths follow the model
configuration.

The last 4 queries are paired in order with the last 4 third-view image tokens
for each sample, e.g. queries 5–8 with tokens 109–112 for a 112-token image.
Targets default to `raw_cosmos`, the configured-depth VLM backbone output
without VL postprocessing, and are detached. Token counts can differ between
samples; fewer target tokens than the requested supervision count is an error.
The loss is the mean of `1 - cosine_similarity` over pairs and samples,
computed in FP32 with `eps=1e-8`, without pooling or negative samples.
Samples whose input state is exactly zero across all dimensions and history
steps, or whose encoded state features are dropped, are excluded from the cosine
loss and its gradients. Eligibility requires both a nonzero input state and no
feature dropout, using the actual per-sample dropout decision. The loss averages
only over eligible samples, or is zero if none are eligible. Flow-matching loss
still includes all samples according to the action mask.

```bash
--use-third-view-aux-loss \
--num-learnable-queries 8 \
--num-supervised-queries 4 \
--query-target-stage raw_cosmos \
--fm-loss-weight 1.0 \
--query-cosine-loss-weight 1.0
```

The total loss is `fm_loss_weight * fm_loss + query_cosine_loss_weight *
query_cosine_loss`; both weights default to 1. The new weight flag replaces
`--query-mse-loss-weight`, and training logs report `query_cosine_loss`.
The supervised count must be between 1 and the total query count. This setup
starts a new experiment from the official base model, initializing queries
and the auxiliary head; old auxiliary checkpoints are not migrated. New
checkpoints save and restore the new configuration and auxiliary weights.

The auxiliary loss backpropagates through the head and the DiT path up to the
supervision point, without detaching intermediate queries. Earlier queries can
receive indirect auxiliary gradients through query self-attention. Later DiT
blocks and the final output projection receive no auxiliary-loss gradient;
flow matching continues to train the full action path. Inference keeps all
queries, skips the auxiliary head, and does not require third-view images.

For datasets already configured for the third-view auxiliary loss, append these
arguments after the `--` separator of `examples/finetune.sh` (or pass them directly
to `launch_finetune.py`):

```bash
--use-third-view-aux-loss \
--use-action-query-gate \
--query-gate-init-prob 0.5
```

Each self-attention block predicts one gate per action token from its existing
AdaLN output. The gate is shared across attention heads and adds
`clamp_min(logsigmoid(logit), log(1e-6))` only to action-row/query-column logits.
Enabling third-view supervision automatically applies block self-attention in
both training and inference: state tokens attend only to state, learnable queries
attend to state and queries, and action tokens attend to all three groups.
Attention within each group is fully connected. Vision-language cross-attention
is unchanged. The gate biases the allowed action-to-query attention; queries
cannot read actions, including indirectly through state across layers. Training
keeps the existing flow-matching and third-view losses; the flow-matching loss
supervises the gates. Gate computation follows the existing mixed-precision policy;
no explicit float32 conversion is added.

The old mask flag has been removed. Remove `--mask-query-action-attention` or
`--no-mask-query-action-attention` from existing launch commands. A residual
`mask_query_action_attention` field in an old checkpoint does not affect behavior.

Gating defaults to off. Enabling it requires third-view supervision and
interleaved self-attention. The gate MLP hidden
width is fixed at 64. The initial probability is configurable; initialization
only applies to new gate weights. Legacy checkpoints initialize missing gates,
while saved gate weights are preserved on reload. Keep the gate flag when
continuing fine-tuning a gated checkpoint; inference reads the gate settings
from the saved model configuration.

Training logs include `query_gate_mean` and `query_gate_mean/layer_<index>` for
each self-attention block, using zero-based DiT block indices. These are gate
probabilities, not measured attention shares. PyTorch inference recomputes gates
at every flow step. The gate also leaves indirect query-to-state-to-action
interactions intact.

### Phase auxiliary prediction

Enable this training objective with the following arguments after the `--`
separator of `examples/finetune.sh`, or directly in `launch_finetune.py`:

```bash
--use-phase-aux-loss \
--phase-label-key phase \
--phase-num-classes 3 \
--phase-mlp-hidden-dim 1024 \
--fm-loss-weight 1.0 \
--phase-loss-weight 1.0
```

Phase supervision is incompatible with `--use-third-view-aux-loss` and
`--use-action-query-gate`, and requires interleaved self-attention. It defaults
to off. Each episode parquet must contain the specified column, with one integer
class ID per frame in `[0, phase_num_classes)`. Missing columns, null labels,
non-integers, and out-of-range IDs are errors. IDs are preserved without remapping;
phases may repeat or move backwards. Labels never enter the model's conditioning
inputs or state/action normalization statistics.

The branch adds exactly four learnable queries, initialized from `Normal(0, 0.02)`.
Each DiT input is `[state tokens, four phase queries, action tokens]`. Self-attention
allows state and phase queries to read each other, but neither group can read
action tokens. Action tokens can read all three groups. In every block, state
and phase queries use plain LayerNorm without timestep-dependent scale or shift;
only action tokens retain AdaLN. The final DiT output normalization likewise
applies time modulation only to action tokens. Vision-language cross-attention
retains its existing behavior. Queries receive no action position embeddings.
They restart from the learned vectors at each flow step; hidden states are not
cached across steps. The third-view query count does not control phase queries.
For fixed observations and state, with dropout disabled, the state/query hidden
states and phase logits are independent of the flow timestep, sampled noise,
and action trajectory. Training dropout and state augmentation can still vary
these features. The attention restriction also blocks indirect action-to-query
information flow through state tokens across layers.

The head reads the four query states from the **last complete DiT block**, after
its attention, FFN, and residuals, but before the final DiT normalization, time
modulation, and output projection. Its architecture is:

```text
[B, 4, D] -> per-token shared RMSNorm(D, eps=1e-6) -> flatten [B, 4D]
          -> Linear(4D, hidden) -> GELU -> Linear(hidden, H*C) -> [B, H, C]
```

RMSNorm starts with unit weights; Linear layers use PyTorch default initialization.
The head adds no dropout. With `D=1536`, `hidden=1024`, `H=40`, and `C=3`, the
queries and head add **6,423,160 parameters**. The output is raw logits, with
position `h` aligned to the corresponding action delta index: the MY_G1 default
is the current frame through `t+39`, not `t+1` through `t+40`.

Sampling remains unchanged: an episode of length `N` contributes
`max(0, N-H+1)` chunk starts for contiguous `0..H-1` action offsets. Starts with
fewer than `H` remaining actions are excluded, including when `allow_padding`
is enabled. The last complete chunk includes the terminal frame. Short input
chunks are padded to the model horizon: action targets are zero-padded and their
loss mask is zero, while phase labels are padded with `-100`. Phase validity is
computed before index clipping, so an out-of-episode phase is ignored rather
than copied from the terminal frame. Existing action index clipping is unchanged.

Training reuses the same sampled flow time and noised actions as flow matching:
`x_tau = (1-tau)*noise + tau*ground_truth_action`. At every sampled time, the
head computes standard FP32 cross entropy on logits, without a preceding softmax,
class weighting, label smoothing, temporal smoothing, or monotonicity constraints.
Only valid labels at valid action time positions count. Cross entropy is summed
over those frames and divided by their count; an all-invalid batch returns a
connected zero loss. Zero input states and state dropout do not exclude samples.

```text
loss = fm_loss_weight * fm_loss + phase_loss_weight * phase_loss
```

Both weights default to 1. FM retains its existing valid-action-element mean;
phase CE uses a valid-frame mean within each rank's current microbatch. Gradient
accumulation and DDP retain the Trainer's behavior, including averaging gradients
across ranks. A zero phase weight removes its contribution but still constructs
the branch, computes its outputs, and requires valid training labels.

Phase loss backpropagates through the head, queries, shared attention/FFN weights
in all DiT blocks, and trainable state and visual-language encoders. It bypasses
the action encoder, action position embeddings, timestep encoder, AdaLN time
projections, final DiT output layers, and action decoder. FM still trains queries
through action-to-context attention, and retains the action/time gradient paths. Setting
`tune_projector=False` freezes queries and the head; `tune_diffusion_model=False`
freezes DiT. Phase prediction now uses the current visual-language observations
and state to predict the chunk's phases, without reading noised ground-truth
actions at any flow time.

Training outputs include `phase_logits`, `phase_loss`, `phase_accuracy`, and
`phase_valid_frames`. Loss and accuracy logs are distributed rank means;
`phase_valid_frames` logs the mean valid-frame count per rank/microbatch, not a
global total. Accuracy uses the same mask as CE.

Base checkpoints initialize the complete missing phase branch; phase checkpoints
restore existing weights and saved configuration. Partial missing weights and
third-view-to-phase checkpoint conversions are rejected by the training pipeline.
When continuing fine-tuning, keep the phase flag and matching head settings in
the launch command. Inference reads the saved configuration, retains all four
queries, and skips the classifier. It needs neither phase labels nor third-view
images and returns the usual action outputs without additional DiT calls.
This context-only attention and normalization behavior applies whenever phase
supervision is enabled. Earlier fully connected phase checkpoints retain compatible
parameter shapes but use the new computation when loaded; this is an architecture
change, not an exact continuation of their original forward behavior.

Context-only phase smoke validation loaded all 537 official action-head tensors from
the local GR00T-N1.7-3B checkpoint (32 DiT blocks). On an NVIDIA L40 with BF16,
batch size 2, 40 action positions, and 96 synthetic backbone tokens:

| Action-head mode | Forward peak allocated MiB | Forward + backward peak allocated MiB |
| --- | ---: | ---: |
| Base | 3377.88 | 6215.34 |
| Phase auxiliary | 3460.48 | 6242.34 |
| Increase | 82.60 | 27.00 |

These are standalone action-head measurements using real checkpoint weights;
they exclude backbone execution, optimizer state, and distributed training.
With dropout disabled, changing flow time and noised actions produced exactly
identical phase logits in the 32-block BF16 checkpoint (maximum difference 0).
Both DiT implementations also passed layer-by-layer context invariance checks
and gradient checks: phase CE has zero gradients into the action/time paths,
while FM retains nonzero gradients into queries and the action/time paths.
Temporary checks also verified phase gradient routing, zero-state/dropout
supervision, masked CE, unchanged inference calls, checkpoint restoration, and
all 20 labeled episodes (14,281 complete-chunk starts). No full fine-tuning run
was started for this validation.

For fine-tuning, use tmux and inject the proxy environment before launching;
tmux does not automatically inherit those variables.
