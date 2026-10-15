"""Small heads and token alignment for LLM-input visual residual supervision."""

import torch
from torch import nn
from torch.nn import functional as F


DELTA_VERSION = 1


def validate_delta_checkpoint_config(config: dict) -> None:
    if config.get("third_delta_loss") and config.get("third_delta_version", 1) != DELTA_VERSION:
        raise RuntimeError(
            "This checkpoint uses the DiT delta-query architecture. "
            "It cannot be loaded or resumed as the LLM-input delta compensation model; "
            "use an original compensation checkpoint or start from the base GR00T model."
        )


def build_third_delta_head(framework: str, dimension: int) -> nn.Module:
    if framework != "mlp":
        raise ValueError(f"Unsupported third_delta_framework: {framework}")
    return nn.Sequential(
        nn.Linear(2 * dimension, 1024),
        nn.GELU(),
        nn.Linear(1024, 1024),
        nn.GELU(),
        nn.Linear(1024, dimension),
    )


def merged_token_counts(grid: torch.Tensor, merge_size: int) -> list[int]:
    if grid.ndim != 2 or grid.shape[1] != 3 or (grid <= 0).any():
        raise ValueError("Invalid image_grid_thw for third_delta_loss")
    if (grid[:, 1:] % merge_size != 0).any():
        raise ValueError("Image grids must be divisible by spatial_merge_size")
    return (grid.prod(dim=-1) // merge_size**2).tolist()


def compensate_visual_tokens(
    features: torch.Tensor,
    grid: torch.Tensor,
    image_map: torch.Tensor,
    head: nn.Module,
    merge_size: int,
    teacher: torch.Tensor | None = None,
    teacher_grid: torch.Tensor | None = None,
    teacher_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pair ego/second/third by sample and observation time, independent of padding."""
    counts = merged_token_counts(grid, merge_size)
    if image_map.shape != (len(counts), 3) or sum(counts) != features.shape[0]:
        raise ValueError("third_delta_image_map must match every main image/merger token")
    rows = image_map.tolist()
    indices = {}
    for index, (sample, time, view) in enumerate(rows):
        if sample < 0 or time < 0 or view not in (0, 1) or (sample, time, view) in indices:
            raise ValueError("Invalid or duplicate third_delta_image_map entry")
        indices[sample, time, view] = index
    pairs = {(sample, time) for sample, time, _ in rows}
    if any((sample, time, view) not in indices for sample, time in pairs for view in (0, 1)):
        raise ValueError("Every observation time requires exactly one ego and second image")
    teacher_indices = {}
    teacher_chunks = None
    if teacher is not None:
        if teacher_grid is None or teacher_map is None:
            raise ValueError("Teacher features require grid and sample/time metadata")
        teacher_counts = merged_token_counts(teacher_grid, merge_size)
        if teacher_map.shape != (len(teacher_counts), 2) or sum(teacher_counts) != teacher.shape[0]:
            raise ValueError("third_delta_teacher_map must match every teacher image/merger token")
        for index, row in enumerate(teacher_map.tolist()):
            key = tuple(row)
            if key in teacher_indices:
                raise ValueError("Duplicate teacher sample/time entry")
            teacher_indices[key] = index
        if set(teacher_indices) != pairs:
            raise ValueError("Teacher sample/time entries must match all ego/second pairs")
        teacher_chunks = list(teacher.detach().split(teacher_counts))
    chunks = list(features.split(counts))
    compensated = list(chunks)
    predictions, targets = [], []
    for sample, time in sorted(pairs):
        ego_index = indices[sample, time, 0]
        second_index = indices[sample, time, 1]
        second = chunks[second_index]
        pooled_ego = chunks[ego_index].mean(dim=0, keepdim=True).expand(second.shape[0], -1)
        delta = head(torch.cat((pooled_ego, second), dim=-1))
        # The LLM scatter requires the original image-embedding dtype.
        compensated[second_index] = second + delta.to(second.dtype)
        if teacher_chunks is not None:
            teacher_index = teacher_indices[sample, time]
            if not torch.equal(grid[second_index], teacher_grid[teacher_index]):
                raise ValueError("second/third image grids must match at each observation time")
            predictions.append(delta)
            targets.append((teacher_chunks[teacher_index] - second).detach())
    metrics = {}
    if predictions:
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        # Respect the enclosing autocast policy without forcing the MSE dtype.
        loss = F.mse_loss(prediction, target)
        with torch.no_grad():
            pred_float, target_float = prediction.float(), target.float()
            zero_mse = target_float.square().mean()
            metrics = {
                "third_delta_cosine_similarity": F.cosine_similarity(
                    pred_float, target_float, dim=-1
                ).mean(),
                "third_delta_relative_zero_mse": loss.detach().float() / zero_mse.clamp_min(1e-12),
                "third_delta_valid_tokens": loss.new_tensor(target.shape[0]),
            }
        metrics["third_delta_loss"] = loss
    return torch.cat(compensated), metrics
