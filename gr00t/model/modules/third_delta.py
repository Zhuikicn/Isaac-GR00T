"""CPU image alignment and detached, ordered targets for DiT delta queries."""

import torch
from torch.nn import functional as F


DELTA_QUERY_COUNT = 8
DELTA_VERSION = 2


def validate_delta_checkpoint_config(config: dict) -> None:
    if config.get("third_delta_loss") and config.get("third_delta_version", 1) != DELTA_VERSION:
        raise RuntimeError(
            "This checkpoint uses the old LLM-input delta compensation architecture. "
            "It cannot be loaded or resumed as the DiT delta-query model; "
            "start a new run from the base GR00T checkpoint."
        )


def merged_token_counts(grid: torch.Tensor, merge_size: int) -> list[int]:
    """Validate processor grids on CPU before the trainer transfers the batch."""
    if grid.device.type != "cpu":
        raise ValueError("Delta image alignment must be prepared on CPU in the collator")
    if grid.ndim != 2 or grid.shape[1] != 3 or (grid <= 0).any():
        raise ValueError("Invalid image_grid_thw for third_delta_loss")
    if (grid[:, 1:] % merge_size != 0).any():
        raise ValueError("Image grids must be divisible by spatial_merge_size")
    return (grid.prod(dim=-1) // merge_size**2).tolist()


def build_third_delta_layout(
    grid: torch.Tensor,
    image_map: list[tuple[int, int, int]],
    batch_size: int,
    merge_size: int,
    teacher_grid: torch.Tensor | None = None,
    teacher_map: list[tuple[int, int]] | None = None,
) -> dict:
    """Store Python token spans in sample order, with no GPU metadata round trips."""
    counts = merged_token_counts(grid, merge_size)
    if len(image_map) != len(counts):
        raise ValueError("Image metadata does not match the processed main image count")
    indices, offsets = {}, []
    offset = 0
    for index, (sample, time, view) in enumerate(image_map):
        if time != 0:
            raise ValueError(
                "DiT delta queries require a single observation time: delta_indices=[0]"
            )
        if not 0 <= sample < batch_size or view not in (0, 1) or (sample, view) in indices:
            raise ValueError("Invalid or duplicate delta image metadata")
        indices[sample, view] = index
        offsets.append(offset)
        offset += counts[index]
    if set(indices) != {(sample, view) for sample in range(batch_size) for view in (0, 1)}:
        raise ValueError("Every sample requires exactly one ego and second image")
    layout = {"main_token_count": offset, "teacher_token_count": 0, "pairs": []}
    if teacher_grid is None:
        return layout
    teacher_counts = merged_token_counts(teacher_grid, merge_size)
    if teacher_map is None or len(teacher_map) != len(teacher_counts):
        raise ValueError("Teacher images require matching sample/time metadata")
    teacher_indices, teacher_offsets = {}, []
    offset = 0
    for index, (sample, time) in enumerate(teacher_map):
        if time != 0 or not 0 <= sample < batch_size or sample in teacher_indices:
            raise ValueError("Every sample requires exactly one teacher at observation time 0")
        teacher_indices[sample] = index
        teacher_offsets.append(offset)
        offset += teacher_counts[index]
    if set(teacher_indices) != set(range(batch_size)):
        raise ValueError("Teacher samples must match all ego/second samples")
    for sample in range(batch_size):
        second_index = indices[sample, 1]
        teacher_index = teacher_indices[sample]
        if not torch.equal(grid[second_index], teacher_grid[teacher_index]):
            raise ValueError("second/third image grids must match for every sample")
        layout["pairs"].append(
            (offsets[second_index], teacher_offsets[teacher_index], counts[second_index])
        )
    layout["teacher_token_count"] = offset
    return layout


@torch.no_grad()
def pooled_delta_target(
    features: torch.Tensor, teacher: torch.Tensor, layout: dict
) -> torch.Tensor:
    """Pool each view to eight ordered groups, then subtract third minus second."""
    if features.shape[0] != layout["main_token_count"]:
        raise ValueError("Main merger token count does not match delta metadata")
    if teacher.shape[0] != layout["teacher_token_count"] or not layout["pairs"]:
        raise ValueError("Teacher merger token count does not match delta metadata")
    if features.shape[1:] != teacher.shape[1:]:
        raise ValueError("second/third merger feature dimensions must match")
    # Group equal-length images so pooling uses one batched operation per size.
    groups = {}
    for sample, (second_start, teacher_start, count) in enumerate(layout["pairs"]):
        groups.setdefault(count, []).append((sample, second_start, teacher_start))
    target = features.new_empty(
        (len(layout["pairs"]), DELTA_QUERY_COUNT, features.shape[-1]), dtype=torch.float32
    )
    for count, rows in groups.items():
        second = torch.stack([features[start : start + count] for _, start, _ in rows]).float()
        third = torch.stack([teacher[start : start + count] for _, _, start in rows]).float()
        second = F.adaptive_avg_pool1d(second.transpose(1, 2), DELTA_QUERY_COUNT).transpose(1, 2)
        third = F.adaptive_avg_pool1d(third.transpose(1, 2), DELTA_QUERY_COUNT).transpose(1, 2)
        target[[sample for sample, _, _ in rows]] = third - second
    return target
