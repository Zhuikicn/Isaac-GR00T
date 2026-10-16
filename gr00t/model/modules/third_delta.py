"""Third-view prediction heads, visual replacement and pooled teacher targets."""

import torch
from torch import nn
from torch.nn import functional as F


THIRD_QUERY_COUNT = 8


def merged_token_counts(grid: torch.Tensor, merge_size: int) -> list[int]:
    """Validate spatial grids and return the merger token count for each image."""
    if grid.ndim != 2 or grid.shape[1] != 3 or (grid <= 0).any():
        raise ValueError("Invalid image_grid_thw for third_loss")
    if (grid[:, 1:] % merge_size != 0).any():
        raise ValueError("Image grids must be divisible by spatial_merge_size")
    return (grid.prod(dim=-1) // merge_size**2).tolist()


def build_third_layout(
    grid: torch.Tensor,
    image_map: list[tuple[int, int, int]],
    batch_size: int,
    merge_size: int,
    teacher_grid: torch.Tensor | None = None,
    teacher_map: list[tuple[int, int]] | None = None,
) -> dict:
    """Store Python token spans in sample order, with no GPU metadata round trips."""
    if grid.device.type != "cpu" or (
        teacher_grid is not None and teacher_grid.device.type != "cpu"
    ):
        raise ValueError("Query alignment must be prepared on CPU in the collator")
    counts = merged_token_counts(grid, merge_size)
    if len(image_map) != len(counts):
        raise ValueError("Image metadata does not match the processed main image count")
    indices, offsets = {}, []
    offset = 0
    for index, (sample, time, view) in enumerate(image_map):
        if time != 0:
            raise ValueError(
                "DiT third-view queries require a single observation time: delta_indices=[0]"
            )
        if not 0 <= sample < batch_size or view not in (0, 1) or (sample, view) in indices:
            raise ValueError("Invalid or duplicate third-view image metadata")
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
def pooled_third_target(
    features: torch.Tensor,
    teacher: torch.Tensor,
    layout: dict,
    prediction_target: str = "third",
) -> dict[str, torch.Tensor]:
    """Pool each view to eight ordered groups and select third or delta targets."""
    if prediction_target not in ("third", "delta"):
        raise ValueError("third_prediction_target must be third or delta")
    if features.shape[0] != layout["main_token_count"]:
        raise ValueError("Main merger token count does not match third-view metadata")
    if teacher.shape[0] != layout["teacher_token_count"] or not layout["pairs"]:
        raise ValueError("Teacher merger token count does not match third-view metadata")
    if features.shape[1:] != teacher.shape[1:]:
        raise ValueError("second/third merger feature dimensions must match")
    groups = {}
    for sample, (second_start, teacher_start, count) in enumerate(layout["pairs"]):
        groups.setdefault(count, []).append((sample, second_start, teacher_start))
    shape = (len(layout["pairs"]), THIRD_QUERY_COUNT, features.shape[-1])
    pooled_second = features.new_empty(shape, dtype=torch.float32)
    pooled_teacher = features.new_empty(shape, dtype=torch.float32)
    for count, rows in groups.items():
        second = torch.stack([features[start : start + count] for _, start, _ in rows]).float()
        third = torch.stack([teacher[start : start + count] for _, _, start in rows]).float()
        second = F.adaptive_avg_pool1d(second.transpose(1, 2), THIRD_QUERY_COUNT).transpose(1, 2)
        third = F.adaptive_avg_pool1d(third.transpose(1, 2), THIRD_QUERY_COUNT).transpose(1, 2)
        samples = [sample for sample, _, _ in rows]
        pooled_second[samples] = second
        pooled_teacher[samples] = third
    return {
        "third_prediction_target": (
            pooled_teacher if prediction_target == "third" else pooled_teacher - pooled_second
        ),
        "third_teacher_target": pooled_teacher,
        "third_second_target": pooled_second,
    }


class ThirdTransformerHead(nn.Module):
    """The experiment's full self-attention block followed by ego cross-attention."""

    def __init__(self, dimension: int = 2048):
        super().__init__()
        self.dimension = dimension
        width = 256
        self.second_projection = nn.Linear(dimension, width)
        self.ego_projection = nn.Linear(dimension, width)
        self.self_norm = nn.RMSNorm(width, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.self_ffn_norm = nn.RMSNorm(width, eps=1e-6)
        self.self_ffn = nn.Sequential(nn.Linear(width, 1024), nn.GELU(), nn.Linear(1024, width))
        self.cross_query_norm = nn.RMSNorm(width, eps=1e-6)
        self.cross_context_norm = nn.RMSNorm(width, eps=1e-6)
        self.cross_attention = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.cross_ffn_norm = nn.RMSNorm(width, eps=1e-6)
        self.cross_ffn = nn.Sequential(nn.Linear(width, 1024), nn.GELU(), nn.Linear(1024, width))
        self.output_norm = nn.RMSNorm(width, eps=1e-6)
        self.output_projection = nn.Linear(width, dimension)

    def reset_parameters(self) -> None:
        """Fully initialize a head omitted by a base-model checkpoint."""
        for child in self.children():
            if isinstance(child, nn.MultiheadAttention):
                # Match MultiheadAttention's constructor order, including its bias zeros.
                child.out_proj.reset_parameters()
                child._reset_parameters()
            elif isinstance(child, nn.Sequential):
                for module in child:
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()
            elif hasattr(child, "reset_parameters"):
                child.reset_parameters()

    def forward(self, ego: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if ego.ndim != 3 or second.ndim != 3:
            raise ValueError("Both ego and second must retain [frames,tokens,dimension] features")
        if (
            ego.shape[0] != second.shape[0]
            or ego.shape[-1] != self.dimension
            or second.shape[-1] != self.dimension
        ):
            raise ValueError("Ego and second must have matching frame count and feature dimension")
        x, context = self.second_projection(second), self.ego_projection(ego)
        normalized = self.self_norm(x)
        x = x + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        x = x + self.self_ffn(self.self_ffn_norm(x))
        query, context = self.cross_query_norm(x), self.cross_context_norm(context)
        x = x + self.cross_attention(query, context, context, need_weights=False)[0]
        x = x + self.cross_ffn(self.cross_ffn_norm(x))
        return self.output_projection(self.output_norm(x))


def build_third_head(framework: str, dimension: int) -> nn.Module:
    if framework == "transformer":
        return ThirdTransformerHead(dimension)
    if framework != "mlp":
        raise ValueError(f"Unsupported visual third_framework: {framework}")
    return nn.Sequential(
        nn.Linear(2 * dimension, 1024),
        nn.GELU(),
        nn.Linear(1024, 1024),
        nn.GELU(),
        nn.Linear(1024, dimension),
    )


def compensate_visual_tokens(
    features: torch.Tensor,
    grid: torch.Tensor,
    image_map: torch.Tensor,
    head: nn.Module,
    merge_size: int,
    teacher: torch.Tensor | None = None,
    teacher_grid: torch.Tensor | None = None,
    teacher_map: torch.Tensor | None = None,
    framework: str = "mlp",
    prediction_target: str = "third",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Predict third/delta and replace second merger tokens with reconstructed third."""
    if framework not in ("mlp", "transformer"):
        raise ValueError("Visual third framework must be mlp or transformer")
    if prediction_target not in ("third", "delta"):
        raise ValueError("third_prediction_target must be third or delta")
    counts = merged_token_counts(grid, merge_size)
    if image_map.shape != (len(counts), 3) or sum(counts) != features.shape[0]:
        raise ValueError("third_image_map must match every main image/merger token")
    rows = image_map.tolist()
    indices = {}
    for index, (sample, time, view) in enumerate(rows):
        if sample < 0 or time < 0 or view not in (0, 1) or (sample, time, view) in indices:
            raise ValueError("Invalid or duplicate third_image_map entry")
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
            raise ValueError("third_teacher_map must match every teacher image/merger token")
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
    groups = {}
    for sample, time in sorted(pairs):
        ego_index = indices[sample, time, 0]
        second_index = indices[sample, time, 1]
        teacher_index = teacher_indices.get((sample, time))
        if teacher_chunks is not None:
            if not torch.equal(grid[second_index], teacher_grid[teacher_index]):
                raise ValueError("second/third image grids must match at each observation time")
        geometry = (counts[ego_index], counts[second_index])
        groups.setdefault(geometry, []).append((ego_index, second_index, teacher_index))
    predictions, targets, seconds, teachers = [], [], [], []
    for rows in groups.values():
        ego = torch.stack([chunks[index] for index, _, _ in rows])
        second = torch.stack([chunks[index] for _, index, _ in rows])
        if framework == "mlp":
            pooled_ego = ego.mean(dim=1, keepdim=True).expand(-1, second.shape[1], -1)
            prediction = head(torch.cat((pooled_ego, second), dim=-1))
        else:
            prediction = head(ego, second)
        reconstructed = prediction if prediction_target == "third" else second + prediction
        for row, (_, second_index, _) in enumerate(rows):
            # The LLM scatter requires the original image-embedding dtype.
            compensated[second_index] = reconstructed[row].to(second.dtype)
        if teacher_chunks is not None:
            teacher_batch = torch.stack([teacher_chunks[index] for _, _, index in rows])
            target = (
                teacher_batch if prediction_target == "third" else teacher_batch - second.detach()
            )
            predictions.append(prediction.flatten(0, 1))
            targets.append(target.detach().flatten(0, 1))
            seconds.append(second.detach().flatten(0, 1))
            teachers.append(teacher_batch.flatten(0, 1))
    metrics = {}
    if predictions:
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        # Preserve feature dtypes; let the enclosing autocast policy handle MSE.
        loss = F.mse_loss(prediction, target)
        metrics = third_prediction_metrics(
            prediction,
            target,
            torch.cat(seconds),
            torch.cat(teachers),
            prediction_target,
        )
        metrics["third_loss"] = loss
        metrics["third_valid_tokens"] = loss.new_tensor(target.shape[0])
    return torch.cat(compensated), metrics


@torch.no_grad()
def third_prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    second: torch.Tensor,
    teacher: torch.Tensor,
    prediction_target: str,
) -> dict[str, torch.Tensor]:
    """Report both the selected objective and reconstructed third in raw feature units."""
    prediction, target = prediction.float(), target.float()
    second, teacher = second.float(), teacher.float()
    reconstructed = prediction if prediction_target == "third" else second + prediction
    mse = F.mse_loss(prediction, target)
    reconstructed_mse = F.mse_loss(reconstructed, teacher)
    return {
        "third_mse": mse,
        "third_cosine_similarity": F.cosine_similarity(prediction, target, dim=-1).mean(),
        "third_relative_zero_mse": mse / target.square().mean().clamp_min(1e-12),
        "third_reconstructed_mse": reconstructed_mse,
        "third_reconstructed_cosine_similarity": F.cosine_similarity(
            reconstructed, teacher, dim=-1
        ).mean(),
        "third_relative_second_mse": reconstructed_mse
        / (teacher - second).square().mean().clamp_min(1e-12),
    }
