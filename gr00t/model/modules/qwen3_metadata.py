"""CPU batch metadata for the image-only Qwen3-VL training path.

This object is deliberately not a Mapping: Trainer/DeepSpeed and dm-tree leave
its CPU metadata on the host when moving the actual batch to a device.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass(frozen=True)
class VisionLayout:
    grid: torch.Tensor
    key: tuple[tuple[int, int, int], ...]
    counts: tuple[int, ...]
    cu_seqlens: torch.Tensor
    max_seqlen: int

    @classmethod
    def from_grid(cls, grid, merge_size):
        if grid.device.type != "cpu":
            raise ValueError("Prepare Qwen metadata in the CPU collator, before device transfer")
        grid = grid.clone()
        if grid.ndim != 2 or grid.shape[1] != 3 or not len(grid) or (grid <= 0).any():
            raise ValueError("Invalid Qwen image grid")
        if (grid[:, 1:] % merge_size).any():
            raise ValueError("Image grid must be divisible by the spatial merge size")
        lengths = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0])
        return cls(
            grid=grid,
            key=tuple(tuple(row) for row in grid.tolist()),
            counts=tuple((grid.prod(-1) // merge_size**2).tolist()),
            cu_seqlens=torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0)),
            max_seqlen=int(lengths.max()),
        )


@dataclass(frozen=True)
class QwenBatchMetadata:
    main: VisionLayout
    teacher: VisionLayout | None
    position_ids: torch.Tensor
    token_config: tuple[int, int, int, int]
    image_map: torch.Tensor | None
    teacher_map: torch.Tensor | None


def prepare_qwen3_metadata(
    batch, merge_size, image_token_id, video_token_id, vision_start_token_id
):
    """Recompute text positions for every batch; never cache tokens or image features."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    ids, mask = batch["input_ids"], batch["attention_mask"]
    if ids.device.type != "cpu" or mask.device.type != "cpu":
        raise ValueError("Qwen position metadata requires CPU input IDs and attention mask")
    # The VLA backbone consumes image inputs only. Keep the original path for
    # video batches instead of applying image assumptions to them.
    if "video_grid_thw" in batch or (ids == video_token_id).any():
        return None
    main = VisionLayout.from_grid(batch["image_grid_thw"], merge_size)
    if int((ids == image_token_id).sum()) != sum(main.counts):
        raise ValueError("Image placeholder count does not match the processed image grids")
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=merge_size),
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
    )
    # Delegate the position convention to the pinned Transformers implementation.
    # It reads only config and the CPU tensors; no model or weights are created.
    positions, _ = Qwen3VLModel.get_rope_index(
        SimpleNamespace(config=config), ids, main.grid, attention_mask=mask
    )
    teacher = None
    if "third_teacher_grid_thw" in batch:
        teacher = VisionLayout.from_grid(batch["third_teacher_grid_thw"], merge_size)
    maps = []
    for key in ("third_image_map", "third_teacher_map"):
        value = batch.get(key)
        if value is not None:
            if value.device.type != "cpu":
                raise ValueError("Third-view alignment metadata must be prepared on CPU")
            value = value.clone()
        maps.append(value)
    return QwenBatchMetadata(
        main,
        teacher,
        positions,
        (merge_size, image_token_id, video_token_id, vision_start_token_id),
        *maps,
    )
