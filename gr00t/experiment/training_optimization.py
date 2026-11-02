"""Training-only storage optimization shared by all loss configurations."""

import logging

import torch


def keep_fused_adamw_on_load(optimizer, state_dict):
    """Retain the selected fused backend when loading older AdamW checkpoints.

    Optimizer state includes execution flags as well as moments/LRs. Updating
    flags before load also lets PyTorch place the step counters on the correct
    device for fused AdamW. Do not copy the potentially large state tensors.
    """
    return {
        **state_dict,
        "param_groups": [
            {**group, "fused": True, "foreach": False} for group in state_dict["param_groups"]
        ],
    }


def cache_frozen_linear_weights(model: torch.nn.Module) -> None:
    """Store frozen Linear parameters in the BF16 format consumed by autocast.

    Call after loading weights and setting requires_grad, before constructing the
    optimizer/distributed engine. Trainable parameters, norms, and embeddings
    retain their existing dtype. This does not change activation gradient paths.
    Runtime unfreezing requires explicitly restoring dtype and optimizer groups.
    """
    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding)
        for parameter in module.parameters()
    }
    count = saved_bytes = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        for parameter in module.parameters(recurse=False):
            if (
                parameter.requires_grad
                or id(parameter) in embedding_ids
                or parameter.dtype != torch.float32
            ):
                continue
            if parameter.is_meta:
                raise ValueError("Load model weights before caching frozen Linear parameters")
            count += parameter.numel()
            saved_bytes += parameter.numel() * 2
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    logging.info(
        "Frozen Linear BF16 storage: %d parameters, %.2f GiB less parameter storage; "
        "trainable parameters unchanged",
        count,
        saved_bytes / 2**30,
    )
