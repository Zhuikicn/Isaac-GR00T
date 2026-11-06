"""Shared Qwen3-VL training optimizations; checkpoint keys and parameters stay intact.

The instance adapters fall back to Transformers outside the prepared training
path. No global Transformers symbols or image-dependent features are cached.
"""

from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from types import MethodType

import torch
from torch.autograd.function import once_differentiable
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3VLModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLVisionAttention,
    Qwen3VLVisionModel,
    apply_rotary_pos_emb_vision,
)

from gr00t.model.modules.qwen3_metadata import QwenBatchMetadata


_GEOMETRY_CACHE_BYTES = 256 * 1024**2
_GEOMETRY_CACHE_ENTRIES = 4


def _clear_geometry(module, *_unused):
    module._gr00t_geometry_cache = OrderedDict()
    module._gr00t_geometry_signature = None


def _vision_apply(self, fn, recurse=True):
    # Invalidate even for a dtype/device round trip between two forwards, where
    # an allocator could reuse an earlier parameter address.
    _clear_geometry(self)
    return Qwen3VLVisionModel._apply(self, fn, recurse=recurse)


def _rms_backward(grad, x, reciprocal, normalized, weight):
    # Preserve the eager graph's BF16 multiplication/cast boundaries.
    g = (grad * weight).to(x.dtype).float()
    xf = x.float()
    direct = g * reciprocal
    dr = (g * xf).sum(-1, keepdim=True)
    dv = dr * (-0.5 * reciprocal.pow(3))
    indirect = (dv / x.shape[-1]) * (2.0 * xf)
    dx = (direct + indirect).to(x.dtype)
    dw = (grad * normalized).to(weight.dtype).reshape(-1, x.shape[-1]).sum(0)
    return dx, dw


@lru_cache(maxsize=1)
def _compiled_rms_backward():
    # Sequence/batch lengths vary during real training. Symbolic sizes avoid
    # compiling one backward graph per padded prompt length.
    return torch.compile(
        _rms_backward,
        fullgraph=True,
        dynamic=True,
        options={"emulate_precision_casts": True, "triton.cudagraphs": False},
    )


class _NativeRMSFusedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        xf = x.float()
        reciprocal = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        normalized = (xf * reciprocal).to(x.dtype)
        ctx.save_for_backward(x, reciprocal, normalized, weight)
        return weight * normalized

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        dx, dw = _compiled_rms_backward()(grad, *ctx.saved_tensors)
        return dx, dw, None


def _rms_forward(self, x):
    if (
        self._gr00t_training_enabled
        and torch.is_grad_enabled()
        and x.is_cuda
        and x.dtype == self.weight.dtype
        and x.dtype in (torch.bfloat16, torch.float32)
        and not torch.jit.is_tracing()
        and not torch.compiler.is_compiling()
    ):
        return _NativeRMSFusedBackward.apply(x, self.weight, self.variance_epsilon)
    return Qwen3VLTextRMSNorm.forward(self, x)


def _vision_rope(q, k, cos, sin):
    tensors = (q, k, cos, sin)
    if (
        all(t.is_cuda and t.device == q.device for t in tensors)
        and q.ndim == 3
        and k.shape == q.shape
        and q.shape[-1] % 2 == 0
        and q.numel() > 0
        and cos.shape == sin.shape == (q.shape[0], q.shape[-1])
        and all(t.dtype in (torch.bfloat16, torch.float16, torch.float32) for t in tensors)
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))
    ):
        from gr00t.model.modules.qwen3_rope import vision_rope

        return vision_rope(q, k, cos, sin)
    return apply_rotary_pos_emb_vision(q, k, cos, sin)


def _vision_attention(
    self,
    hidden_states,
    cu_seqlens,
    rotary_pos_emb=None,
    position_embeddings=None,
    vision_max_seqlen=None,
    **kwargs,
):
    if vision_max_seqlen is None or self.config._attn_implementation != "flash_attention_2":
        return Qwen3VLVisionAttention.forward(
            self, hidden_states, cu_seqlens, rotary_pos_emb, position_embeddings, **kwargs
        )
    # Transformers 4.57.3's FA2 branch, with a CPU-prepared maximum length and
    # paired RoPE. Attention scaling, layout, dropout and projection are native.
    seq_length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    q, k = _vision_rope(q, k, *position_embeddings)
    q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
    output, _ = ALL_ATTENTION_FUNCTIONS["flash_attention_2"](
        self,
        q,
        k,
        v,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=vision_max_seqlen,
        max_length_k=vision_max_seqlen,
        is_causal=False,
        **kwargs,
    )
    return self.proj(output.reshape(seq_length, -1).contiguous())


def _geometry(self, layout):
    weight, frequency = self.pos_embed.weight, self.rotary_pos_emb.inv_freq
    signature = tuple(
        (id(t), t.data_ptr(), t._version, t.dtype, t.device) for t in (weight, frequency)
    ) + (
        self.spatial_merge_size,
        self.num_grid_per_side,
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
    )
    if getattr(self, "_gr00t_geometry_signature", None) != signature:
        self._gr00t_geometry_cache = OrderedDict()
        self._gr00t_geometry_signature = signature
    cache = self._gr00t_geometry_cache
    if layout.key in cache:
        cache.move_to_end(layout.key)
        return cache[layout.key][0]
    with torch.no_grad():
        pos = self.fast_pos_embed_interpolate(layout.grid)
        rotary = self.rot_pos_emb(layout.grid)
        emb = torch.cat((rotary, rotary), dim=-1)
        value = (pos, (emb.cos(), emb.sin()), layout.cu_seqlens.to(weight.device))
    size = sum(t.numel() * t.element_size() for t in (value[0], *value[1], value[2]))
    if size <= _GEOMETRY_CACHE_BYTES:
        while cache and (
            len(cache) >= _GEOMETRY_CACHE_ENTRIES
            or sum(v[1] for v in cache.values()) + size > _GEOMETRY_CACHE_BYTES
        ):
            cache.popitem(last=False)
        cache[layout.key] = (value, size)
    return value


def _vision_forward(self, hidden_states, grid_thw, metadata=None, collect_deepstack=True, **kwargs):
    if metadata is None:
        return Qwen3VLVisionModel.forward(self, hidden_states, grid_thw, **kwargs)
    expected = sum(t * h * w for t, h, w in metadata.key)
    if hidden_states.shape[0] != expected:
        raise ValueError("Vision metadata does not match the current pixel batch")
    pos, position_embeddings, cu_seqlens = _geometry(self, metadata)
    hidden_states = self.patch_embed(hidden_states) + pos
    hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
    deepstack = []
    for index, block in enumerate(self.blocks):
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            vision_max_seqlen=metadata.max_seqlen,
            **kwargs,
        )
        if collect_deepstack and index in self.deepstack_visual_indexes:
            deepstack.append(
                self.deepstack_merger_list[self.deepstack_visual_indexes.index(index)](
                    hidden_states
                )
            )
    return self.merger(hidden_states), deepstack


def _image_features(self, pixel_values, image_grid_thw=None):
    metadata = getattr(self, "_gr00t_batch_metadata", None)
    if metadata is None:
        return Qwen3VLModel.get_image_features(self, pixel_values, image_grid_thw)
    features, deepstack = self.visual(
        pixel_values.type(self.visual.dtype), grid_thw=image_grid_thw, metadata=metadata.main
    )
    return features.split(metadata.main.counts), deepstack


def _placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
    metadata = getattr(self, "_gr00t_batch_metadata", None)
    if metadata is None or input_ids is None or video_features is not None:
        return Qwen3VLModel.get_placeholder_mask(
            self, input_ids, inputs_embeds, image_features, video_features
        )
    # Placeholder count was checked on CPU in the collator, for this batch.
    if image_features is not None and image_features.numel() != (
        sum(metadata.main.counts) * inputs_embeds.shape[-1]
    ):
        raise ValueError("Image features do not match CPU-validated placeholder count")
    image = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    video = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    return image, video


@contextmanager
def training_batch(model, inputs, enabled):
    metadata = inputs.get("qwen_metadata")
    eligible = (
        enabled
        and isinstance(metadata, QwenBatchMetadata)
        and inputs["input_ids"].is_cuda
        and not torch.jit.is_tracing()
        and not torch.compiler.is_compiling()
        and not any(p.requires_grad for p in model.visual.parameters())
    )
    if not eligible:
        metadata = None
        # Do not keep frozen weight-derived tensors across visual unfreezing.
        _clear_geometry(model.visual)
    else:
        config = model.config
        expected = (
            config.vision_config.spatial_merge_size,
            config.image_token_id,
            config.video_token_id,
            config.vision_start_token_id,
        )
        if metadata.token_config != expected:
            raise ValueError("Collator Qwen token/grid configuration differs from the model")
        if metadata.position_ids.shape != (3, *inputs["input_ids"].shape):
            raise ValueError("Qwen metadata does not match the current token batch")
    previous = getattr(model.model, "_gr00t_batch_metadata", None)
    model.model._gr00t_batch_metadata = metadata
    try:
        yield
    finally:
        model.model._gr00t_batch_metadata = previous


def install(model, enabled=True):
    """Install once on this instance, before constructing the distributed engine."""
    if not getattr(model, "_gr00t_training_installed", False):
        model.visual.forward = MethodType(_vision_forward, model.visual)
        model.visual._apply = MethodType(_vision_apply, model.visual)
        model.visual.register_load_state_dict_post_hook(_clear_geometry)
        for block in model.visual.blocks:
            block.attn.forward = MethodType(_vision_attention, block.attn)
        model.model.get_image_features = MethodType(_image_features, model.model)
        model.model.get_placeholder_mask = MethodType(_placeholder_mask, model.model)
        for module in model.language_model.modules():
            if isinstance(module, Qwen3VLTextRMSNorm):
                module.forward = MethodType(_rms_forward, module)
        model._gr00t_training_installed = True
    for module in model.language_model.modules():
        if isinstance(module, Qwen3VLTextRMSNorm):
            module._gr00t_training_enabled = enabled
    if not enabled:
        _clear_geometry(model.visual)
