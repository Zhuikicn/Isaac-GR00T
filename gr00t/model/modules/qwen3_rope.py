"""Paired vision RoPE, preserving eager FP32 products and addition rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_pair_kernel(
    Q,
    K,
    COS,
    SIN,
    OQ,
    OK,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    d = i % (D // 2)
    h = (i // (D // 2)) % H
    t = i // (H * (D // 2))
    qbase = t * QS[0] + h * QS[1] + d * QS[2]
    kbase = t * KS[0] + h * KS[1] + d * KS[2]
    q0 = tl.load(Q + qbase, valid, 0).to(tl.float32)
    q1 = tl.load(Q + qbase + (D // 2) * QS[2], valid, 0).to(tl.float32)
    k0 = tl.load(K + kbase, valid, 0).to(tl.float32)
    k1 = tl.load(K + kbase + (D // 2) * KS[2], valid, 0).to(tl.float32)
    c0 = tl.load(COS + t * CS[0] + d * CS[1], valid, 0).to(tl.float32)
    c1 = tl.load(COS + t * CS[0] + (d + D // 2) * CS[1], valid, 0).to(tl.float32)
    s0 = tl.load(SIN + t * SS[0] + d * SS[1], valid, 0).to(tl.float32)
    s1 = tl.load(SIN + t * SS[0] + (d + D // 2) * SS[1], valid, 0).to(tl.float32)
    out = t * (H * D) + h * D + d
    tl.store(OQ + out, q0 * c0 + (-q1) * s0, valid)
    tl.store(OQ + out + D // 2, q1 * c1 + q0 * s1, valid)
    tl.store(OK + out, k0 * c0 + (-k1) * s0, valid)
    tl.store(OK + out + D // 2, k1 * c1 + k0 * s1, valid)


def vision_rope(q, k, cos, sin):
    out_q = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    out_k = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    elements = q.numel() // 2
    with torch.cuda.device(q.device):
        _rope_pair_kernel[(triton.cdiv(elements, 256),)](
            q,
            k,
            cos,
            sin,
            out_q,
            out_k,
            elements,
            q.shape[1],
            q.shape[2],
            q.stride(),
            k.stride(),
            cos.stride(),
            sin.stride(),
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out_q, out_k
