"""Runtime patch: chunk the FeedForward over the token dim during sampling.

At ~60k tokens × hidden=16384, the FFN intermediate alone is ~3.7 GB per block
and is the dominant transient activation. ComfyUI's `LTXVChunkFeedForward` and
the community LTX-2 VRAM project both attack this exact tensor; chunking it
along the token dim drops per-block FFN peak by `num_chunks`× at the cost of
running the FFN as N smaller calls (same total FLOPs).

LTX-2 transformer blocks call ``self.ff(x)`` and ``self.audio_ff(x)`` directly
with no chunking machinery. This module provides a context manager that, for
the duration of a sampling pass, replaces each block's ``ff`` / ``audio_ff``
forward with a chunked variant. Restored on context exit so training is
untouched.
"""

from contextlib import contextmanager

import torch

from diffusers.models import attention_dispatch


# Patch diffusers' sage backend to fall back to native SDPA when an attn_mask
# is present. SageAttention's CUDA INT8/FP8 kernels don't accept masks (the
# Triton FP16 variant does, but diffusers doesn't route to it), and diffusers
# raises ValueError on encountering one. In LTX-2.3 the only masked call is
# the connector's text self-attention (q_len=1024, mask=text padding) — tiny
# relative to the video self-attention (q_len=~60k, mask=None), so the mixed
# dispatch keeps the sage win where it matters.
_SAGE_FALLBACK_PATCH_ATTR = "_ot_sage_mask_fallback_applied"
if not getattr(attention_dispatch, _SAGE_FALLBACK_PATCH_ATTR, False):
    _orig_sage = attention_dispatch._sage_attention
    _native_fn = attention_dispatch._native_attention

    def _sage_or_native(*args, attn_mask=None, **kwargs):
        if attn_mask is not None:
            return _native_fn(*args, attn_mask=attn_mask, **kwargs)
        return _orig_sage(*args, attn_mask=None, **kwargs)

    attention_dispatch._sage_attention = _sage_or_native
    # The dispatcher registry captured the original by-value at import; update.
    from diffusers.models.attention_dispatch import (
        AttentionBackendName as _ABN,
        _AttentionBackendRegistry as _ABR,
    )
    if _ABN.SAGE in _ABR._backends:
        _ABR._backends[_ABN.SAGE] = _sage_or_native
    setattr(attention_dispatch, _SAGE_FALLBACK_PATCH_ATTR, True)


@contextmanager
def attention_backend(transformer, backend: str | None):
    """Temporarily set the diffusers attention backend on ``transformer``.

    No-ops if ``backend`` is ``None``. Reverts to whatever the transformer
    had previously on exit (typically ``"native"``). Sage attention may not
    support backward — wrapping per-sampling-pass keeps training paths on the
    default backend.
    """
    if backend is None:
        yield
        return
    # The first attention module's _attention_backend is representative; capture
    # it as the previous value to restore. set_attention_backend walks the tree.
    prev = None
    for m in transformer.modules():
        if hasattr(m, "_attention_backend"):
            prev = m._attention_backend
            break
    transformer.set_attention_backend(backend)
    try:
        yield
    finally:
        if prev is None:
            # Diffusers uses None as "use registry default"; setting "native"
            # is the explicit equivalent.
            transformer.set_attention_backend("native")
        else:
            transformer.set_attention_backend(prev)


def _make_chunked_forward(original_forward, chunk_size: int, chunk_dim: int = 1):
    """Build a closure suitable for assigning to ``module.forward`` (no self arg)."""
    def chunked_forward(hidden_states, *args, **kwargs):
        n = hidden_states.shape[chunk_dim]
        if n <= chunk_size:
            return original_forward(hidden_states, *args, **kwargs)
        outs = [
            original_forward(part, *args, **kwargs)
            for part in torch.split(hidden_states, chunk_size, dim=chunk_dim)
        ]
        return torch.cat(outs, dim=chunk_dim)
    return chunked_forward


@contextmanager
def chunked_ffn(transformer, chunk_size: int):
    """Patch every block's ``ff`` and ``audio_ff`` (if present) to chunk along token dim."""
    if chunk_size is None or chunk_size <= 0:
        yield
        return

    patched = []  # list of (module, original_forward) to restore
    for block in getattr(transformer, "transformer_blocks", []):
        for name in ("ff", "audio_ff"):
            ff = getattr(block, name, None)
            if ff is None:
                continue
            original = ff.forward
            ff.forward = _make_chunked_forward(original, chunk_size, 1)
            patched.append((ff, original))

    try:
        yield
    finally:
        for ff, original in patched:
            ff.forward = original
