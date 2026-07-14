"""Runtime patch: chunk the FeedForward over the token dim during sampling.

At ~60k tokens x hidden=16384, the FFN intermediate alone is ~3.7 GB per block and is the
dominant transient activation -- not a cosmetic VRAM optimization but the difference between
a sample fitting in dedicated VRAM and silently spilling into Windows' shared-system-memory
fallback (PCIe-bandwidth-bound, effectively a slow-motion OOM that doesn't raise). ComfyUI's
`LTXVChunkFeedForward` and the community LTX-2 VRAM project both attack this exact tensor;
chunking it along the token dim drops per-block FFN peak by `num_chunks`x at the cost of
running the FFN as N smaller calls (same total FLOPs, same output).

LTX-2 transformer blocks call ``self.ff(x)`` and ``self.audio_ff(x)`` directly with no
chunking machinery. This module provides a context manager that, for the duration of a
sampling pass, replaces each block's ``ff``/``audio_ff`` forward with a chunked variant.
Restored on context exit so training is untouched.
"""

from contextlib import contextmanager

import torch


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
