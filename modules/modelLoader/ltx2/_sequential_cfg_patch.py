"""Runtime patch: split batched-CFG transformer calls into two sequential passes.

The diffusers LTX2 pipeline runs CFG by concatenating uncond+cond into a single
batch=2 transformer forward (``pipeline_ltx2.py:1213-1246``). At high resolution
(e.g. 1920x1080 x 241 frames, ~60k tokens) every transient activation buffer in
the transformer is doubled, which drives sampling peak VRAM well above ComfyUI's
sequential-CFG footprint at the same shape.

This module wraps ``LTX2VideoTransformer3DModel.forward`` so that, when a flag
is set on the instance, a batch=2 call is split into two sequential batch=1
forwards (uncond first, cond second, matching the pipeline's downstream
``chunk(2)`` order). Activations from the first pass free before the second
runs. Engaged only via the ``sequential_cfg`` context manager so training and
non-CFG sampling paths are untouched.
"""

from contextlib import contextmanager

import torch

from diffusers import LTX2VideoTransformer3DModel


_PATCH_ATTR = "_ot_sequential_cfg_patch_applied"
_FLAG_ATTR = "_ot_sequential_cfg"

# Forward kwargs that carry a leading batch dimension and must be split.
_BATCH_KWARGS = (
    "hidden_states",
    "audio_hidden_states",
    "encoder_hidden_states",
    "audio_encoder_hidden_states",
    "timestep",
    "audio_timestep",
    "sigma",
    "audio_sigma",
    "encoder_attention_mask",
    "audio_encoder_attention_mask",
    "video_coords",
    "audio_coords",
    "perturbation_mask",
)


def _split_half(t: torch.Tensor | None, half: int):
    if t is None:
        return None, None
    if t.shape[0] != 2 * half:
        # Not a CFG-doubled tensor (e.g. broadcasted scalar timestep); pass through.
        return t, t
    return t[:half], t[half:]


def _sequential_cfg_forward(self, *args, **kwargs):
    if not getattr(self, _FLAG_ATTR, False):
        return _original_forward(self, *args, **kwargs)

    # Normalize to all-kwargs so we can split by name. The first positional, if
    # any, is hidden_states per the forward signature.
    if args:
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = args[0]
        args = args[1:]
        # The forward signature only takes hidden_states positionally in practice
        # (the pipeline calls it all-kwargs), but guard against extras.
        if args:
            return _original_forward(self, *args, **kwargs)

    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None or hidden_states.shape[0] != 2:
        return _original_forward(self, **kwargs)

    # return_dict=True would require reconstructing an output object; the
    # pipeline always passes False, so fall through if someone asks for True.
    if kwargs.get("return_dict", True):
        return _original_forward(self, **kwargs)

    half = 1
    uncond_kwargs = dict(kwargs)
    cond_kwargs = dict(kwargs)
    for name in _BATCH_KWARGS:
        if name in kwargs:
            u, c = _split_half(kwargs[name], half)
            uncond_kwargs[name] = u
            cond_kwargs[name] = c

    out_uncond = _original_forward(self, **uncond_kwargs)
    out_cond = _original_forward(self, **cond_kwargs)

    # Both are tuples (return_dict=False). Concatenate each tensor element along
    # the batch dim so the pipeline's downstream chunk(2) sees the same layout
    # it would have from a single batched call.
    if isinstance(out_uncond, tuple):
        return tuple(
            torch.cat([u, c], dim=0) if (u is not None and c is not None) else (u if u is not None else c)
            for u, c in zip(out_uncond, out_cond, strict=True)
        )
    return torch.cat([out_uncond, out_cond], dim=0)


@contextmanager
def sequential_cfg(transformer):
    """Engage sequential-CFG splitting on ``transformer`` for the duration of the block."""
    prev = getattr(transformer, _FLAG_ATTR, False)
    setattr(transformer, _FLAG_ATTR, True)
    try:
        yield
    finally:
        setattr(transformer, _FLAG_ATTR, prev)


if not getattr(LTX2VideoTransformer3DModel, _PATCH_ATTR, False):
    _original_forward = LTX2VideoTransformer3DModel.forward
    LTX2VideoTransformer3DModel.forward = _sequential_cfg_forward
    setattr(LTX2VideoTransformer3DModel, _PATCH_ATTR, True)
