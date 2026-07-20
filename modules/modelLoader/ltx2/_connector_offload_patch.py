"""Runtime helper: keep the text connectors off dedicated VRAM except for their one call.

The diffusers LTX2 pipeline calls ``self.connectors(...)`` exactly once per pipeline()
invocation, before the denoising loop starts, to project the text-encoder output into the
transformer's conditioning embeddings -- see pipeline_ltx2.py's single reference to
``self.connectors``. Everything after that (all denoising steps, both stages of a
two-stage sample) never touches it again.

The old branch's design (docs/LTX2.3_SPEC_PLAN.md: ``Ltx2ConnectorLoader.py`` /
``EncodeLtx2Connectors.py``, "cache-at-text-caching-time") ran connectors once during text
caching and never needed them GPU-resident during sampling at all. This module doesn't go
that far (it would require threading pre-computed embeddings through the pipeline call),
but gets the same VRAM benefit for sampling specifically: a forward pre-hook brings the
module to the training device right before its one call, and a forward hook sends it back
to the temp device immediately after -- so it only occupies dedicated VRAM for the single
moment it's actually needed, not for the whole denoising loop.
"""

from contextlib import contextmanager

import torch
from torch import nn


def _pre_hook(module: nn.Module, args, kwargs, *, train_device: torch.device):
    if next(module.parameters()).device != train_device:
        module.to(train_device)
    return args, kwargs


def _post_hook(module: nn.Module, args, output, *, temp_device: torch.device):
    module.to(temp_device)
    return output


@contextmanager
def connector_offload(connectors: nn.Module, train_device: torch.device, temp_device: torch.device):
    """Auto-offload ``connectors`` to ``temp_device`` immediately after each forward call.

    Engage for the duration of a sampling call so ``connectors`` only occupies dedicated
    VRAM for the instant it's actually invoked, instead of for the whole denoising loop.
    """
    pre_handle = connectors.register_forward_pre_hook(
        lambda module, args, kwargs: _pre_hook(module, args, kwargs, train_device=train_device),
        with_kwargs=True,
    )
    post_handle = connectors.register_forward_hook(
        lambda module, args, output: _post_hook(module, args, output, temp_device=temp_device),
    )
    try:
        yield
    finally:
        pre_handle.remove()
        post_handle.remove()
