"""Runtime patch for diffusers' LTX2 single-file converter.

The pinned diffusers commit (see requirements-global.txt) lacks key renames for
``audio_prompt_adaln_single`` and ``prompt_adaln_single``. Without them,
``LTX2VideoTransformer3DModel.from_single_file(...)`` — used by the LTX2 GGUF
transformer load path — produces a state dict with unmapped names and fails to
load. We patch the converter at import time so a fresh clone of the repo works
without local edits to diffusers.

Apply by importing this module once before any LTX2 single-file load.
"""

from diffusers.loaders import single_file_model, single_file_utils

_PATCH_RENAMES = (
    # Order matters: the longer key contains the shorter as a substring, so
    # rename it first to avoid corrupting it on the second pass.
    ("audio_prompt_adaln_single", "audio_prompt_adaln"),
    ("prompt_adaln_single", "prompt_adaln"),
)

_PATCH_ATTR = "_ot_ltx2_adaln_patch_applied"


def _patched_convert_ltx2_transformer_to_diffusers(checkpoint, **kwargs):
    for old, new in _PATCH_RENAMES:
        for key in [k for k in checkpoint if old in k]:
            checkpoint[key.replace(old, new)] = checkpoint.pop(key)
    return _original_converter(checkpoint, **kwargs)


if not getattr(single_file_utils, _PATCH_ATTR, False):
    _original_converter = single_file_utils.convert_ltx2_transformer_to_diffusers
    single_file_utils.convert_ltx2_transformer_to_diffusers = (
        _patched_convert_ltx2_transformer_to_diffusers
    )
    # The original reference is captured by-value into SINGLE_FILE_LOADABLE_CLASSES
    # at diffusers import time, so patching the module attribute alone is not
    # enough — we have to override the dict entry too.
    _entry = single_file_model.SINGLE_FILE_LOADABLE_CLASSES.get(
        "LTX2VideoTransformer3DModel"
    )
    if _entry is not None and "checkpoint_mapping_fn" in _entry:
        _entry["checkpoint_mapping_fn"] = _patched_convert_ltx2_transformer_to_diffusers
    setattr(single_file_utils, _PATCH_ATTR, True)
