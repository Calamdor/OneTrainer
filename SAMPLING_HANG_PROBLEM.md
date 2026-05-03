# LTX-2.3 Sampling Loop Hang — Problem Statement for Analysis

## Context

OneTrainer (OT) is a model training tool that also runs inference ("sampling") to preview training
progress. This document describes a sampling hang/failure when sampling LTX-2.3 at
1920×1088×121 frames on an RTX 5090 (32 GB VRAM) with a GGUF-quantized transformer and a
distilled LoRA companion file.

The same generation works in ComfyUI on a 4080 Super (16 GB VRAM). OT hangs at `0%|` — tqdm
shows the denoising loop has started but never completes step 0.

---

## System

- **GPU**: RTX 5090, 32 GB VRAM (shows as 34.19 GB total in PyTorch due to virtual addressing)
- **OS**: Windows 11 — Task Manager shows pinned CPU memory as "Shared VRAM"
- **Resolution**: 1920×1088×121 frames (full 1080p, ~5 seconds at 24fps)

---

## Model Components

| Component | Format | Size | Device during diffusion |
|-----------|--------|------|-------------------------|
| Transformer | GGUF Q4_K_M (or similar) | ~22 GB on CPU pinned | CPU pinned buffer, 1 block at a time in GPU |
| Distilled LoRA | BF16 safetensors | ~9.56 GB | GPU throughout denoising |
| Connectors (LTX2TextConnectors) | BF16 | ~500 MB | GPU |
| Text encoder (Gemma3-12B) | BF16 | ~24 GB | CPU (offloaded after encoding) |
| VAE | FP32 | ~1 GB | CPU (used only after denoising) |

---

## Observed VRAM Log (with `LTX2_VRAM_DEBUG=1`)

```
[Ltx2 LoRA] 1660 pairs loaded from ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors
[Ltx2 VRAM] entry: alloc=0.00 peak=2.68 reserved=0.04 free/total=32.33/34.19 GB
[Ltx2 VRAM] after TE→GPU: alloc=23.63 peak=23.63 reserved=23.89 free/total=8.47/34.19 GB
[Ltx2 VRAM] after prompt encode: alloc=24.41 peak=24.80 reserved=25.52 free/total=0.00/34.19 GB
[Ltx2 VRAM] after TE→CPU + gc: alloc=0.78 peak=24.41 reserved=0.80 free/total=31.56/34.19 GB
[Ltx2 VRAM] after diffusion components→GPU: alloc=11.93 peak=11.93 reserved=11.95 free/total=20.41/34.19 GB
[Ltx2 LoRA] applying 1660 distilled LoRA patches (strength=0.3)
  0%|           ← hangs here, never advances
```

The `free=20.41 GB` reading before the pipeline starts suggests VRAM is NOT the bottleneck. The
hang occurs during the first denoising step (step 0).

**Breakdown of the 11.93 GB dedicated VRAM before pipeline:**
- Distilled LoRA BF16 tensors: ~9.56 GB (1660 pairs, avg rank 105)
- Conductor GPU static buffer (2 blocks × ~416 MB): ~832 MB
- Non-block transformer params (time_embed, scale_shift, etc.): ~670 MB
- Connectors: ~500 MB
- Total: ~11.56 GB ≈ 11.93 GB ✓

The GGUF transformer weights (~22 GB) are in CPU **pinned** memory (shows in Task Manager
"Shared VRAM", but NOT in `torch.cuda.memory_allocated()`).

---

## Architecture: LayerOffloadConductor

OT's layer offload system (`modules/util/LayerOffloadConductor.py`) works as follows:

1. **Setup** (`conductor.to(train_device)`): Iterates all 48 transformer blocks.
   - Temporarily loads each block to GPU (to establish dtype/size).
   - Calls `offload_quantized(module, cpu, allocator=pinned_allocator.allocate_like)` — moves
     each block's weights into a pre-allocated **pinned CPU buffer** (via `tensor.data =
     new_tensor`). The pinned buffer is the source of the 22 GB "Shared VRAM" in Task Manager.
   - Calls `torch.cuda.empty_cache()` after each block.
   - Also calls `__module_to_device_except_layers()` — moves all non-block transformer
     parameters (time_embed, scale_shift_table, av_cross_attn modules, etc.) to GPU permanently.
     These ~670 MB stay on GPU throughout.

2. **GPU static buffer**: Pre-allocated `torch.zeros(..., dtype=int8, device=cuda)` at size =
   max(2 blocks × sizeof_block). Sized for the rolling window. **Not a copy of the weights — a
   reusable scratchpad.**

3. **Per-step inference** (`OffloadCheckpointLayer.forward`, else branch — `torch.is_grad_enabled()=False`):
   ```
   block 0: start_forward(False) → empty_cache()
            before_layer(0) → copy block 0 weights from CPU pinned → GPU static buffer
            LTX2VideoTransformerBlock.forward(...)
            after_layer(0)
   block 1: before_layer(1) → schedule block 2 load + block 0 offload (async if enabled)
            LTX2VideoTransformerBlock.forward(...)
            after_layer(1)
   ...
   block N: ...
   ```

4. **`offload_quantized`** (`modules/util/quantization_util.py:308`):
   ```python
   for tensor in get_offload_tensors(module):
       new_tensor = allocator(tensor)   # allocate slice of static buffer
       new_tensor.copy_(tensor.data)   # copy from pinned CPU
       tensor.data = new_tensor        # in-place rebind
   ```
   For GGUF weights, `tensor` is a `GGUFParameter`. The `quant_type` Python attribute is
   preserved through `.data` rebinding because `GGUFParameter.__torch_function__` propagates it.

---

## Architecture: Distilled LoRA Patching

The distilled LoRA (1660 paired down/up matrices, avg rank 105, BF16) is NOT a proper LoRA
adapter in the diffusers sense. It is applied via **forward method patching** on individual
`nn.Linear` modules inside the transformer blocks.

### Load time (`_apply_distilled_lora`, `BaseLtx2Setup.py:452`)

```python
handles.append((target_linear, target_linear.forward, {"down": d, "up": u}))
```

- `target_linear` is a `nn.Linear` (or `LinearGGUFA8`/`GGUFLinear`) inside a transformer block.
- `target_linear.forward` is the original CLASS method bound to `target_linear`.
- `d`, `u` are detached BF16 tensors (initially on CPU).

**Order during setup**:
1. `setup_optimizations()` — wraps each `LTX2VideoTransformerBlock` with `OffloadCheckpointLayer`:
   ```python
   # non-compile path (compile=False forced for LTX2):
   layer = OffloadCheckpointLayer(orig_module=None, orig_forward=block.forward, ...)
   block.forward = layer.forward   # patches the BLOCK's forward
   ```
2. `setup_model()` — calls `_apply_distilled_lora()`. The LoRA traversal navigates the transformer's
   submodule tree to reach individual linears. It does NOT go through `.forward` — it uses
   `getattr(block, "attn")` etc., accessing PyTorch's `_modules` dict. The traversal handles
   `OffloadCheckpointLayer` by checking for `.checkpoint` attribute (only relevant in compile mode).

So at LoRA load time, `target_linear.forward` = the linear's original method (unpatched). The
block's `.forward` is already patched to `OffloadCheckpointLayer.forward`, but this doesn't
affect the linear's own `.forward`.

### Sample time — resume (`_resume_distilled_lora_hooks`, `Ltx2Model.py:121`)

```python
for handle in self.distilled_lora_handles:
    module, payload = handle[0], handle[2]
    current_fwd = module.forward            # snapshot current forward
    payload["_pre_resume_fwd"] = current_fwd
    
    def _make_patched(base, _d, _u, _s):
        def patched(x):
            return base(x) + F.linear(
                F.linear(x, _d.to(x.device, x.dtype)),
                _u.to(x.device, x.dtype),
            ) * _s
        return patched
    
    module.forward = _make_patched(current_fwd, d, u, strength)
```

**Result**: `module.forward` (an INSTANCE attribute) is set to `patched`. When
`LTX2VideoTransformerBlock.forward` calls `self.attn.to_q(x)`, PyTorch's `__call__` invokes
`to_q.forward = patched`. Inside `patched`, `base(x)` calls the ORIGINAL method (bound to the
module instance).

### Sample time — order in `__sample_base` (`Ltx2Sampler.py:394`)

```python
# 1. Text encode, then offload TE
self.model.text_encoder_to(train_device)
...encode...
self.model.text_encoder_to(temp_device); torch_gc()

# 2. Load diffusion components
self.model.connectors_to(train_device)
self.model.transformer_to(train_device)      # ← conductor.to(train_device)
self.model.distilled_lora_to(train_device)   # ← BF16 LoRA d/u tensors to GPU
torch_gc()
# alloc=11.93 GB, free=20.41 GB

# 3. Run pipeline
pipeline = self.model.create_pipeline()
self.model._resume_distilled_lora_hooks()    # ← patch linears' .forward
video_latents, audio_latents = pipeline(...)  # ← HANGS AT 0%
```

---

## Potential Conflict Vectors

### Vector 1: Static buffer ownership collision

The conductor's GPU static buffer is a pre-allocated `int8` tensor that is sliced and assigned to
linear weights via `tensor.data = new_tensor` during `before_layer`. The weight's `.data` now
points INTO the static buffer.

The distilled LoRA's `patched` forward does:
```python
return base(x) + F.linear(F.linear(x, _d), _u)
```

`base(x)` calls `LinearGGUFA8.forward` (or `GGUFLinear.forward`), which reads `self.weight`.
At this point, `self.weight.data` is the slice of the GPU static buffer. This is fine — the
weight is readable.

**BUT**: is there a race or aliasing issue where the allocator `allocate_like` creates a new
tensor that ALIASES memory with the LoRA weight tensors (`_d`, `_u`)? The LoRA weights were
moved to GPU via `distilled_lora_to(train_device)` which uses `t.data = t.data.to(device)` — a
plain `torch.Tensor.to()` call that allocates fresh CUDA memory from the CUDA caching allocator.

The static buffer is a separate pre-allocated `torch.zeros(..., dtype=int8, device=cuda)` — its
memory should NOT alias the LoRA tensors. But if `torch.cuda.empty_cache()` was called between
`distilled_lora_to` and `_resume_distilled_lora_hooks`, could the allocator reuse the memory? No
— the LoRA tensors are live (referenced by `payload["down"]` etc.).

### Vector 2: GGUFParameter + .data rebinding + dequantize chain

The GGUF transformer's linear weights are `GGUFParameter` instances (from diffusers). They carry
a Python-level `quant_type` attribute that `dequantize_gguf_tensor` checks.

After `offload_quantized` moves the weight from CPU pinned → GPU static buffer:
```python
tensor.data = new_tensor   # new_tensor is a uint8 slice of the GPU static buffer
```

`GGUFParameter.__torch_function__` preserves `quant_type` through `.data` assignment. So
`dequantize_gguf_tensor(self.weight.detach())` should still work.

`LinearGGUFA8.forward` checks:
```python
if x.shape[0] > 16 and hasattr(self.weight, 'quant_type') and self.weight.quant_type not in UNQUANTIZED_TYPES:
```

When the weight's `.data` is the static buffer's uint8 tensor, does `.detach()` on the
`GGUFParameter` still return something with `quant_type`? `.detach()` creates a new tensor that
shares storage but has no grad_fn. `GGUFParameter` inherits from `nn.Parameter` which inherits
from `torch.Tensor`. The `__torch_function__` override propagates `quant_type` through
`torch.Tensor.detach()`... or does it?

Looking at diffusers `GGUFParameter.__torch_function__` — it intercepts calls and returns
`GGUFParameter` instances with `quant_type` set for many operations. If `detach()` is NOT in
its handled operations, it returns a plain `torch.Tensor` WITHOUT `quant_type`. Then `hasattr`
returns False, the `if` branch is NOT taken, and `F.linear(x, w, bias)` is called with the raw
uint8 dequantized float tensor from `dequantize_gguf_tensor`. This path was explicitly added as
a guard — so this specific case is HANDLED.

### Vector 3: OffloadCheckpointLayer wraps block forward AFTER LoRA is set up — NO, BEFORE

Confirmed: `setup_optimizations` (block wrapping) runs before `setup_model` (LoRA setup). LoRA
handles store `target_linear` (the actual module reference) and `target_linear.forward` (the
original method) at LoRA-load time. When the block's `.forward` is patched to
`OffloadCheckpointLayer.forward`, the linears inside the block are NOT affected. The LoRA
traversal reaches linears via `getattr()` on submodules, not via `.forward`.

### Vector 4: `patched` function calls `base(x)` — does `base` reference the CONDUCTOR-WRAPPED forward?

At load time (in `_apply_distilled_lora`):
```python
handles.append((target, target.forward, ...))
```
`target.forward` for a linear is the class method. Not the OffloadCheckpointLayer wrapper (which
is on the BLOCK, not the LINEAR).

At resume time:
```python
current_fwd = module.forward
```
`module` is the linear. Its `.forward` at this point:
- First resume: still the class method (if nothing else patched it)
- Subsequent resumes: the bound method set by `_pause` which restores `pre_resume_fwd`

No reference to OffloadCheckpointLayer. ✓

### Vector 5: `patched` forward signature vs how the linear is called

```python
def patched(x):
    return base(x) + F.linear(F.linear(x, _d.to(...)), _u.to(...)) * _s
```

`patched` accepts only ONE positional argument `x`. `nn.Module.__call__` calls `self.forward(*input,
**kwargs)` where `input` is the positional args tuple. For a linear, the call is always
`linear(x)` — single tensor. This is fine.

But `base(x)` calls `LinearGGUFA8.forward(x)` or `GGUFLinear.forward(x)`. Inside
`LinearGGUFA8.forward`:
```python
def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
    x = x_orig.reshape(-1, x_orig.shape[-1])
    w = dequantize_gguf_tensor(self.weight.detach())
    ...
    return y.reshape(x_orig.shape[:-1] + (y.shape[-1],))
```

`base` is the BOUND METHOD captured when `_apply_distilled_lora` ran. When we call `base(x)`,
Python calls `LinearGGUFA8.forward(self=target_linear, x_orig=x)`. The method accesses
`self.weight` on the module instance — the same reference that the conductor updates in-place.

The OUTPUT of `base(x)` has shape `x_orig.shape[:-1] + (out_features,)`. Then:
- `F.linear(x, _d)` where `_d.shape = (rank, in_features)` → shape `x.shape[:-1] + (rank,)`
- `F.linear(..., _u)` where `_u.shape = (out_features, rank)` → shape `x.shape[:-1] + (out_features,)`

Both outputs have the same leading dimensions → addition works. ✓

### Vector 6: Memory layout of the GPU static buffer vs what dequantize expects

`offload_quantized` with the static allocator does:
```python
new_tensor = allocator(tensor)   # allocate_like: creates uint8 tensor of same numel×element_size
new_tensor.copy_(tensor.data)    # copy from CPU pinned to GPU
tensor.data = new_tensor
```

`allocate_like` creates a **uint8 view** of the static buffer, sized to `tensor.numel() *
tensor.element_size()` bytes — regardless of original dtype. Then `.copy_` copies raw bytes. For
a GGUF quantized weight (which was already stored as uint8 `GGUFParameter`), this is a raw byte
copy. The `quant_type` attribute is preserved on the Python object.

`dequantize_gguf_tensor` then calls `tensor.as_tensor()` (GGUF lookup-table dequant) on the
GPU uint8 data. This should produce correct float values.

BUT: `StaticLayerAllocator.allocate_like` returns a plain `torch.Tensor` (uint8 view), NOT a
`GGUFParameter`. When `tensor.data = new_tensor` is called, the PARAMETER OBJECT remains a
`GGUFParameter` (the Python wrapper isn't replaced), but its `.data` now points to a plain
`torch.Tensor`. The `GGUFParameter.quant_type` attribute is on the Python wrapper, not on
`.data`. So `hasattr(weight, 'quant_type')` → True (Python attribute on the wrapper). ✓

BUT: does `weight.detach()` return a `GGUFParameter` with `quant_type` set? `detach()` is a
torch operation. `GGUFParameter.__torch_function__` may or may not intercept it. If it does NOT,
`detach()` returns a plain `torch.Tensor` → `hasattr(result, 'quant_type')` → False.

In `LinearGGUFA8.forward`:
```python
w = dequantize_gguf_tensor(self.weight.detach())
```

If `self.weight.detach()` returns a plain tensor without `quant_type`, then
`dequantize_gguf_tensor` checks `hasattr(tensor, 'quant_type')` → False → falls through to
return the tensor as-is (raw uint8 data). Then `F.linear(x, w, bias)` tries to do a matmul with
a uint8 weight → **RuntimeError: expected scalar type Float but found Byte** or similar.

This path was previously guarded with `hasattr(self.weight, 'quant_type')` (added in a prior
fix). The new guard is on `self.weight` (the parameter), not `self.weight.detach()`. So the
outer if-check passes, but the inner `dequantize_gguf_tensor` gets a plain tensor from `.detach()`.

**This is the most likely crash vector.**

---

## The Critical Question

Does `GGUFParameter.detach()` return a `GGUFParameter` with `quant_type`, or a plain `torch.Tensor`?

Relevant file: `venv/src/diffusers/src/diffusers/quantizers/gguf/utils.py`

The `GGUFParameter.__torch_function__` typically handles `.to()`, `.cuda()`, `.cpu()` by
returning a `GGUFParameter` with the attribute preserved. But `.detach()` is often NOT in the
handled list for custom `__torch_function__` implementations — it returns `NotImplemented` or a
plain tensor.

If `.detach()` returns a plain tensor WITHOUT `quant_type`, then when the conductor has moved the
weight to the GPU static buffer (making `.data` a plain uint8 tensor), `dequantize_gguf_tensor`
receives raw bytes and cannot dequantize → crash.

**Before the conductor is active**, the weight's `.data` is the original `GGUFParameter` storage
(which may have been loaded from the GGUF file in a way where `.detach()` was previously tested
to work). **After the conductor does `.data = new_tensor`**, the storage changes to a plain uint8
slice of the static buffer. The Python `GGUFParameter` wrapper still exists, but `.detach()` now
operates on different underlying storage.

---

## Secondary Issue: Text Encoder Fills All VRAM

```
after TE→GPU:     free=8.47 GB
after encode:     free=0.00 GB  ← 34.19 GB dedicated VRAM completely saturated
after TE→CPU+gc:  free=31.56 GB ← recovered
```

The full BF16 Gemma3-12B text encoder requires ~24 GB dedicated VRAM. On the 5090 this fits (just),
but saturates the entire VRAM pool. This works but creates VRAM fragmentation risk and leaves no
room for concurrent operations during encoding. Not the cause of the `0%` hang, but a risk factor.

---

## ComfyUI Reference

ComfyUI handles 1920×1088×121 with the same distilled LoRA on a 4080 Super (16 GB):

- **Total "VRAM"**: 29.4 GB (dedicated + pageable — pageable does NOT show as "Shared VRAM")
- **Architecture**: per-weight on-demand loading via `cast_bias_weight` / `LowVramPatch`
  callbacks. Each matmul: copy weight from PAGEABLE CPU RAM → GPU → dequantize → matmul →
  free GPU copy. No pinned buffer.
- **Distilled LoRA**: applied via ComfyUI's `ModelPatcher` which modifies the weight patch
  callback, so LoRA delta is FUSED into the same dequantize pass. No separate LoRA GPU tensors.
- **No forward-method patching**: LoRA weights are never stored separately in GPU memory;
  they're applied as a delta to the dequantized weight before each matmul.

---

## Key Files

```
modules/modelSampler/Ltx2Sampler.py          — sampling loop, _vram_log, pipeline call
modules/model/Ltx2Model.py                   — distilled LoRA handle management (lines 90-195)
modules/modelSetup/BaseLtx2Setup.py          — _apply_distilled_lora (line 452), setup order
modules/util/LayerOffloadConductor.py        — conductor, static buffer, before/after_layer
modules/util/checkpointing_util.py           — OffloadCheckpointLayer (line 132)
modules/util/quantization_util.py            — offload_quantized, get_offload_tensors (line 308)
modules/module/quantized/LinearGGUFA8.py     — GGUF+A8 quantized linear with LoRA compatibility
venv/src/diffusers/src/diffusers/quantizers/gguf/utils.py  — GGUFParameter, __torch_function__
```

---

## What We Need

1. **Identify the root cause** of the hang at `0%` — is it the `GGUFParameter.detach()` issue
   above, or a different interaction between the conductor's static buffer and the LoRA patching?

2. **Fix the conflict** between:
   - `LayerOffloadConductor` moving GGUF weights in-place via `tensor.data = new_tensor`
   - Distilled LoRA patching calling `base(x)` which calls `LinearGGUFA8.forward` which does
     `dequantize_gguf_tensor(self.weight.detach())`

3. **Optionally**: reduce the 9.56 GB LoRA footprint on GPU during denoising, as ComfyUI avoids
   this by fusing LoRA application into the dequantize pass rather than holding all LoRA weights
   in GPU simultaneously.

The fix should work without changing the conductor architecture (which works correctly for other
models like Flux.2 and Wan2.2).
