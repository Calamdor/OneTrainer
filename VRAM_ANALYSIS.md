# What Is Kept in VRAM During LTX-2.3 Sampling — Analysis

## Updated Picture (after running with `LTX2_VRAM_DEBUG=1`)

The sampling **completed successfully** but at catastrophic speed:
- 20 steps in **13:08** (39.41 sec/step)
- ComfyUI does the same generation in a fraction of this time on a smaller GPU
- VRAM hits `free=0.00/34.19 GB` during diffusion → 100% saturated
- Reserved=31.34 GB, peak alloc=26.95 GB
- System RAM: **135.44/136.53 GB** → ~99% saturated → OS page-faulting

The bottleneck is **VRAM saturation forcing CUDA to thrash**, not a hard hang. Each step waits on memory transfers between dedicated VRAM, pinned CPU, and (when even pinned spills) pageable system RAM.

---

## VRAM Budget Breakdown (Static, Before Pipeline Starts)

From the log: `after diffusion components→GPU: alloc=11.93 GB`

| Consumer | Size | Notes |
|----------|------|-------|
| **Distilled LoRA `down`/`up` BF16 tensors** | **~9.15 GB** | 1660 pairs, full LoRA resident in dedicated VRAM throughout diffusion |
| Conductor GPU static buffer (rolling window of 2 blocks) | ~832 MB | Pre-allocated `int8` scratchpad for the currently-loaded blocks |
| Non-block transformer params (`time_embed`, `scale_shift_table`, `caption_projection`, `proj_in`, etc.) | ~670 MB | Permanently on GPU (not managed by conductor) |
| Connectors (`LTX2TextConnectors`) | ~500 MB | Used once per step to derive cross-attention KV |
| Cached `prompt_embeds` left after TE offload | ~0.78 GB | `[2, 1024, 184320]` BF16 — Gemma3-12B 48 layers stacked |
| **Total static** | **~11.93 GB** | ✓ matches log |

**Important**: the GGUF transformer is NOT a VRAM consumer here. With 100% layer offload, the conductor keeps only 2 blocks (~832 MB of static-buffer space) on GPU at any moment. The rest is in pinned CPU memory, which is irrelevant to dedicated VRAM pressure. Transformer size doesn't factor into the bottleneck.

---

## VRAM Budget Breakdown (Peak During Diffusion)

Peak `alloc=26.95 GB`, `reserved=31.34 GB` — meaning ~15 GB of activations on top of the 11.93 GB static budget.

LTX-2.3 transformer config for 1920×1088×121:
- Latent grid: 16 × 34 × 60 = **32,640 video tokens**
- Audio grid: ~smaller (maybe ~3,000 tokens)
- Hidden dim: `num_attention_heads (32) × attention_head_dim (128) = 4096` (video)
- Hidden dim: `audio_num_attention_heads (32) × audio_attention_head_dim (64) = 2048` (audio)
- 48 transformer blocks
- CFG enabled → **batch dim = 2** for the entire forward

Per-block intermediate activation memory (one block, one layer's worth):
- Hidden states `[2, 32640, 4096]` BF16 = **534 MB**
- Q, K, V tensors (3×): ~1.6 GB
- FF intermediate (4× expansion `[2, 32640, 16384]` BF16) = **2.1 GB**
- Per-block transient peak: **~3-4 GB**

Connector outputs (computed once, kept resident across all 48 blocks × all 20 steps):
- `connector_prompt_embeds` for cross-attention K/V on every block
- Plus `connector_audio_prompt_embeds` for the audio branch
- These are NOT recomputed each step — live the whole denoising loop

**The 15 GB activation peak is on the high side but normal for this resolution and CFG=2.**

---

## The Real Problem: LoRA Co-Resides With Activations

The ~9.15 GB LoRA sits in dedicated VRAM **simultaneously** with:
- The conductor's currently-loaded block (~832 MB)
- The 15 GB activation peak

Total live in VRAM at peak = `9.15 + 0.83 + 0.67 + 0.5 + 0.78 + ~15 ≈ 27 GB allocated`, plus the CUDA caching allocator's internal padding/fragmentation pushes reserved to 31.34 GB out of 34.19 GB total.

**With only 3 GB of headroom**, every transient allocation (Q, K, V projections; FFN intermediate; LoRA delta computation) competes for slim free VRAM. The CUDA caching allocator is forced to:
1. Frequently call `cudaMalloc`/`cudaFree` for new activations
2. Wait for in-flight ops to complete before reusing memory
3. (In the worst case) trigger CUDA's internal "out-of-memory" recovery, which on Windows falls back to using shared system RAM (the 135 GB system RAM at 99% saturation)

**Each step takes 39 seconds because the GPU spends most of its time waiting on host-managed memory operations**, not on actual matmul compute.

---

## Why ComfyUI Doesn't Have This Problem

ComfyUI's architecture for the same workload:
- **GGUF weights pinned to pageable CPU RAM** (not pinned). On-demand per-matmul transfer.
- **LoRA delta is FUSED into the dequant/cast pass** via `LowVramPatch` callbacks. The flow is:
  1. `cast_bias_weight` is called inside each linear's forward
  2. It copies the GGUF weight to GPU
  3. Calls `weight.dequantize()` to get the float weight
  4. Calls the registered `weight_function` callbacks (e.g. LoRA patch) on the dequant'd weight, modifying it in place
  5. Does the matmul against the LoRA-merged weight
  6. Frees the GPU copy after the matmul
- **Net GPU LoRA footprint: ZERO** — LoRA d/u tensors live in pageable CPU RAM, only one linear's LoRA is touched at a time, and even then the merge happens in-place against an already-allocated weight buffer.

OT's per-block conductor was designed for the GGUF transformer alone. It works perfectly for that. But the **distilled LoRA was bolted on as forward-method patches that hold all 1660 d/u pairs in VRAM at once** — independent of which block is currently loaded.

---

## Why the LoRA Is "Bolted On" Today

Looking at `Ltx2Model.py:_resume_distilled_lora_hooks` and `BaseLtx2Setup.py:_apply_distilled_lora`:
- Each `(target_linear, original_forward, {"down": d, "up": u})` handle is independent of the conductor.
- `distilled_lora_to(train_device)` moves ALL `d`/`u` tensors to GPU at once.
- The patched forward closure captures `_d`/`_u` as full tensors (not as references to a per-block buffer).

This was the simplest design — works with arbitrary linears, no coupling to the conductor. The cost is the 9.15 GB resident overhead.

---

## Three Possible Fixes (Ordered by Effort vs Payoff)

### Fix 1: Match LoRA windowing to the conductor's block windowing — **MEDIUM EFFORT, HIGH PAYOFF**

When the conductor brings block `i` to GPU (`before_layer(i)`), also move that block's LoRA pairs to GPU. When it offloads block `i`, also move block `i`'s LoRA pairs back to CPU.

Implementation:
1. Group `distilled_lora_handles` by which transformer block they belong to (parse from `module_path`).
2. Hook into the conductor's `before_layer`/`after_layer` to swap the corresponding LoRA pairs.
3. The patched forward's `_d.to(x.device, x.dtype)` already triggers a copy if the tensor is on CPU — but that's per-call, expensive. Better: move the per-block LoRA into the static buffer alongside the GGUF weights, or maintain a dedicated rolling LoRA buffer.

**Expected savings**: 9.15 GB → ~400 MB (only 2 blocks' worth of LoRA on GPU). Frees ~8.75 GB of headroom. Should drop step time from 39s to a few seconds.

### Fix 2: Fuse LoRA delta into the dequant pass (ComfyUI-style) — **HIGH EFFORT, MAXIMUM PAYOFF**

Replace the forward-method patching with a dequantize-time merge:
1. The conductor's `before_layer` already loads quantized weights to GPU.
2. Add a hook that runs after dequant (or as part of the matmul) which adds `up @ down * strength` to the dequantized float weight.
3. The matmul then uses the merged weight; no separate LoRA forward.

This requires modifying `LinearGGUFA8.forward` (and `GGUFLinear.forward`) to accept an optional LoRA delta function that runs against the dequantized weight. Or, the cleanest path: precompute the LoRA delta into the dequantized weight in the static buffer at `before_layer` time.

**Expected savings**: 9.15 GB → 0 GB resident (LoRA only briefly touched per-block). Best perf. Most invasive code change.

### Fix 3: Quantize the LoRA itself — **LOW EFFORT, MODERATE PAYOFF**

The LoRA is BF16. If quantized to FP8 (E4M3 or E5M2):
- 9.15 GB → 4.58 GB
- Still 4.58 GB resident, but frees ~4.6 GB

Use `LinearFp8.quantize_axiswise` or similar to quantize `d`/`u` once at load time, dequantize on-the-fly in the patched forward. Risks: quality regression at strength=0.6 (stage 2), need to verify.

**Expected savings**: 9.15 GB → 4.58 GB. Step time should drop from 39s to maybe 15s (still tight, but workable).

---

## Recommendation

**Fix 1** is the right entry point: it preserves the existing forward-method patching design (which is simple and verified) and just adds conductor-aware swapping. The conductor already has the infrastructure to schedule per-block CPU↔GPU transfers — extending it to cover LoRA tensors is a localized change.

If Fix 1 still doesn't deliver speed parity with ComfyUI, then move to Fix 2 (full fusion).

**Fix 3** is a minor footprint reduction but doesn't address the underlying architectural mismatch (LoRA still scales with model size on GPU, just with a constant factor).

---

## Other Observations

### Transformer size is IRRELEVANT during sampling
At 100% layer offload, only 2 blocks (the rolling window) ever live in dedicated VRAM at any moment. Whether the GGUF transformer is 13 GB (Q4) or 22 GB (Q8) makes no difference to the GPU memory budget — those bytes sit in CPU pinned memory and only one block's worth (~416 MB) is in the static buffer at a time. The conductor IS doing its job correctly. The transformer is not the problem.

### The 24 GB text encoder peak is uncomfortable but transient
The full BF16 Gemma3-12B fills `free=0.00/34.19` during prompt encode, then offloads to ~0.8 GB. Not in the diffusion hot loop; not a bottleneck for step time.

### Connector outputs are kept resident across all steps
Computed once before the loop, used every step for cross-attention K/V. Hundreds of MB up to ~1 GB. Correct design (per-step recompute would waste time), but counts toward baseline VRAM during diffusion.
