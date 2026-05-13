# OneTrainer — Dev Environment

## Environment
- **Venv python**: `D:/AI/OTTest/Cala/OneTrainer/venv/Scripts/python.exe`
- **Activate**: `source D:/AI/OTTest/Cala/OneTrainer/venv/Scripts/activate`
- Always use the venv for anything requiring torch, safetensors, diffusers, transformers, accelerate, etc.

## Commands
```bash
# Fast import sanity check after edits
D:/AI/OTTest/Cala/OneTrainer/venv/Scripts/python.exe -c "import modules.modelSetup.WanLoRASetup"

# Training
D:/AI/OTTest/Cala/OneTrainer/venv/Scripts/python.exe trainer_train.py -c <config.json>

# Standalone sampler
D:/AI/OTTest/Cala/OneTrainer/venv/Scripts/python.exe ../wan_sampler_gui.py
```

## OT-First Principle — CRITICAL
Before implementing anything, find how OT already does it and match that pattern exactly. OT devs (Nerogar/DXQB) are excellent; deviating from their patterns is a common source of subtle bugs.

- **Factory registration**: `factory.register(BaseClass, Impl, EnumVal)` at module bottom; auto-discovered via `factory.import_dir()` in `modules/util/create.py` — no manual central registration
- **Loader factories**: `make_fine_tune_model_loader(...)` / `make_lora_model_loader(...)`
- **Forward patching**: `m.forward = patched_fn` — NOT `register_forward_hook` (silently bypassed by `torch.compile(fullgraph=True)`)

## Git Workflow — CRITICAL
- **Dev work**: `D:/AI/OTTest/Cala/OneTrainer/` — local commits only
- **GitHub push**: from `D:/AI/OTTest/Cala/WAN2.2_Github_Work/OneTrainer` → `myfork` remote → `feature/wan2.2-t2v-a14b`
- **NEVER push** directly from the dev `OneTrainer/` repo

## Key Architecture Notes
- `ModelWeightDtypes` has `transformer` but NOT `transformer_2` — both Wan experts share the `transformer` dtype field
- Resolution quantization minimum: VAE×8 × patch×2 = **16** (not 64)
- Companion LoRA handles are 4-tuples: `(module, orig_forward, rot_mod_or_None, lora_buffer_names_or_None)`. LoRA down/up are registered as non-persistent buffers on the target submodule so the parent transformer's `.to(device)` moves them — closure-captured tensors get stranded on the original device when the inactive expert is later promoted to GPU for sampling.
- `LoRALoaderMixin.__load_internal` loads safetensors directly without `convert_to_diffusers` — keys are already in OT format
- GGUF non-quantized layers (norms, biases) must use `train_dtype`, never hardcode BF16 — FP16 compute + BF16 weights = NaN
- OFT `patched_forward` MUST NOT call `rot_mod.to(x.device)` — module `.to()` is untraceable by `torch.compile(fullgraph=True)`

## Verification After Changes
Always run an import test after editing any module:
```bash
D:/AI/OTTest/Cala/OneTrainer/venv/Scripts/python.exe -c "import modules.<changed_module_path>"
```
For sampler/setup changes, also do a short test run to catch runtime errors not caught by import.

## Subagents
For codebase exploration ("how does OT do X?"), delegate to a subagent — keeps your main context clean for implementation.
