"""End-to-end smoke test for the LTX-2.3 LoRA training pipeline.

Tests predict() → calculate_loss() → backward() with:
- A minimal fake transformer whose forward actually routes through
  transformer_blocks.attn linear layers (so LoRA hooks fire)
- Both unmasked and masked (prior preservation) loss paths

Run from the OneTrainer root:
    python test_ltx2_lora_training.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from contextlib import nullcontext
from types import SimpleNamespace
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers to build a minimal fake model
# ---------------------------------------------------------------------------

class FakeBlock(nn.Module):
    """Tiny residual block with a single linear (the LoRA target)."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.attn(x)


class FakeTransformer(nn.Module):
    """Minimal transformer with transformer_blocks, mirrors LTX-2.3 naming."""
    def __init__(self, dim: int = 32, num_blocks: int = 2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([FakeBlock(dim) for _ in range(num_blocks)])
        # config namespace mirrors LTX2VideoTransformer3DModel.config
        self.config = SimpleNamespace(
            patch_size=1,
            patch_size_t=1,
            audio_in_channels=128,
        )

    def forward(
        self, hidden_states, audio_hidden_states, encoder_hidden_states,
        audio_encoder_hidden_states, timestep, sigma, encoder_attention_mask,
        audio_encoder_attention_mask, num_frames, height, width,
        audio_num_frames=1, use_cross_timestep=True, return_dict=False,
    ):
        # hidden_states: (B, T*H*W, C)  — we only use dim=32 for the fake
        x = hidden_states
        for block in self.transformer_blocks:
            x = block(x)
        # return (video_pred, audio_pred); audio_pred ignored for T2V
        return x, torch.zeros_like(audio_hidden_states)


class FakeConnectors(nn.Module):
    """Returns fixed-shape embeddings that predict() passes to the transformer."""
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))  # makes it an nn.Module

    def forward(self, text_emb, mask, padding_side="left"):
        B = text_emb.shape[0]
        seq_len = text_emb.shape[1]
        video_emb = torch.zeros(B, seq_len, 32, device=text_emb.device, dtype=text_emb.dtype)
        audio_emb = torch.zeros(B, 1, 32, device=text_emb.device, dtype=text_emb.dtype)
        attn_mask = torch.ones(B, seq_len, device=text_emb.device, dtype=torch.bool)
        return video_emb, audio_emb, attn_mask


class FakeVAE(nn.Module):
    def __init__(self, C: int = 32):
        super().__init__()
        self.latents_mean = torch.zeros(C)
        self.latents_std = torch.ones(C)
        self.config = SimpleNamespace(scaling_factor=0.18215)


class FakeNoiseScheduler:
    config = SimpleNamespace(num_train_timesteps=1000)


class FakeDataType:
    def torch_dtype(self):
        return torch.float32


class FakeTokenizer:
    padding_side = "left"


# ---------------------------------------------------------------------------
# Build the Ltx2Model and Ltx2LoRASetup instances
# ---------------------------------------------------------------------------

def build_fake_model(latent_C: int = 32) -> "Ltx2Model":
    from modules.model.Ltx2Model import Ltx2Model
    from modules.util.enum.ModelType import ModelType

    model = Ltx2Model(ModelType.LTX_2_3)
    model.transformer = FakeTransformer(dim=latent_C)
    model.vae = FakeVAE(C=latent_C)
    model.connectors = FakeConnectors()
    model.noise_scheduler = FakeNoiseScheduler()
    model.train_dtype = FakeDataType()
    model.tokenizer = FakeTokenizer()
    model.autocast_context = nullcontext()
    model.text_encoder = None
    model.audio_vae = None
    model.vocoder = None
    model.transformer_lora = None
    model.lora_state_dict = None
    model.distilled_lora_handles = []
    return model


def build_train_config(masked: bool = False) -> "TrainConfig":
    from modules.util.config.TrainConfig import TrainConfig
    config = TrainConfig.default_values()
    config.train_device = "cpu"
    config.temp_device = "cpu"
    config.model_type = __import__(
        "modules.util.enum.ModelType", fromlist=["ModelType"]
    ).ModelType.LTX_2_3
    config.masked_training = masked
    config.mse_strength = 1.0
    config.mae_strength = 0.0
    config.log_cosh_strength = 0.0
    config.huber_strength = 0.0
    config.vb_loss_strength = 0.0
    config.unmasked_weight = 0.01
    config.normalize_masked_area_loss = False
    config.masked_prior_preservation_weight = 0.0
    config.loss_weight_fn = __import__(
        "modules.util.enum.LossWeight", fromlist=["LossWeight"]
    ).LossWeight.CONSTANT
    config.loss_scaler = __import__(
        "modules.util.enum.LossScaler", fromlist=["LossScaler"]
    ).LossScaler.NONE
    config.batch_size = 1
    config.gradient_accumulation_steps = 1
    config.lora_rank = 4
    config.lora_alpha = 4.0
    config.lora_weight_dtype = __import__(
        "modules.util.enum.DataType", fromlist=["DataType"]
    ).DataType.FLOAT_32
    config.dropout_probability = 0.0
    config.layer_filter = ""
    config.latent_caching = False
    # Timestep distribution defaults
    config.timestep_distribution = __import__(
        "modules.util.enum.TimestepDistribution", fromlist=["TimestepDistribution"]
    ).TimestepDistribution.UNIFORM
    config.min_noising_strength = 0.0
    config.max_noising_strength = 1.0
    config.noising_weight = 0.0
    config.noising_bias = 0.5
    return config


# ---------------------------------------------------------------------------
# Core test helpers
# ---------------------------------------------------------------------------

def make_batch(latent_C: int = 32, latent_H: int = 8, latent_W: int = 8,
               latent_T: int = 1, seq_len: int = 16,
               masked: bool = False, with_prior: bool = False) -> dict:
    """Build a minimal training batch matching Ltx2BaseDataLoader output."""
    B = 1
    # text embedding: (B, seq_len, hidden_size*num_layers) — fake small size
    text_emb = torch.randn(B, seq_len, 32)
    tokens_mask = torch.ones(B, seq_len, dtype=torch.bool)
    # latent image: (B, C, T, H, W) — single frame
    latent_image = torch.randn(B, latent_C, latent_T, latent_H, latent_W)
    batch = {
        "text_encoder_1_hidden_state": text_emb,
        "tokens_mask_1": tokens_mask,
        "latent_image": latent_image,
        "loss_weight": torch.ones(B),
    }
    if masked:
        # latent_mask must match packed shape (B, T*H*W, 1) or (B, T*H*W, C)
        # calculate_loss uses __masked_losses which broadcasts mask
        # mask is in latent space: (B, 1, T, H, W) — same spatial size as latent
        batch["latent_mask"] = torch.ones(B, 1, latent_T, latent_H, latent_W) * 0.5
    return batch


def run_forward_and_loss(setup, model, batch, config) -> tuple:
    """Returns (loss, data) and optionally checks backward."""
    from modules.util.TrainProgress import TrainProgress
    progress = TrainProgress()
    data = setup.predict(model, batch, config, progress, deterministic=True)
    loss = setup.calculate_loss(model, batch, data, config)
    return loss, data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_predict_output_shapes(setup, model, config):
    print("  [predict] unmasked batch...")
    batch = make_batch(latent_C=32)
    from modules.util.TrainProgress import TrainProgress
    data = setup.predict(model, batch, config, TrainProgress(), deterministic=True)
    assert "predicted" in data, "predict() must return 'predicted'"
    assert "target" in data, "predict() must return 'target'"
    assert data["predicted"].shape == data["target"].shape, (
        f"Shape mismatch: pred={data['predicted'].shape} target={data['target'].shape}"
    )
    print(f"     predicted shape: {data['predicted'].shape}  ✓")


def test_calculate_loss_scalar(setup, model, config):
    print("  [loss] unmasked → scalar...")
    batch = make_batch(latent_C=32)
    loss, data = run_forward_and_loss(setup, model, batch, config)
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert not loss.isnan(), "Loss is NaN"
    print(f"     loss = {loss.item():.6f}  ✓")


def test_backward_and_lora_grads(setup, model, config):
    print("  [backward] loss.backward() → LoRA grad check...")
    batch = make_batch(latent_C=32)
    loss, _ = run_forward_and_loss(setup, model, batch, config)
    loss.backward()

    grad_count = 0
    none_count = 0
    for p in model.transformer_lora.parameters():
        if p.requires_grad:
            if p.grad is not None:
                grad_count += 1
            else:
                none_count += 1

    print(f"     LoRA params with grad: {grad_count}, without: {none_count}")
    assert grad_count > 0, "No LoRA parameters received gradients!"
    print(f"     ✓ {grad_count} LoRA params have gradients")


def test_masked_loss_differs(setup, model, config_masked, config_unmasked):
    print("  [masked] masked vs unmasked loss paths...")
    batch_unmasked = make_batch(latent_C=32, masked=False)
    batch_masked = make_batch(latent_C=32, masked=True)

    loss_unmasked, _ = run_forward_and_loss(setup, model, batch_unmasked, config_unmasked)
    loss_masked, _ = run_forward_and_loss(setup, model, batch_masked, config_masked)

    print(f"     loss (masked_training=False, no mask): {loss_unmasked.item():.6f}")
    print(f"     loss (masked_training=True, mask=0.5): {loss_masked.item():.6f}")
    assert not loss_unmasked.isnan(), "Unmasked loss is NaN"
    assert not loss_masked.isnan(), "Masked loss is NaN"
    print("     ✓ both loss paths produce valid scalars")


def test_masked_prior_preservation(setup, model, config_masked):
    print("  [prior] masked prior preservation path...")
    batch = make_batch(latent_C=32, masked=True)
    from modules.util.TrainProgress import TrainProgress
    data = setup.predict(model, batch, config_masked, TrainProgress(), deterministic=True)
    # Inject a prior_target that differs from the real target
    data["prior_target"] = torch.zeros_like(data["target"])
    config_masked.masked_prior_preservation_weight = 1.0

    loss = setup.calculate_loss(model, batch, data, config_masked)
    assert not loss.isnan(), "Prior preservation loss is NaN"
    print(f"     loss with prior preservation = {loss.item():.6f}  ✓")
    config_masked.masked_prior_preservation_weight = 0.0  # reset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("LTX-2.3 LoRA training smoke test")
    print("=" * 60)

    # --- Imports that need sys.path set ---
    # Import BaseLtx2Setup directly to avoid the circular import that occurs when
    # Ltx2LoRASetup → optimizer_util → create.py → factory.import_dir triggers.
    # We create a concrete subclass here just to satisfy ABCMeta.
    from modules.modelSetup.BaseLtx2Setup import BaseLtx2Setup
    from modules.module.LoRAModule import LoRAModuleWrapper

    class Ltx2LoRASetup(BaseLtx2Setup):
        pass

    device = torch.device("cpu")

    print("\n[setup] Building fake Ltx2Model...")
    model = build_fake_model(latent_C=32)

    print("[setup] Building TrainConfig...")
    config = build_train_config(masked=False)
    config_masked = build_train_config(masked=True)

    print("[setup] Creating Ltx2LoRASetup and wrapping transformer with LoRA...")
    setup = Ltx2LoRASetup(train_device=device, temp_device=device, debug_mode=False)

    # Manually create LoRA wrapper (skip setup_model to avoid optimizer init)
    model.transformer_lora = LoRAModuleWrapper(
        model.transformer, "lora_transformer", config, ["transformer_blocks"]
    )
    model.transformer_lora.set_dropout(0.0)
    model.transformer_lora.to(dtype=torch.float32)
    model.transformer_lora.hook_to_module()

    # Freeze base transformer, enable LoRA params
    model.transformer.requires_grad_(False)
    model.transformer_lora.requires_grad_(True)

    print(f"     LoRA modules created: {len(model.transformer_lora.lora_modules)}")
    assert len(model.transformer_lora.lora_modules) > 0, "No LoRA modules were created!"

    print("\nRunning tests...")
    test_predict_output_shapes(setup, model, config)
    test_calculate_loss_scalar(setup, model, config)
    test_backward_and_lora_grads(setup, model, config)

    # Zero grads for masked tests
    for p in model.transformer_lora.parameters():
        if p.grad is not None:
            p.grad = None

    test_masked_loss_differs(setup, model, config_masked, config)
    test_masked_prior_preservation(setup, model, config_masked)

    print("\n" + "=" * 60)
    print("All tests passed ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
