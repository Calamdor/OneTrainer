from contextlib import nullcontext

import torch

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeLtx2Connectors(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Cache LTX2TextConnectors outputs alongside text encoder embeddings.

    Runs once per item during the text-caching pass so that predict() can
    consume the pre-projected video_emb / audio_emb / attn_mask directly,
    eliminating the ~30-50 ms connector forward call per training step.

    Input ``hidden_state_in_name`` must be the full-length (max_token_length)
    flattened Gemma3 embedding — shape (seq_len, hidden_size * num_layers).
    This module must be placed BEFORE PruneMaskedTokens in the pipeline so
    that the sequence length is constant (max_token_length) for all items.
    The fixed length means the cached tensors need no per-batch re-padding.
    """

    def __init__(
            self,
            hidden_state_in_name: str,
            attention_mask_in_name: str,
            video_emb_out_name: str,
            audio_emb_out_name: str,
            attn_mask_out_name: str,
            connectors,
            padding_side: str = "left",
            autocast_contexts: list[torch.autocast | None] | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_state_in_name = hidden_state_in_name
        self.attention_mask_in_name = attention_mask_in_name
        self.video_emb_out_name = video_emb_out_name
        self.audio_emb_out_name = audio_emb_out_name
        self.attn_mask_out_name = attn_mask_out_name
        self.connectors = connectors
        self.padding_side = padding_side
        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.hidden_state_in_name)

    def get_inputs(self) -> list[str]:
        return [self.hidden_state_in_name, self.attention_mask_in_name]

    def get_outputs(self) -> list[str]:
        return [self.video_emb_out_name, self.audio_emb_out_name, self.attn_mask_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        hidden_state = self._get_previous_item(variation, self.hidden_state_in_name, index)
        attn_mask = self._get_previous_item(variation, self.attention_mask_in_name, index)

        # Add batch dimension for connector forward, then remove it from outputs.
        hs = hidden_state.unsqueeze(0)    # (1, seq_len, hidden_size*num_layers)
        mask = attn_mask.unsqueeze(0)     # (1, seq_len)

        with torch.no_grad(), self._all_contexts(self.autocast_contexts):
            video_emb, audio_emb, out_mask = self.connectors(
                hs, mask, padding_side=self.padding_side,
            )

        video_emb = video_emb.squeeze(0)   # (seq_len, video_dim)
        audio_emb = audio_emb.squeeze(0)   # (seq_len, audio_dim)
        out_mask = out_mask.squeeze(0)     # (seq_len,)

        if self.dtype is not None:
            video_emb = video_emb.to(dtype=self.dtype)
            audio_emb = audio_emb.to(dtype=self.dtype)

        return {
            self.video_emb_out_name: video_emb,
            self.audio_emb_out_name: audio_emb,
            self.attn_mask_out_name: out_mask,
        }
