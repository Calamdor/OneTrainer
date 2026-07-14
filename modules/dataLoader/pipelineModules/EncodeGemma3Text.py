from contextlib import nullcontext

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

from transformers import Gemma3ForConditionalGeneration


class EncodeGemma3Text(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """LTX-2.3's Gemma3 text encoding: every decoder layer's hidden state, stacked and
    flattened into a single embedding.

    Upstream mgds' ``EncodeGemmaText`` only selects a single layer's hidden state
    (``hidden_state_output_index``) -- LTX-2.3 instead uses every layer's hidden state
    stacked along a new dimension and flattened into one ``[seq_len, hidden_size *
    num_layers]`` tensor (see ``Ltx2Model.encode_text()``, which this module mirrors
    exactly so cached and live-encoded embeddings match).
    """

    def __init__(
            self,
            tokens_in_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            text_encoder: Gemma3ForConditionalGeneration,
            autocast_contexts: list[torch.autocast | None] | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.tokens_in_name = tokens_in_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.text_encoder = text_encoder

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.tokens_in_name)

    def get_inputs(self) -> list[str]:
        return [self.tokens_in_name, self.tokens_attention_mask_in_name]

    def get_outputs(self) -> list[str]:
        return [self.hidden_state_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.tokens_in_name, index)
        tokens = tokens.unsqueeze(0)

        if self.tokens_attention_mask_in_name is not None:
            attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index)
            attention_mask = attention_mask.unsqueeze(0)
        else:
            attention_mask = None

        with torch.no_grad(), self._all_contexts(self.autocast_contexts):
            outputs = self.text_encoder(
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Every layer's hidden state, stacked along a new dim and flattened into a single
        # 3D tensor [batch, seq_len, hidden_size * num_layers] -- matches
        # Ltx2Model.encode_text() exactly.
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        embedding = hidden_states.flatten(2, 3)
        embedding = embedding.squeeze(0)  # (seq_len, hidden_size * num_layers)

        if self.dtype is not None:
            embedding = embedding.to(dtype=self.dtype)

        return {
            self.hidden_state_out_name: embedding,
        }
