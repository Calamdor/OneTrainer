import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.BaseModel import BaseModel
from modules.model.Ltx2Model import Ltx2Model
from modules.modelSetup.BaseLtx2Setup import BaseLtx2Setup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import cuda_memory_profile
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.EncodeGemma3Text import EncodeGemma3Text
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.PadMaskedTokens import PadMaskedTokens
from mgds.pipelineModules.PruneMaskedTokens import PruneMaskedTokens
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize
from modules.dataLoader.pipelineModules.EncodeLtx2Connectors import EncodeLtx2Connectors


class Ltx2BaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def _preparation_modules(self, config: TrainConfig, model: Ltx2Model):
        rescale_image = RescaleImageChannels(
            image_in_name='image', image_out_name='image',
            in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1,
        )
        encode_image = EncodeVAE(
            in_name='image', out_name='latent_image_distribution',
            vae=model.vae,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype(),
        )
        image_sample = SampleVAEDistribution(
            in_name='latent_image_distribution', out_name='latent_image', mode='mean',
        )
        # LTX-2.3 VAE spatial compression ratio is 32; mask must be downscaled to match.
        downscale_mask = ScaleImage(
            in_name='mask', out_name='latent_mask', factor=1 / 32,
        )

        # Gemma3 uses left-padding for chat-style prompts. Ensure the tokenizer is
        # configured correctly before tokenization so PruneMaskedTokens/PadMaskedTokens
        # work with the right padding positions.
        if model.tokenizer is not None:
            model.tokenizer.padding_side = "left"
            if model.tokenizer.pad_token is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token

        tokenize_prompt = Tokenize(
            in_name='prompt', tokens_out_name='tokens_1', mask_out_name='tokens_mask_1',
            tokenizer=model.tokenizer,
            max_token_length=1024,
        )
        encode_prompt = EncodeGemma3Text(
            tokens_in_name='tokens_1',
            tokens_attention_mask_in_name='tokens_mask_1',
            hidden_state_out_name='text_encoder_1_hidden_state',
            text_encoder=model.text_encoder,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype(),
        )
        # Optional CUDA memory profile (env: LTX2_MEMORY_PROFILE=1). Records
        # allocations during the first few text-encoder forwards, dumps a
        # pickle for https://pytorch.org/memory_viz, and prints periodic
        # memory_stats summaries. No-op when the env var is unset.
        if cuda_memory_profile.is_enabled():
            cuda_memory_profile.start()
            cuda_memory_profile.print_stats("preparation_modules entry")
            cuda_memory_profile.install_caching_probe(
                model.text_encoder,
                label="gemma3_te",
                stats_every=25,
                dump_after=5,
            )
            if model.connectors is not None:
                cuda_memory_profile.install_caching_probe(
                    model.connectors,
                    label="connectors",
                    stats_every=25,
                    dump_after=5,
                )
        # Run connectors here — before PruneMaskedTokens — so every item has the
        # same fixed sequence length (max_token_length=1024). Cached outputs are
        # consumed directly in predict(), eliminating the per-step connector call.
        padding_side = getattr(model.tokenizer, "padding_side", "left") if model.tokenizer is not None else "left"
        encode_connectors = EncodeLtx2Connectors(
            hidden_state_in_name='text_encoder_1_hidden_state',
            attention_mask_in_name='tokens_mask_1',
            video_emb_out_name='connector_video_emb',
            audio_emb_out_name='connector_audio_emb',
            attn_mask_out_name='connector_attn_mask',
            connectors=model.connectors,
            padding_side=padding_side,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype(),
        )
        prune_masked_tokens = PruneMaskedTokens(
            tokens_name='tokens_1', tokens_mask_name='tokens_mask_1',
            hidden_state_name='text_encoder_1_hidden_state',
        )

        modules = [rescale_image, encode_image, image_sample]
        if config.masked_training:
            modules.append(downscale_mask)

        modules.append(tokenize_prompt)
        # Text encoder is never trained for LTX-2.3 LoRA — always cache embeddings.
        modules.append(encode_prompt)
        # Connector encoding must come before PruneMaskedTokens (fixed 1024-length inputs).
        modules.append(encode_connectors)

        if config.latent_caching:
            modules.append(prune_masked_tokens)

        return modules

    def _cache_modules(self, config: TrainConfig, model: Ltx2Model, model_setup: BaseLtx2Setup):
        image_split_names = ['latent_image', 'original_resolution', 'crop_offset']

        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        image_aggregate_names = ['crop_resolution', 'image_path']

        # Text encoder is never trained for LTX-2.3 LoRA — always cache.
        # Connector outputs are fixed-length (max_token_length=1024) so no
        # per-batch padding is required — cache them alongside text embeddings.
        text_split_names = [
            'tokens_1', 'tokens_mask_1', 'text_encoder_1_hidden_state',
            'connector_video_emb', 'connector_audio_emb', 'connector_attn_mask',
        ]

        sort_names = image_aggregate_names + image_split_names + [
            'prompt', 'tokens_1', 'tokens_mask_1', 'text_encoder_1_hidden_state',
            'connector_video_emb', 'connector_audio_emb', 'connector_attn_mask',
            'concept',
        ]

        return self._cache_modules_from_names(
            model, model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=True,
        )

    def _output_modules(self, config: TrainConfig, model: Ltx2Model, model_setup: BaseLtx2Setup):
        output_names = [
            'image_path', 'latent_image',
            'prompt',
            'tokens_1', 'tokens_mask_1',
            'text_encoder_1_hidden_state',
            'connector_video_emb', 'connector_audio_emb', 'connector_attn_mask',
            'original_resolution', 'crop_resolution', 'crop_offset',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        pad_masked_tokens = PadMaskedTokens(
            tokens_name='tokens_1', tokens_mask_name='tokens_mask_1',
            hidden_state_name='text_encoder_1_hidden_state', max_length=1024,
        )

        output_module_list = self._output_modules_from_out_names(
            model, model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

        if config.latent_caching:
            output_module_list = [pad_masked_tokens] + output_module_list

        return output_module_list

    def _debug_modules(self, config: TrainConfig, model: Ltx2Model):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.vae_to(self.train_device)

        decode_image = DecodeVAE(
            in_name='latent_image', out_name='decoded_image',
            vae=model.vae,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype(),
        )
        decode_prompt = DecodeTokens(
            in_name='tokens_1', out_name='decoded_prompt', tokenizer=model.tokenizer,
        )
        save_prompt = SaveText(
            text_in_name='decoded_prompt', original_path_in_name='image_path',
            path=debug_dir, before_save_fun=before_save_fun,
        )

        modules = [decode_image, decode_prompt, save_prompt]

        if config.masked_training:
            upscale_mask = ScaleImage(in_name='latent_mask', out_name='decoded_mask', factor=32)
            save_mask = SaveImage(
                image_in_name='decoded_mask', original_path_in_name='image_path',
                path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun,
            )
            modules += [upscale_mask, save_mask]

        return modules

    def _create_dataset(
            self,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        return DataLoaderText2ImageMixin._create_dataset(
            self, config, model, model_setup, train_progress, is_validation,
            # LTX-2.3 VAE spatial compression is 32x; pixel dims must be divisible by 32.
            aspect_bucketing_quantization=32,
            frame_dim_enabled=True,
            allow_video_files=True,
            vae_frame_dim=True,
        )


factory.register(BaseDataLoader, Ltx2BaseDataLoader, ModelType.LTX_2_3)
