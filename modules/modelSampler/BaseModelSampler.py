import io
import os
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from pathlib import Path

from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.VideoFormat import VideoFormat

import torch

import av
from PIL import Image


class ModelSamplerOutput:
    def __init__(
            self,
            file_type: FileType,
            data: Image.Image | torch.Tensor | bytes,
            fps: int = 24,
            audio: torch.Tensor | None = None,
            audio_sample_rate: int | None = None,
    ):
        self.file_type = file_type
        self.fps = fps
        # Optional audio waveform muxed into the same container as a video file.
        # Shape conventions accepted by save_sampler_output: (channels, samples) or (samples,).
        self.audio = audio
        self.audio_sample_rate = audio_sample_rate
        if isinstance(data, bytes):
            assert file_type == FileType.IMAGE
            self.data = Image.open(io.BytesIO(data))
        else:
            self.data = data

    #Reduce to a JPEG bytestream for cloud training:
    def __reduce__(self):
        match self.file_type:
            case FileType.IMAGE:
                b = io.BytesIO()
                self.data.save(b, format='JPEG')
                return ModelSamplerOutput, (self.file_type, b.getvalue())
            case FileType.VIDEO:
                #do not transfer videos; they are not shown anyway
                #the video sample file is transferred via workspace sync
                return ModelSamplerOutput, (self.file_type, None)
            case FileType.AUDIO:
                # TODO
                return ModelSamplerOutput, (self.file_type, None)
            case _:
                return ModelSamplerOutput, (self.file_type, None)


class BaseModelSampler(metaclass=ABCMeta):

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
    ):
        super().__init__()

        self.train_device = train_device
        self.temp_device = temp_device

    @abstractmethod
    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat,
            video_format: VideoFormat,
            audio_format: AudioFormat,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        pass

    @staticmethod
    def quantize_resolution(resolution: int, quantization: int) -> int:
        return round(resolution / quantization) * quantization

    @staticmethod
    def save_sampler_output(
            sampler_output: ModelSamplerOutput,
            destination: str,
            image_format: ImageFormat | None,
            video_format: VideoFormat | None,
            audio_format: AudioFormat | None,
            fps: int = 24,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        if sampler_output.file_type == FileType.IMAGE:
            if image_format is None:
                raise ValueError("Image format required for sampling an image")
            image = sampler_output.data
            image.save(destination + image_format.extension(), format=image_format.pil_format())
        elif sampler_output.file_type == FileType.VIDEO:
            if video_format is None:
                raise ValueError("Video format required for sampling a video")

            if isinstance(sampler_output.data, torch.Tensor):
                video_tensor = sampler_output.data.detach().cpu()

                if len(video_tensor.shape) == 4:
                    shape = video_tensor.shape
                    # Normalize to (T, H, W, C) — permute (C, T, H, W) if needed.
                    if shape[-1] != 3:
                        video_tensor = video_tensor.permute(1, 2, 3, 0).contiguous()

                    # Ensure uint8 in [0, 255] regardless of source dtype/range.
                    if video_tensor.dtype != torch.uint8:
                        if video_tensor.dtype.is_floating_point and float(video_tensor.max()) <= 1.0:
                            video_tensor = (video_tensor * 255).round().clamp(0, 255).to(torch.uint8)
                        else:
                            video_tensor = video_tensor.to(torch.uint8)

                    output_path = destination + video_format.extension()
                    has_audio = (
                        sampler_output.audio is not None
                        and sampler_output.audio_sample_rate is not None
                    )

                    if has_audio:
                        # Delegate to diffusers' battle-tested AAC muxer. Handles
                        # AAC stream creation, channel layout, format resampling,
                        # and proper interleaving with the video stream. Pass the
                        # torch.Tensor directly (uint8) — encode_video's numpy
                        # branch warns when given a non-[0,1] ndarray, but the
                        # tensor branch goes straight to av.VideoFrame.
                        from diffusers.pipelines.ltx2.export_utils import encode_video
                        encode_video(
                            video=video_tensor,
                            fps=int(sampler_output.fps),
                            audio=sampler_output.audio.detach().cpu(),
                            audio_sample_rate=int(sampler_output.audio_sample_rate),
                            output_path=output_path,
                        )
                    else:
                        frames = video_tensor.numpy()
                        with av.open(output_path, 'w') as container:
                            stream = container.add_stream('libx264', rate=sampler_output.fps)
                            stream.options = {'crf': '17'}
                            stream.width = frames.shape[2]
                            stream.height = frames.shape[1]
                            stream.pix_fmt = 'yuv420p'  # Required pixel format for H.264

                            for frame_data in frames:
                                frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
                                for packet in stream.encode(frame):
                                    container.mux(packet)

                            for packet in stream.encode():
                                container.mux(packet)
                else:
                    raise ValueError(f"Expected 4D video tensor (T, H, W, C) or (C, T, H, W), got shape {video_tensor.shape}")
        elif sampler_output.file_type == FileType.AUDIO:
            pass # TODO
