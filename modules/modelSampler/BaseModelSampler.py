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

# Use PyAV for video writing as alternative to deprecated torchvision.io.write_video
HAS_PYAV = False  # noqa: F821
try:
    import av as pyav  # noqa: F401

    HAS_PYAV = True
except ImportError:
    pass

from PIL import Image


class ModelSamplerOutput:
    def __init__(
            self,
            file_type: FileType,
            data: Image.Image | torch.Tensor | bytes,

    ):
        self.file_type = file_type
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
            fps: int = 24,  # Target FPS for video output (default 24 for backward compatibility)
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

            # Handle video writing with PyAV as alternative to deprecated torchvision.io.write_video
            try:
                import av  # Import locally to avoid LSP errors

                # Only attempt video writing if data is a tensor (not an image)
                if isinstance(sampler_output.data, torch.Tensor):
                    # Convert tensor to numpy format expected by PyAV
                    video_tensor = sampler_output.data.detach().cpu()

                    # Ensure we have the right shape for video processing - should be 4D
                    if len(video_tensor.shape) == 4:
                        # Handle different input formats:
                        # - (T, H, W, C): already in frame order format (e.g., from HunyuanVideoSampler)
                        # - (C, T, H, W): channel-first format
                        shape = video_tensor.shape
                        if shape[-1] == 3 and shape[0] > 10:  # Likely (T, H, W, C) with many frames
                            # Already in correct order, just convert to numpy
                            frames = video_tensor.numpy()
                        else:
                            # Convert from (C, T, H, W) to (T, H, W, C)
                            frames = video_tensor.permute(1, 2, 3, 0).numpy()

                        # Normalize values to [0, 255] range if needed
                        frames = (
                            (frames * 255).astype('uint8')
                            if frames.max() <= 1.0
                            else frames.astype('uint8')
                        )

                        # Write video with PyAV using the specified FPS
                        with av.open(destination + video_format.extension(), 'w') as container:
                            stream = container.add_stream('libx264', rate=fps)

                            # Configure stream properties (required for libx264)
                            stream.width = frames.shape[2]
                            stream.height = frames.shape[1]
                            stream.pix_fmt = 'yuv420p'  # Required pixel format for H.264

                            # Ensure frames are in correct format (T, H, W, C)
                            if len(frames.shape) == 4 and frames.shape[-1] == 3:  # RGB format
                                for frame_data in frames:
                                    frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
                                    for packet in stream.encode(frame):
                                        container.mux(packet)

                                # Flush the encoder after all frames are written
                                try:
                                    for packet in stream.encode():  # Empty call to flush (no argument)
                                        container.mux(packet)
                                except Exception as flush_e:
                                    print(f"Warning: Could not flush video encoder: {flush_e}")
                            else:
                                raise ValueError(f"Unsupported video frame shape: {frames.shape}")

            except Exception as e:
                print(f"Error writing video with PyAV (fallback): {e}")
                # Fallback to no-op if we can't write the video
        elif sampler_output.file_type == FileType.AUDIO:
            pass # TODO
