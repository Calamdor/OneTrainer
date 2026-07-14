from enum import Enum


class LtxMultiScaleMode(Enum):
    FULL_SIZE = 'FULL_SIZE'   # Single-pass at the final W x H (no upsampler)
    X1_5      = 'X1_5'        # Stage 1 at (final W/1.5) x (final H/1.5) -> upsample 1.5x -> stage 2 at final W x H
    X2        = 'X2'          # Stage 1 at (final W/2)   x (final H/2)   -> upsample 2x   -> stage 2 at final W x H

    def __str__(self) -> str:
        return self.value

    def is_two_stage(self) -> bool:
        return self != LtxMultiScaleMode.FULL_SIZE

    def upscale_factor(self) -> float:
        return {
            LtxMultiScaleMode.X1_5: 1.5,
            LtxMultiScaleMode.X2: 2.0,
        }.get(self, 1.0)
