from enum import Enum


class LtxMultiScaleMode(Enum):
    FULL_SIZE = 'FULL_SIZE'   # Single-pass at user's W×H (no upsampler)
    X1_5      = 'X1_5'        # Stage 1 at W/1.5 × H/1.5 → upsample 1.5× → stage 2
    X2        = 'X2'          # Stage 1 at W/2   × H/2   → upsample 2×   → stage 2

    def __str__(self) -> str:
        return self.value

    def is_two_stage(self) -> bool:
        return self != LtxMultiScaleMode.FULL_SIZE

    def reduction_factor(self) -> float:
        return {
            LtxMultiScaleMode.X1_5: 1.5,
            LtxMultiScaleMode.X2: 2.0,
        }.get(self, 1.0)
