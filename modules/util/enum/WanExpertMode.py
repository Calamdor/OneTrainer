from enum import Enum


class WanExpertMode(Enum):
    BOTH = 'BOTH'
    HIGH_NOISE = 'HIGH_NOISE'
    LOW_NOISE = 'LOW_NOISE'

    def __str__(self):
        return self.value
