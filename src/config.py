"""Training and architecture configuration for ResSENet / FER2013 benchmark."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    quick_run: bool = False        # True: 5-epoch smoke test
    epochs: int = 100              # overridden to 5 if quick_run
    batch_size: int = 64
    seed: int = 42
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10             # early stopping patience
    lr_patience: int = 5           # ReduceLROnPlateau patience
    dropout: float = 0.5
    se_ratio: int = 16             # squeeze-and-excitation reduction ratio
    grad_clip: float = 5.0
    num_classes: int = 7
    img_size: int = 48
    emotion_labels: List[str] = field(
        default_factory=lambda: ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    )

    def __post_init__(self):
        if self.quick_run:
            self.epochs = 5


DEFAULT_CONFIG = Config()
