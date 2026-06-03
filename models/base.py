from dataclasses import dataclass
from typing import Protocol, Tuple

import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    family: str
    image_size: int
    feature_shape: Tuple[int, ...]

    @property
    def feature_dim(self) -> int:
        dim = 1
        for value in self.feature_shape:
            dim *= value
        return dim


class ModelAdapter(Protocol):
    spec: ModelSpec

    def load_image_encoder(
        self,
        model_path: str | None = None,
        device: str = "cuda",
    ) -> tuple[nn.Module, object]:
        ...
