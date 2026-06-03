import copy
import gc

import torch
import torch.nn as nn

from models.base import ModelSpec


DEFAULT_MODEL_PATH = "liuhaotian/llava-v1.5-7b"


class LlavaImageEncoder(nn.Module):
    """Wrap the LLaVA vision tower so gradients can flow through it."""

    def __init__(self, vision_tower, device="cuda"):
        super().__init__()
        self.vision_tower = vision_tower
        self.device = torch.device(device)
        self._forward_impl = getattr(self.vision_tower.forward, "__wrapped__", None)

    def forward(self, x):
        x = x.to(device=self.device)
        if self._forward_impl is not None:
            return self._forward_impl(self.vision_tower, x)
        return self.vision_tower.forward(x)


class Llava15Adapter:
    spec = ModelSpec(
        family="llava-1.5",
        image_size=336,
        feature_shape=(576, 1024),
    )

    def load_image_encoder(self, model_path: str | None = None, device: str = "cuda"):
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        resolved_model_path = model_path or DEFAULT_MODEL_PATH
        resolved_device = torch.device(device)
        print(
            f"Initializing {self.spec.family} image encoder from {resolved_model_path} on {resolved_device} ..."
        )

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=resolved_model_path,
            model_base=None,
            model_name=get_model_name_from_path(resolved_model_path),
        )

        vision_tower_copy = copy.deepcopy(model.model.vision_tower)
        image_encoder = LlavaImageEncoder(vision_tower_copy, device=device).to(resolved_device)
        image_encoder.eval()

        total_params = sum(p.numel() for p in image_encoder.parameters())
        print(f"Vision tower total parameters: {total_params:,}")

        del model
        del tokenizer
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
        print("Image encoder ready.")

        return image_encoder, image_processor
