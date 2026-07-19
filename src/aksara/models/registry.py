"""Name -> model factory.

Every model in the experiment matrix is addressed by a string, so the matrix
itself stays pure configuration and adding an architecture never touches the
training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn

from .cnn_baselines import LeNet5, SimpleCNN


@dataclass(frozen=True)
class ModelSpec:
    """Static facts a model needs from the harness before it is built."""

    name: str
    builder: Callable[..., nn.Module]
    family: str  # "scratch" | "cnn_pretrained" | "transformer"
    default_image_size: int = 64
    grayscale: bool = False
    # Transformers have fixed positional embeddings; timm can interpolate them
    # but only at model-construction time, so the harness must know in advance.
    requires_fixed_size: bool = False
    kwargs: dict = field(default_factory=dict)


def _timm_builder(timm_name: str) -> Callable[..., nn.Module]:
    def build(num_classes: int, in_channels: int = 3, pretrained: bool = True, image_size: int | None = None):
        import timm

        extra = {}
        # Only pass img_size to architectures that accept it; CNNs reject it.
        if image_size is not None and timm_name.startswith(("vit_", "swin_", "deit_")):
            extra["img_size"] = image_size

        return timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
            **extra,
        )

    return build


REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    if spec.name in REGISTRY:
        raise ValueError(f"Duplicate model name: {spec.name}")
    REGISTRY[spec.name] = spec


# --- From-scratch baselines -------------------------------------------------
register(ModelSpec("lenet5", LeNet5, "scratch", 32, grayscale=True))
for _depth in (2, 3, 4):
    register(
        ModelSpec(
            f"simplecnn_d{_depth}",
            SimpleCNN,
            "scratch",
            64,
            kwargs={"depth": _depth},
        )
    )
register(
    ModelSpec(
        "simplecnn_d3_nobn",
        SimpleCNN,
        "scratch",
        64,
        kwargs={"depth": 3, "batch_norm": False},
    )
)

# --- Pretrained CNN backbones ----------------------------------------------
for _name, _timm in [
    ("resnet18", "resnet18"),
    ("resnet50", "resnet50"),
    ("efficientnet_b0", "efficientnet_b0"),
    ("mobilenetv3_small", "mobilenetv3_small_100"),
    ("densenet121", "densenet121"),
    ("convnext_tiny", "convnext_tiny"),
]:
    register(ModelSpec(_name, _timm_builder(_timm), "cnn_pretrained", 224))

# --- Transformers -----------------------------------------------------------
# Fixed at 224: these checkpoints' positional embeddings and window sizes are
# tied to that resolution, so they sit out the image-size ablation.
for _name, _timm in [
    ("vit_tiny", "vit_tiny_patch16_224"),
    ("vit_small", "vit_small_patch16_224"),
    ("deit_small", "deit_small_patch16_224"),
    ("swin_tiny", "swin_tiny_patch4_window7_224"),
]:
    register(ModelSpec(_name, _timm_builder(_timm), "transformer", 224, requires_fixed_size=True))


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    image_size: int | None = None,
) -> tuple[nn.Module, ModelSpec]:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(REGISTRY)}")

    spec = REGISTRY[name]
    in_channels = 1 if spec.grayscale else 3
    kwargs = dict(spec.kwargs)

    if spec.family == "scratch":
        # Scratch models have no pretrained weights to load, so the flag is
        # meaningless for them and passing it would raise.
        model = spec.builder(num_classes=num_classes, in_channels=in_channels, **kwargs)
    else:
        model = spec.builder(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            image_size=image_size,
            **kwargs,
        )

    return model, spec


def models_by_family(family: str) -> list[str]:
    return sorted(n for n, s in REGISTRY.items() if s.family == family)
