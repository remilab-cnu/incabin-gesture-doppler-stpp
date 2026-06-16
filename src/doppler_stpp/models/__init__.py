"""Lightweight model zoo for in-cabin Doppler gesture recognition.

All builders share the signature ``build(input_shape, num_classes) -> keras.Model``
and default to the paper's input ``(100, 30, 3)`` and 6 gesture classes.
"""
from .shufflenetv2 import ShuffleNetV2
from .mobilenetv3 import MobileNetV3Small, MobileNetV3Large
from .mobilenetv2 import MobileNetV2
from .ghostnet import GhostNet
from .mobilevit import MobileViT

# Registry keyed by the names used in the manuscript (Table 7).
MODELS = {
    "ShuffleNetV2": lambda input_shape=(100, 30, 3), num_classes=6: ShuffleNetV2(
        input_shape=input_shape, num_classes=num_classes, scale=1.0),
    "MobileNetV3Small": MobileNetV3Small,
    "MobileNetV3Large": MobileNetV3Large,
    "MobileNetV2": lambda input_shape=(100, 30, 3), num_classes=6: MobileNetV2(
        input_shape=input_shape, num_classes=num_classes, alpha=1.0),
    "GhostNet": lambda input_shape=(100, 30, 3), num_classes=6: GhostNet(
        input_shape=input_shape, num_classes=num_classes, width_mult=1.0),
    "MobileViT": MobileViT,
}


def build_model(name, input_shape=(100, 30, 3), num_classes=6):
    """Instantiate a model by its manuscript name (see ``MODELS``)."""
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. choices: {list(MODELS)}")
    return MODELS[name](input_shape=input_shape, num_classes=num_classes)


__all__ = ["MODELS", "build_model", "ShuffleNetV2", "MobileNetV3Small",
           "MobileNetV3Large", "MobileNetV2", "GhostNet", "MobileViT"]
