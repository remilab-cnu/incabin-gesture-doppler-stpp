"""MobileNetV3-Small and MobileNetV3-Large (Howard et al., ICCV 2019)
adapted for (100, 30, 3) Doppler spectra.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, activations


class HSwish(layers.Layer):
    def call(self, inputs):
        return inputs * activations.relu(inputs + 3.0, max_value=6.0) / 6.0


class SEBlock(layers.Layer):
    """Squeeze-and-excitation. ``activation`` is the gating non-linearity."""

    def __init__(self, in_channels, reduction=4, gate="sigmoid"):
        super().__init__()
        self.reduce = layers.Conv2D(in_channels // reduction, 1, activation="relu")
        self.expand = layers.Conv2D(in_channels, 1, activation=gate)

    def call(self, inputs):
        c = inputs.shape[-1]
        x = layers.GlobalAveragePooling2D()(inputs)
        x = layers.Reshape((1, 1, c))(x)
        x = self.reduce(x)
        x = self.expand(x)
        return inputs * x


class Bottleneck(layers.Layer):
    """MBConv block: expand -> depthwise -> (SE) -> project, optional residual."""

    def __init__(self, in_c, out_c, k, s, exp, se=False, nl="RE", se_gate="sigmoid",
                 act_after_dw=False):
        super().__init__()
        self.use_res = (s == 1 and in_c == out_c)
        self.act_after_dw = act_after_dw
        self.pw = layers.Conv2D(exp, 1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.dw = layers.DepthwiseConv2D(k, strides=s, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.se = SEBlock(exp, gate=se_gate) if se else layers.Lambda(lambda x: x)
        self.project = layers.Conv2D(out_c, 1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.act = HSwish() if nl == "HS" else layers.ReLU()

    def call(self, x):
        out = self.act(self.bn1(self.pw(x)))
        out = self.bn2(self.dw(out))
        if self.act_after_dw:
            out = self.act(out)
        out = self.se(out)
        out = self.bn3(self.project(out))
        return x + out if self.use_res else out


def MobileNetV3Small(input_shape=(100, 30, 3), num_classes=6):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = HSwish()(x)

    cfg = [
        (16, 16, 3, 2, 16, True, "RE"),
        (16, 24, 3, 2, 72, False, "RE"),
        (24, 24, 3, 1, 88, False, "RE"),
        (24, 40, 5, 2, 96, True, "HS"),
        (40, 40, 5, 1, 240, True, "HS"),
        (40, 40, 5, 1, 240, True, "HS"),
        (40, 48, 5, 1, 120, True, "HS"),
        (48, 48, 5, 1, 144, True, "HS"),
        (48, 96, 5, 2, 288, True, "HS"),
        (96, 96, 5, 1, 576, True, "HS"),
        (96, 96, 5, 1, 576, True, "HS"),
    ]
    for in_c, out_c, k, s, exp, se, nl in cfg:
        x = Bottleneck(in_c, out_c, k, s, exp, se, nl)(x)

    x = layers.Conv2D(576, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = HSwish()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 576))(x)
    x = layers.Conv2D(1024, 1)(x)
    x = HSwish()(x)
    x = layers.Conv2D(num_classes, 1)(x)
    x = layers.Flatten()(x)
    out = layers.Softmax()(x)
    return models.Model(inputs, out, name="MobileNetV3Small")


def MobileNetV3Large(input_shape=(100, 30, 3), num_classes=6):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = HSwish()(x)

    cfg = [
        (16, 16, 3, 1, 16,  False, "RE"),
        (16, 24, 3, 2, 64,  False, "RE"),
        (24, 24, 3, 1, 72,  False, "RE"),
        (24, 40, 5, 2, 72,  True,  "RE"),
        (40, 40, 5, 1, 120, True,  "RE"),
        (40, 40, 5, 1, 120, True,  "RE"),
        (40, 80, 3, 2, 240, False, "HS"),
        (80, 80, 3, 1, 200, False, "HS"),
        (80, 80, 3, 1, 184, False, "HS"),
        (80, 80, 3, 1, 184, False, "HS"),
        (80, 112, 3, 1, 480, True,  "HS"),
        (112, 112, 3, 1, 672, True,  "HS"),
        (112, 160, 5, 2, 672, True,  "HS"),
        (160, 160, 5, 1, 960, True,  "HS"),
        (160, 160, 5, 1, 960, True,  "HS"),
    ]
    # MobileNetV3-Large uses hard-sigmoid SE gating and applies the
    # non-linearity after the depthwise conv (V3 practice).
    for in_c, out_c, k, s, exp, use_se, nl in cfg:
        x = Bottleneck(in_c, out_c, k, s, exp, use_se, nl,
                       se_gate="hard_sigmoid", act_after_dw=True)(x)

    x = layers.Conv2D(960, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = HSwish()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 960))(x)
    x = layers.Conv2D(1280, 1, use_bias=True)(x)
    x = HSwish()(x)
    x = layers.Conv2D(num_classes, 1, use_bias=True)(x)
    x = layers.Flatten()(x)
    outputs = layers.Softmax()(x)
    return models.Model(inputs, outputs, name="MobileNetV3Large")
