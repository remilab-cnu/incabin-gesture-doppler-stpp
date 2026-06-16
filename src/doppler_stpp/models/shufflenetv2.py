"""ShuffleNetV2 (Ma et al., ECCV 2018) for (100, 30, 3) Doppler spectra.

This is the best accuracy/compute trade-off backbone in the paper
(1.28 M params, 21.26 MFLOPs).
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def channel_shuffle(x, groups=2):
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    assert c % groups == 0, "Channels must be divisible by groups"
    ch = c // groups
    x = tf.reshape(x, [-1, h, w, groups, ch])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, h, w, c])
    return x


def conv1x1_bn_relu(x, out_channels):
    x = layers.Conv2D(out_channels, 1, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def dwconv3x3_bn(x, stride):
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return x


def shufflenet_v2_unit(x, out_channels):
    """Stride-1 block: channel split -> branch2(1x1,DW,1x1) -> concat -> shuffle."""
    assert out_channels % 2 == 0
    c_half = out_channels // 2
    in_c = x.shape[-1]
    if in_c != out_channels:
        x = conv1x1_bn_relu(x, out_channels)
    x1 = layers.Lambda(lambda t: t[:, :, :, :c_half])(x)
    x2 = layers.Lambda(lambda t: t[:, :, :, c_half:])(x)
    x2 = conv1x1_bn_relu(x2, c_half)
    x2 = dwconv3x3_bn(x2, stride=1)
    x2 = conv1x1_bn_relu(x2, c_half)
    out = layers.Concatenate(axis=-1)([x1, x2])
    out = layers.Lambda(channel_shuffle)(out)
    return out


def shufflenet_v2_downsample(x, out_channels):
    """Stride-2 block: two downsampling branches -> concat -> shuffle."""
    assert out_channels % 2 == 0
    c_half = out_channels // 2
    b1 = dwconv3x3_bn(x, stride=2)
    b1 = conv1x1_bn_relu(b1, c_half)
    b2 = conv1x1_bn_relu(x, c_half)
    b2 = dwconv3x3_bn(b2, stride=2)
    b2 = conv1x1_bn_relu(b2, c_half)
    out = layers.Concatenate(axis=-1)([b1, b2])
    out = layers.Lambda(channel_shuffle)(out)
    return out


def ShuffleNetV2(input_shape=(100, 30, 3), num_classes=6, scale=1.0):
    """ShuffleNetV2 with channels per Table 5 of the original paper."""
    cfg = {
        0.5: dict(stage2=48,  stage3=96,  stage4=192, conv5=1024),
        1.0: dict(stage2=116, stage3=232, stage4=464, conv5=1024),
        1.5: dict(stage2=176, stage3=352, stage4=704, conv5=1024),
        2.0: dict(stage2=244, stage3=488, stage4=976, conv5=2048),
    }
    if scale not in cfg:
        raise ValueError("scale must be one of {0.5, 1.0, 1.5, 2.0}")
    ch = cfg[scale]

    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(24, 3, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    x = shufflenet_v2_downsample(x, ch["stage2"])
    for _ in range(3):
        x = shufflenet_v2_unit(x, ch["stage2"])

    x = shufflenet_v2_downsample(x, ch["stage3"])
    for _ in range(7):
        x = shufflenet_v2_unit(x, ch["stage3"])

    x = shufflenet_v2_downsample(x, ch["stage4"])
    for _ in range(3):
        x = shufflenet_v2_unit(x, ch["stage4"])

    x = layers.Conv2D(ch["conv5"], 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name=f"ShuffleNetV2_{scale}x")
