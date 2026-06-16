"""GhostNet (Han et al., CVPR 2020) for (100, 30, 3) Doppler spectra."""
from tensorflow.keras import layers, models


def se_block(x, se_ratio=0.25, name="se"):
    in_ch = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Reshape((1, 1, in_ch), name=f"{name}_reshape")(se)
    se = layers.Conv2D(int(in_ch * se_ratio), 1, activation="relu", use_bias=True,
                       name=f"{name}_reduce")(se)
    se = layers.Conv2D(in_ch, 1, activation="hard_sigmoid", use_bias=True,
                       name=f"{name}_expand")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def ghost_module(x, out_channels, ratio=2, dw_kernel_size=3, use_relu=True, name="ghost"):
    """Ghost module: cheap depthwise ops generate redundant feature maps."""
    init_channels = int(round(out_channels / ratio))
    y = layers.Conv2D(init_channels, 1, padding="same", use_bias=False,
                      name=f"{name}_pw")(x)
    y = layers.BatchNormalization(name=f"{name}_pw_bn")(y)
    if use_relu:
        y = layers.ReLU(name=f"{name}_pw_relu")(y)
    cheap = layers.DepthwiseConv2D(dw_kernel_size, padding="same", use_bias=False,
                                   name=f"{name}_dw")(y)
    cheap = layers.BatchNormalization(name=f"{name}_dw_bn")(cheap)
    if use_relu:
        cheap = layers.ReLU(name=f"{name}_dw_relu")(cheap)
    out = layers.Concatenate(name=f"{name}_concat")([y, cheap])
    out = layers.Lambda(lambda z: z[:, :, :, :out_channels], name=f"{name}_slice")(out)
    return out


def ghost_bottleneck(x, out_channels, exp_channels, stride=1, use_se=False,
                     se_ratio=0.25, name="gbneck"):
    in_channels = x.shape[-1]
    y = ghost_module(x, exp_channels, ratio=2, dw_kernel_size=3, use_relu=True,
                     name=f"{name}_expand")
    if stride != 1:
        y = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False,
                                   name=f"{name}_dw_s")(y)
        y = layers.BatchNormalization(name=f"{name}_dw_s_bn")(y)
    if use_se:
        y = se_block(y, se_ratio=se_ratio, name=f"{name}_se")
    y = ghost_module(y, out_channels, ratio=2, dw_kernel_size=3, use_relu=False,
                     name=f"{name}_project")
    if stride == 1 and in_channels == out_channels:
        shortcut = x
    else:
        shortcut = layers.DepthwiseConv2D(3, strides=stride, padding="same",
                                          use_bias=False, name=f"{name}_sc_dw")(x)
        shortcut = layers.BatchNormalization(name=f"{name}_sc_dw_bn")(shortcut)
        shortcut = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                                 name=f"{name}_sc_pw")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_sc_pw_bn")(shortcut)
    return layers.Add(name=f"{name}_add")([y, shortcut])


def GhostNet(input_shape=(100, 30, 3), num_classes=6, width_mult=1.0, se_ratio=0.25):
    def c(ch):
        return max(1, int(round(ch * width_mult)))

    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(c(16), 3, strides=2, padding="same", use_bias=False,
                      name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    cfgs = [
        (16, 16, False, 1),
        (48, 24, False, 2),
        (72, 24, False, 1),
        (72, 40, True, 2),
        (120, 40, True, 1),
        (240, 80, False, 2),
        (200, 80, False, 1),
        (184, 80, False, 1),
        (184, 80, False, 1),
        (480, 112, True, 1),
        (672, 112, True, 1),
        (672, 160, True, 2),
        (960, 160, False, 1),
        (960, 160, True, 1),
        (960, 160, False, 1),
        (960, 160, True, 1),
    ]
    for i, (exp, out, use_se, s) in enumerate(cfgs, 1):
        x = ghost_bottleneck(x, out_channels=c(out), exp_channels=c(exp), stride=s,
                             use_se=use_se, se_ratio=se_ratio, name=f"gb{i}")

    x = layers.Conv2D(c(960), 1, padding="same", use_bias=False, name="head_conv")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.ReLU(name="head_relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Reshape((1, 1, c(960)), name="post_gap_reshape")(x)
    x = layers.Conv2D(1280, 1, padding="same", use_bias=True, name="conv_final")(x)
    x = layers.ReLU(name="conv_final_relu")(x)
    x = layers.Flatten(name="flatten")(x)
    out = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return models.Model(inp, out, name=f"GhostNet_{width_mult}x")
