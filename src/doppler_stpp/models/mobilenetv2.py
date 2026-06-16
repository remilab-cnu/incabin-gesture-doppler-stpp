"""MobileNetV2 (Sandler et al., CVPR 2018) for (100, 30, 3) Doppler spectra."""
from tensorflow.keras import layers, models


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def inverted_residual(x, t, c, s, n, block_id_start):
    """Inverted residual (linear bottleneck) stage with ``n`` repeats."""
    in_ch = x.shape[-1]
    out_ch = c
    for i in range(n):
        stride = s if i == 0 else 1
        shortcut = x
        expanded = x
        expanded_ch = int(in_ch * t)
        if t != 1:
            expanded = layers.Conv2D(expanded_ch, 1, padding="same", use_bias=False,
                                     name=f"block{block_id_start+i}_expand")(expanded)
            expanded = layers.BatchNormalization(
                name=f"block{block_id_start+i}_expand_bn")(expanded)
            expanded = layers.ReLU(max_value=6.0,
                                   name=f"block{block_id_start+i}_expand_relu")(expanded)
        dw = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False,
                                    name=f"block{block_id_start+i}_dw")(expanded)
        dw = layers.BatchNormalization(name=f"block{block_id_start+i}_dw_bn")(dw)
        dw = layers.ReLU(max_value=6.0, name=f"block{block_id_start+i}_dw_relu")(dw)
        proj = layers.Conv2D(out_ch, 1, padding="same", use_bias=False,
                             name=f"block{block_id_start+i}_project")(dw)
        proj = layers.BatchNormalization(name=f"block{block_id_start+i}_project_bn")(proj)
        if (stride == 1) and (in_ch == out_ch):
            x = layers.Add(name=f"block{block_id_start+i}_add")([shortcut, proj])
        else:
            x = proj
        in_ch = x.shape[-1]
    return x


def MobileNetV2(input_shape=(100, 30, 3), num_classes=6, alpha=1.0,
                include_top=True, pooling="avg", dropout=0.0):
    inputs = layers.Input(shape=input_shape)
    first_channels = _make_divisible(32 * alpha) if alpha != 1.0 else 32
    x = layers.Conv2D(first_channels, 3, strides=2, padding="same", use_bias=False,
                      name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6.0, name="stem_relu")(x)

    def C(ch):
        ch = int(ch * alpha)
        return _make_divisible(ch) if alpha != 1.0 else ch

    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
    block_id = 0
    for (t, c, n, s) in cfg:
        x = inverted_residual(x, t=t, c=C(c), s=s, n=n, block_id_start=block_id)
        block_id += n

    x = layers.Conv2D(1280, 1, padding="same", use_bias=False, name="last_conv")(x)
    x = layers.BatchNormalization(name="last_bn")(x)
    x = layers.ReLU(max_value=6.0, name="last_relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avgpool")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name="dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    else:
        if pooling == "avg":
            outputs = layers.GlobalAveragePooling2D(name="avgpool")(x)
        elif pooling == "max":
            outputs = layers.GlobalMaxPooling2D(name="maxpool")(x)
        else:
            outputs = x
    return models.Model(inputs, outputs, name=f"MobileNetV2_alpha{alpha}")
