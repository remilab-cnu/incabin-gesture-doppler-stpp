"""MobileViT-XS (Mehta & Rastegari, ICLR 2022) for (100, 30, 3) Doppler spectra.

MobileViT block: local 3x3 conv -> 1x1 projection -> unfold into non-overlapping
2x2 patches -> Transformer x L -> fold -> 1x1 -> concat with input -> 3x3 fuse.
Layout follows the XS configuration (patch=2, FFN ratio=2.0, Swish).
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def conv_bn_act(x, filters, k=3, s=1, act=True, name=None):
    x = layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False,
                      name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("swish", name=f"{name}_swish")(x) if act else x


class InvertedResidual(layers.Layer):
    """MobileNetV2 block; expansion t=4 per MobileViT-XS."""

    def __init__(self, out_ch, s=1, t=4, name=None):
        super().__init__(name=name)
        self.out_ch, self.s, self.t = out_ch, s, t
        self.seq = None
        self.use_res = None

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        hid = int(in_ch * self.t)
        self.use_res = (self.s == 1 and in_ch == self.out_ch)
        self.seq = models.Sequential(name=f"{self.name}_seq")
        self.seq.add(layers.Conv2D(hid, 1, padding="same", use_bias=False,
                                   name=f"{self.name}_exp"))
        self.seq.add(layers.BatchNormalization(name=f"{self.name}_exp_bn"))
        self.seq.add(layers.Activation("swish", name=f"{self.name}_exp_swish"))
        self.seq.add(layers.DepthwiseConv2D(3, strides=self.s, padding="same",
                                            use_bias=False, name=f"{self.name}_dw"))
        self.seq.add(layers.BatchNormalization(name=f"{self.name}_dw_bn"))
        self.seq.add(layers.Activation("swish", name=f"{self.name}_dw_swish"))
        self.seq.add(layers.Conv2D(self.out_ch, 1, padding="same", use_bias=False,
                                   name=f"{self.name}_proj"))
        self.seq.add(layers.BatchNormalization(name=f"{self.name}_proj_bn"))

    def call(self, x):
        y = self.seq(x)
        return x + y if self.use_res else y


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads=4, ffn_ratio=2.0, drop=0.0, name=None):
        super().__init__(name=name)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln1")
        self.attn = layers.MultiHeadAttention(num_heads=num_heads,
                                              key_dim=d_model // num_heads,
                                              dropout=drop, name=f"{name}_mha")
        self.ln2 = layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln2")
        self.ffn1 = layers.Dense(int(d_model * ffn_ratio), activation="swish",
                                 name=f"{name}_ffn1")
        self.ffn2 = layers.Dense(d_model, name=f"{name}_ffn2")

    def call(self, x):
        h = x
        x = self.ln1(x)
        x = self.attn(x, x)
        x = x + h
        h = x
        x = self.ln2(x)
        x = self.ffn2(self.ffn1(x))
        x = x + h
        return x


class UnfoldPatches(layers.Layer):
    """[B,H,W,d] -> (seq:[B*P,N,d], Hn, Wn, H, W), non-overlapping patches."""

    def __init__(self, patch=2, name=None):
        super().__init__(name=name)
        self.patch = int(patch)

    def call(self, t):
        B = tf.shape(t)[0]
        H = tf.shape(t)[1]
        W = tf.shape(t)[2]
        C = tf.shape(t)[3]
        p = self.patch
        patches = tf.image.extract_patches(
            images=t, sizes=[1, p, p, 1], strides=[1, p, p, 1],
            rates=[1, 1, 1, 1], padding="SAME")
        Hn = tf.shape(patches)[1]
        Wn = tf.shape(patches)[2]
        P = p * p
        patches = tf.reshape(patches, [B, Hn * Wn, P, C])
        patches = tf.transpose(patches, [0, 2, 1, 3])
        seq = tf.reshape(patches, [B * P, Hn * Wn, C])
        return seq, Hn, Wn, H, W


class FoldPatches(layers.Layer):
    """Inverse of UnfoldPatches via depth_to_space, then crop to (H, W)."""

    def __init__(self, d_model, patch=2, name=None):
        super().__init__(name=name)
        self.d_model = int(d_model)
        self.patch = int(patch)
        self.P = self.patch ** 2

    def call(self, inputs):
        seq, Hn, Wn, H, W = inputs
        B = tf.shape(seq)[0] // self.P
        d = tf.shape(seq)[2]
        z = tf.reshape(seq, [B, self.P, tf.shape(seq)[1], d])
        z = tf.transpose(z, [0, 2, 1, 3])
        z = tf.reshape(z, [B, Hn, Wn, self.P * d])
        z = tf.nn.depth_to_space(z, block_size=self.patch)
        z = z[:, :H, :W, :]
        return z


class MobileViTBlock(layers.Layer):
    def __init__(self, in_ch, d_model, L=2, patch=2, name=None):
        super().__init__(name=name)
        self.local3 = layers.Conv2D(in_ch, 3, padding="same", use_bias=False,
                                    name=f"{name}_local3")
        self.local3_bn = layers.BatchNormalization(name=f"{name}_local3_bn")
        self.local3_act = layers.Activation("swish", name=f"{name}_local3_swish")
        self.proj1 = layers.Conv2D(d_model, 1, padding="same", use_bias=False,
                                   name=f"{name}_proj1")
        self.proj1_bn = layers.BatchNormalization(name=f"{name}_proj1_bn")
        self.proj1_act = layers.Activation("swish", name=f"{name}_proj1_swish")
        self.unfold = UnfoldPatches(patch=patch, name=f"{name}_unfold")
        self.tx = [TransformerBlock(d_model, num_heads=4, ffn_ratio=2.0, drop=0.0,
                                    name=f"{name}_tx{i}") for i in range(L)]
        self.fold = FoldPatches(d_model=d_model, patch=patch, name=f"{name}_fold")
        self.proj_back = layers.Conv2D(d_model, 1, padding="same", use_bias=False,
                                       name=f"{name}_proj_back")
        self.proj_back_bn = layers.BatchNormalization(name=f"{name}_proj_back_bn")
        self.proj_back_act = layers.Activation("swish", name=f"{name}_proj_back_swish")
        self.fuse = layers.Conv2D(in_ch, 3, padding="same", use_bias=False,
                                  name=f"{name}_fuse3")
        self.fuse_bn = layers.BatchNormalization(name=f"{name}_fuse3_bn")
        self.fuse_act = layers.Activation("swish", name=f"{name}_fuse3_swish")

    def call(self, x):
        h = x
        x = self.local3(x)
        x = self.local3_bn(x)
        x = self.local3_act(x)
        x = self.proj1(x)
        x = self.proj1_bn(x)
        x = self.proj1_act(x)
        seq, Hn, Wn, H, W = self.unfold(x)
        for blk in self.tx:
            seq = blk(seq)
        xg = self.fold([seq, Hn, Wn, H, W])
        xg = self.proj_back(xg)
        xg = self.proj_back_bn(xg)
        xg = self.proj_back_act(xg)
        y = layers.Concatenate(name=f"{self.name}_concat")([h, xg])
        y = self.fuse(y)
        y = self.fuse_bn(y)
        y = self.fuse_act(y)
        return y


def MobileViT(input_shape=(100, 30, 3), num_classes=6, patch=2):
    """MobileViT-XS layout fixed for (100, 30, 3) inputs."""
    if tuple(input_shape) != (100, 30, 3):
        raise ValueError("This MobileViT-XS layout is fixed for (100, 30, 3).")
    inp = layers.Input(shape=(100, 30, 3), name="input")
    x = conv_bn_act(inp, 16, k=3, s=2, name="stem")
    x = InvertedResidual(32, s=1, t=4, name="s0_mv2")(x)
    x = InvertedResidual(48, s=2, t=4, name="s1_mv2_down")(x)
    x = InvertedResidual(48, s=1, t=4, name="s1_mv2_1")(x)
    x = InvertedResidual(48, s=1, t=4, name="s1_mv2_2")(x)
    x = InvertedResidual(64, s=2, t=4, name="s2_mv2_down")(x)
    x = MobileViTBlock(in_ch=64, d_model=96, L=2, patch=patch, name="s2_mvit")(x)
    x = InvertedResidual(80, s=2, t=4, name="s3_mv2_down")(x)
    x = MobileViTBlock(in_ch=80, d_model=120, L=4, patch=patch, name="s3_mvit")(x)
    x = InvertedResidual(96, s=2, t=4, name="s4_mv2_down")(x)
    x = MobileViTBlock(in_ch=96, d_model=144, L=3, patch=patch, name="s4_mvit")(x)
    x = conv_bn_act(x, 384, k=1, s=1, name="head_1x1")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    out = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return models.Model(inp, out, name="MobileViT_XS_100x30")
