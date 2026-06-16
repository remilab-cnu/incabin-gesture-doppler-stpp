"""MSFE-GAM-SPointNet implementation in TensorFlow/Keras.

Reference:
  Li, W., Guo, Z., Han, Z.
  Millimeter-Wave Radar Point Cloud Gesture Recognition Based on Multiscale Feature Extraction.
  Electronics 14(2), 371 (2025). https://doi.org/10.3390/electronics14020371

The paper adds three modifications on top of SequentialPointNet:
  (1) Multiscale Feature Extraction (MSFE): 1x1 + 3x3 conv parallel branches → concat
  (2) Global Attention Module (GAM) replacing CBAM
  (3) Separable MLP: single-layer MLP on neighbor features, 2-layer MLP on point features, with residual

Input  : (B, F, N, C_in)   where C_in = 4 for (x, y, z, v)
Output : (B, num_classes)  softmax logits

The paper omits exact channel counts; we use values inspired by PointNet++.
"""
import tensorflow as tf
from tensorflow.keras import layers


def _knn_indices(points, k):
    """Return indices of k nearest neighbors for each point.

    points : (B, N, 3)
    return : (B, N, k) int32 indices
    """
    inner = -2.0 * tf.matmul(points, tf.transpose(points, [0, 2, 1]))
    sq = tf.reduce_sum(points * points, axis=-1, keepdims=True)
    dist = sq + inner + tf.transpose(sq, [0, 2, 1])
    _, idx = tf.nn.top_k(-dist, k=k)
    return idx


def _gather_neighbors(features, idx):
    """Gather neighbor features using indices.

    features : (B, N, C)
    idx      : (B, N, k)
    return   : (B, N, k, C)
    """
    B = tf.shape(features)[0]
    N = tf.shape(features)[1]
    k = tf.shape(idx)[-1]
    batch_idx = tf.reshape(tf.range(B), (B, 1, 1))
    batch_idx = tf.tile(batch_idx, (1, N, k))
    gather_idx = tf.stack([batch_idx, idx], axis=-1)
    return tf.gather_nd(features, gather_idx)


class MSFEBlock(layers.Layer):
    """Multiscale feature extraction: parallel 1x1 + 3x3 conv → concat."""
    def __init__(self, channels, **kw):
        super().__init__(**kw)
        self.c1 = layers.Conv2D(channels, (1, 1), padding='same')
        self.b1 = layers.BatchNormalization()
        self.c3 = layers.Conv2D(channels, (3, 3), padding='same')
        self.b3 = layers.BatchNormalization()

    def call(self, x, training=False):
        a = tf.nn.relu(self.b1(self.c1(x), training=training))
        b = tf.nn.relu(self.b3(self.c3(x), training=training))
        return tf.concat([a, b], axis=-1)


class GAMAttention(layers.Layer):
    """Global Attention Module (channel attention + spatial attention)."""
    def __init__(self, channels, reduction=4, **kw):
        super().__init__(**kw)
        red = max(channels // reduction, 4)
        self.ca_fc1 = layers.Dense(red, activation='relu')
        self.ca_fc2 = layers.Dense(channels, activation='sigmoid')
        self.sa_c1 = layers.Conv2D(red, (7, 7), padding='same')
        self.sa_b1 = layers.BatchNormalization()
        self.sa_c2 = layers.Conv2D(channels, (7, 7), padding='same')
        self.sa_b2 = layers.BatchNormalization()
        self.channels = channels

    def call(self, x, training=False):
        ca = tf.reduce_mean(x, axis=[1, 2])  # (B, C)
        ca = self.ca_fc1(ca)
        ca = self.ca_fc2(ca)
        ca = tf.reshape(ca, (-1, 1, 1, self.channels))
        x_ca = x * ca
        sa = tf.nn.relu(self.sa_b1(self.sa_c1(x_ca), training=training))
        sa = tf.nn.sigmoid(self.sa_b2(self.sa_c2(sa), training=training))
        return x_ca * sa


class SeparableMLPBlock(layers.Layer):
    """Separable MLP: neighbor-feature MLP + point-feature MLP, with residual."""
    def __init__(self, channels_list, use_msfe=False, **kw):
        super().__init__(**kw)
        c0, c1, c2 = channels_list
        self.nb_c = layers.Conv2D(c0, (1, 1), padding='same')
        self.nb_b = layers.BatchNormalization()
        if use_msfe:
            self.pt1 = MSFEBlock(c1 // 2)
            self.pt1_is_msfe = True
        else:
            self.pt1_conv = layers.Conv2D(c1, (1, 1), padding='same')
            self.pt1_bn = layers.BatchNormalization()
            self.pt1_is_msfe = False
        self.pt2_conv = layers.Conv2D(c2, (1, 1), padding='same')
        self.pt2_bn = layers.BatchNormalization()
        self.res_proj = layers.Conv2D(c2, (1, 1), padding='same') if c0 != c2 else None

    def call(self, x, training=False):
        nb = tf.nn.relu(self.nb_b(self.nb_c(x), training=training))
        if self.pt1_is_msfe:
            pt = self.pt1(nb, training=training)
        else:
            pt = tf.nn.relu(self.pt1_bn(self.pt1_conv(nb), training=training))
        pt = self.pt2_bn(self.pt2_conv(pt), training=training)
        res = self.res_proj(nb) if self.res_proj is not None else nb
        return tf.nn.relu(pt + res)


class SAModule(layers.Layer):
    """PointNet++-style Set Abstraction layer with separable MLP + optional GAM."""
    def __init__(self, k, channels_list, use_msfe=False, use_gam=False, **kw):
        super().__init__(**kw)
        self.k = k
        self.sep_mlp = SeparableMLPBlock(channels_list, use_msfe=use_msfe)
        self.use_gam = use_gam
        self.gam = GAMAttention(channels_list[2]) if use_gam else None

    def call(self, inputs, training=False):
        xyz = inputs[..., :3]
        idx = _knn_indices(xyz, self.k)
        neighbors = _gather_neighbors(inputs, idx)
        center = tf.expand_dims(inputs, 2)
        rel_xyz = neighbors[..., :3] - center[..., :3]
        nb_feat = tf.concat([rel_xyz, neighbors[..., 3:]], axis=-1)
        feat = self.sep_mlp(nb_feat, training=training)
        if self.gam is not None:
            feat = self.gam(feat, training=training)
        feat = tf.reduce_max(feat, axis=2)
        return tf.concat([xyz, feat], axis=-1)


def positional_encoding(length, d_model):
    """Standard sinusoidal positional encoding."""
    pos = tf.cast(tf.range(length)[:, None], tf.float32)
    i = tf.cast(tf.range(d_model)[None, :], tf.float32)
    angle = pos / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    sin = tf.sin(angle[:, 0::2])
    cos = tf.cos(angle[:, 1::2])
    # interleave
    pe = tf.stack([sin, cos], axis=2)
    pe = tf.reshape(pe, (length, -1))[:, :d_model]
    return pe  # (length, d_model)


def build_msfe_gam_spointnet(
    n_frames=32,
    n_points=32,
    n_features=4,
    n_classes=6,
    sa1_k=8,
    sa2_k=4,
    sa1_channels=(32, 32, 64),
    sa2_channels=(64, 64, 128),
    temporal_channels=(128, 128),
    fc_channels=(128, 64),
    dropout=0.5,
    use_msfe=True,
    use_gam=True,
    use_separable_mlp=True,
    name='msfe_gam_spointnet',
):
    """Construct MSFE-GAM-SPointNet model.

    use_msfe, use_gam, use_separable_mlp are toggles for ablation experiments.
    When all three are False, this approximates the SequentialPointNet baseline.
    """
    inp = layers.Input(shape=(n_frames, n_points, n_features), name='pcd_input')

    # ---- Per-frame spatial feature extraction ----
    # Merge frame dim into batch so SA layers run per-frame.
    class MergeFrames(layers.Layer):
        def call(self, t):
            return tf.reshape(t, (-1, n_points, n_features))

    class SplitFrames(layers.Layer):
        def __init__(self, F, **kw):
            super().__init__(**kw)
            self.F = F
        def call(self, t):
            c = tf.shape(t)[-1]
            return tf.reshape(t, (-1, self.F, c))

    class AddPE(layers.Layer):
        def __init__(self, pe, **kw):
            super().__init__(**kw)
            self.pe = tf.constant(pe, dtype=tf.float32)
        def call(self, t):
            return t + self.pe[None, ...]

    class PerFrameMaxPool(layers.Layer):
        def call(self, t):
            # t: (B*F, N, C+3) where last channels include xyz first; we drop xyz here
            return tf.reduce_max(t[..., 3:], axis=1)

    x_flat = MergeFrames(name='merge_frames')(inp)

    sa1 = SAModule(
        k=sa1_k,
        channels_list=list(sa1_channels),
        use_msfe=use_msfe if use_separable_mlp else False,
        use_gam=use_gam,
        name='sa1',
    )(x_flat)
    sa2 = SAModule(
        k=sa2_k,
        channels_list=list(sa2_channels),
        use_msfe=use_msfe if use_separable_mlp else False,
        use_gam=use_gam,
        name='sa2',
    )(sa1)
    # Per-frame global feature: max pool over points
    per_frame = PerFrameMaxPool(name='per_frame_maxpool')(sa2)  # (B*F, C)
    per_frame_c = per_frame.shape[-1]
    per_frame = SplitFrames(n_frames, name='split_frames')(per_frame)

    # ---- Positional encoding (temporal) ----
    pe = positional_encoding(n_frames, per_frame_c).numpy()
    per_frame = AddPE(pe, name='add_pe')(per_frame)

    # ---- Temporal feature extraction (net4DV_T1, net4DV_T2 analogue) ----
    # Use 1D conv across frames. Optionally MSFE-style multiscale (1 + 3 kernel).
    if use_msfe:
        t1 = layers.Conv1D(temporal_channels[0] // 2, 1, padding='same', name='t1_1')(per_frame)
        t1 = layers.BatchNormalization(name='t1_1_bn')(t1)
        t1 = layers.ReLU(name='t1_1_relu')(t1)
        t3 = layers.Conv1D(temporal_channels[0] // 2, 3, padding='same', name='t1_3')(per_frame)
        t3 = layers.BatchNormalization(name='t1_3_bn')(t3)
        t3 = layers.ReLU(name='t1_3_relu')(t3)
        temporal = layers.Concatenate(axis=-1, name='t1_concat')([t1, t3])
    else:
        temporal = layers.Conv1D(temporal_channels[0], 3, padding='same', name='t1_conv')(per_frame)
        temporal = layers.BatchNormalization(name='t1_bn')(temporal)
        temporal = layers.ReLU(name='t1_relu')(temporal)

    temporal = layers.Conv1D(temporal_channels[1], 1, padding='same', name='t2_conv')(temporal)
    temporal = layers.BatchNormalization(name='t2_bn')(temporal)
    temporal = layers.ReLU(name='t2_relu')(temporal)

    # ---- Aggregation: avg + max pool over frames ----
    gap = layers.GlobalAveragePooling1D(name='gap')(temporal)
    gmp = layers.GlobalMaxPooling1D(name='gmp')(temporal)
    feat = layers.Concatenate(axis=-1, name='agg')([gap, gmp])

    # ---- FC head ----
    h = feat
    for i, fc in enumerate(fc_channels):
        h = layers.Dense(fc, name=f'fc{i+1}')(h)
        h = layers.BatchNormalization(name=f'fc{i+1}_bn')(h)
        h = layers.ReLU(name=f'fc{i+1}_relu')(h)
        h = layers.Dropout(dropout, name=f'fc{i+1}_drop')(h)
    logits = layers.Dense(n_classes, activation='softmax', name='logits')(h)

    return tf.keras.Model(inputs=inp, outputs=logits, name=name)


def build_sequentialpointnet_baseline(**kwargs):
    """SequentialPointNet baseline = MSFE-GAM-SPointNet with all toggles off."""
    return build_msfe_gam_spointnet(
        use_msfe=False,
        use_gam=False,
        use_separable_mlp=False,
        name='sequentialpointnet_baseline',
        **kwargs,
    )


if __name__ == '__main__':
    model = build_msfe_gam_spointnet()
    model.summary(line_length=120)
    # quick forward pass
    import numpy as np
    x = np.random.randn(2, 32, 64, 4).astype(np.float32)
    y = model(x, training=False)
    print(f'\nOutput shape: {y.shape}, sum per row: {y.numpy().sum(axis=1)}')
