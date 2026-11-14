"""Periodic-padding version of the CosmoMap U-Net for exact full-box inference.

Every 3x3x3 convolution uses wrap-around padding instead of zero padding, so a whole periodic
simulation box can be predicted in one pass with no tiling boundaries. Layer names and weight
shapes are identical to model.build(), so trained weights are copied over unchanged.
Used by ``predict.py --periodic`` (the "full box" rows of the inference-scheme table).
"""
from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

from cosmomap.models.blocks import conditioning_network  # noqa: E402
from cosmomap.models.film import FiLM3D  # noqa: E402

from model import build, norm_layer  # noqa: E402


@keras.utils.register_keras_serializable(package="cosmomap")
class PeriodicPad3D(layers.Layer):
    """Pad the three spatial axes of a (B, D, H, W, C) tensor by `p` voxels with periodic wrap."""

    def __init__(self, p=1, **kw):
        super().__init__(**kw)
        self.p = p

    def call(self, x):
        p = self.p
        x = tf.concat([x[:, -p:], x, x[:, :p]], axis=1)
        x = tf.concat([x[:, :, -p:], x, x[:, :, :p]], axis=2)
        return tf.concat([x[:, :, :, -p:], x, x[:, :, :, :p]], axis=3)

    def get_config(self):
        return {**super().get_config(), "p": self.p}


def conv3(x, filters, name):
    return layers.Conv3D(filters, 3, padding="valid", name=name)(PeriodicPad3D(1, name=f"{name}_wrap")(x))


def res_block_periodic(x, cond, filters, groups, dropout, name, convs=1, norm="group"):
    g = min(groups, filters)
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = layers.Conv3D(filters, 1, padding="same", name=f"{name}_proj")(x)
    h = x
    for j in range(convs):
        h = conv3(h, filters, f"{name}_conv{j + 1}")
        h = norm_layer(norm, g, f"{name}_gn{j + 1}")(h)
        if j < convs - 1:
            h = layers.Activation("relu", name=f"{name}_relu{j + 1}")(h)
        h = FiLM3D(name=f"{name}_film{j + 1}")([h, cond])
    if dropout > 0:
        h = layers.SpatialDropout3D(dropout, name=f"{name}_drop")(h)
    h = layers.Add(name=f"{name}_add")([shortcut, h])
    return layers.Activation("relu", name=f"{name}_out")(h)


def build_periodic(n_in=2, n_out=2, n_cond=7, base=64, depth=3, groups=8, cond_hidden=192, dropout=0.1,
                   convs=1, bottleneck_convs=2, norm="group"):
    image = keras.Input(shape=(None, None, None, n_in), name="image")
    cosmo = keras.Input(shape=(n_cond,), name="cosmo")
    cond = conditioning_network(cosmo, cond_hidden)
    filt = [base * 2**i for i in range(depth)]
    x = conv3(image, base, "stem_conv")
    x = norm_layer(norm, groups, "stem_gn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    skips = []
    for i, f in enumerate(filt):
        x = res_block_periodic(x, cond, f, groups, dropout, f"enc{i}", convs, norm)
        skips.append(x)
        x = layers.MaxPooling3D(2, name=f"enc{i}_pool")(x)
    x = res_block_periodic(x, cond, filt[-1], groups, dropout, "bottleneck", bottleneck_convs, norm)
    for i, f in enumerate(reversed(filt)):
        x = layers.Conv3DTranspose(f, 2, strides=2, padding="same", name=f"dec{i}_up")(x)
        x = layers.Concatenate(name=f"dec{i}_concat")([x, skips.pop()])
        x = res_block_periodic(x, cond, f, groups, dropout, f"dec{i}", convs, norm)
    out = layers.Conv3D(2 * n_out, 1, padding="same", name="prediction", dtype="float32")(x)
    return keras.Model({"image": image, "cosmo": cosmo}, out, name="cosmomap_periodic")


def copy_weights(src, dst):
    """Copy every weighted layer of `src` into the layer of the same name in `dst`."""
    n = 0
    for layer in src.layers:
        w = layer.get_weights()
        if w:
            dst.get_layer(layer.name).set_weights(w)
            n += 1
    assert n == sum(1 for layer in dst.layers if layer.get_weights()), "weight copy incomplete"
    return dst


def load_periodic(weights_path, base=64, groups=8, norm="group"):
    """Build the standard model, load trained weights, and copy them into the periodic model."""
    src = build(base=base, groups=groups, norm=norm)
    src.load_weights(weights_path)
    return copy_weights(src, build_periodic(base=base, groups=groups, norm=norm))
