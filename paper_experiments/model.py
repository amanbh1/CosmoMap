"""3D FiLM U-Net (fully convolutional, so it runs on any size divisible by 8).

Architecture: 3 levels (64/128/256 channels),
residual FiLM blocks with GroupNorm(8), spatial dropout, heteroscedastic head.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

from cosmomap.models.blocks import conditioning_network, res_block_3d  # noqa: E402
from cosmomap.models.film import FiLM3D  # noqa: E402


def norm_layer(kind, groups, name):
    """GroupNorm (statistics of the current input window) or BatchNorm (fixed running statistics at
    inference, so predictions do not depend on the window size)."""
    if kind == "batch":
        return layers.BatchNormalization(momentum=0.99, name=name)
    return layers.GroupNormalization(groups=groups, name=name)


def res_block_norm(x, cond, filters, groups, dropout, name, convs=1, norm="group"):
    """res_block_3d with a selectable normalisation; identical to it for norm="group"."""
    if norm == "group":
        return res_block_3d(x, cond, filters, groups, dropout, name=name, convs=convs)
    g = min(groups, filters)
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = layers.Conv3D(filters, 1, padding="same", name=f"{name}_proj")(x)
    h = x
    for j in range(convs):
        h = layers.Conv3D(filters, 3, padding="same", name=f"{name}_conv{j + 1}")(h)
        h = norm_layer(norm, g, f"{name}_gn{j + 1}")(h)
        if j < convs - 1:
            h = layers.Activation("relu", name=f"{name}_relu{j + 1}")(h)
        h = FiLM3D(name=f"{name}_film{j + 1}")([h, cond])
    if dropout > 0:
        h = layers.SpatialDropout3D(dropout, name=f"{name}_drop")(h)
    h = layers.Add(name=f"{name}_add")([shortcut, h])
    return layers.Activation("relu", name=f"{name}_out")(h)


def build(n_in=2, n_out=2, n_cond=7, base=64, depth=3, groups=8, cond_hidden=192, dropout=0.1,
          convs=1, bottleneck_convs=2, norm="group"):
    image = keras.Input(shape=(None, None, None, n_in), name="image")
    cosmo = keras.Input(shape=(n_cond,), name="cosmo")
    cond = conditioning_network(cosmo, cond_hidden)
    filt = [base * 2**i for i in range(depth)]

    x = layers.Conv3D(base, 3, padding="same", name="stem_conv")(image)
    x = norm_layer(norm, groups, "stem_gn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    skips = []
    for i, f in enumerate(filt):
        x = res_block_norm(x, cond, f, groups, dropout, f"enc{i}", convs, norm)
        skips.append(x)
        x = layers.MaxPooling3D(2, name=f"enc{i}_pool")(x)
    x = res_block_norm(x, cond, filt[-1], groups, dropout, "bottleneck", bottleneck_convs, norm)
    for i, f in enumerate(reversed(filt)):
        x = layers.Conv3DTranspose(f, 2, strides=2, padding="same", name=f"dec{i}_up")(x)
        x = layers.Concatenate(name=f"dec{i}_concat")([x, skips.pop()])
        x = res_block_norm(x, cond, f, groups, dropout, f"dec{i}", convs, norm)
    out = layers.Conv3D(2 * n_out, 1, padding="same", name="prediction", dtype="float32")(x)
    return keras.Model({"image": image, "cosmo": cosmo}, out, name="cosmomap")
