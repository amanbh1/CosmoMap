import numpy as np
import pytest
import tensorflow as tf

from cosmomap.data import make_dataset, make_synthetic_camels
from cosmomap.models import build_model


def _first_batch(config, fit_stats):
    data = make_synthetic_camels(config, n_sims=2, seed=0)
    ds = make_dataset(data, np.arange(data.n_samples), config, training=False, stats=fit_stats(data, config))
    return next(iter(ds))[0]


def test_forward_pass_shape(tiny_config, fit_stats):
    inputs = _first_batch(tiny_config, fit_stats)
    model = build_model(tiny_config)
    out = model(inputs, training=False)
    p = tiny_config.data.patch_size
    assert tuple(out.shape[1:]) == (p, p, p, tiny_config.out_channels)


def test_model_is_conditioned_on_parameters(tiny_config, fit_stats):
    """Changing the conditioning vector must change the prediction (FiLM works)."""
    inputs = _first_batch(tiny_config, fit_stats)
    model = build_model(tiny_config)
    out_a = model(inputs, training=False).numpy()
    perturbed = dict(inputs)
    perturbed["cosmo"] = inputs["cosmo"] + 5.0
    out_b = model(perturbed, training=False).numpy()
    assert not np.allclose(out_a, out_b)


def test_patch_divisibility_guard(tiny_config):
    tiny_config.data.patch_size = 9  # not divisible by 2**depth
    with pytest.raises(ValueError):
        build_model(tiny_config)


@pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"),
    reason="mixed-precision (float16) 3D ops require a GPU",
)
def test_mixed_precision_save_reload(tiny_config, fit_stats, tmp_path):
    """Regression guard for the FiLM mixed-precision dtype bug: a model saved under
    mixed_float16 must reload (under the default float32 policy) and run its FiLM
    layers without a float16/float32 mismatch."""
    from tensorflow import keras

    import cosmomap.models  # noqa: F401  -- register FiLM3D for load

    inputs = _first_batch(tiny_config, fit_stats)

    path = tmp_path / "mp_model.keras"
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        build_model(tiny_config).save(path)
    finally:
        keras.mixed_precision.set_global_policy("float32")

    reloaded = keras.models.load_model(path)
    out = reloaded(inputs, training=False)
    p = tiny_config.data.patch_size
    assert tuple(out.shape[1:]) == (p, p, p, tiny_config.out_channels)
