import numpy as np
import pytest

from cosmomap.data import (
    denormalize_field,
    fit_normalization,
    load_normalization,
    make_dataset,
    make_synthetic_camels,
    normalize_field,
    save_normalization,
    split_indices,
    split_simulations,
)
from cosmomap.data.camels import load_camels
from cosmomap.data.dataset import crop_periodic, cube_symmetry


def test_synthetic_shapes(tiny_config):
    data = make_synthetic_camels(tiny_config, n_sims=3, seed=0)
    assert data.n_samples == 3 * len(tiny_config.data.redshifts)
    r = tiny_config.data.resolution
    for f in tiny_config.data.fields_in + tiny_config.data.fields_out:
        assert data.fields[f].shape == (data.n_samples, r, r, r)
        assert (data.fields[f] > tiny_config.data.log_floors[f]).all()  # floors lie below the data
    assert data.cond.shape == (data.n_samples, tiny_config.n_cond)
    assert np.isfinite(data.fields["ne"]).all()


def test_dataset_batch_shapes(tiny_config, fit_stats):
    data = make_synthetic_camels(tiny_config, n_sims=2, seed=0)
    ds = make_dataset(data, np.arange(data.n_samples), tiny_config, training=False, stats=fit_stats(data, tiny_config))
    inputs, targets = next(iter(ds))
    p = tiny_config.data.patch_size
    bs = tiny_config.train.batch_size
    assert tuple(inputs["image"].shape) == (bs, p, p, p, tiny_config.n_in)
    assert tuple(inputs["cosmo"].shape) == (bs, tiny_config.n_cond)
    assert tuple(targets.shape) == (bs, p, p, p, tiny_config.n_out)


def test_training_dataset_is_standardised_and_augmented(tiny_config, fit_stats):
    data = make_synthetic_camels(tiny_config, n_sims=4, seed=0)
    ds = make_dataset(data, np.arange(data.n_samples), tiny_config, training=True, stats=fit_stats(data, tiny_config))
    batches = [b for b, _ in zip(ds, range(8), strict=False)]
    images = np.concatenate([b[0]["image"].numpy() for b in batches])
    targets = np.concatenate([b[1].numpy() for b in batches])
    assert np.isfinite(images).all() and np.isfinite(targets).all()
    assert abs(float(targets.mean())) < 1.0 and 0.2 < float(targets.std()) < 3.0  # roughly unit scale


def test_split_indices_partition():
    train, val, test = split_indices(100, [0.8, 0.1, 0.1], seed=1)
    allidx = np.concatenate([train, val, test])
    assert len(allidx) == 100
    assert len(set(allidx.tolist())) == 100  # disjoint, covering


def test_simulation_split_keeps_redshifts_together(tiny_config):
    data = make_synthetic_camels(tiny_config, n_sims=10, seed=0)
    parts = split_simulations(data.sim_ids, tiny_config.data.train_val_test_split, seed=3)
    sims = [set(data.sim_ids[p].tolist()) for p in parts]
    assert not (sims[0] & sims[1]) and not (sims[0] & sims[2]) and not (sims[1] & sims[2])
    assert sum(len(p) for p in parts) == data.n_samples


# --------------------------------------------------------------------------- #
# Normalisation: constants come from the training samples and nothing else
# --------------------------------------------------------------------------- #
def test_normalization_roundtrip():
    rng = np.random.default_rng(0)
    x = (10.0 ** (-7 + 0.5 * rng.standard_normal((4, 8, 8, 8)))).astype(np.float32)
    stats = fit_normalization({"ne": x}, [0, 1, 2])
    normed = normalize_field(x[3], stats["ne"])
    assert np.allclose(denormalize_field(normed, stats["ne"]), x[3], rtol=1e-3)
    train = np.log10(x[:3])
    assert stats["ne"]["mean"] == pytest.approx(float(train.mean()), abs=1e-4)
    assert stats["ne"]["std"] == pytest.approx(float(train.std()), abs=1e-4)


def test_normalization_ignores_held_out_samples():
    """Changing validation/test data must not change the constants (no target statistics at test time)."""
    rng = np.random.default_rng(0)
    fields = {"T": (10.0 ** (5 + rng.standard_normal((6, 4, 4, 4)))).astype(np.float32)}
    before = fit_normalization(fields, [0, 1, 2, 3])
    fields["T"][4:] *= 1000.0  # held-out samples become very different
    assert fit_normalization(fields, [0, 1, 2, 3]) == before


def test_normalization_preserves_amplitude_differences():
    """A box that is ten times hotter stays one dex higher after normalisation."""
    rng = np.random.default_rng(0)
    x = (10.0 ** (5 + 0.3 * rng.standard_normal((2, 4, 4, 4)))).astype(np.float32)
    stats = fit_normalization({"T": x}, [0, 1])["T"]
    shift = normalize_field(10 * x[0], stats) - normalize_field(x[0], stats)
    assert np.allclose(shift, 1.0 / stats["std"], atol=1e-4)


def test_normalization_floor_and_persistence(tmp_path):
    x = np.array([[[[0.0, 1e-9], [1e-7, 1e-6]]]], np.float32).repeat(2, 2)  # contains a zero
    stats = fit_normalization({"ne": x}, [0])
    assert stats["ne"]["floor"] == 1e-12 and np.isfinite(normalize_field(x[0], stats["ne"])).all()
    save_normalization(stats, tmp_path / "normalization.json")
    assert load_normalization(tmp_path / "normalization.json") == stats
    with pytest.raises(FileNotFoundError):
        load_normalization(tmp_path / "missing.json")
    with pytest.raises(ValueError):
        fit_normalization({"ne": x}, [])


# --------------------------------------------------------------------------- #
# Patch sampling
# --------------------------------------------------------------------------- #
def test_periodic_crop_wraps():
    box = np.arange(4**3).reshape(4, 4, 4)
    patch = crop_periodic(box, (3, 0, 2), 2)
    assert patch.shape == (2, 2, 2)
    assert patch[0, 0, 0] == box[3, 0, 2] and patch[1, 1, 1] == box[0, 1, 3]
    assert np.array_equal(crop_periodic(box, (0, 0, 0), 4), box)


def test_cube_symmetries_keep_inputs_and_targets_aligned():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 6, 6, 4))
    seen = set()
    for _ in range(1000):
        y = cube_symmetry(x, rng)
        assert y.shape == x.shape
        assert np.allclose(np.sort(y.ravel()), np.sort(x.ravel()))  # a rearrangement of the same voxels
        assert np.array_equal(np.sort(y[..., 0].ravel()), np.sort(x[..., 0].ravel()))  # channels stay together
        seen.add(y[..., 0].tobytes())
    assert len(seen) == 48  # every symmetry of the cube occurs


# --------------------------------------------------------------------------- #
# CAMELS loader (on small files in the public naming scheme)
# --------------------------------------------------------------------------- #
def test_load_camels_public_layout(tiny_config, tmp_path):
    rng = np.random.default_rng(0)
    n_sims, r = 3, 8
    tiny_config.data.data_root = str(tmp_path)
    tiny_config.data.resolution = r
    params = rng.uniform(0.2, 1.0, (n_sims, 6))
    np.savetxt(tmp_path / "params_LH_IllustrisTNG.txt", params)  # no header, CAMELS column order
    arrays = {}
    for f in ("Mcdm", "Vcdm", "T", "ne"):
        for z in tiny_config.data.redshifts:
            arrays[(f, z)] = rng.uniform(1.0, 2.0, (n_sims, r, r, r)).astype(np.float32)
            np.save(tmp_path / f"Grids_{f}_IllustrisTNG_LH_{r}_z={z:.1f}.npy", arrays[(f, z)])
    data = load_camels(tiny_config, max_sims=2)
    n_z = len(tiny_config.data.redshifts)
    assert data.n_sims == 2 and data.n_samples == 2 * n_z
    assert data.sim_ids.tolist() == [0, 0, 1, 1]  # simulation outer, redshift inner
    assert np.array_equal(data.fields["T"][1 * n_z + 1], arrays[("T", 1.0)][1])
    assert data.cond.shape == (2 * n_z, 7) and data.cond_names[-1] == "z"
    tiny_config.data.cosmo_params = ["sigma_8", "Omega_m"]  # a subset, in another order
    tiny_config.data.include_redshift_in_cond = False
    assert load_camels(tiny_config).cond.shape == (n_sims * n_z, 2)
