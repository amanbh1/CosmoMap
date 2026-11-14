"""Build dark matter velocity-modulus grids (128^3) from CAMELS particle snapshots.

validate: read PartType1 of one IllustrisTNG hydro snapshot (remote, HTTP range reads) and compare candidate
          gridding schemes with the CMD Vcdm grid of the same simulation, to reproduce its definition.
nbody:    grid the matched IllustrisTNG_DM (gravity-only) z=0 snapshots of the test simulations with the scheme
          that reproduces CMD (PCS, mass-weighted mean particle speed: log-corr 0.998 over all cells, median deviation 0.7%;
          see validate_kernels.py) and write RAW/Grids_Vcdm_Nbody_test_pcs_z=0.0.npy (rows in split()["test"] order),
          the file passed to ``predict.py --nbody-vel``.

    python grid_velocity.py validate [sim]
    python grid_velocity.py nbody
"""
import sys
import time

import fsspec
import h5py
import hdf5plugin  # noqa: F401  (IllustrisTNG_DM snapshots use a compression filter plugin)
import numpy as np

from common import RAW, split
from kernels import deposit_kernel

BASE = "https://users.flatironinstitute.org/~camels/Sims"
N, L = 128, 25000.0  # grid cells, box in kpc/h


def read_dm(url):
    with fsspec.open(url, "rb", block_size=2**24, cache_type="readahead") as f, h5py.File(f, "r") as h:
        a = float(h["Header"].attrs["Time"])
        pos = h["PartType1/Coordinates"][:].astype(np.float32)
        vel = h["PartType1/Velocities"][:].astype(np.float32) * np.float32(np.sqrt(a))  # Gadget units -> peculiar km/s
    return pos, vel


def cic_weights(pos):
    x = pos / L * N - 0.5  # cell centres at (i + 0.5) L/N
    i0 = np.floor(x).astype(np.int64)
    d = x - i0
    out = []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (d[:, 0] if dx else 1 - d[:, 0]) * (d[:, 1] if dy else 1 - d[:, 1]) * (d[:, 2] if dz else 1 - d[:, 2])
                idx = (((i0[:, 0] + dx) % N) * N + (i0[:, 1] + dy) % N) * N + (i0[:, 2] + dz) % N
                out.append((idx, w))
    return out


def deposit(cw, val):
    g = np.zeros(N ** 3)
    for idx, w in cw:
        g += np.bincount(idx, weights=w * val, minlength=N ** 3)
    return g.reshape(N, N, N)


def cic_mass_speed(pos, speed):
    """CIC-deposit mass (equal particle masses) and mass-weighted speed corner by corner, without holding all
    eight weight arrays in memory; returns the mass-weighted mean speed per cell."""
    x = pos / L * N - 0.5
    i0 = np.floor(x).astype(np.int32)
    d = (x - i0).astype(np.float32)
    del x
    m = np.zeros(N ** 3); v = np.zeros(N ** 3)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (d[:, 0] if dx else 1 - d[:, 0]) * (d[:, 1] if dy else 1 - d[:, 1]) * (d[:, 2] if dz else 1 - d[:, 2])
                idx = (((i0[:, 0] + dx) % N).astype(np.int64) * N + (i0[:, 1] + dy) % N) * N + (i0[:, 2] + dz) % N
                m += np.bincount(idx, weights=w, minlength=N ** 3)
                v += np.bincount(idx, weights=w * speed, minlength=N ** 3)
                del w, idx
    return (v / np.maximum(m, 1e-12)).reshape(N, N, N)


def ngp_idx(pos):
    i = np.floor(pos / L * N).astype(np.int64) % N
    return (i[:, 0] * N + i[:, 1]) * N + i[:, 2]


def schemes(pos, vel):
    cw = cic_weights(pos)
    m = deposit(cw, np.ones(len(pos)))
    vx, vy, vz = (deposit(cw, vel[:, k]) for k in range(3))
    vmod = np.linalg.norm(vel, axis=1)
    out = {"cic_vec_mod": np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) / np.maximum(m, 1e-12),
           "cic_mod": deposit(cw, vmod) / np.maximum(m, 1e-12)}
    ni = ngp_idx(pos)
    mn = np.bincount(ni, minlength=N ** 3)
    vn = [np.bincount(ni, weights=vel[:, k], minlength=N ** 3) for k in range(3)]
    out["ngp_vec_mod"] = (np.sqrt(sum(v ** 2 for v in vn)) / np.maximum(mn, 1)).reshape(N, N, N)
    out["ngp_mod"] = (np.bincount(ni, weights=vmod, minlength=N ** 3) / np.maximum(mn, 1)).reshape(N, N, N)
    return out, m


def validate(sim):
    t0 = time.time()
    pos, vel = read_dm(f"{BASE}/IllustrisTNG/LH/LH_{sim}/snapshot_090.hdf5")
    print(f"read {len(pos)} particles in {time.time() - t0:.0f}s; pos range {pos.min():.1f}-{pos.max():.1f}", flush=True)
    ref = np.load(str(RAW / "Grids_Vcdm_IllustrisTNG_LH_128_z=0.0.npy"), mmap_mode="r")[sim].astype(np.float64)
    print(f"CMD Vcdm: min {ref.min():.3g} median {np.median(ref):.3g} max {ref.max():.3g} zero-frac {(ref <= 0).mean():.4f}")
    out, m = schemes(pos, vel)
    ok = ref > 0
    for k, g in out.items():
        for tn, tr in (("id", lambda a: a), ("T", lambda a: a.transpose(2, 1, 0))):
            gg = tr(g)
            good = ok & (gg > 0)
            r = np.corrcoef(np.log10(gg[good]), np.log10(ref[good]))[0, 1]
            rel = np.median(np.abs(gg[good] / ref[good] - 1))
            print(f"{k:12s} {tn:2s} log-corr {r:.5f}  median |ratio-1| {rel:.4f}  zero-frac {(gg <= 0).mean():.4f}", flush=True)


def nbody():
    import os
    sims = split()["test"]
    path = str(RAW / "Grids_Vcdm_Nbody_test_pcs_z=0.0.npy")
    out = np.lib.format.open_memmap(path, "r+" if os.path.exists(path) else "w+", np.float32, (len(sims), N, N, N))
    done = str(RAW / "vcdm_nbody_pcs_done.txt")
    have = set(open(done).read().split()) if os.path.exists(done) else set()
    for k, sim in enumerate(sims):
        if str(sim) in have:
            continue
        t0 = time.time()
        for _attempt in range(5):
            try:
                pos, vel = read_dm(f"{BASE}/IllustrisTNG_DM/LH/LH_{sim}/snapshot_090.hdf5")
                break
            except OSError as e:  # transient HTTP errors
                print("retry", sim, e, flush=True); time.sleep(30)
        else:
            raise RuntimeError(f"could not read sim {sim}")
        # scheme validated against the CMD Vcdm grid on all cells: PCS mass-weighted mean particle speed
        # (CIC matches only on occupied cells and leaves about 3 per cent of void cells empty)
        speed = np.linalg.norm(vel, axis=1).astype(np.float32)
        del vel
        out[k] = deposit_kernel(pos, speed, 4)[0].astype(np.float32)
        del pos, speed
        out.flush()
        with open(done, "a") as f:
            f.write(f"{sim}\n")
        print(f"{k + 1}/{len(sims)} sim {sim} {time.time() - t0:.0f}s", flush=True)
    print("NBODY_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate(int(sys.argv[2]) if len(sys.argv) > 2 else int(split()["test"][0]))
    else:
        nbody()
