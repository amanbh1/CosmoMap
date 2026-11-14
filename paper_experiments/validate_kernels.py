"""Compare CIC, TSC and PCS gridding of the dark matter particle speeds of one hydrodynamical snapshot with the
published CMD velocity grid of the same simulation (log correlation over all cells, median deviation).
PCS reproduces the published grid; it is the scheme grid_velocity.py uses for the gravity-only runs.

    python validate_kernels.py <simulation index>
"""
import sys
import numpy as np
from common import RAW
from grid_velocity import BASE, read_dm, cic_mass_speed
from kernels import deposit_kernel

sim = int(sys.argv[1])
pos, vel = read_dm(f"{BASE}/IllustrisTNG/LH/LH_{sim}/snapshot_090.hdf5")
sp = np.linalg.norm(vel.astype(np.float64), axis=1)
ref = np.load(str(RAW / "Grids_Vcdm_IllustrisTNG_LH_128_z=0.0.npy"), mmap_mode="r")[sim].astype(np.float64)
for name, g in (("cic", cic_mass_speed(pos, sp)), ("tsc", deposit_kernel(pos, sp, 3)[0]), ("pcs", deposit_kernel(pos, sp, 4)[0])):
    zero = g <= 0
    lg = np.log10(np.maximum(g, 1.0)); lr = np.log10(ref)
    print(f"{name}: zero-frac {zero.mean():.4f}  log-corr all cells {np.corrcoef(lg.ravel(), lr.ravel())[0, 1]:.5f}  "
          f"median |ratio-1| (nonzero) {np.median(np.abs(g[~zero] / ref[~zero] - 1)):.4f}", flush=True)
cz = cic_mass_speed(pos, sp) <= 0
print("CMD values in CIC-empty cells: median %.1f, vs overall median %.1f" % (np.median(ref[cz]), np.median(ref)))
