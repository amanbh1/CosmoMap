"""Higher-order mass-assignment kernels (TSC, PCS) for the velocity-grid validation, memory-lean."""
import numpy as np

N, L = 128, 25000.0


def deposit_kernel(pos, val, order):
    x = pos.astype(np.float64) / L * N - 0.5
    if order == 3:
        ic = np.rint(x).astype(np.int64)
        d = x - ic  # in [-0.5, 0.5]
        offs = (-1, 0, 1)
        ws = [0.5 * (0.5 - d) ** 2, 0.75 - d ** 2, 0.5 * (0.5 + d) ** 2]
    else:  # PCS (cubic B-spline), cells i0-1..i0+2
        ic = np.floor(x).astype(np.int64)
        d = x - ic  # in [0,1)
        offs = (-1, 0, 1, 2)
        def b(s):
            s = np.abs(s)
            return np.where(s < 1, (4 - 6 * s ** 2 + 3 * s ** 3) / 6, np.where(s < 2, (2 - s) ** 3 / 6, 0.0))
        ws = [b(d + 1).astype(np.float32), b(d).astype(np.float32), b(d - 1).astype(np.float32), b(d - 2).astype(np.float32)]
        del d, x
    m = np.zeros(N ** 3); v = np.zeros(N ** 3)
    for a, oa in enumerate(offs):
        for bb, ob in enumerate(offs):
            for c, oc in enumerate(offs):
                w = ws[a][:, 0] * ws[bb][:, 1] * ws[c][:, 2]
                idx = (((ic[:, 0] + oa) % N) * N + (ic[:, 1] + ob) % N) * N + (ic[:, 2] + oc) % N
                m += np.bincount(idx, weights=w, minlength=N ** 3)
                v += np.bincount(idx, weights=w * val, minlength=N ** 3)
    return (v / np.maximum(m, 1e-300)).reshape(N, N, N), m.reshape(N, N, N)
