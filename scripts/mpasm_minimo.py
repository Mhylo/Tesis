"""MPASM, Zhao et al., Opt. Lett. 45, 5937 (2020). Solo el propagador.

gpu=True corre el mismo codigo con CuPy (como MatrixDftGPU del codigo de
referencia). Las fases se calculan siempre en float64; solo el fasor ya
calculado baja al dtype de trabajo, que en GPU es complex64 por defecto.
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def kf(N, delta, lamb, z, s):
    """Ec. (14) con (sN·lamb)^4, recortada a Kf >= 1."""
    if z == 0:
        return 1.0
    z = abs(z)
    A = (s * N * lamb) ** 2
    B = 64 * (s * N * lamb * z) ** 2
    fmax = np.sqrt(np.sqrt(A**2 + B) - A) / (4 * np.sqrt(2) * z * lamb)
    return max(1.0, (1 / (2 * delta)) / fmax)


def mpasm(U0, delta, lamb, z, s=1, r=1, mag=1.0, gpu=False, dtype=None):
    """Campo U0 (M filas = y, N columnas = x) propagado una distancia z.

    Con gpu=True devuelve un array de CuPy; bajalo con .get().
    """
    if gpu and cp is None:
        raise RuntimeError("gpu=True pero CuPy no esta instalado")
    xp = cp if gpu else np
    dtype = dtype or (np.complex64 if gpu else np.complex128)

    M, N = U0.shape
    Kfy, Kfx = kf(M, delta, lamb, z, s), kf(N, delta, lamb, z, s)
    # la salida (r·N·delta·mag) debe caber en el periodo del espectro (s·N·delta·Kf)
    if r * mag > s * min(Kfy, Kfx) * (1 + 1e-12):
        raise ValueError(f"r*mag = {r * mag:g} > s*Kf = {s * min(Kfy, Kfx):g}: "
                         f"la salida traeria copias periodicas del campo")

    x = (xp.arange(N) - N / 2) * delta
    y = (xp.arange(M) - M / 2) * delta
    fx = (xp.arange(s * N) - s * N / 2) / (Kfx * s * N * delta)
    fy = (xp.arange(s * M) - s * M / 2) / (Kfy * s * M * delta)
    x1 = (xp.arange(r * N) - r * N / 2) * delta * mag
    y1 = (xp.arange(r * M) - r * M / 2) * delta * mag

    def fasor(fase):
        return xp.exp(1j * fase).astype(dtype, copy=False)

    U = xp.asarray(U0, dtype=dtype)
    U = fasor(-2 * np.pi * xp.outer(fy, y)) @ U
    F = U @ fasor(-2 * np.pi * xp.outer(x, fx))
    del U

    arg = 1 - (lamb * fx[None, :]) ** 2 - (lamb * fy[:, None]) ** 2
    F *= xp.where(arg > 0, fasor(2 * np.pi / lamb * z * xp.sqrt(xp.maximum(arg, 0))), 0)
    del arg

    F = fasor(2 * np.pi * xp.outer(y1, fy)) @ F
    U = F @ fasor(2 * np.pi * xp.outer(fx, x1))
    return U / (M * N * Kfx * Kfy * s**2)
