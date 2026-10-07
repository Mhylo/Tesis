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


def mpasm(U0, delta, lamb, z, s=1, r=1, mag=1.0, gpu=False, dtype=None,
          ventana=None):
    """Campo U0 (M filas = y, N columnas = x) propagado una distancia z.

    Con gpu=True devuelve un array de CuPy; bajalo con .get().

    ventana = (fila, columna, alto, ancho) calcula SOLO ese trozo de la malla
    de salida (r*M x r*N), y devuelve un array alto x ancho. La entrada sigue
    siendo U0 entero: no se pierde nada del holograma ni cambia Kf. Abarata los
    dos productos de salida; el espectro sM x sN se arma igual.
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
    if ventana is not None:
        f, c, alto, ancho = ventana
        if not (0 <= f and 0 <= c and alto > 0 and ancho > 0
                and f + alto <= r * M and c + ancho <= r * N):
            raise ValueError(f"ventana {ventana} fuera de la salida {r * M}x{r * N}")
        y1, x1 = y1[f:f + alto], x1[c:c + ancho]

    def fasor(fase):
        # cos + i sin escritos directo en el dtype de trabajo: exp(1j*fase)
        # pasaba por un complex128 del tamano entero, el doble que la fase
        out = xp.empty(fase.shape, dtype=dtype)
        out.real = xp.cos(fase)
        out.imag = xp.sin(fase)
        return out

    U = xp.asarray(U0, dtype=dtype)
    U = fasor(-2 * np.pi * xp.outer(fy, y)) @ U
    F = U @ fasor(-2 * np.pi * xp.outer(x, fx))
    del U

    # La funcion de transferencia por bloques de filas y en su sitio. Entera
    # eran varios float64 y complex128 de sM x sN a la vez: con 3000x4000 y
    # s = 3, mas de 4 GB de temporales encima de F, y en Windows el driver no
    # falla, desborda a la RAM y cada z se arrastra. La fase sigue en float64:
    # z*sqrt(arg)/lamb son ~1e5 rad y en float32 se perderia.
    kx2 = (lamb * fx[None, :]) ** 2
    paso = max(1, 2**23 // (s * N))   # ~64 MB de float64 por temporal
    for i in range(0, s * M, paso):
        arg = 1 - kx2 - (lamb * fy[i:i + paso, None]) ** 2
        H = fasor(2 * np.pi / lamb * z * xp.sqrt(xp.maximum(arg, 0)))
        # por el producto y no con H[arg <= 0] = 0: la mascara booleana obliga a
        # CuPy a sincronizar, la llamada dejaba de volver enseguida y el
        # encadenado de barrido_foco (SOLAPAR) perdia la GPU que adelantar
        H *= arg > 0
        F[i:i + paso] *= H
    del kx2, arg, H

    F = fasor(2 * np.pi * xp.outer(y1, fy)) @ F
    U = F @ fasor(2 * np.pi * xp.outer(fx, x1))
    return U / (M * N * Kfx * Kfy * s**2)
