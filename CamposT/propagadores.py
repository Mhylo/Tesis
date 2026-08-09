"""Propagadores de campo óptico bajo una interfaz común, en CPU o en GPU.

Port del código de Zhao et al., Opt. Lett. 45, 5937 (2020), más FFT-ASM y BLAS
para comparación. Punto de partida del Objetivo 1.

Todas las funciones aceptan device='gpu'|'cpu'|'auto' y un dtype de trabajo.
El cuerpo del algoritmo es el mismo en ambos casos (ver backend.py): lo único
que cambia es si `xp` es CuPy o NumPy, de modo que los tiempos comparados
miden el dispositivo y no dos implementaciones distintas.

Correspondencia con la notación del paper y del código original:
    s   <-> k1 en el código, S1=S2 en el paper. Sobremuestreo en frecuencia.
    Kf  <-> k2 en el código, K_fx=K_fy en el paper. Compresión del intervalo espectral.
    r   <-> k3 en el código, r1=r2 en el paper. Nº de puntos del plano de salida.
    mag <-> k4 en el código. Razón Δx'/Δx (tamaño de la ventana de observación).

OJO: el código original calcula Kf con `A**2 * (8*N*lamb*z)**2` donde la Ec. (14)
del paper dice `A**2 + (8*N*lamb*z)**2`. Es un error de tipeo (`*+` en vez de `+`):
la versión con producto no es dimensionalmente consistente. Aquí se usa la suma.
Ver kf_auto(..., formula='codigo') para reproducir el comportamiento original.
"""

import numpy as np

from CamposT import backend as bk
from CamposT.backend import (a_dispositivo, a_numpy, bloques, dtype_por_defecto,
                             fase_a_complejo, get_xp, phasor)


# ---------------------------------------------------------------- Kf automático
def kf_auto(N, delta, lamb, z, s=1, formula="paper"):
    """Coeficiente de compresión frecuencial K_f, Ec. (14) del paper.

    Es aritmética escalar: se hace siempre en doble precisión en CPU, sin
    importar el backend con el que luego se propague.
    """
    if z == 0:
        return 1.0
    Ns = s * N
    A = (Ns * lamb) ** 2
    B = (8 * Ns * lamb * z) ** 2
    inner = A**2 + B if formula == "paper" else A**2 * B   # 'codigo' = bug original
    fmax = np.sqrt(np.sqrt(inner) - A) / (4 * np.sqrt(2) * z * lamb)
    return max(1.0, (1 / (2 * delta)) / fmax)


# ---------------------------------------------------------------------- MPASM
def memoria_mpasm(M, N, s=10, r=1, dtype=np.complex64):
    """Bytes de pico estimados para mpasm(). Útil antes de lanzar en GPU."""
    itemsize = np.dtype(dtype).itemsize
    Ms, Ns = s * M, s * N
    elementos = (Ms * Ns          # espectro F
                 + M * Ns         # intermedio My @ U0
                 + N * Ns         # Mx
                 + Ms * M         # My
                 + Ns * (r * N)   # Mx1
                 + (r * M) * Ms   # My1
                 + (r * M) * Ns)  # intermedio My1 @ F
    return elementos * itemsize + bk.PRESUPUESTO_BLOQUE


def mpasm(U0, delta, lamb, z, s=10, Kf=None, r=1, mag=1.0, formula="paper",
          device="auto", dtype=None):
    """Espectro angular por producto matricial.

    Devuelve (campo_salida, Kf_usado). Con s=Kf=r=mag=1 es equivalente a FFT-ASM.
    El campo devuelto vive en el dispositivo pedido; usa backend.a_numpy() para
    bajarlo.

    Es el método que más gana con CUDA: el grueso del coste son dos productos
    matriciales densos encadenados, justo lo que cuBLAS hace bien.
    """
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)
    U0 = a_dispositivo(U0, xp, dtype)

    M, N = U0.shape
    Ms, Ns = s * M, s * N
    if Kf is None:
        Kf = kf_auto(N, delta, lamb, z, s=s, formula=formula)

    if dev == "gpu":
        bk.comprobar_memoria(memoria_mpasm(M, N, s, r, dtype),
                             f"mpasm(N={N}, s={s}, r={r}, {np.dtype(dtype).name})",
                             dtype=dtype)

    # mallas en doble precisión: son coordenadas, no datos de campo
    x = (np.arange(N) - N / 2) * delta
    y = (np.arange(M) - M / 2) * delta
    fx = (np.arange(Ns) - Ns / 2) / (s * delta * N) / Kf
    fy = (np.arange(Ms) - Ms / 2) / (s * delta * M) / Kf

    # DFT como triple producto matricial, Ec. (3)
    Mx = phasor(x, fx, -1, xp, dtype)                    # (N, Ns)
    My = phasor(fy, y, -1, xp, dtype)                    # (Ms, M)
    norma = float(s**2 * M * N * Kf**2)
    F = ((My @ U0 @ Mx) / norma).astype(dtype, copy=False)
    del Mx, My

    # H se aplica por bloques de filas y nunca se materializa entera: a s alto
    # duplicaría el pico de memoria del espectro sin necesidad
    aplicar_transferencia(F, fx, fy, lamb, z, xp)

    # DFT inversa con muestreo espacial de salida independiente, Ecs. (6)-(8)
    x1 = (np.arange(r * N) - r * N / 2) * delta * mag
    y1 = (np.arange(r * M) - r * M / 2) * delta * mag
    Mx1 = phasor(fx, x1, +1, xp, dtype)                  # (Ns, rN)
    My1 = phasor(y1, fy, +1, xp, dtype)                  # (rM, Ms)
    return My1 @ F @ Mx1, Kf


# ------------------------------------------------------- función de transferencia
def transfer_function(fx, fy, lamb, z, xp=np, dtype=np.complex128):
    """H(fx,fy;z) para propagación en espacio libre. Ondas evanescentes a cero.

    La fase k·z·sqrt(1-λ²f²) puede valer ~1e12 rad, así que se evalúa siempre
    en float64 y sólo se convierte el fasor resultante al dtype de trabajo.
    """
    H = xp.empty((len(fy), len(fx)), dtype=dtype)
    _llenar_transferencia(H, fx, fy, lamb, z, xp)
    return H


def aplicar_transferencia(F, fx, fy, lamb, z, xp):
    """Multiplica F por H(fx,fy;z) in situ, bloque de filas a bloque de filas."""
    _llenar_transferencia(F, fx, fy, lamb, z, xp, multiplicar=True)


def _llenar_transferencia(dst, fx, fy, lamb, z, xp, multiplicar=False):
    fx = xp.asarray(fx, dtype=np.float64)
    fy = xp.asarray(fy, dtype=np.float64)
    k = 2 * np.pi / lamb
    uu = (lamb * fx)[None, :]
    for i0, i1 in bloques(fy.size, fx.size):
        vv = (lamb * fy[i0:i1])[:, None]
        arg = 1.0 - uu**2 - vv**2
        propagante = arg > 0
        H = xp.where(propagante,
                     fase_a_complejo(k * z * xp.sqrt(xp.where(propagante, arg, 0.0)),
                                     dst.dtype, xp),
                     0)
        if multiplicar:
            dst[i0:i1] *= H
        else:
            dst[i0:i1] = H


# -------------------------------------------------------------------- FFT-ASM
def fft_asm(U0, delta, lamb, z, device="auto", dtype=None):
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)
    U0 = a_dispositivo(U0, xp, dtype)

    M, N = U0.shape
    fx = (np.arange(N) - N / 2) / (delta * N)
    fy = (np.arange(M) - M / 2) / (delta * M)
    H = transfer_function(fx, fy, lamb, z, xp, dtype)
    F = xp.fft.fftshift(xp.fft.fft2(U0))
    return xp.fft.ifft2(xp.fft.ifftshift(F * H))


# ----------------------------------------------------------------------- BLAS
def blas(U0, delta, lamb, z, device="auto", dtype=None):
    """Espectro angular de banda limitada, Matsushima & Shimobaba (2009)."""
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)
    U0 = a_dispositivo(U0, xp, dtype)

    M, N = U0.shape
    fx = (np.arange(N) - N / 2) / (delta * N)
    fy = (np.arange(M) - M / 2) / (delta * M)
    H = transfer_function(fx, fy, lamb, z, xp, dtype)
    flim_x = 1 / (lamb * np.sqrt((2 * z / (delta * N)) ** 2 + 1))
    flim_y = 1 / (lamb * np.sqrt((2 * z / (delta * M)) ** 2 + 1))
    FX, FY = xp.meshgrid(xp.asarray(fx), xp.asarray(fy))
    H *= ((xp.abs(FX) < flim_x) & (xp.abs(FY) < flim_y))
    F = xp.fft.fftshift(xp.fft.fft2(U0))
    return xp.fft.ifft2(xp.fft.ifftshift(F * H))


# ------------------------------------------------------- referencias analíticas
def gauss_beam(N, delta, w0, device="auto", dtype=None):
    """Devuelve (U0 en el dispositivo, X, Y en NumPy para las referencias)."""
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)
    x = (np.arange(N) - N / 2) * delta
    X, Y = np.meshgrid(x, x)
    U0 = np.exp(-(X**2 + Y**2) / w0**2)
    return a_dispositivo(U0, xp, dtype), X, Y


def gauss_analytic(X, Y, w0, lamb, z):
    """Amplitud de un haz gaussiano propagado (solución exacta)."""
    zR = np.pi * w0**2 / lamb
    wz = w0 * np.sqrt(1 + (z / zR) ** 2)
    return (w0 / wz) * np.exp(-(X**2 + Y**2) / wz**2)


# ------------------------------------------------------------------- métricas
def sam(U, ref):
    """Error RMS entre amplitudes normalizadas. Reemplazar por el SAM del paper
    una vez se confirme su definición exacta. Acepta arrays de CPU o de GPU."""
    a = np.abs(a_numpy(U)).astype(np.float64)
    b = np.abs(a_numpy(ref)).astype(np.float64)
    a /= a.max()
    b /= b.max()
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    from CamposT.backend import cronometrar, gpu_disponible, info_gpu, liberar_memoria

    # Parámetros de la Tabla 1 del paper (sin la lente)
    L0, N, w0, lamb = 5.0, 512, 1.0, 632.8e-6      # mm
    delta = L0 / (N - 1)
    Z = (500, 2000, 6000, 12000, 30000, 80000, 200000)

    if gpu_disponible():
        gi = info_gpu()
        print(f"GPU: {gi['nombre']} (cc {gi['capacidad']}), "
              f"{gi['VRAM total [GB]']:.1f} GB, CuPy {gi['cupy']}")
    else:
        print("Sin GPU: todo en CPU.")

    # --- exactitud contra la solución analítica -------------------------------
    U0_cpu, X, Y = gauss_beam(N, delta, w0, device="cpu")
    print("\nExactitud (SAM contra el gaussiano analítico), CPU complex128:")
    print(f"{'z [mm]':>8} {'Kf':>7} {'MPASM':>9} {'FFT-ASM':>9} {'BLAS':>9}")
    for z in Z:
        ref = gauss_analytic(X, Y, w0, lamb, z)
        Um, Kf = mpasm(U0_cpu, delta, lamb, z, device="cpu")
        print(f"{z:8d} {Kf:7.2f} {sam(Um, ref):9.5f} "
              f"{sam(fft_asm(U0_cpu, delta, lamb, z, device='cpu'), ref):9.5f} "
              f"{sam(blas(U0_cpu, delta, lamb, z, device='cpu'), ref):9.5f}")

    # --- coste de la precisión simple en GPU ----------------------------------
    if gpu_disponible():
        print("\nEfecto del dtype en GPU (SAM contra el analítico):")
        print(f"{'z [mm]':>8} {'c64':>9} {'c128':>9} {'CPU c128':>9}")
        for z in Z:
            ref = gauss_analytic(X, Y, w0, lamb, z)
            U64, _ = mpasm(U0_cpu, delta, lamb, z, device="gpu", dtype=np.complex64)
            U128, _ = mpasm(U0_cpu, delta, lamb, z, device="gpu", dtype=np.complex128)
            Ucpu, _ = mpasm(U0_cpu, delta, lamb, z, device="cpu")
            print(f"{z:8d} {sam(U64, ref):9.5f} {sam(U128, ref):9.5f} "
                  f"{sam(Ucpu, ref):9.5f}")
            liberar_memoria()

        # --- tiempos ----------------------------------------------------------
        z = 12000
        print(f"\nTiempos a z = {z} mm, N = {N}, s = 10 (con sincronización):")
        print(f"{'método':>16} {'CPU c128 [s]':>13} {'GPU c64 [s]':>12} {'x':>7}")
        for nombre, fn in (("MPASM", mpasm), ("FFT-ASM", fft_asm), ("BLAS", blas)):
            _, t_cpu = cronometrar(fn, U0_cpu, delta, lamb, z, device="cpu")
            _, t_gpu = cronometrar(fn, U0_cpu, delta, lamb, z, device="gpu",
                                   dtype=np.complex64)
            print(f"{nombre:>16} {t_cpu:13.3f} {t_gpu:12.3f} {t_cpu / t_gpu:7.1f}")
            liberar_memoria()
