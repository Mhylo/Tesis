"""
Propagadores de campo óptico bajo una interfaz común.

Port a NumPy (CPU) del código de Zhao et al., Opt. Lett. 45, 5937 (2020),
más FFT-ASM y BLAS para comparación. Punto de partida del Objetivo 1.

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


# ---------------------------------------------------------------- Kf automático
def kf_auto(N, delta, lamb, z, s=1, formula="paper"):
    """Coeficiente de compresión frecuencial K_f, Ec. (14) del paper."""
    if z == 0:
        return 1.0
    Ns = s * N
    A = (Ns * lamb) ** 2
    B = (8 * Ns * lamb * z) ** 2
    inner = A**2 + B if formula == "paper" else A**2 * B   # 'codigo' = bug original
    fmax = np.sqrt(np.sqrt(inner) - A) / (4 * np.sqrt(2) * z * lamb)
    return max(1.0, (1 / (2 * delta)) / fmax)


# ---------------------------------------------------------------------- MPASM
def mpasm(U0, delta, lamb, z, s=1, Kf=None, r=1, mag=1.0, formula="paper"):
    """Espectro angular por producto matricial.

    Devuelve (campo_salida, Kf_usado). Con s=Kf=r=mag=1 es equivalente a FFT-ASM.
    """
    M, N = U0.shape
    Ms, Ns = s * M, s * N
    if Kf is None:
        Kf = kf_auto(N, delta, lamb, z, s=s, formula=formula)

    x = (np.arange(N) - N / 2) * delta
    y = (np.arange(M) - M / 2) * delta
    fx = (np.arange(Ns) - Ns / 2) / (s * delta * N) / Kf
    fy = (np.arange(Ms) - Ms / 2) / (s * delta * M) / Kf

    # DFT como triple producto matricial, Ec. (3)
    Mx = np.exp(-2j * np.pi * np.outer(x, fx))
    My = np.exp(-2j * np.pi * np.outer(fy, y))
    F = (My @ U0 @ Mx) / (s**2 * M * N) / Kf**2

    F *= transfer_function(fx, fy, lamb, z)

    # DFT inversa con muestreo espacial de salida independiente, Ecs. (6)-(8)
    x1 = (np.arange(r * N) - r * N / 2) * delta * mag
    y1 = (np.arange(r * M) - r * M / 2) * delta * mag
    Mx1 = np.exp(2j * np.pi * np.outer(fx, x1))
    My1 = np.exp(2j * np.pi * np.outer(y1, fy))
    return My1 @ F @ Mx1, Kf


def transfer_function(fx, fy, lamb, z):
    """H(fx,fy;z) para propagación en espacio libre. Ondas evanescentes a cero."""
    uu, vv = np.meshgrid(lamb * fx, lamb * fy)
    arg = 1 - uu**2 - vv**2
    return np.where(arg > 0, np.exp(1j * 2 * np.pi / lamb * z * np.sqrt(np.abs(arg))), 0)


# -------------------------------------------------------------------- FFT-ASM
def fft_asm(U0, delta, lamb, z):
    M, N = U0.shape
    fx = (np.arange(N) - N / 2) / (delta * N)
    fy = (np.arange(M) - M / 2) / (delta * M)
    H = transfer_function(fx, fy, lamb, z)
    F = np.fft.fftshift(np.fft.fft2(U0))
    return np.fft.ifft2(np.fft.ifftshift(F * H))


# ----------------------------------------------------------------------- BLAS
def blas(U0, delta, lamb, z):
    """Espectro angular de banda limitada, Matsushima & Shimobaba (2009)."""
    M, N = U0.shape
    fx = (np.arange(N) - N / 2) / (delta * N)
    fy = (np.arange(M) - M / 2) / (delta * M)
    H = transfer_function(fx, fy, lamb, z)
    flim_x = 1 / (lamb * np.sqrt((2 * z / (delta * N)) ** 2 + 1))
    flim_y = 1 / (lamb * np.sqrt((2 * z / (delta * M)) ** 2 + 1))
    FX, FY = np.meshgrid(fx, fy)
    H = H * ((np.abs(FX) < flim_x) & (np.abs(FY) < flim_y))
    F = np.fft.fftshift(np.fft.fft2(U0))
    return np.fft.ifft2(np.fft.ifftshift(F * H))


# ------------------------------------------------------- referencias analíticas
def gauss_beam(N, delta, w0):
    x = (np.arange(N) - N / 2) * delta
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / w0**2), X, Y


def gauss_analytic(X, Y, w0, lamb, z):
    """Amplitud de un haz gaussiano propagado (solución exacta)."""
    zR = np.pi * w0**2 / lamb
    wz = w0 * np.sqrt(1 + (z / zR) ** 2)
    return (w0 / wz) * np.exp(-(X**2 + Y**2) / wz**2)


# ------------------------------------------------------------------- métricas
def sam(U, ref):
    """Error RMS entre amplitudes normalizadas. Reemplazar por el SAM del paper
    una vez se confirme su definición exacta."""
    a = np.abs(U) / np.abs(U).max()
    b = np.abs(ref) / np.abs(ref).max()
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    # Parámetros de la Tabla 1 del paper (sin la lente)
    L0, N, w0, lamb = 5.0, 512, 1.0, 632.8e-6      # mm
    delta = L0 / (N - 1)
    U0, X, Y = gauss_beam(N, delta, w0)

    print(f"{'z [mm]':>8} {'Kf':>7} {'MPASM':>9} {'FFT-ASM':>9} {'BLAS':>9}")
    for z in (500, 2000, 6000, 12000, 30000, 80000, 200000):
        ref = gauss_analytic(X, Y, w0, lamb, z)
        Um, Kf = mpasm(U0, delta, lamb, z)
        print(f"{z:8d} {Kf:7.2f} {sam(Um, ref):9.5f} "
              f"{sam(fft_asm(U0, delta, lamb, z), ref):9.5f} "
              f"{sam(blas(U0, delta, lamb, z), ref):9.5f}")