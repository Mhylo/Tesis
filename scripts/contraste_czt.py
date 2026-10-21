"""Contraste término a término del MPASM con la transformada Z chirp (CZT).

Tarea 32 del cronograma, la parte que faltaba ("falta el contraste término a
término MPASM vs CZT"). Cada matriz de la DFT del MPASM es exp(±i2π a_n b_k)
con a_n = a_0 + n·da y b_k = b_0 + k·db uniformes, y eso es una CZT:

    sum_n u_n exp(±i2π a_n b_k) = exp(±i2π a_0 b_k) · sum_n u_n z_k^-n,
    z_k = A W^-k,  A = exp(∓i2π da b_0),  W = exp(±i2π da db),

es decir, la CZT de Rabiner, Schafer y Rader (scipy.signal.czt, con la misma
convención) por un factor de fase que centra la malla. Aquí se calcula el
MPASM de CamposT.propagadores.mpasm por los dos caminos, con las mismas
frecuencias, el mismo K_f y la misma H, y se comparan:

    espectro      F de las matrices frente a F de dos CZT (columnas y filas)
    propagación   el campo de mpasm() frente al de la CZT, ida completa
    tiempo        una propagación por cada camino: matrices en CPU (complex128)
                  y en GPU (complex64), CZT en CPU (complex128; SciPy no la
                  tiene en GPU)

Escenario: la Tabla 1 de Zhao (CamposT.montaje.TABLA1), el haz gaussiano.
Salida: resultados/contraste_czt/{acuerdo,tiempos}.csv.

RESULTADO (2026-10-21). Los dos caminos dan el mismo espectro y el mismo
campo: diferencia relativa máxima de 4e-12 a 4e-10 en las nueve
combinaciones de z y s, redondeo. En CPU y en doble cuestan lo mismo en
orden: con N = 512 ganan las matrices (0.11 frente a 0.15 s, s = 1); desde
N = 1024 la CZT, hasta 1.5 veces con N = 2048 (2.2 frente a 3.4 s). La GPU no
estaba visible ese día (cudaErrorNoDevice): para ella, la Tabla 4.1
(scripts/comparacion.py), con el mismo mpasm().

    Tesis_env/Scripts/python.exe -m scripts.contraste_czt
"""

import csv
import pathlib
import time

import numpy as np
from scipy.signal import czt

from CamposT.backend import get_xp, liberar_memoria, sincronizar
from CamposT.montaje import TABLA1
from CamposT.propagadores import aplicar_transferencia, kf_auto, mpasm
from CamposT.referencias import gauss_beam

RAIZ = pathlib.Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "resultados" / "contraste_czt"

#: distancias del acuerdo [mm]: corta, donde el haz sale de la ventana, larga
DISTANCIAS = (500.0, 12e3, 200e3)
SOBREMUESTREOS = (1, 4, 10)
#: mallas del cronómetro
LADOS = (512, 1024, 2048)


def dft_czt(u, entrada, salida, signo, axis):
    """sum_n u_n exp(signo·i2π e_n s_k) a lo largo de axis, con una CZT.

    entrada (e_n) y salida (s_k) son mallas uniformes: es la matriz
    phasor(...) de CamposT.backend aplicada como CZT en lugar de como producto.
    """
    e0, de = entrada[0], entrada[1] - entrada[0]
    s0, ds = salida[0], salida[1] - salida[0]
    a = np.exp(-signo * 2j * np.pi * de * s0)
    w = np.exp(signo * 2j * np.pi * de * ds)
    X = czt(u, m=salida.size, w=w, a=a, axis=axis)
    fase = np.exp(signo * 2j * np.pi * e0 * salida)
    forma = [1, 1]
    forma[axis] = salida.size
    return X * fase.reshape(forma)


def mallas(N, delta, s, Kf):
    """Las de mpasm() con r = mag = 1: x, y, fx, fy (CamposT.propagadores)."""
    x = (np.arange(N) - N / 2) * delta
    f = (np.arange(s * N) - s * N / 2) / (s * delta * N) / Kf
    return x, x.copy(), f, f.copy()


def espectro_czt(U0, x, y, fx, fy, norma):
    """F = My @ U0 @ Mx / norma, con una CZT por eje."""
    G = dft_czt(U0, x, fx, -1, axis=1)                  # (M, Ns)
    return dft_czt(G, y, fy, -1, axis=0) / norma         # (Ms, Ns)


def espectro_matrices(U0, x, y, fx, fy, norma):
    """El de mpasm(), en NumPy y en doble."""
    Mx = np.exp(-2j * np.pi * np.outer(x, fx))
    My = np.exp(-2j * np.pi * np.outer(fy, y))
    return My @ U0 @ Mx / norma


def mpasm_czt(U0, delta, lamb, z, s, Kf):
    """mpasm() con r = mag = 1, con las dos DFT hechas por CZT."""
    N = U0.shape[0]
    x, y, fx, fy = mallas(N, delta, s, Kf)
    F = espectro_czt(U0, x, y, fx, fy, float(s**2 * N * N * Kf * Kf))
    aplicar_transferencia(F, fx, fy, lamb, z, np)
    G = dft_czt(F, fx, x, +1, axis=1)                   # (Ms, N)
    return dft_czt(G, fy, y, +1, axis=0)                 # (N, N)


def relativa(a, b):
    """max|a - b| / max|b|."""
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def acuerdo():
    N, delta, lamb, w0 = TABLA1.N, TABLA1.delta, TABLA1.lamb, TABLA1.W0
    U0 = np.asarray(gauss_beam(N, delta, w0, device="cpu", dtype=np.complex128)[0])
    filas = []
    print(f"Tabla 1: N = {N}, delta = {delta * 1e3:.2f} um, lamb = {lamb * 1e6:.1f} nm")
    print(f"{'z [m]':>7} {'s':>3} {'Kf':>6} {'espectro':>10} {'propagación':>12}")
    for z in DISTANCIAS:
        for s in SOBREMUESTREOS:
            Kf = kf_auto(N, delta, lamb, z, s=s)
            x, y, fx, fy = mallas(N, delta, s, Kf)
            norma = float(s**2 * N * N * Kf * Kf)
            e_F = relativa(espectro_czt(U0, x, y, fx, fy, norma),
                           espectro_matrices(U0, x, y, fx, fy, norma))
            U_mat, kf_usado = mpasm(U0, delta, lamb, z, s=s, Kf=Kf, device="cpu",
                                    dtype=np.complex128)
            e_U = relativa(mpasm_czt(U0, delta, lamb, z, s, Kf), np.asarray(U_mat))
            print(f"{z / 1e3:7.1f} {s:3d} {Kf:6.2f} {e_F:10.1e} {e_U:12.1e}")
            filas.append(dict(z_mm=z, s=s, kf=f"{Kf:.4f}", error_espectro=f"{e_F:.3e}",
                              error_propagacion=f"{e_U:.3e}"))
    return filas


def _cronometro(fn, sincronizar_gpu=False, repeticiones=2):
    """Mediana de repeticiones, tras una pasada de calentamiento."""
    fn()
    if sincronizar_gpu:
        sincronizar()
    ts = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fn()
        if sincronizar_gpu:
            sincronizar()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def tiempos(z=12e3):
    lamb, w0, L0 = TABLA1.lamb, TABLA1.W0, TABLA1.L0
    hay_gpu = get_xp("auto")[1] == "gpu"
    filas = []
    print(f"\nTiempo de una propagación (z = {z / 1e3:g} m, K_f automático), en s")
    print(f"{'N':>5} {'s':>3} {'matrices CPU':>13} {'CZT CPU':>9} {'matrices GPU':>13}")
    for N in LADOS:
        delta = L0 / N
        for s in (1, 4):
            Kf = kf_auto(N, delta, lamb, z, s=s)
            U0 = np.asarray(gauss_beam(N, delta, w0, device="cpu",
                                       dtype=np.complex128)[0])
            fila = dict(N=N, s=s)
            try:
                fila["matrices_cpu_s"] = _cronometro(
                    lambda: mpasm(U0, delta, lamb, z, s=s, Kf=Kf, device="cpu",
                                  dtype=np.complex128), repeticiones=1)
            except MemoryError:
                fila["matrices_cpu_s"] = float("nan")
            try:
                fila["czt_cpu_s"] = _cronometro(lambda: mpasm_czt(U0, delta, lamb, z, s, Kf))
            except MemoryError:
                fila["czt_cpu_s"] = float("nan")
            fila["matrices_gpu_s"] = float("nan")
            if hay_gpu:
                xp = get_xp("gpu")[0]
                Ug = xp.asarray(U0.astype(np.complex64))
                try:
                    fila["matrices_gpu_s"] = _cronometro(
                        lambda: mpasm(Ug, delta, lamb, z, s=s, Kf=Kf, device="gpu"),
                        sincronizar_gpu=True)
                except Exception as e:                  # memoria de la tarjeta
                    print(f"  GPU N = {N}, s = {s}: {type(e).__name__}")
                del Ug
                liberar_memoria()
            print(f"{N:5d} {s:3d} {fila['matrices_cpu_s']:13.3f} {fila['czt_cpu_s']:9.3f} "
                  f"{fila['matrices_gpu_s']:13.4f}")
            filas.append({k: (f"{v:.4g}" if isinstance(v, float) else v)
                          for k, v in fila.items()})
    return filas


def guardar(nombre, filas):
    SALIDA.mkdir(parents=True, exist_ok=True)
    with open(SALIDA / nombre, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"  -> {SALIDA / nombre}")


if __name__ == "__main__":
    guardar("acuerdo.csv", acuerdo())
    guardar("tiempos.csv", tiempos())
