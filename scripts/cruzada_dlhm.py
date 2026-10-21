"""Ida y vuelta DLHM con onda esferica, cruzando la cadena de Carlos y el MPASM esferico.

El hermano esferico de scripts/comparacion_cruzada.py, con la geometria del
banco de Carlos: 528 nm, 1.83 um, L = 10 mm, un sensor simulado de N x N y la
muestra a varias z. Cuatro combinaciones con fase (holograma complejo 1 + V,
sin imagen gemela):

    ida    "ASM"    ida_carlos(): el ESPEJO de la cadena de Carlos
                    (reconstruction_dlhm.py, copiada en mi_prueba_dlhm.py). Sus
                    pasos al reves, sobre la malla remuestreada de
                    malla_remuestreada() que cubre la ventana N*delta: la
                    perturbacion e^{ikr}/r (t - 1) en el plano de la muestra;
                    su espectro angular por L - z HACIA el sensor -su
                    angular_spectrum es e^{-ikz}, asi que va con -(L - z)-;
                    division por la esferica en el sensor, remuestreo a la
                    malla del sensor con su resize(), y + 1: la referencia va
                    analitica (ver ida_carlos()).
           "MPASM"  ida_mpasm(): CamposT.dlhm.holograma(complejo=True).
    vuelta "ASM"    vuelta_carlos(): reconstruir() + onda_de_referencia(),
                    U/U0 y la huella en la malla delta*z/L. Es MotorCarlos.campo
                    sin su normalizacion a [0, 1], que con un holograma
                    complejo cambiaria la escala y el error maximo.
           "MPASM"  vuelta_mpasm(): CamposT.dlhm.Reconstructor.

Como en comparacion_cruzada.py, la ida y vuelta con un mismo metodo no mide su
exactitud; las cruzadas si.

    Tesis_env/Scripts/python.exe -m scripts.cruzada_dlhm

-> resultados/cruzada_dlhm/resumen.csv y los .npy de z = 1 y z = 3.
UNIDADES: milimetros.
"""

import csv
import pathlib

import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter

from CamposT.backend import a_numpy, dtype_por_defecto, get_xp
from CamposT.dlhm import Reconstructor, holograma
from CamposT.metricas import sam
from scripts import mi_prueba_dlhm as mp
from scripts.hologramas_carlos import geometria

RAIZ = pathlib.Path(__file__).resolve().parents[1]
OBJETO = RAIZ / "resultados" / "campos" / "entrada.png"
SALIDA = RAIZ / "resultados" / "cruzada_dlhm"
LAMB, DELTA, L = 528e-6, 1.83e-3, 10.0      # el banco de Carlos (CamposT/montaje.py)
N = 1024
ZS = (1.0, 2.0, 3.0, 5.0, 7.0)
GUARDAR = (1.0, 3.0)
BANDA = 500.0       # mm^-1: holograma() con mallas finas (ver dlhm-mpasm-esferico)


def _xp(device):
    xp, dev = get_xp(device)
    return xp, (np.complex128 if xp is np else dtype_por_defecto(dev))


def ida_carlos(t, z, L, lamb, delta, N, device="auto"):
    """El holograma complejo 1 + V en la malla del sensor (N x N), con el espejo
    de la cadena de Carlos. t esta en la malla delta*z/L, N x N."""
    xp, dtype = _xp(device)
    n_r, _ = mp.malla_remuestreada(N, N, z, lamb, delta)
    W = N * delta
    pr, pt = W / n_r, delta * z / L
    # SOLO SE PROPAGA LA PERTURBACION S_z (t - 1), y la referencia S_L se suma
    # analitica en el sensor. Propagar la iluminacion entera no sirve: en la
    # ventana W la esferica de la muestra diverge a M W en el sensor y la FFT
    # la reinyecta por los bordes (sin objeto, en el centro |h - 1| llegaba a 5.7); y
    # recortarla a la huella difracta en su borde (0.74). La referencia del
    # banco no esta recortada en la muestra. Es lo mismo que hace CamposT.dlhm.
    # El objeto solo cubre la huella W z/L: t - 1 va en una submalla central y
    # fuera vale 0 (interpolar_centrado no extrapola).
    m = 2 * int(np.floor((N / 2 - 1) * pt / pr))
    A = xp.zeros((n_r, n_r), dtype=dtype)
    i0 = n_r // 2 - m // 2
    A[i0:i0 + m, i0:i0 + m] = mp.interpolar_centrado(
        xp.asarray(np.asarray(t) - 1, dtype=dtype), pt, (m, m), pr, xp)
    k = 2 * np.pi / lamb
    S_z = mp.fuente_puntual(n_r, n_r, z, 0, 0, lamb, pr, xp=xp, dtype=dtype)
    U = mp.espectro_angular(S_z * A, W, W, k, -(L - z), xp=xp, dtype=dtype)
    del A, S_z
    U /= mp.fuente_puntual(n_r, n_r, L, 0, 0, lamb, pr, xp=xp, dtype=dtype)
    return 1 + mp.resize(a_numpy(U).astype(np.complex128), N, N)


def vuelta_carlos(h, z, L, lamb, delta, device="auto"):
    """t en la malla delta*z/L con la cadena de Carlos: U/U0 y la huella."""
    xp, dtype = _xp(device)
    Rec = mp.reconstruir(h, z, lamb, delta, L, xp=xp, dtype=dtype, sobremuestreo=None)
    U0 = mp.onda_de_referencia(h.shape, z, lamb, delta, L, xp=xp, dtype=dtype,
                               sobremuestreo=None)
    Rec /= xp.abs(U0) ** 2
    n = h.shape[0]
    t = mp.interpolar_centrado(Rec, delta * n / Rec.shape[0], (n, n), delta * z / L, xp)
    return a_numpy(t).astype(np.complex128)


def ida_mpasm(t, z, L, lamb, delta, N, device="auto"):
    """1 + V con la ida exacta de CamposT.dlhm."""
    V = holograma(t, delta * z / L, lamb, z, L, (N, N), delta, complejo=True,
                  banda=BANDA, device=device)
    return 1 + a_numpy(V).astype(np.complex128)


def vuelta_mpasm(h, z, L, lamb, delta, device="auto"):
    return a_numpy(Reconstructor(h, delta, lamb, L, z, device=device).campo(z)).astype(np.complex128)


def centro(a, frac=0.5):
    n = a.shape[0]
    m = int(n * frac) // 2
    return a[n // 2 - m:n // 2 + m, n // 2 - m:n // 2 + m]


def medir(t, ref):
    return {"sam_db": sam(t, ref), "sam_db_centro": sam(centro(t), centro(ref)),
            "error_max": float(np.abs(t - ref).max() / np.abs(ref).max()),
            "error_max_centro": float(np.abs(centro(t - ref)).max() / np.abs(centro(ref)).max())}


def objeto():
    """El USAF binario de comparacion_cruzada.py, en N x N, suavizado 1 px:
    barras opacas sobre fondo transparente (t = 1 en el borde)."""
    g = cv.imread(str(OBJETO), cv.IMREAD_GRAYSCALE).astype(np.float64) / 255
    g = cv.resize(g, (N, N), interpolation=cv.INTER_NEAREST)
    return (1 - gaussian_filter(g, 1.0)).astype(np.complex128)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    t = objeto()
    idas = {"asm": ida_carlos, "mpasm": ida_mpasm}
    vueltas = {"asm": vuelta_carlos, "mpasm": vuelta_mpasm}
    filas = []
    for z in ZS:
        D = geometria.D_de(z, L)
        c = geometria.cociente_ventana(z, L, DELTA, N, LAMB)
        hs = {nom: f(t, z, L, LAMB, DELTA, N) for nom, f in idas.items()}
        for i, h in hs.items():
            if z in GUARDAR:
                np.save(SALIDA / f"z{z:g}_ida_{i}.npy", h.astype(np.complex64))
            for v, f in vueltas.items():
                r = f(h, z, L, LAMB, DELTA)
                m = medir(r, t)
                filas.append(dict(z_mm=z, D_mm=f"{D:.3f}", cociente=f"{c:.3f}", ida=i,
                                  vuelta=v, **{k: f"{x:.6g}" for k, x in m.items()}))
                print(f"z = {z:g} ({c:.2f}) {i:>5s} -> {v:<5s}: SAM {m['sam_db']:6.2f} dB "
                      f"(centro {m['sam_db_centro']:6.2f}), error max {m['error_max']:.2e}",
                      flush=True)
                if z in GUARDAR:
                    np.save(SALIDA / f"z{z:g}_{i}_{v}.npy", r.astype(np.complex64))
    np.save(SALIDA / "objeto.npy", t.astype(np.complex64))
    with open(SALIDA / "resumen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


if __name__ == "__main__":
    main()
