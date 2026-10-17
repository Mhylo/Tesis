"""Ida y vuelta CON FASE, cruzando FFT-ASM y MPASM, a cada distancia de ZS.

A cada z propaga el objeto a +z con los dos metodos y guarda los dos campos
COMPLEJOS. Luego retropropaga cada uno a -z con los dos metodos: cuatro
reconstrucciones por distancia, todas guardadas, medidas contra el objeto y
pintadas en una figura de 2 x 3.

    ida FFT-ASM  ->  vuelta FFT-ASM      vuelta MPASM
    ida MPASM    ->  vuelta FFT-ASM      vuelta MPASM

Hace en una sola corrida lo que antes pedia tres scripts y editar sus
constantes en cada z: retro_fft_angular.py y retro_mpasm.py para la ida con
fase, y retro_holograma.py para la vuelta. No lleva copia de ningun
propagador: MPASM es scripts/mpasm_minimo.py y FFT-ASM es el de
CamposT.propagadores, el mismo que llama retro_holograma.py.

POR QUE CRUZAR. Ida y vuelta con el MISMO metodo no mide su exactitud. FFT-ASM
es la identidad -H(+z)*H(-z) = 1 en toda la malla- aunque su H este aliada, asi
que deshace exactamente su propio error: a 1000 mm su ida es ruido y aun asi la
vuelta devuelve el objeto con 1e-7. Las combinaciones cruzadas no tienen esa
coartada. Ida FFT -> vuelta MPASM dice cuanto se equivoco la ida FFT, e ida
MPASM -> vuelta FFT cuanto se equivoca la vuelta FFT.

LAS TRES MEDIDAS, todas contra el objeto de partida:

    sam [dB]        CamposT.metricas.sam, sobre intensidades y con escala
                    ajustada. MAS ALTO ES MEJOR. Mide la forma, no la fase.
    rms_amplitud    CamposT.metricas.rms_amplitud, amplitudes normalizadas por
                    su maximo. MAS BAJO ES MEJOR.
    error_max       max|U - U0| / max|U0| sobre el campo complejo, la misma que
                    imprimen los retro_*.py. Es la unica de las tres que ve la
                    fase.

periodo_min es el periodo mas fino cuya luz difractada todavia cae dentro de
la ventana del sensor, Ec. (12): lambda * sqrt(4 z^2 + (n delta)^2) / (n delta).
Lo que el objeto tenga mas fino que eso sale por los bordes y no hay vuelta
que lo recupere, con fase o sin ella.

LO QUE SALIO (17/10) sobre resultados/campos/entrada.png, 512x512, 405 nm,
3.45 um, s = 1, GPU en complex64. Las barras del USAF tienen periodos de ~28 a
~90 um:

    z [mm]  periodo_min   ida->vuelta      sam [dB]  rms_amp  error_max
     100       46 um      fft->fft           66.19    0.0000   8.1e-07
                          fft->mpasm          3.26    0.1594   1.06
                          mpasm->fft          3.26    0.1606   1.05
                          mpasm->mpasm        3.28    0.1594   1.05
     200       92 um      fft->fft           66.03    0.0000   9.6e-07
                          las otras tres     1.94-1.99  0.190    1.05
    1000      459 um      fft->fft           66.06    0.0000   9.0e-07
                          las otras tres     0.67-0.72  0.31-0.32  1.00

Las tres combinaciones que no son fft->fft dan lo mismo entre si, y lo que
dan lo fija la ventana, no el propagador: a 100 mm sobreviven las barras
gruesas, a 200 mm las barras se leen en la fila de arriba (~90 um, justo en
el corte) y a medias en la segunda, y a 1000 mm quedan manchas en la
cuadricula de los bloques. fft->fft sale
perfecta a las tres z porque deshace su propio aliasing, no porque acierte.
Los .npy de la ida coinciden con los de retro_fft_angular.py y retro_mpasm.py,
y las vueltas con las de retro_holograma.py, hasta 3e-6 (el piso de complex64).

SIN RELLENO DE CEROS. mpasm_minimo no lo tiene, y para que la comparacion solo
cambie el propagador FFT-ASM tampoco lo lleva: los dos ven la misma malla.

LOS PNG VAN SIN GAMMA, como en guardar_holograma() de los retro_*.py: un PNG
guardado con gamma y releido como intensidad mueve la z a la que enfoca. La
fase esta en los .npy, que guardan el campo complejo tal cual.

UNIDADES: milimetros para todo.  633 nm -> 633e-6    3.45 um -> 3.45e-3
"""

import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mpasm_minimo import cp, kf, mpasm

from CamposT.backend import a_numpy, liberar_memoria
from CamposT.metricas import rms_amplitud, sam
from CamposT.propagadores import fft_asm

RAIZ = pathlib.Path(__file__).resolve().parents[1]

RUTA = RAIZ / "resultados" / "campos" / "entrada.png"   # el OBJETO, no un holograma
LAMB = 405e-6       # mm
DELTA = 3.45e-3     # mm, paso de pixel
ZS = (100.0, 200.0, 1000.0)   # mm, POSITIVAS: la ida va a +z y la vuelta a -z
#: "intensidad": A = sqrt(img), la convencion de los retro_*.py.
#: "amplitud":   A = img. Sobre un objeto binario da lo mismo.
ENTRADA = "intensidad"
S = 1
GPU = cp is not None   # True/False para forzar; con True y sin CuPy aborta
SALIDA = RAIZ / "resultados" / "comparacion_cruzada" / RUTA.stem

METODOS = ("fft", "mpasm")
NOMBRES = {"fft": "FFT-ASM", "mpasm": "MPASM"}


def propagar(metodo, U, z):
    """U a una distancia z con signo, en el dispositivo y dtype por defecto."""
    if metodo == "fft":
        return fft_asm(U, DELTA, LAMB, z, device="gpu" if GPU else "cpu")
    return mpasm(U, DELTA, LAMB, z, s=S, gpu=GPU)


def guardar(U, base):
    """Campo complejo -> base.npy tal cual y base.png con |U|^2 sin gamma."""
    A = a_numpy(U)
    np.save(f"{base}.npy", A)
    I = np.abs(A) ** 2
    m = I.max()
    I = I / m if m > 0 else np.zeros_like(I)
    Image.fromarray((I * 255).astype(np.uint8)).save(f"{base}.png")


def periodo_minimo(n, z):
    """Periodo mas fino [mm] cuya luz cae en una ventana de n pixeles a z."""
    return LAMB * np.sqrt(4 * z**2 + (n * DELTA) ** 2) / (n * DELTA)


if ENTRADA not in ("intensidad", "amplitud"):
    raise SystemExit(f'ENTRADA = "{ENTRADA}" no existe: "intensidad" o "amplitud".')
if GPU and cp is None:
    raise SystemExit("GPU = True pero CuPy no esta instalado")
if min(ZS) <= 0:
    raise SystemExit("ZS son distancias POSITIVAS: el signo lo pone la ida o la vuelta.")

img = np.asarray(Image.open(RUTA).convert("L"), dtype=np.float64) / 255.0
U0 = np.sqrt(img) if ENTRADA == "intensidad" else img
U0 = U0.astype(complex)
M, N = U0.shape

SALIDA.mkdir(parents=True, exist_ok=True)
with open(SALIDA / "parametros.txt", "w", encoding="utf-8") as f:
    for clave, valor in (("objeto", RUTA), ("malla", f"{M}x{N}"),
                         ("lambda [mm]", LAMB), ("delta [mm]", DELTA),
                         ("ZS [mm]", ZS), ("ENTRADA", ENTRADA), ("S", S),
                         ("dispositivo", "gpu" if GPU else "cpu")):
        f.write(f"{clave} = {valor}\n")

print(f"{RUTA.name}: {M}x{N}, lambda {LAMB * 1e6:.0f} nm, delta {DELTA * 1e3:.2f} um, "
      f"s {S}, {'GPU' if GPU else 'CPU'}, sin relleno")

filas = []
for z in ZS:
    Kfy, Kfx = kf(M, DELTA, LAMB, z, S), kf(N, DELTA, LAMB, z, S)
    p_min = max(periodo_minimo(M, z), periodo_minimo(N, z))
    carpeta = SALIDA / f"z{z:08.3f}"
    carpeta.mkdir(exist_ok=True)
    print(f"\nz = {z:g} mm | Kf de MPASM (y, x) = ({Kfy:.3f}, {Kfx:.3f}) | "
          f"periodo minimo en la ventana {p_min * 1e3:.1f} um")
    print(f"    {'ida':6s} {'vuelta':6s} {'sam [dB]':>9s} {'rms_amp':>8s} {'error_max':>10s}")

    ida = {m: propagar(m, U0, +z) for m in METODOS}
    for m, U in ida.items():
        guardar(U, carpeta / f"ida_{m}")

    vuelta = {}
    for o in METODOS:
        for v in METODOS:
            U = propagar(v, ida[o], -z)
            guardar(U, carpeta / f"ida_{o}_vuelta_{v}")
            A = a_numpy(U)
            medida = {"sam_db": sam(A, U0), "rms_amplitud": rms_amplitud(A, U0),
                      "error_max": float(np.abs(A - U0).max() / np.abs(U0).max())}
            vuelta[o, v] = (A, medida)
            filas.append({"z_mm": z, "Kf_y": Kfy, "Kf_x": Kfx,
                          "periodo_min_um": p_min * 1e3, "ida": o, "vuelta": v,
                          **medida})
            print(f"    {o:6s} {v:6s} {medida['sam_db']:9.2f} "
                  f"{medida['rms_amplitud']:8.4f} {medida['error_max']:10.2e}")

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 9.8), layout="constrained")
    for k, o in enumerate(METODOS):
        ax[k, 0].imshow(np.abs(a_numpy(ida[o])) ** 2, cmap="gray")
        ax[k, 0].set_title(f"ida {NOMBRES[o]}, |U|^2 a +{z:g} mm", fontsize=10)
        for j, v in enumerate(METODOS, start=1):
            A, medida = vuelta[o, v]
            ax[k, j].imshow(np.abs(A) ** 2, cmap="gray")
            ax[k, j].set_title(f"ida {NOMBRES[o]} -> vuelta {NOMBRES[v]}\n"
                               f"SAM {medida['sam_db']:.1f} dB | "
                               f"rms {medida['rms_amplitud']:.3f}", fontsize=10)
    for a in ax.flat:
        a.set_axis_off()
    fig.suptitle(f"Ida y vuelta con fase  -  z = {z:g} mm, lambda {LAMB * 1e6:.0f} nm, "
                 f"delta {DELTA * 1e3:.2f} um, Kf = ({Kfy:.3f}, {Kfx:.3f}), "
                 f"periodo minimo {p_min * 1e3:.0f} um", fontsize=11)
    fig.savefig(carpeta / "figura.png", dpi=120)
    plt.close(fig)
    print(f"    -> {carpeta}")

    del ida, vuelta
    liberar_memoria()

with open(SALIDA / "resumen.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0]))
    w.writeheader()
    w.writerows(filas)
print(f"\n-> {SALIDA / 'resumen.csv'}")
