"""Ida y vuelta CON FASE, cruzando FFT-ASM y MPASM, a cada distancia de ZS.

A cada z propaga el objeto a +z con los dos metodos y guarda los dos campos
COMPLEJOS. Luego retropropaga cada uno a -z con los dos metodos: cuatro
reconstrucciones por distancia, todas guardadas, medidas contra el objeto y
pintadas en una figura. Con R > 1 anade una tercera ida, MPASM con la ventana
ampliada (ver abajo).

    ida FFT-ASM        ->  vuelta FFT-ASM      vuelta MPASM
    ida MPASM          ->  vuelta FFT-ASM      vuelta MPASM
    ida MPASM ampliada ->        -             vuelta MPASM      (R > 1)

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

LA VENTANA AMPLIADA (R > 1). En la zona de Fresnel la luz se abre mucho mas
que el sensor, y lo que sale de la ventana no vuelve con ninguna vuelta: eso
fija la resolucion, no el propagador. MPASM puede evaluar la ida en una malla
de salida distinta de la de entrada -R*M x R*N puntos de paso DELTA*mag-, o
sea simular un sensor mas grande sin tocar el objeto. Su vuelta es MPASM con
mag = 1/mag, que regresa al paso DELTA del objeto, y se recorta el centro.

    mag   el mayor con el que ese sensor todavia muestrea las franjas. La
          frecuencia local en el sensor llega a (W + a)/(lambda z), con W y a
          los semianchos del sensor y del objeto, y tiene que caber en
          1/(2 DELTA mag): mag (R mag + 1) <= lambda z / (n DELTA^2).
    s     el menor que cumple R*mag <= s*Kf, la guarda de mpasm_minimo: por
          encima la salida trae copias periodicas del campo.

La vuelta va con s = S: con R = 4, s = 2 en vez de 1 cambia el SAM 0.02 dB.
Y la regla de mag se comprobo a 192.6 y 1926 mm: 0.6, 1.5 y 2.5 veces ese mag
dan peor SAM. Por encima entra mas energia, pero el sensor ya no muestrea las
franjas.

NO ES EL MISMO DATO que la ventana fija: es un sensor de R*n pixeles de
DELTA*mag. Lo que mide es lo que gana quien puede simularlo. FFT-ASM no puede:
su malla de salida es la de entrada, y para la misma ventana sin aliar
necesitaria rellenar hasta max(R*mag, z/z_c)*n puntos por lado, con
z_c = n DELTA^2 / lambda. El script lo imprime.

LAS TRES MEDIDAS, todas contra el objeto de partida:

    sam [dB]        CamposT.metricas.sam, sobre intensidades y con escala
                    ajustada. MAS ALTO ES MEJOR. Mide la forma, no la fase.
    rms_amplitud    CamposT.metricas.rms_amplitud, amplitudes normalizadas por
                    su maximo. MAS BAJO ES MEJOR.
    error_max       max|U - U0| / max|U0| sobre el campo complejo, la misma que
                    imprimen los retro_*.py. Es la unica de las tres que ve la
                    fase.

Y dos de la ida: energia, la fraccion de la energia del objeto que cae en la
ventana del sensor (FFT-ASM da 1 siempre: lo que sale por un borde reentra por
el opuesto), y periodo_min, el periodo mas fino cuya luz difractada todavia
cae en ella, Ec. (12): lambda * sqrt(4 z^2 + ancho^2) / ancho. Lo que el
objeto tenga mas fino que eso no lo recupera ninguna vuelta.

N_F es el numero de Fresnel de la ventana del sensor, (n DELTA / 2)^2 /
(lambda z). Los ZS por defecto son N_F = 10, 5, 2 y 1: la zona de Fresnel de
este montaje, hasta su borde con Fraunhofer.

LO QUE SALIO (17/10), con R = 1, sobre resultados/campos/entrada.png, 512x512,
405 nm, 3.45 um, s = 1, GPU en complex64. Las barras del USAF tienen periodos
de ~28 a ~90 um:

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
cuadricula de los bloques. fft->fft sale perfecta a las tres z porque deshace
su propio aliasing, no porque acierte. Con S = 4 el Kf baja a la mitad y las
medidas no cambian en la milesima. Los .npy de la ida coinciden con los de
retro_fft_angular.py y retro_mpasm.py, y las vueltas con las de
retro_holograma.py, hasta 3e-6 (el piso de complex64).

LO QUE SALIO (17/10) EN LA ZONA DE FRESNEL, con los ZS de defecto y R = 4, el
resto igual. La ventana fija es la mejor de las tres cruzadas (mpasm->mpasm):

                   ventana fija (1.77 mm)    ventana ampliada (MPASM, R = 4)
    N_F  z [mm]    p_min  energia  sam    mag  s  ventana  p_min  energia  sam
    10    192.6    88 um   0.53   2.01   1.67  4  11.8 mm  13 um   0.96   6.27
     5    385.2   177 um   0.36   1.38   2.41  4  17.0 mm  18 um   0.94   5.35
     2    963.0   442 um   0.20   0.73   3.88  4  27.4 mm  29 um   0.89   4.11
     1   1926.0   883 um   0.12   0.34   5.53  4  39.1 mm  40 um   0.84   3.05

Con la ventana fija las cruzadas siguen empatadas entre si y a N_F = 1 solo
quedan manchas. La ampliada recupera las barras: a N_F = 1 se leen las de las
dos filas de arriba. Para la misma ventana sin aliar FFT-ASM pediria de 6553^2
puntos (0.3 GB) a N_F = 10 hasta 65535^2 (32 GB) a N_F = 1. R = 1 reproduce la
tabla de arriba con diferencia 0.

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
#: mm, POSITIVAS: la ida va a +z y la vuelta a -z. Las de defecto son N_F =
#: 10, 5, 2 y 1 para 512 x 3.45 um a 405 nm.
ZS = (192.6, 385.2, 963.0, 1926.0)
#: "intensidad": A = sqrt(img), la convencion de los retro_*.py.
#: "amplitud":   A = img. Sobre un objeto binario da lo mismo.
ENTRADA = "intensidad"
S = 1
#: Lado de la ventana ampliada en pixeles, en multiplos del objeto. 1 la
#: apaga. Cada paso cuesta R^2 en la malla de salida y en el .npy que se guarda.
R = 4
GPU = cp is not None   # True/False para forzar; con True y sin CuPy aborta
#: con R > 1 en su propia carpeta: resumen.csv no mezcla corridas con y sin ella
SALIDA = (RAIZ / "resultados" / "comparacion_cruzada"
          / (RUTA.stem + (f"_r{R}" if R > 1 else "")))

METODOS = ("fft", "mpasm")
NOMBRES = {"fft": "FFT-ASM", "mpasm": "MPASM", "mpasm_amp": "MPASM ampliada"}


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


def periodo_minimo(ancho, z):
    """Periodo mas fino [mm] cuya luz cae en una ventana de ancho [mm] a z."""
    return LAMB * np.sqrt(4 * z**2 + ancho**2) / ancho


def energia(U, paso):
    """Fraccion de la energia del objeto en U, con U muestreado a paso [mm]."""
    return float(np.sum(np.abs(a_numpy(U)) ** 2) * paso**2
                 / (np.sum(np.abs(U0) ** 2) * DELTA**2))


def ventana_ampliada(z):
    """(mag, s) de la ida ampliada a z. Ver LA VENTANA AMPLIADA arriba."""
    q = z * LAMB / (max(M, N) * DELTA**2)
    mag = max(1.0, (np.sqrt(1 + 4 * R * q) - 1) / (2 * R))
    s = S
    while s * min(kf(M, DELTA, LAMB, z, s), kf(N, DELTA, LAMB, z, s)) < R * mag:
        s += 1
    return mag, s


def medir(U):
    """Las tres medidas de una reconstruccion contra U0."""
    A = a_numpy(U)
    return A, {"sam_db": sam(A, U0), "rms_amplitud": rms_amplitud(A, U0),
               "error_max": float(np.abs(A - U0).max() / np.abs(U0).max())}


if ENTRADA not in ("intensidad", "amplitud"):
    raise SystemExit(f'ENTRADA = "{ENTRADA}" no existe: "intensidad" o "amplitud".')
if GPU and cp is None:
    raise SystemExit("GPU = True pero CuPy no esta instalado")
if min(ZS) <= 0:
    raise SystemExit("ZS son distancias POSITIVAS: el signo lo pone la ida o la vuelta.")
if R < 1 or int(R) != R:
    raise SystemExit(f"R = {R}: tiene que ser un entero >= 1.")

img = np.asarray(Image.open(RUTA).convert("L"), dtype=np.float64) / 255.0
U0 = np.sqrt(img) if ENTRADA == "intensidad" else img
U0 = U0.astype(complex)
M, N = U0.shape
n = max(M, N)
# el recorte central de la vuelta ampliada empieza en (R - 1) * M / 2: si no es
# entero, la reconstruccion sale corrida medio pixel respecto del objeto
if (R - 1) * M % 2 or (R - 1) * N % 2:
    raise SystemExit(f"R = {R} con una malla {M}x{N}: (R - 1) por cada lado "
                     f"tiene que ser par, o el recorte cae a medio pixel.")

SALIDA.mkdir(parents=True, exist_ok=True)
with open(SALIDA / "parametros.txt", "w", encoding="utf-8") as f:
    for clave, valor in (("objeto", RUTA), ("malla", f"{M}x{N}"),
                         ("lambda [mm]", LAMB), ("delta [mm]", DELTA),
                         ("ZS [mm]", ZS), ("ENTRADA", ENTRADA), ("S", S),
                         ("R", R), ("dispositivo", "gpu" if GPU else "cpu")):
        f.write(f"{clave} = {valor}\n")

print(f"{RUTA.name}: {M}x{N}, lambda {LAMB * 1e6:.0f} nm, delta {DELTA * 1e3:.2f} um, "
      f"s {S}, R {R}, {'GPU' if GPU else 'CPU'}, sin relleno")

filas = []
for z in ZS:
    N_F = (n * DELTA / 2) ** 2 / (LAMB * z)
    Kfy, Kfx = kf(M, DELTA, LAMB, z, S), kf(N, DELTA, LAMB, z, S)
    p_min = periodo_minimo(min(M, N) * DELTA, z)
    carpeta = SALIDA / f"z{z:08.3f}"
    carpeta.mkdir(exist_ok=True)
    print(f"\nz = {z:g} mm | N_F {N_F:.2f} | ventana del sensor {n * DELTA:.2f} mm, "
          f"periodo minimo {p_min * 1e3:.1f} um | Kf (y, x) = ({Kfy:.3f}, {Kfx:.3f})")

    # cada ida: (campo, paso de su malla, r, mag, s, Kf o None)
    ida = {m: (propagar(m, U0, +z), DELTA, 1, 1.0, S,
               (Kfy, Kfx) if m == "mpasm" else None) for m in METODOS}
    if R > 1:
        mag, s_amp = ventana_ampliada(z)
        ida["mpasm_amp"] = (mpasm(U0, DELTA, LAMB, +z, s=s_amp, r=R, mag=mag, gpu=GPU),
                            DELTA * mag, R, mag, s_amp,
                            (kf(M, DELTA, LAMB, z, s_amp), kf(N, DELTA, LAMB, z, s_amp)))
        P = max(R * mag, z * LAMB / (n * DELTA**2))
        print(f"MPASM ampliada: r {R}, mag {mag:.2f}, s {s_amp}, ventana "
              f"{R * mag * n * DELTA:.1f} mm, periodo minimo "
              f"{periodo_minimo(R * mag * min(M, N) * DELTA, z) * 1e3:.1f} um | "
              f"FFT-ASM necesitaria {P * n:.0f}^2 puntos "
              f"({(P * n) ** 2 * 8 / 2**30:.1f} GB en complex64)")
    for m, (U, *_) in ida.items():
        guardar(U, carpeta / f"ida_{m}")

    print(f"    {'ida':9s} {'vuelta':6s} {'energia':>7s} {'sam [dB]':>9s} "
          f"{'rms_amp':>8s} {'error_max':>10s}")
    vuelta = {}
    for o, (U_ida, paso, r, mag, s, Kf) in ida.items():
        e = energia(U_ida, paso)
        for v in METODOS:
            if o == "mpasm_amp":
                if v == "fft":
                    continue          # FFT-ASM no cambia el paso de su malla
                f0, c0 = (R - 1) * M // 2, (R - 1) * N // 2
                U = mpasm(U_ida, paso, LAMB, -z, s=S, mag=1 / mag, gpu=GPU,
                          ventana=(f0, c0, M, N))
            else:
                U = propagar(v, U_ida, -z)
            guardar(U, carpeta / f"ida_{o}_vuelta_{v}")
            A, medida = medir(U)
            vuelta[o, v] = (A, medida)
            ancho = r * mag * min(M, N) * DELTA
            filas.append({"z_mm": z, "N_F": N_F, "ida": o, "vuelta": v,
                          "r": r, "mag": mag, "s": s,
                          "Kf_y": "" if Kf is None else Kf[0],
                          "Kf_x": "" if Kf is None else Kf[1],
                          "ventana_mm": ancho,
                          "periodo_min_um": periodo_minimo(ancho, z) * 1e3,
                          "energia": e, **medida})
            print(f"    {o:9s} {v:6s} {e:7.3f} {medida['sam_db']:9.2f} "
                  f"{medida['rms_amplitud']:8.4f} {medida['error_max']:10.2e}")

    fig, ax = plt.subplots(len(ida), 3, figsize=(13.5, 4.9 * len(ida)),
                           layout="constrained")
    for k, (o, (U_ida, paso, r, mag, s, Kf)) in enumerate(ida.items()):
        ancho = r * mag * min(M, N) * DELTA
        ax[k, 0].imshow(np.abs(a_numpy(U_ida)) ** 2, cmap="gray")
        ax[k, 0].set_title(f"ida {NOMBRES[o]}, |U|^2 a +{z:g} mm\n"
                           f"ventana {ancho:.1f} mm | energia {energia(U_ida, paso):.2f}",
                           fontsize=10)
        for j, v in enumerate(METODOS, start=1):
            if (o, v) not in vuelta:
                ax[k, j].text(0.5, 0.5, "FFT-ASM no cambia\nel paso de su malla",
                              ha="center", va="center", fontsize=10)
                continue
            A, medida = vuelta[o, v]
            ax[k, j].imshow(np.abs(A) ** 2, cmap="gray")
            ax[k, j].set_title(f"ida {NOMBRES[o]} -> vuelta {NOMBRES[v]}\n"
                               f"SAM {medida['sam_db']:.1f} dB | "
                               f"rms {medida['rms_amplitud']:.3f}", fontsize=10)
    for a in ax.flat:
        a.set_axis_off()
    fig.suptitle(f"Ida y vuelta con fase  -  z = {z:g} mm, N_F = {N_F:.2f}, "
                 f"lambda {LAMB * 1e6:.0f} nm, delta {DELTA * 1e3:.2f} um, "
                 f"periodo minimo del sensor {p_min * 1e3:.0f} um", fontsize=11)
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
