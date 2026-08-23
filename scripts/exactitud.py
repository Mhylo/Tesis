"""Exactitud de los propagadores contra la solución analítica: la Figura 4.

Barre z sobre siete órdenes de magnitud con los parámetros de la Tabla 1 y
mide SAM [dB] de cada método contra el gaussiano analítico. Más alto es mejor.

Es donde se ve el argumento del Objetivo 1: MPASM se mantiene entre 31 y 48 dB
en todo el rango, FFT-ASM se desploma a números negativos en cuanto el haz deja
de caber en la ventana, y BL-ASM se degrada y luego se recupera cuando su
máscara de banda empieza a actuar. A z corto los tres empatan, o FFT-ASM gana:
cada método tiene su régimen, y ése es el resultado.

Imprime además el efecto del dtype en GPU y los tiempos, para que la exactitud
y el coste se lean juntos.

Escribe en resultados/exactitud/:

    sam_vs_z.csv   una fila por distancia, con el SAM de los tres métodos y el
                   Kf que MPASM eligió en cada una.
    sam_vs_z.png   la curva. Es la reproducción de la Figura 4 del paper, con
                   la Ec. (16) corregida (ver metricas.alfa_sam).

Uso:
    python -m scripts.exactitud
"""

import csv
import pathlib

import matplotlib
matplotlib.use("Agg")            # sin ventana: el script guarda, no muestra
import matplotlib.pyplot as plt
import numpy as np

from CamposT.backend import (cronometrar, gpu_disponible, info_gpu,
                             liberar_memoria)
from CamposT.metricas import sam
from CamposT.propagadores import blas, fft_asm, mpasm
from CamposT.referencias import gauss_analytic, gauss_beam

#: raíz del repo, para que la salida no dependa del directorio de invocación
RAIZ = pathlib.Path(__file__).resolve().parent.parent

#: Parámetros de la Tabla 1 del paper (sin la lente). Unidades: mm.
L0, N, W0, LAMB = 5.0, 512, 1.0, 632.8e-6
DELTA = L0 / (N - 1)

#: Las siete distancias de la tabla del paper, que son las que se imprimen.
#: La curva añade una rejilla logarítmica entre ellas: siete puntos sobre siete
#: órdenes de magnitud no dibujan una curva, dibujan una poligonal.
Z_TABLA = (500, 2000, 6000, 12000, 30000, 80000, 200000)
PUNTOS = 25

#: Un trazo por método, fijo: el color no cambia si algún día se añade o se
#: quita una serie. Los dos primeros colores son los de figura_escenario.py; el
#: tercero se eligió midiendo, no a ojo: es el que mantiene ΔE ≥ 15 entre los
#: tres con visión normal y ≥ 8 bajo protanopia, deuteranopia y tritanopia
#: (OKLab ×100). Cada método lleva además su tipo de línea y su marcador, para
#: que la figura se siga leyendo impresa en blanco y negro.
TINTA, SUAVE = "#1a1a1a", "#8a8a8a"
METODOS = (
    ("MPASM",   "sam_mpasm_dB", "#1f4e79", "-",  "o"),
    ("FFT-ASM", "sam_fft_dB",   "#c1543a", "--", "s"),
    ("BL-ASM",  "sam_blas_dB",  "#2a9d8f", "-.", "^"),
)
COLUMNAS = ("z_mm", "Kf", "sam_mpasm_dB", "sam_fft_dB", "sam_blas_dB")


# ----------------------------------------------------------------- el barrido
def malla_z(puntos=PUNTOS):
    """Distancias del barrido: la rejilla logarítmica más las de la tabla.

    Los puntos de la rejilla que caen sobre un valor de la tabla se descartan
    en favor del valor exacto. Sin eso, el logspace devuelve 500.0000000000001
    donde la tabla dice 500 y np.unique los toma por distintos: dos filas y dos
    propagaciones para la misma distancia.
    """
    tabla = np.asarray(Z_TABLA, dtype=float)
    fina = np.logspace(np.log10(tabla[0]), np.log10(tabla[-1]), puntos)
    propios = ~np.any(np.isclose(fina[:, None], tabla[None, :], rtol=1e-9),
                      axis=1)
    return np.sort(np.concatenate([fina[propios], tabla]))


def z_llena_la_ventana():
    """z al que el radio 1/e del haz alcanza el borde de la ventana.

    w(z) = w0·sqrt(1 + (z/zR)²) con zR = π·w0²/λ. Igualando w(z) = L0/2 y
    despejando. Es la distancia a la que FFT-ASM se queda sin sitio para el
    haz, así que explica el desplome de su curva en vez de sólo mostrarlo.
    """
    zR = np.pi * W0**2 / LAMB
    razon = (L0 / 2 / W0) ** 2 - 1
    return zR * np.sqrt(razon) if razon > 0 else np.nan


def barrido_sam(U0, X, Y, zs):
    """SAM de los tres métodos contra el analítico, distancia a distancia.

    Todo en CPU y complex128 a propósito: es la fila de referencia, y el coste
    de la doble precisión aquí no importa. El efecto de bajar a complex64 se
    mide aparte, más abajo.
    """
    filas = []
    for z in zs:
        ref = gauss_analytic(X, Y, W0, LAMB, z)
        Um, Kf = mpasm(U0, DELTA, LAMB, z, device="cpu")
        filas.append({
            "z_mm": float(z),
            "Kf": float(Kf),
            "sam_mpasm_dB": sam(Um, ref),
            "sam_fft_dB": sam(fft_asm(U0, DELTA, LAMB, z, device="cpu"), ref),
            "sam_blas_dB": sam(blas(U0, DELTA, LAMB, z, device="cpu"), ref),
            "en_tabla": bool(np.any(np.isclose(z, Z_TABLA))),
        })
    return filas


# ------------------------------------------------------------------- salidas
def guardar_csv(filas, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    print(f"  -> {path}")


def figura_sam(filas, path):
    """La curva de la Figura 4: SAM contra z, un trazo por método.

    Los marcadores no van en todos los puntos, sólo en las siete distancias de
    la tabla del paper: son los valores contrastables, y el resto de la rejilla
    está para que la curva sea una curva.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": SUAVE, "axes.labelcolor": TINTA,
        "xtick.color": SUAVE, "ytick.color": SUAVE,
        "axes.linewidth": 0.8, "figure.facecolor": "white",
    })
    z = [f["z_mm"] for f in filas]
    marcas = [i for i, f in enumerate(filas) if f["en_tabla"]]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.set_xscale("log")
    ax.grid(axis="y", color=SUAVE, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

    # Contexto antes que datos: las dos referencias van detrás y en gris.
    ax.axhline(0, color=SUAVE, lw=0.8, zorder=1)
    z_v = z_llena_la_ventana()
    if np.isfinite(z_v):
        ax.axvline(z_v, color=SUAVE, lw=0.9, ls=(0, (4, 3)), zorder=1)

    for nombre, clave, color, ls, marca in METODOS:
        ax.plot(z, [f[clave] for f in filas], label=nombre, color=color,
                ls=ls, lw=1.8, marker=marca, ms=4.5, markevery=marcas,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)

    ax.set_xlabel("z [mm]")
    ax.set_ylabel("SAM [dB]   (más alto, mejor)")
    ax.set_title("Exactitud contra el gaussiano analítico, Tabla 1 del paper",
                 fontsize=10.5, color=TINTA, pad=10)
    if np.isfinite(z_v):
        ax.annotate(f"el haz 1/e llena\nla ventana ({z_v/1000:.1f} m)",
                    xy=(z_v, ax.get_ylim()[1]), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, color=SUAVE,
                    va="top")
    ax.annotate("0 dB: el error iguala a la señal", xy=(z[0], 0),
                xytext=(2, 4), textcoords="offset points", fontsize=8,
                color=SUAVE)
    # arriba a la izquierda: es la única esquina que ninguna curva ocupa, y ahí
    # no pisa el rótulo de los 0 dB, que vive pegado a su propia línea
    ax.legend(frameon=False, loc="upper left", fontsize=9,
              labelcolor=TINTA, handlelength=2.6)
    fig.text(0.5, -0.02, f"N = {N}, δ = {DELTA*1e3:.2f} µm, λ = {LAMB*1e6:.1f} nm, "
             f"w₀ = {W0:.0f} mm · CPU, complex128 · Ec. (16) corregida",
             ha="center", fontsize=8, color=SUAVE)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def main():
    salida = RAIZ / "resultados" / "exactitud"

    if gpu_disponible():
        gi = info_gpu()
        print(f"GPU: {gi['nombre']} (cc {gi['capacidad']}), "
              f"{gi['VRAM total [GB]']:.1f} GB, CuPy {gi['cupy']}")
    else:
        print("Sin GPU: todo en CPU.")

    # --- exactitud contra la solución analítica -------------------------------
    U0_cpu, X, Y = gauss_beam(N, DELTA, W0, device="cpu")
    zs = malla_z()
    print(f"\nBarriendo {len(zs)} distancias de {zs[0]:.0f} a {zs[-1]:.0f} mm "
          f"con los tres métodos, CPU complex128...")
    filas = barrido_sam(U0_cpu, X, Y, zs)

    print("\nExactitud: SAM [dB] contra el gaussiano analítico, CPU complex128.")
    print("Más alto es mejor. Ec. (15) con la (16) corregida; ver alfa_sam().")
    print(f"{'z [mm]':>8} {'Kf':>7} {'MPASM':>9} {'FFT-ASM':>9} {'BL-ASM':>9}")
    for f in filas:
        if f["en_tabla"]:
            print(f"{f['z_mm']:8.0f} {f['Kf']:7.2f} {f['sam_mpasm_dB']:9.2f} "
                  f"{f['sam_fft_dB']:9.2f} {f['sam_blas_dB']:9.2f}")

    print()
    guardar_csv(filas, salida / "sam_vs_z.csv")
    figura_sam(filas, salida / "sam_vs_z.png")

    # --- coste de la precisión simple en GPU ----------------------------------
    if gpu_disponible():
        print("\nEfecto del dtype en GPU (SAM [dB] contra el analítico):")
        print(f"{'z [mm]':>8} {'c64':>9} {'c128':>9} {'CPU c128':>9}")
        for z in Z_TABLA:
            ref = gauss_analytic(X, Y, W0, LAMB, z)
            U64, _ = mpasm(U0_cpu, DELTA, LAMB, z, device="gpu", dtype=np.complex64)
            U128, _ = mpasm(U0_cpu, DELTA, LAMB, z, device="gpu", dtype=np.complex128)
            Ucpu, _ = mpasm(U0_cpu, DELTA, LAMB, z, device="cpu")
            print(f"{z:8d} {sam(U64, ref):9.2f} {sam(U128, ref):9.2f} "
                  f"{sam(Ucpu, ref):9.2f}")
            liberar_memoria()

        # --- tiempos ----------------------------------------------------------
        z = 12000
        print(f"\nTiempos a z = {z} mm, N = {N}, s = 10 (con sincronización):")
        print(f"{'método':>16} {'CPU c128 [s]':>13} {'GPU c64 [s]':>12} {'x':>7}")
        for nombre, fn in (("MPASM", mpasm), ("FFT-ASM", fft_asm), ("BL-ASM", blas)):
            _, t_cpu = cronometrar(fn, U0_cpu, DELTA, LAMB, z, device="cpu")
            _, t_gpu = cronometrar(fn, U0_cpu, DELTA, LAMB, z, device="gpu",
                                   dtype=np.complex64)
            print(f"{nombre:>16} {t_cpu:13.3f} {t_gpu:12.3f} {t_cpu / t_gpu:7.1f}")
            liberar_memoria()


if __name__ == "__main__":
    main()
