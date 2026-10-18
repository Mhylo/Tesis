"""Figuras del manuscrito que salen de datos ya guardados o de una fórmula.

Ninguna vuelve a medir ni a propagar nada: leen los CSV que dejaron los
scripts que hicieron el cálculo, o evalúan kf_auto(), y dibujan la versión de
la tesis (sin título ni pie, que los pone el caption de LaTeX; mismo estilo que
la Figura 4.1 de scripts/exactitud.py).

    kf_vs_z.pdf             K_f automático contra z, escenario de la Tabla 1.
                            Sección 2.4.
    mpasm_esferico.pdf      correlación con el objeto contra el ángulo, MPASM
                            esférico frente al escalado. Datos:
                            resultados/dlhm_esferico/campo_N3000.csv, de
                            scripts/verif_dlhm_esferico.py campo --n 3000.
                            Sección 3.2.2.
    errata_kf.pdf           la Ec. (14) impresa frente a la del código
                            publicado, s = 10. Sección 4.1.
    tiempos.pdf             tiempo contra N y contra s. Datos:
                            resultados/escalado_N.csv y escalado_s.csv, de
                            scripts/comparacion.py. Los mismos de la Tabla 4.1.

    Tesis_env/Scripts/python.exe -m scripts.figuras_manuscrito

Escribe en manuscrito/figuras/.
"""

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from CamposT.montaje import TABLA1
from CamposT.propagadores import kf_auto

RAIZ = pathlib.Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "manuscrito" / "figuras"
RESULTADOS = RAIZ / "resultados"

# Paleta de la Figura 4.1 (scripts/exactitud.py): un color por método.
TINTA, SUAVE = "#1a1a1a", "#8a8a8a"
COLOR = {"mpasm": "#1f4e79", "fft": "#c1543a", "blas": "#2a9d8f"}
NOMBRE = {"mpasm": "MPASM", "fft": "FFT-ASM", "blas": "BLAS"}


def estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": SUAVE, "axes.labelcolor": TINTA,
        "xtick.color": SUAVE, "ytick.color": SUAVE,
        "axes.linewidth": 0.8, "figure.facecolor": "white",
    })


def limpiar(ax, rejilla="y"):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis=rejilla, which="major", color=SUAVE, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)


def guardar(fig, nombre):
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / nombre, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {SALIDA / nombre}")


def leer_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- K_f contra z
def figura_kf_vs_z():
    """K_f automático contra z, un trazo por s, con su asíntota.

    A distancias largas la Ec. (kf-auto) tiende a (1/delta)·sqrt(lambda z/(sN)),
    que cruza K_f = 1 en z = s N delta²/lambda.
    """
    N, d, lamb = TABLA1.N, TABLA1.delta, TABLA1.lamb
    zs = np.logspace(1, np.log10(3e5), 400)           # mm
    eses = (1, 2, 4, 10)
    tonos = plt.cm.viridis(np.linspace(0.15, 0.8, len(eses)))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    limpiar(ax, rejilla="both")
    ax.axhline(1.0, color=SUAVE, lw=0.8, zorder=1)
    for s, tono in zip(eses, tonos):
        kf = [kf_auto(N, d, lamb, z, s=s) for z in zs]
        ax.plot(zs * 1e-3, kf, color=tono, lw=1.8, label=f"s = {s}", zorder=3)
        asintota = np.sqrt(lamb * zs / (s * N)) / d
        sube = asintota < 1
        ax.plot(zs[sube] * 1e-3, asintota[sube], color=tono, lw=0.9,
                ls=(0, (2, 2)), zorder=2)
    # el punto que cita la Sección 4.1: s = 10 a 200 m
    kf200 = kf_auto(N, d, lamb, 2e5, s=10)
    ax.plot(200, kf200, "o", color=tonos[-1], ms=5, zorder=4)
    ax.annotate(f"{kf200:.1f}", (200, kf200), xytext=(-6, 6),
                textcoords="offset points", ha="right", fontsize=8,
                color=TINTA)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("$K_f$ automático")
    ax.set_ylim(0.08, 70)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    guardar(fig, "kf_vs_z.pdf")
    print(f"     s = 10 a 200 m: K_f = {kf200:.2f}; "
          f"umbral s N delta²/lambda = {10 * N * d**2 / lamb / 1e3:.2f} m")


# ------------------------------------------------- MPASM esférico vs escalado
def figura_mpasm_esferico():
    """Correlación con el objeto por tesela contra el ángulo desde la fuente."""
    filas = leer_csv(RESULTADOS / "dlhm_esferico" / "campo_N3000.csv")
    series = (("exacto", "MPASM con la esférica exacta", COLOR["mpasm"], "-", "o"),
              ("escalado", "escalado paraxial", COLOR["fft"], "--", "s"))
    entradas = (("campo complejo", "(a) entrada: campo complejo"),
                ("intensidad", "(b) entrada: intensidad"))

    fig, ejes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, (clave, titulo) in zip(ejes, entradas):
        sel = [f for f in filas if f["entrada"] == clave]
        x = [(float(f["grados_desde"]) + float(f["grados_hasta"])) / 2 for f in sel]
        for col, nombre, color, ls, marca in series:
            ax.plot(x, [float(f[col]) for f in sel], color=color, ls=ls, lw=1.8,
                    marker=marca, ms=4.5, markeredgecolor="white",
                    markeredgewidth=0.6, label=nombre)
        limpiar(ax)
        ax.set_title(titulo, fontsize=9, color=TINTA)
        ax.set_xlabel("ángulo desde la fuente [grados]")
        ax.set_xlim(0, 21)
        ax.set_ylim(0.55, 1.02)
        print(f"     {clave}: exacto {float(sel[0]['exacto']):.3f} -> "
              f"{float(sel[-1]['exacto']):.3f}; escalado "
              f"{float(sel[0]['escalado']):.3f} -> {float(sel[-1]['escalado']):.3f}")
    ejes[0].set_ylabel("correlación con el objeto (por tesela)")
    ejes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    guardar(fig, "mpasm_esferico.pdf")


# ----------------------------------------------------------- errata Ec. (14)
def figura_errata_kf(s=10, z_marca=80e3):
    """La Ec. (14) impresa (A² + B) frente a la del código publicado (A²·B)."""
    N, d, lamb = TABLA1.N, TABLA1.delta, TABLA1.lamb
    zs = np.logspace(2, np.log10(3e5), 300)            # mm, como el código
    paper = [kf_auto(N, d, lamb, z, s=s) for z in zs]
    codigo = [kf_auto(N, d, lamb, z, s=s, formula="codigo") for z in zs]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    limpiar(ax, rejilla="both")
    ax.axhline(1.0, color=SUAVE, lw=0.8, zorder=1)
    ax.plot(zs * 1e-3, paper, color=COLOR["mpasm"], lw=1.8,
            label="Ec. (14) del artículo")
    ax.plot(zs * 1e-3, codigo, color=COLOR["fft"], lw=1.8, ls=(0, (4, 2)),
            label="código publicado")
    kp = kf_auto(N, d, lamb, z_marca, s=s)
    kc = kf_auto(N, d, lamb, z_marca, s=s, formula="codigo")
    for k in (kp, kc):
        ax.plot(z_marca * 1e-3, k, "o", color=TINTA, ms=4, zorder=4)
        ax.annotate(f"{k:.1f}", (z_marca * 1e-3, k), xytext=(6, -2),
                    textcoords="offset points", fontsize=8, color=TINTA)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("$K_f$ automático")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    guardar(fig, "errata_kf.pdf")
    print(f"     s = {s}, z = {z_marca / 1e3:.0f} m: artículo {kp:.2f}, código {kc:.2f}, "
          f"cociente {kc / kp:.3f}; 1/(sN lambda) = {1 / (s * N * lamb):.3f}")


# ----------------------------------------------------------------- tiempos
def figura_tiempos():
    """(a) tiempo contra N con s = 1, CPU complex128 y GPU complex64;
    (b) MPASM contra s con N fijo. Los mismos CSV que la Tabla 4.1."""
    filas_N = leer_csv(RESULTADOS / "escalado_N.csv")
    filas_s = leer_csv(RESULTADOS / "escalado_s.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for metodo in ("fft", "blas", "mpasm"):
        sel = [f for f in filas_N if f["metodo"] == metodo]
        n = [int(f["N"]) for f in sel]
        for col, ls, marca, disp in (("cpu_128", "-", "o", "CPU"),
                                     ("gpu_64", "--", "s", "GPU")):
            a.plot(n, [1e3 * float(f[col]) for f in sel], color=COLOR[metodo],
                   ls=ls, lw=1.6, marker=marca, ms=4, markeredgecolor="white",
                   markeredgewidth=0.6, label=f"{NOMBRE[metodo]}, {disp}")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("N (puntos por lado)")
    a.set_ylabel("tiempo por propagación [ms]")
    a.set_title("(a) s = 1", fontsize=9, color=TINTA)
    a.legend(frameon=False, fontsize=7, loc="upper left", ncol=1)
    limpiar(a, rejilla="both")

    s = [int(f["s"]) for f in filas_s]
    for col, ls, marca, disp in (("cpu_128", "-", "o", "CPU"),
                                 ("gpu_64", "--", "s", "GPU")):
        b.plot(s, [1e3 * float(f[col]) for f in filas_s], color=COLOR["mpasm"],
               ls=ls, lw=1.6, marker=marca, ms=4, markeredgecolor="white",
               markeredgewidth=0.6, label=f"MPASM, {disp}")
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xticks(s, [str(v) for v in s])
    b.set_xlabel("s")
    b.set_title(f"(b) MPASM, N = {filas_s[0]['N']}", fontsize=9, color=TINTA)
    b.legend(frameon=False, fontsize=7, loc="upper left")
    limpiar(b, rejilla="both")
    fig.tight_layout()
    guardar(fig, "tiempos.pdf")
    g = [float(f["gpu_64"]) for f in filas_s]
    print(f"     MPASM GPU, s = {s[0]} -> {s[-1]}: {1e3 * g[0]:.1f} -> {1e3 * g[-1]:.1f} ms "
          f"(x{g[-1] / g[0]:.1f})")


def main():
    estilo()
    figura_kf_vs_z()
    figura_mpasm_esferico()
    figura_errata_kf()
    figura_tiempos()


if __name__ == "__main__":
    main()
