"""Figuras nuevas de la guía de lectura del manuscrito.

Son solo para la guía (docs/guia_manuscrito/guia_manuscrito.tex). Dibujan
esquemas con los números del manuscrito o leen los resultados guardados; la
única que propaga algo es figura_campo_ventana(), con CamposT en CPU, y no
sustituye ningún resultado del manuscrito. El estilo es el de
scripts/figuras_manuscrito.py, para que se lean como las demás.

    Tesis_env/Scripts/python.exe docs/guia_manuscrito/figuras_guia.py [nombre...]

Escribe en docs/guia_manuscrito/figuras/.
"""

import csv
import math
import pathlib
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from CamposT.campos import usaf_like
from CamposT.dlhm import ventana_chirp
from CamposT.propagadores import fft_asm, mpasm

RAIZ = pathlib.Path(__file__).resolve().parents[2]
SALIDA = pathlib.Path(__file__).resolve().parent / "figuras"
SALIDA.mkdir(exist_ok=True)

# Los mismos colores que el manuscrito: un color por método.
TINTA, SUAVE = "#1a1a1a", "#8a8a8a"
COLOR = {"mpasm": "#1f4e79", "fft": "#c1543a", "blas": "#2a9d8f"}


def estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": SUAVE, "axes.labelcolor": TINTA,
        "xtick.color": SUAVE, "ytick.color": SUAVE,
        "xtick.labelcolor": TINTA, "ytick.labelcolor": TINTA,
        "axes.linewidth": 0.8, "figure.facecolor": "white",
        "pdf.fonttype": 42,
    })


def limpiar(ax, rejilla="y"):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    if rejilla:
        ax.grid(axis=rejilla, which="major", color=SUAVE, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)


def guardar(fig, nombre):
    fig.savefig(SALIDA / f"{nombre}.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  ", nombre)


# ------------------------------------------------------------ el campo real
def figura_campo_ventana():
    """El objeto de la ida y vuelta, propagado de verdad en una malla rellenada."""
    N, d, lamb = 512, 3.45e-3, 405e-6
    W = N * d
    zlim = N * d**2 / lamb
    t = usaf_like(N)
    P = 4096
    borde = (P - N) // 2
    t_pad = np.zeros((P, P), dtype=np.complex64)
    t_pad[borde:borde + N, borde:borde + N] = t

    def verdad(z):
        return fft_asm(t_pad, d, lamb, z, device="cpu", dtype=np.complex64)

    fig = plt.figure(figsize=(6.5, 4.75))
    rejilla = fig.add_gridspec(2, 3, hspace=0.26, wspace=0.08)
    medio = P // 2
    lado = int(3.5 * N)
    for j, z in enumerate((5.0, 15.0, 75.0)):
        letra = "abc"[j]
        ax = fig.add_subplot(rejilla[0, j])
        U = verdad(z)
        I = np.abs(U[medio - lado:medio + lado, medio - lado:medio + lado])**2
        I = I.reshape(I.shape[0] // 4, 4, I.shape[1] // 4, 4).mean(axis=(1, 3))
        ax.imshow(np.log10(I / I.max() + 1e-6), cmap="gray", vmin=-4.5, vmax=0,
                  extent=[-3.5, 3.5, -3.5, 3.5], interpolation="antialiased")
        ax.plot([-.5, .5, .5, -.5, -.5], [-.5, -.5, .5, .5, -.5], color="white",
                lw=0.9)
        ax.set_title(f"({letra}) z = {z:g} mm,  q = {z / zlim:.2f}" if z < 20 else
                     f"({letra}) z = {z:g} mm,  q = {z / zlim:.1f}", fontsize=8.5,
                     color=TINTA)
        ax.set_xticks([])
        ax.set_yticks([])
        for lado_ax in ax.spines.values():
            lado_ax.set_visible(False)

    z = 75.0
    U = verdad(z)
    ref = U[borde:borde + N, borde:borde + N]
    Ufft = fft_asm(t.astype(np.complex64), d, lamb, z, device="cpu",
                   dtype=np.complex64)
    Um, Kf = mpasm(t.astype(np.complex64), d, lamb, z, s=1, device="cpu",
                   dtype=np.complex64)
    Um4, Kf4 = mpasm(t.astype(np.complex64), d, lamb, z, s=4, device="cpu",
                     dtype=np.complex64)
    norma = np.linalg.norm(ref)
    casos = (("(d) campo correcto", ref, None),
             ("(e) FFT-ASM sin relleno", Ufft, np.linalg.norm(Ufft - ref) / norma),
             (f"(f) MPASM, s = 1, $K_f$ = {Kf:.2f}", Um,
              np.linalg.norm(Um - ref) / norma))
    print(f"   K_f(s=4) = {Kf4:.2f}, eps(s=4) = "
          f"{np.linalg.norm(Um4 - ref) / norma:.3f}")
    vmax = np.percentile(np.abs(ref)**2, 99.8)
    for j, (titulo, campo, eps) in enumerate(casos):
        ax = fig.add_subplot(rejilla[1, j])
        ax.imshow(np.abs(campo)**2, cmap="gray", vmin=0, vmax=vmax,
                  interpolation="antialiased")
        ax.set_title(titulo, fontsize=8.5, color=TINTA)
        ax.set_xticks([])
        ax.set_yticks([])
        if eps is not None:
            ax.set_xlabel(f"error del campo: {eps:.2g}", fontsize=8.5, color=TINTA)
            print(f"   {titulo}: eps = {eps:.3f}")
        for lado_ax in ax.spines.values():
            lado_ax.set_visible(False)
    guardar(fig, "g_campo_ventana")


# ------------------------------------------------------- reentrada y relleno
def figura_rayos():
    fig, ejes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    for ax, P, titulo in zip(ejes, (1, 2), ("(a) sin relleno: periodo = 1 ventana",
                                            "(b) relleno ×2: periodo = 2 ventanas")):
        limpiar(ax, rejilla=None)
        ax.axhspan(-.5, .5, color=COLOR["mpasm"], alpha=0.10, lw=0)
        ax.text(1.45 if P == 1 else 0.1, -0.47, "ventana del sensor", ha="left",
                va="bottom", fontsize=8, color=COLOR["mpasm"])
        if P > 1:
            for y in (-P / 2, P / 2):
                ax.axhline(y, color=SUAVE, lw=0.8, ls=(0, (4, 3)))
            ax.text(3.95, 1.04, "borde del periodo", ha="right", va="bottom",
                    fontsize=8, color=TINTA)
        q = np.linspace(0, 4, 4001)
        x = q / 2
        plegado = (x + P / 2) % P - P / 2
        salto = np.abs(np.diff(plegado, prepend=plegado[0])) > 0.5
        plegado = plegado.copy()
        plegado[salto] = np.nan
        ax.plot(q, plegado, color=TINTA, lw=1.6)
        qr = 2 * P - 1
        ax.plot(qr, -0.5, "o", color=COLOR["fft"], ms=6, zorder=4)
        ax.axvline(qr, color=COLOR["fft"], lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.annotate(f"reentra en el sensor:\nq = {qr}", (qr, -0.5),
                    xytext=(7, -24), textcoords="offset points", fontsize=8,
                    color=COLOR["fft"], ha="left")
        if P > 1:
            ax.plot(1, .5, "o", mfc="white", mec=TINTA, ms=4.5, zorder=4)
            ax.text(0.05, .72, "sale del sensor", fontsize=8, color=TINTA, ha="left")
        ax.set_xlim(0, 4)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(titulo, fontsize=9, color=TINTA)
        ax.set_xlabel(r"cociente $q = z\,/\,(N\delta^2/\lambda)$")
    ejes[0].set_ylabel("posición del rayo / ventana")
    fig.tight_layout()
    guardar(fig, "g_rayos")


# -------------------------------------------------------- muestrear H
def figura_muestreo_h():
    N, q = 64, 4.0
    fase = lambda u: -np.pi * (q * N / 4) * u**2
    fino = np.linspace(-1, 1, 12001)
    casos = (
        ("(a) FFT: 64 muestras hasta Nyquist", (np.arange(N) - N / 2) * 2 / N,
         COLOR["fft"], 1.0),
        ("(b) MPASM, $K_f$ = 2: las mismas 64 muestras, en media banda",
         (np.arange(N) - N / 2) * 2 / N / 2, COLOR["mpasm"], 0.5),
        ("(c) MPASM, s = 4: 256 muestras hasta Nyquist",
         (np.arange(4 * N) - 2 * N) * 2 / (4 * N), COLOR["mpasm"], 1.0),
    )
    fig, ejes = plt.subplots(3, 1, figsize=(6.5, 4.6), sharex=True)
    for ax, (titulo, u, color, banda) in zip(ejes, casos):
        limpiar(ax, rejilla=None)
        ax.plot(fino, np.cos(fase(fino)), color=SUAVE, lw=0.35, alpha=0.75, zorder=1)
        ax.plot(u, np.cos(fase(u)), "-", color=color, lw=0.7, zorder=2)
        ax.plot(u, np.cos(fase(u)), "o", color=color, ms=2.3, zorder=3)
        if banda < 1:
            for signo in (-1, 1):
                ax.axvspan(signo * banda, signo * 1, color=SUAVE, alpha=0.14, lw=0)
            ax.text(0.75, -1.5, "fuera de la banda", ha="center", fontsize=8,
                    color=TINTA)
        ax.set_ylim(-1.9, 1.25)
        ax.set_yticks([-1, 0, 1])
        ax.set_ylabel("Re $H$")
        ax.set_title(titulo, fontsize=9, color=TINTA, loc="left")
    for signo in (-1, 1):
        ejes[0].axvline(signo * 0.25, color=COLOR["fft"], lw=1.0, ls=(0, (3, 2)))
    ejes[0].annotate("más allá de ±0.25, las muestras\nya no siguen a $H$", (0.25, -1.45),
                     xytext=(6, 0), textcoords="offset points", fontsize=8,
                     color=COLOR["fft"], va="center")
    ejes[2].set_xlim(-1, 1)
    ejes[2].set_xlabel(r"frecuencia / Nyquist,  $f\,/\,[1/(2\delta)]$")
    fig.tight_layout()
    guardar(fig, "g_muestreo_h")


# ------------------------------------------------------ banda bien muestreada
def figura_fmax():
    N, d, lamb = 512, 5.0 / 512, 632.8e-6        # escenario de Zhao, en mm
    nyq = 1 / (2 * d)
    z = np.logspace(1, np.log10(3e5), 500)
    fft = N * d / (lamb * np.sqrt(4 * z**2 + (N * d)**2))
    blas = 1 / (lamb * np.sqrt((z / (N * d))**2 + 1))

    def mp(s):
        a = (s * N * lamb)
        return np.sqrt(np.sqrt(a**4 + 64 * (a * z)**2) - a**2) / (4 * np.sqrt(2) * z * lamb)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    limpiar(ax, rejilla="both")
    ax.axhline(nyq, color=SUAVE, lw=0.9, ls=(0, (4, 3)))
    ax.text(1.2e-2, nyq * 1.1, "Nyquist, 51 mm⁻¹", fontsize=8, color=TINTA)
    ax.plot(z * 1e-3, np.minimum(fft, nyq), color=COLOR["fft"], lw=1.8,
            ls="--", label="FFT-ASM (Ec. 2.5)")
    ax.plot(z * 1e-3, np.minimum(blas, nyq), color=COLOR["blas"], lw=1.8,
            ls="-.", label="BLAS, relleno ×2 (Ec. 2.7)")
    ax.plot(z * 1e-3, np.minimum(mp(1), nyq), color=COLOR["mpasm"], lw=1.3,
            ls=(0, (1, 1)), label="MPASM, s = 1 (Ec. 2.11)")
    ax.plot(z * 1e-3, np.minimum(mp(10), nyq), color=COLOR["mpasm"], lw=1.8,
            label="MPASM, s = 10 (Ec. 2.11)")
    ax.annotate(r"cae como $1/z$", (40, fft[np.argmin(abs(z - 4e4))]), xytext=(-52, -14),
                textcoords="offset points", fontsize=8.5, color=COLOR["fft"])
    ax.annotate(r"cae como $1/\sqrt{z}$", (40, mp(10)[np.argmin(abs(z - 4e4))]),
                xytext=(-16, 8), textcoords="offset points", fontsize=8.5,
                color=COLOR["mpasm"])
    ax.set_xlabel("distancia z [m]")
    ax.set_ylabel("frecuencia máxima bien muestreada [mm⁻¹]")
    ax.set_xlim(1e-2, 3e2)
    ax.set_ylim(0.03, 120)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    guardar(fig, "g_fmax")


# ----------------------------------------------------- tres lecturas del escalado
def figura_tres_lecturas():
    fig, ejes = plt.subplots(3, 1, figsize=(6.5, 4.7),
                             gridspec_kw=dict(height_ratios=[1.25, 0.72, 1.03]))
    limites = ((-1.42, 1.0), (-0.95, 0.42), (-1.25, 0.88))
    for ax, (y0, y1) in zip(ejes, limites):
        ax.set_xlim(0, 10)
        ax.set_ylim(y0, y1)
        ax.axis("off")

    def cota(ax, x0, x1, y, texto):
        ax.annotate("", (x0, y), (x1, y),
                    arrowprops=dict(arrowstyle="<->", color=TINTA, lw=0.8))
        ax.text((x0 + x1) / 2, y - 0.08, texto, ha="center", va="top", fontsize=8.5,
                color=TINTA)

    # (a) montaje real
    ax = ejes[0]
    ax.set_title("(a) El montaje real: onda esférica", loc="left", fontsize=9, color=TINTA)
    xf, xm, xs = 0.7, 2.9, 6.6
    for signo in (-1, 1):
        ax.plot([xf, xs], [0, signo * 0.8], color=COLOR["mpasm"], lw=0.8)
    ax.plot(xf, 0, "o", color=TINTA, ms=4)
    ax.plot([xm, xm], [-.3, .3], color=TINTA, lw=2.2)
    ax.plot([xs, xs], [-.8, .8], color=TINTA, lw=2.2)
    ax.text(xf, 0.2, "fuente", ha="center", fontsize=8, color=TINTA)
    ax.text(xm, 0.38, "muestra", ha="center", fontsize=8, color=TINTA)
    ax.text(xs + 0.1, 0.68, "sensor", ha="left", fontsize=8, color=TINTA)
    cota(ax, xf, xm, -0.98, "z = 3 mm")
    cota(ax, xm, xs, -0.98, "L − z = 7 mm")
    ax.text(7.5, -0.05, "píxel  δ = 1.83 µm\n" r"$M = L/z = 3.33$", fontsize=8.5,
            color=TINTA, va="center")

    # (b) y (c): onda plana
    def plana(ax, x0, x1, media, titulo, cota_txt, txt):
        ax.set_title(titulo, loc="left", fontsize=9, color=TINTA)
        for y in np.linspace(-media, media, 5):
            ax.annotate("", (x0, y), (x0 - 0.5, y),
                        arrowprops=dict(arrowstyle="-|>", color=COLOR["mpasm"], lw=0.7))
            ax.plot([x0, x1], [y, y], color=COLOR["mpasm"], lw=0.6, alpha=0.6)
        ax.plot([x0, x0], [-media, media], color=TINTA, lw=2.2)
        ax.plot([x1, x1], [-media, media], color=TINTA, lw=2.2)
        cota(ax, x0, x1, -media - 0.2, cota_txt)
        ax.text(7.5, 0, txt, fontsize=8.5, color=TINTA, va="center")

    plana(ejes[1], 1.2, 3.0, 0.3, "(b) Leído en la malla de la muestra: onda plana",
          r"$z_\mathrm{eq} = z(L-z)/L$ = 2.1 mm",
          "paso  δ/M = 0.55 µm\nventana  Nδ/M")
    plana(ejes[2], 1.2, 6.6, 0.8, "(c) Leído en la malla del sensor: la misma propagación",
          r"$D = M^2 z_\mathrm{eq} = L(L-z)/z$ = 23.3 mm",
          "paso  δ = 1.83 µm\nventana  Nδ")
    fig.tight_layout(h_pad=0.2)
    guardar(fig, "g_tres_lecturas")


# ------------------------------------------------------- q contra la muestra
def figura_q():
    lamb = 528e-6

    def q(z, N, d, L):
        return lamb * L * (L - z) / (N * d**2 * z)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    limpiar(ax, rejilla="both")
    z = np.logspace(np.log10(0.3), np.log10(9), 300)
    d_exp, L_exp = 3.80e-3, 18.23
    for y, txt in ((1, "q = 1: sin relleno, el campo deja de caber"),
                   (3.2, "q ≈ 3: falla el FFT-ASM ×2")):
        ax.axhline(y, color=SUAVE, lw=0.9, ls=(0, (4, 3)))
        ax.text(0.31, y * 0.93, txt, fontsize=8, color=TINTA, ha="left", va="top")
    # geometría inferida, con el rango real de la platina en trazo grueso
    rango = (0.413, 2.001)
    for N, ls_, tono, etiqueta in ((4640, "-", COLOR["mpasm"], "hologramas, sensor entero (4640 px)"),
                                   (512, "-", "#6f93b5", "hologramas, recorte común (512 px)")):
        ax.plot(z, q(z, N, d_exp, L_exp), color=tono, lw=0.8, ls=ls_)
        zr = np.logspace(np.log10(rango[0]), np.log10(rango[1]), 100)
        ax.plot(zr, q(zr, N, d_exp, L_exp), color=tono, lw=2.4, label=etiqueta)
    ax.plot(z, q(z, 1024, 1.83e-3, 10.0), color=TINTA, lw=1.3, ls=(0, (5, 2)),
            label="banco nominal, ida y vuelta (1024 px)")
    ax.plot(3, q(3, 1024, 1.83e-3, 10.0), "o", color=TINTA, ms=5, zorder=4)
    ax.annotate("punto nominal\nq = 3.6", (3, q(3, 1024, 1.83e-3, 10.0)), xytext=(8, 2),
                textcoords="offset points", fontsize=8, color=TINTA)
    for zr, N in ((rango[0], 4640), (rango[1], 4640)):
        v = q(zr, N, d_exp, L_exp)
        ax.plot(zr, v, "o", color=COLOR["mpasm"], ms=5, zorder=4)
        ax.annotate(f"{v:.1f}", (zr, v), xytext=(0, -13), textcoords="offset points",
                    fontsize=8.5, color=COLOR["mpasm"], ha="center")
    ax.set_xlabel("z: distancia de la muestra a la fuente [mm]")
    ax.set_ylabel(r"cociente $q = \lambda L(L-z)\,/\,(N\delta^2 z)$")
    ax.set_xlim(0.3, 9)
    ax.set_ylim(0.1, 200)
    ax.set_xticks([0.3, 0.5, 1, 2, 3, 5, 8], ["0.3", "0.5", "1", "2", "3", "5", "8"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    guardar(fig, "g_q")


# ---------------------------------------------------------- el MPASM esférico
def figura_esferica():
    lamb, d, L = 528e-6, 1.83e-3, 10.0
    X = 1500 * d
    nyq = 1 / (2 * d)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.95),
                               gridspec_kw=dict(width_ratios=[1, 1.45]))
    limpiar(a)
    x = np.linspace(-X, X, 400)
    a.axhspan(-nyq, nyq, color=COLOR["mpasm"], alpha=0.10, lw=0)
    a.plot(x, x / (lamb * np.sqrt(x**2 + L**2)), color=TINTA, lw=1.7)
    a.text(-X, nyq * 0.78, "Nyquist del píxel: ±273", fontsize=8, ha="left",
           color=COLOR["mpasm"])
    a.annotate("501 mm⁻¹", (X, X / (lamb * np.hypot(X, L))), xytext=(-48, -2),
               textcoords="offset points", fontsize=8.5, color=TINTA)
    a.set_xlabel("posición en el sensor [mm]")
    a.set_ylabel("frecuencia local de la esférica [mm⁻¹]")
    a.set_title("(a) muestreada, se aliasa", fontsize=9, color=TINTA)

    limpiar(b, rejilla=None)
    f = np.linspace(-850, 850, 3401)
    centros = np.array([-1500, 0, 1500]) * d
    A = np.abs(ventana_chirp(centros, f, d, lamb, L)) / d
    for fila, xn in zip(A, centros):
        b.plot(f, fila, color=COLOR["mpasm"], lw=0.9)
        b.fill_between(f, 0, fila, color=COLOR["mpasm"], alpha=0.10, lw=0)
        rotulo = "$x_n$ = 0" if xn == 0 else f"$x_n$ = {xn:+.1f} mm".replace("-", "−")
        b.text(xn / (lamb * L), 1.26, rotulo, fontsize=8, ha="center", color=TINTA)
    b.vlines((-nyq, nyq), -0.03, 1.2, color=SUAVE, lw=0.9, ls=(0, (4, 3)))
    borde = X / (lamb * L) + nyq
    b.annotate("", (-borde, -0.17), (borde, -0.17),
               arrowprops=dict(arrowstyle="<->", color=TINTA, lw=0.8))
    b.text(0, -0.15, f"banda necesaria: ±{borde:.0f} mm⁻¹, $K_f$ ≈ {nyq / borde:.2f}",
           ha="center", va="bottom", fontsize=8, color=TINTA,
           bbox=dict(facecolor="white", edgecolor="none", pad=1))
    b.set_ylim(-0.26, 1.42)
    b.set_yticks([0, 1])
    b.set_xlabel("frecuencia f [mm⁻¹]")
    b.set_ylabel(r"$|W(f - x_n/\lambda L)|\,/\,\delta$")
    b.set_title("(b) calculada: un bloque por píxel", fontsize=9, color=TINTA)
    fig.tight_layout()
    guardar(fig, "g_esferica")


# ------------------------------------------------- órdenes de una red y ventana
def figura_ordenes():
    lamb, z = 405e-6, 192.6
    fija, ampliada = 1.766 / 2, 11.8 / 2
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    limpiar(ax, rejilla=None)
    for p, tono in ((88.3, TINTA), (27.6, TINTA), (13.2, TINTA)):
        y = lamb * z / (p * 1e-3)
        for signo in (-1, 1):
            ax.plot([0, z], [0, signo * y], color=tono, lw=0.9)
        ax.text(z + 3, y, f"periodo {p:g} µm", fontsize=8, color=TINTA, va="center")
    ax.plot([0, z], [0, 0], color=SUAVE, lw=0.7)
    ax.plot([z, z], [-ampliada, ampliada], color=COLOR["mpasm"], lw=5, alpha=0.35,
            solid_capstyle="butt", label="ventana ampliada, 11.8 mm")
    ax.plot([z, z], [-fija, fija], color=TINTA, lw=5, solid_capstyle="butt",
            label="ventana fija, 1.77 mm")
    ax.plot([0, 0], [-0.883, 0.883], color=TINTA, lw=2.2)
    ax.text(-3, 1.3, "objeto", fontsize=8, color=TINTA, ha="left")
    ax.set_xlim(-6, 232)
    ax.set_ylim(-7, 7)
    ax.set_xlabel("distancia z [mm]")
    ax.set_ylabel("posición lateral [mm]")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    guardar(fig, "g_ordenes")


# --------------------------------------------------------- la recta M contra D
def figura_recta():
    filas = list(csv.DictReader(open(RAIZ / "resultados" / "hologramas_carlos" /
                                     "L_implicito.csv")))
    D = np.array([float(f["D"]) for f in filas])
    M = np.array([float(f["M"]) for f in filas])
    e = np.array([float(f["eM"]) for f in filas])
    p, L = 3.80 / 1.83, 18.23
    x = np.linspace(0, 200, 10)
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    limpiar(ax, rejilla="both")
    ax.plot(x, 1 + x / 10, color=SUAVE, lw=1.6, ls=(0, (4, 2)),
            label=r"nominal: $\delta$ = 1.83 µm, $L$ = 10 mm")
    ax.plot(x, 1 / p + p / L * x, color=COLOR["mpasm"], lw=1.8,
            label=r"adoptada: $\delta$ = 3.80 µm, $L$ = 18.23 mm")
    ax.errorbar(D, M, yerr=e, fmt="o", color=TINTA, ms=4.5, lw=1, capsize=2,
                label="medido con el USAF, grupos 4, 6 y 7", zorder=4)
    ax.plot(35.142, 4.46, "o", mfc="white", mec=TINTA, ms=9, zorder=3,
            label="grupo 5, fuera del ajuste")
    for f, d_, m_ in zip(filas, D, M):
        ax.annotate(f"posición {f['k']}", (d_, m_), xytext=(-8, 7),
                    textcoords="offset points", fontsize=8, color=TINTA, ha="right")
    ax.set_xlabel(r"foco medido con el píxel nominal, $D_\mathrm{medido}$ [mm]")
    ax.set_ylabel(r"magnificación medida, $M_\mathrm{medida}$")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 24)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    guardar(fig, "g_recta")


# ------------------------------------------------------- memoria de la cadena
def figura_memoria():
    lado = [512, 768, 1024, 2048]
    gb = [1.1, 2.6, 4.8, 19]
    malla = [4994, 7666, 10309, 20787]
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    limpiar(ax)
    pos = np.arange(len(lado))
    ax.bar(pos, gb, width=0.42, color=COLOR["fft"], alpha=0.9)
    ax.axhline(4, color=TINTA, lw=1.0, ls=(0, (4, 3)))
    ax.text(-0.42, 4.5, "GPU de 4 GB", fontsize=8.5, color=TINTA)
    for x_, g, m in zip(pos, gb, malla):
        ax.text(x_, g + 0.45, f"{g:g} GB", ha="center", fontsize=8.5, color=TINTA,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=5)
    ax.set_xticks(pos, [f"recorte de {l} px\nmalla de {m:,}".replace(",", " ")
                        for l, m in zip(lado, malla)])
    ax.set_ylabel("memoria que pide [GB]")
    ax.set_ylim(0, 22)
    ax.set_xlim(-0.5, 3.5)
    guardar(fig, "g_memoria")


# ------------------------------------------------- dónde cae cada caso en q
def figura_resumen_q():
    filas = list(csv.DictReader(open(RAIZ / "resultados" / "barrido_obj2" / "barrido.csv")))
    lamb, d, L = 0.528e-3, 1.83e-3, 10.0
    qs = [lamb * L * (L - float(f["z"])) / (int(f["N"]) * d * d * float(f["z"])) for f in filas]
    casos = [
        ("DHM con objetivo, reenfoque de 1 a 10 mm", 1 / 45.9, 10 / 45.9),
        ("Barrido del banco", min(qs), max(qs)),
        ("Ida y vuelta, banco DLHM (1024 px)", 0.66, 13.9),
        ("Ida y vuelta, onda plana", 0.33, 128),
        ("Hologramas, sensor entero", 1.2, 6.2),
        ("Hologramas, recorte común de 512 px", 10.6, 56),
        ("Anexo: interferometría simulada", 0.5, 20),
        ("Holografía de Fresnel a 1 m", 21.8, 21.8),
        ("Escenario de Zhao, 0.5 a 200 m", 500 / 77.4, 200000 / 77.4),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.set_xscale("log")
    limpiar(ax, rejilla="x")
    ax.spines["left"].set_visible(False)
    y = np.arange(len(casos))[::-1]
    for y_, (nombre, q0, q1) in zip(y, casos):
        if q1 > q0:
            ax.plot([q0, q1], [y_, y_], color="#4a4a4a", lw=3.2,
                    solid_capstyle="round")
        else:
            ax.plot(q0, y_, "o", color="#4a4a4a", ms=5.5)
    ax.plot(2.59, y[1], "o", mfc="white", mec="#4a4a4a", ms=5.5, zorder=5)
    ax.annotate("punto nominal, 2.6", (2.59, y[1]), xytext=(0, 6),
                textcoords="offset points", fontsize=7.5, color=TINTA, ha="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    for x_, txt, ha_ in ((1, "q = 1", "right"), (3.2, "q ≈ 3", "left")):
        ax.vlines(x_, -0.6, len(casos) - 0.55, color=COLOR["fft"], lw=0.9,
                  ls=(0, (4, 3)), zorder=1)
        ax.text(x_, len(casos) - 0.48, txt, fontsize=8.5, color=COLOR["fft"], ha=ha_,
                va="bottom")
    ax.set_yticks(y, [c[0] for c in casos])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(casos) + 0.05)
    ax.set_xlim(0.01, 5000)
    ax.set_xlabel("cociente q: distancia equivalente frente a la ventana")
    guardar(fig, "g_resumen_q")


# ---------------------------------------------------------------- s = 1
def figura_s1():
    filas = list(csv.DictReader(open(RAIZ / "resultados" / "barrido_obj2" / "barrido.csv")))
    T = {}
    for f in filas:
        if f["metodo"] == "mpasm" and f["valido"] == "True":
            T[(float(f["z"]), int(f["N"]), int(float(f["s"])), float(f["kf_factor"]))] = f
    w = 2.5
    # Una misma rampa por s, de claro a oscuro, como en la Figura 2.1.
    tonos = {1: "#7fb3d5", 2: "#3a7ca5", 3: "#14344f"}
    marcas = {1: "o", 2: "s", 3: "^"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.15))
    pares = []
    for (z, N, s, k), f in T.items():
        if k != 1.0:
            continue
        kf, e, piso = float(f["kf"]), float(f["err_rel"]), float(f["err_ref"])
        if kf > 1.0001 and e > 5 * piso:
            pares.append((s, math.exp(-(math.pi * w * 0.5 / kf)**2), e))
    limpiar(a, rejilla="both")
    a.set_xscale("log")
    a.set_yscale("log")
    for s in (1, 2, 3):
        x = [p[1] for p in pares if p[0] == s]
        y = [p[2] for p in pares if p[0] == s]
        a.plot(x, y, marcas[s], color=tonos[s], ms=4.5, mec="white", mew=0.4,
               label=f"s = {s}")
    lx = np.log10([p[1] for p in pares])
    ly = np.log10([p[2] for p in pares])
    r = np.corrcoef(lx, ly)[0, 1]
    print(f"   n = {len(pares)}, r = {r:.3f}")
    a.text(0.97, 0.05, f"r = {r:.2f} en log-log, n = {len(pares)}", transform=a.transAxes,
           ha="right", fontsize=8, color=TINTA)
    a.set_xlabel("amplitud del espectro en el borde de la banda")
    a.set_ylabel(r"error relativo $\varepsilon$")
    a.set_title("(a) el error sigue al borde de la banda", fontsize=9, color=TINTA)
    a.legend(frameon=False, fontsize=8, loc="upper left")

    limpiar(b, rejilla="both")
    b.set_xscale("log")
    b.set_yscale("log")
    xs, ys, rel = [], [], []
    for (z, N, s, k), f in T.items():
        if s != 1 or k != 1.0:
            continue
        g = T.get((z, N, 3, 2.0))
        h = T.get((z, N, 3, 1.0))
        if not g or not h or float(h["kf"]) <= 1.0001:
            continue
        xs.append(float(f["err_rel"]))
        ys.append(float(g["err_rel"]))
        rel.append(3 * float(g["kf"]) / float(f["kf"]))
    print(f"   n = {len(xs)}, periodo x{statistics.median(rel):.2f}, "
          f"razon mediana {statistics.median(y / x for x, y in zip(xs, ys)):.2f}")
    lim = [min(xs + ys) * 0.7, max(xs + ys) * 1.5]
    b.plot(lim, lim, color=SUAVE, lw=0.9, ls=(0, (4, 3)))
    b.text(lim[1] * 0.8, lim[1] * 0.42, "mismo error", fontsize=8, color=TINTA, rotation=38,
           ha="right", va="top")
    b.plot(xs, ys, "o", color=COLOR["mpasm"], ms=4.5, mec="white", mew=0.4)
    b.set_xlim(lim)
    b.set_ylim(lim)
    b.set_xlabel(r"$\varepsilon$ con s = 1 y $K_f$ automático")
    b.set_ylabel(r"$\varepsilon$ con s = 3 y $K_f$ al doble")
    b.set_title("(b) casi la misma banda, periodo 3.5× mayor", fontsize=9, color=TINTA)
    fig.tight_layout()
    guardar(fig, "g_s1")


FIGURAS = {
    "campo": figura_campo_ventana, "rayos": figura_rayos,
    "muestreo": figura_muestreo_h, "fmax": figura_fmax,
    "lecturas": figura_tres_lecturas, "q": figura_q,
    "esferica": figura_esferica, "ordenes": figura_ordenes,
    "recta": figura_recta, "memoria": figura_memoria,
    "resumen": figura_resumen_q, "s1": figura_s1,
}

if __name__ == "__main__":
    estilo()
    for clave in sys.argv[1:] or FIGURAS:
        print(clave)
        FIGURAS[clave]()
