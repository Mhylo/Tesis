"""Figura del escenario de la Tabla 1: apertura y frente de onda propagado.

Reproduce el modelo de simulación del paper (Zhao et al., Opt. Lett. 45, 5937)
y lo dibuja: la apertura de entrada, su frente de onda, y ambos después de
propagar.

Sobre la "lente" del paper. La Tabla 1 pide una lente de f = -300 mm y la
Fig. 2 se titula "Gaussian beam through lens", lo que choca de frente con una
tesis sobre holografía SIN lente. No hay contradicción: aplicada sobre la
cintura, esa lente deja el haz con radio de curvatura R = 300 mm exactos y la
cintura intacta, es decir, divergiendo como si naciera en un punto 300 mm
detrás. Es un pinhole virtual. Por eso aquí no se modela ninguna lente, sino
la primitiva que de verdad hay: un frente esférico divergente de radio R.
Comprobación dentro del propio script: propagar a z devuelve un radio 1/e
igual a w0*(R+z)/R, el factor geométrico de una fuente puntual.

Dos decisiones que la figura hace explícitas:

La malla de salida es MAG veces la de entrada. A z = 1200 mm el haz mide
10 mm de diámetro y la ventana de entrada son 5 mm: no cabe. Se usa el
parámetro mag de mpasm (los r1, r2 de las Ecs. (7)-(8) del paper), que FFT-ASM
no tiene: su ventana de salida está atada a la de entrada.

El frente de onda propagado se recorta al radio 1/e. No es estética. Con el
paso de salida delta*MAG las franjas dejan de estar muestreadas más allá de
ese radio, y el mapa completo mostraría aliasing con aspecto de estructura
real.

Uso:
    Tesis_env/Scripts/python.exe -m scripts.figura_escenario
    Tesis_env/Scripts/python.exe -m scripts.figura_escenario --z 400 --mag 3
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")            # sin ventana: el script guarda, no muestra
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from CamposT.propagadores import mpasm

#: raíz del repo, para que la salida no dependa del directorio de invocación
RAIZ = pathlib.Path(__file__).resolve().parent.parent

# --- Tabla 1 del paper -------------------------------------------------------
L0 = 5.0            # mm, ventana de entrada
W0 = 1.0            # mm, radio de cintura
LAMB = 632.8e-6     # mm, HeNe
R = 300.0           # mm, radio del frente esférico (la lente de f = -300 mm)

TINTA, SUAVE, AZUL, TEJA = "#1a1a1a", "#8a8a8a", "#1f4e79", "#c1543a"


def campo_entrada(N):
    """Campo de la Tabla 1: envolvente gaussiana por frente esférico divergente.

    Devuelve (U0, delta, x). El signo positivo del exponente es el que diverge;
    con el negativo el haz enfoca.
    """
    delta = L0 / (N - 1)
    x = (np.arange(N) - N / 2) * delta
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    U0 = np.exp(-r2 / W0**2) * np.exp(1j * (2 * np.pi / LAMB) * r2 / (2 * R))
    return U0, delta, x


def _mapa(fig, ax, campo, coord, titulo, clase, recorte=None):
    """Pinta amplitud (gris, secuencial) o fase (twilight, cíclica).

    La fase envuelve en ±pi, así que un mapa secuencial inventaría un salto
    donde no lo hay; y se enmascara donde no hay luz, porque allí es ruido.
    """
    ext = [coord[0], coord[-1], coord[0], coord[-1]]
    amp = np.abs(campo)
    if clase == "amp":
        im = ax.imshow(amp / amp.max(), extent=ext, origin="lower",
                       cmap="gray_r", vmin=0, vmax=1)
        etiqueta = "amplitud normalizada"
    else:
        fase = np.ma.masked_where(amp < 0.01 * amp.max(), np.angle(campo))
        cm = plt.get_cmap("twilight").copy()
        cm.set_bad("#f2f2f2")
        im = ax.imshow(fase, extent=ext, origin="lower", cmap=cm,
                       vmin=-np.pi, vmax=np.pi)
        etiqueta = "fase [rad]"
    if recorte is not None:
        ax.set_xlim(-recorte, recorte)
        ax.set_ylim(-recorte, recorte)
    ax.set_title(titulo, fontsize=9.5, color=TINTA, pad=7)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(etiqueta, fontsize=8, color=TINTA)
    cb.outline.set_edgecolor(SUAVE)
    cb.ax.tick_params(labelsize=7, color=SUAVE)
    if clase == "fase":
        cb.set_ticks([-np.pi, 0, np.pi])
        cb.set_ticklabels(["-pi", "0", "+pi"])


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=512, help="lado de la malla")
    p.add_argument("--z", type=float, default=1200.0,
                   help="distancia de propagación en mm (la Fig. 3 usa 1200)")
    p.add_argument("--s", type=int, default=4, help="sobremuestreo de MPASM")
    p.add_argument("--mag", type=float, default=6.0,
                   help="ampliación de la malla de salida")
    p.add_argument("--salida", type=pathlib.Path,
                   default=RAIZ / "resultados" / "apertura_frente.png")
    args = p.parse_args(argv)

    U0, delta, x = campo_entrada(args.n)
    U, Kf = mpasm(U0, delta, LAMB, args.z, s=args.s, mag=args.mag, device="cpu")
    U = np.asarray(U)
    delta_out = delta * args.mag
    x_out = (np.arange(U.shape[0]) - U.shape[0] / 2) * delta_out
    radio = W0 * (R + args.z) / R          # radio 1/e por óptica geométrica

    print(f"delta entrada {delta:.5f} mm (Tabla 1: 0.0098) | "
          f"delta salida {delta_out:.5f} mm | Kf = {Kf:.3f}")
    print(f"ventana entrada {L0:.2f} mm | ventana salida "
          f"{U.shape[0] * delta_out:.2f} mm | radio 1/e esperado {radio:.2f} mm")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": SUAVE, "axes.labelcolor": TINTA,
        "xtick.color": SUAVE, "ytick.color": SUAVE,
        "axes.linewidth": 0.8, "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(9.2, 9.6))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.72],
                  hspace=0.42, wspace=0.28)

    _mapa(fig, fig.add_subplot(gs[0, 0]), U0, x,
          f"Apertura de entrada  ·  ventana {L0:.0f} mm", "amp")
    _mapa(fig, fig.add_subplot(gs[0, 1]), U0, x,
          f"Frente de onda de entrada  ·  R = {R:.0f} mm divergente", "fase")
    _mapa(fig, fig.add_subplot(gs[1, 0]), U, x_out,
          f"Amplitud propagada  ·  z = {args.z:.0f} mm", "amp")
    _mapa(fig, fig.add_subplot(gs[1, 1]), U, x_out,
          "Frente de onda propagado  ·  zoom al radio 1/e", "fase",
          recorte=radio)

    ax = fig.add_subplot(gs[2, :])
    a0 = np.abs(U0[args.n // 2]); a0 /= a0.max()
    a1 = np.abs(U[U.shape[0] // 2]); a1 /= a1.max()
    ax.plot(x, a0, lw=2, color=AZUL, solid_capstyle="round")
    ax.plot(x_out, a1, lw=2, color=TEJA, solid_capstyle="round")
    ax.text(1.15, 0.62, "entrada", color=AZUL, fontsize=9, fontweight="bold")
    ax.text(radio + 1.6, 0.72, f"propagado a z = {args.z:.0f} mm", color=TEJA,
            fontsize=9, fontweight="bold")
    ax.axvline(radio, color=SUAVE, lw=1, ls=(0, (4, 3)))
    ax.annotate(f"radio 1/e geometrico\nw0(R+z)/R = {radio:.1f} mm",
                xy=(radio, 0.37), xytext=(radio + 1.8, 0.20),
                fontsize=8, color=TINTA,
                arrowprops=dict(arrowstyle="-", color=SUAVE, lw=0.8))
    ax.set_xlim(-3 * radio, 3 * radio)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("amplitud normalizada")
    ax.set_title(f"Corte horizontal por el centro  ·  el haz se expande "
                 f"{radio / W0:.0f}x", fontsize=9.5, color=TINTA, pad=7)
    ax.grid(axis="y", color="#e8e8e8", lw=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

    fig.suptitle("Escenario de la Tabla 1: apertura y frente de onda propagado",
                 fontsize=12, color=TINTA, y=0.975)
    fig.text(0.5, 0.945,
             f"gaussiano w0 = {W0:.0f} mm  ·  lambda = 632.8 nm  ·  "
             f"MPASM s = {args.s}, Kf = {Kf:.2f}, malla de salida x{args.mag:.0f}",
             ha="center", fontsize=8.5, color=SUAVE)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.salida, dpi=200, bbox_inches="tight")
    print("guardado:", args.salida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
