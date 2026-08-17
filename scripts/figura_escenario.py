"""Figuras del escenario de la Tabla 1, una por archivo.

Reproduce el modelo de simulación del paper (Zhao et al., Opt. Lett. 45, 5937)
y lo dibuja en tres figuras independientes, en resultados/escenario/:

    apertura.png          la forma de la apertura de entrada
    frente_propagado.png  el frente de onda del haz tras propagar
    corte_axial.png       la sección longitudinal x-z: cómo diverge el haz

Sobre la "lente" del paper. La Tabla 1 pide una lente de f = -300 mm y la
Fig. 2 se titula "Gaussian beam through lens", lo que choca de frente con una
tesis sobre holografía SIN lente. No hay contradicción: aplicada sobre la
cintura, esa lente deja el haz con radio de curvatura R = 300 mm exactos y la
cintura intacta, es decir, divergiendo como si naciera en un punto 300 mm
detrás. Es un pinhole virtual. Por eso aquí no se modela ninguna lente, sino
la primitiva que de verdad hay: un frente esférico divergente de radio R.

El corte axial es la comprobación de esa afirmación, y no una ilustración: el
radio 1/e medido sobre el barrido se contrasta punto a punto contra el factor
geométrico w0*(R+z)/R de una fuente puntual. La desviación máxima se imprime.

Dos decisiones que las figuras hacen explícitas:

La malla de salida es MAG veces la de entrada. A z = 1200 mm el haz mide
10 mm de diámetro y la ventana de entrada son 5 mm: no cabe. Se usa el
parámetro mag de mpasm (los r1, r2 de las Ecs. (7)-(8) del paper), que FFT-ASM
no tiene: su ventana de salida está atada a la de entrada.

Cada figura elige su propio MAG, porque cada una responde a otra pregunta y el
muestreo que le conviene es distinto. El caso que lo obliga es el frente de
onda: se recorta al radio 1/e, así que su ventana debe ajustarse a ese recorte.
Con MAG = 6 la ventana son 30 mm de los que se muestran 10, y en el borde
quedan 3.2 muestras por franja: por encima de Nyquist —el dato es correcto—
pero ilegible, porque con un mapa cíclico eso produce un batido con aspecto de
tablero que se confunde con aliasing. Ajustando MAG al recorte quedan ~9
muestras por franja. muestras_por_franja() calcula la cifra y el pie de la
figura la reporta, para que no haya que fiarse del ojo.

Uso:
    Tesis_env/Scripts/python.exe -m scripts.figura_escenario
    Tesis_env/Scripts/python.exe -m scripts.figura_escenario --z 400
    Tesis_env/Scripts/python.exe -m scripts.figura_escenario --nz 40   # rápido
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")            # sin ventana: el script guarda, no muestra
import matplotlib.pyplot as plt
import numpy as np

from CamposT.backend import a_numpy, gpu_disponible, liberar_memoria
from CamposT.propagadores import mpasm

#: raíz del repo, para que la salida no dependa del directorio de invocación
RAIZ = pathlib.Path(__file__).resolve().parent.parent

# --- Tabla 1 del paper -------------------------------------------------------
L0 = 5.0            # mm, ventana de entrada
W0 = 1.0            # mm, radio de cintura
LAMB = 632.8e-6     # mm, HeNe
R = 300.0           # mm, radio del frente esférico (la lente de f = -300 mm)

TINTA, SUAVE, AZUL, TEJA = "#1a1a1a", "#8a8a8a", "#1f4e79", "#c1543a"


def radio_geometrico(z):
    """Radio 1/e por óptica geométrica: el haz diverge desde un punto a -R."""
    return W0 * (R + z) / R


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


def estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": SUAVE, "axes.labelcolor": TINTA,
        "xtick.color": SUAVE, "ytick.color": SUAVE,
        "axes.linewidth": 0.8, "figure.facecolor": "white",
    })


def _limpiar(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def _barra(fig, ax, im, etiqueta):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(etiqueta, fontsize=8, color=TINTA)
    cb.outline.set_edgecolor(SUAVE)
    cb.ax.tick_params(labelsize=7, color=SUAVE)
    return cb


# ------------------------------------------------------------ 1) la apertura
def figura_apertura(U0, x, destino):
    """Amplitud del campo de entrada: la forma de la apertura."""
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    amp = np.abs(U0)
    ext = [x[0], x[-1], x[0], x[-1]]
    im = ax.imshow(amp / amp.max(), extent=ext, origin="lower",
                   cmap="gray_r", vmin=0, vmax=1)
    # el radio 1/e de la cintura, que es la escala del problema
    ax.add_patch(plt.Circle((0, 0), W0, fill=False, color=TEJA, lw=1.4,
                            ls=(0, (5, 3))))
    ax.annotate(f"w0 = {W0:.0f} mm", xy=(W0 * 0.71, W0 * 0.71),
                xytext=(1.55, 1.75), fontsize=8.5, color=TEJA,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEJA, lw=0.8))
    _barra(fig, ax, im, "amplitud normalizada")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Apertura de entrada  ·  ventana {L0:.0f} mm",
                 fontsize=10.5, color=TINTA, pad=8)
    _limpiar(ax)
    fig.text(0.5, -0.02,
             f"gaussiano w0 = {W0:.0f} mm  ·  lambda = 632.8 nm  ·  "
             f"frente esférico divergente R = {R:.0f} mm",
             ha="center", fontsize=8, color=SUAVE)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ->", destino)


# ------------------------------------------------- 2) el frente de onda propagado
def muestras_por_franja(radio, z, delta_out):
    """Cuántas muestras caen en un periodo de franja en el borde del recorte.

    El frente propagado es esférico de radio R+z, así que su fase vale
    k·r²/2(R+z) y el paso entre píxeles vecinos, k·r·delta_out/(R+z). Nyquist
    admite hasta 2 muestras por franja, pero eso ya no se LEE como un frente:
    con un mapa cíclico, 3 muestras por periodo producen un batido con aspecto
    de tablero que parece aliasing sin serlo. Se pide holgura.
    """
    paso = (2 * np.pi / LAMB) * radio * delta_out / (R + z)
    return 2 * np.pi / paso


def figura_frente(U, x_out, z, radio, Kf, s, mag, destino):
    """Fase del campo propagado: el frente de onda del haz gaussiano.

    Se pinta con un mapa cíclico porque la fase envuelve en ±pi, y se enmascara
    donde no hay luz, porque allí es ruido. Encima van las curvas de nivel de
    amplitud, para que se vea a qué parte del haz corresponde cada franja.
    """
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    amp = np.abs(U)
    amp_n = amp / amp.max()
    ext = [x_out[0], x_out[-1], x_out[0], x_out[-1]]

    fase = np.ma.masked_where(amp_n < 0.01, np.angle(U))
    cm = plt.get_cmap("twilight").copy()
    cm.set_bad("#f2f2f2")
    im = ax.imshow(fase, extent=ext, origin="lower", cmap=cm,
                   vmin=-np.pi, vmax=np.pi)

    X, Y = np.meshgrid(x_out, x_out)
    cs = ax.contour(X, Y, amp_n, levels=[np.exp(-2), np.exp(-1)],
                    colors=[SUAVE, TINTA], linewidths=[0.9, 1.3],
                    linestyles=[(0, (4, 3)), "solid"])
    ax.clabel(cs, fmt={np.exp(-2): "1/e²", np.exp(-1): "1/e"},
              fontsize=7.5, inline=True)

    cb = _barra(fig, ax, im, "fase [rad]")
    cb.set_ticks([-np.pi, 0, np.pi])
    cb.set_ticklabels(["-pi", "0", "+pi"])

    ax.set_xlim(-radio, radio)
    ax.set_ylim(-radio, radio)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Frente de onda propagado  ·  z = {z:.0f} mm",
                 fontsize=10.5, color=TINTA, pad=8)
    _limpiar(ax)
    mpf = muestras_por_franja(radio, z, x_out[1] - x_out[0])
    fig.text(0.5, -0.02,
             f"MPASM s = {s}, Kf = {Kf:.2f}, malla de salida x{mag:.2f}  ·  "
             f"recortado al radio 1/e = {radio:.1f} mm  ·  "
             f"{mpf:.1f} muestras por franja en el borde",
             ha="center", fontsize=8, color=SUAVE)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ->", destino)


# --------------------------------------------------------- 3) el corte axial
def barrido_axial(U0, delta, z_max, nz, s, mag, device):
    """Propaga a nz distancias y devuelve (Z, x_out, mapa) del plano x-z.

    Se usa un mag FIJO para todas las distancias: si cambiara con z, cada
    columna del mapa tendría una escala transversal distinta y la figura
    mentiría sobre la divergencia, que es justo lo que se quiere medir.
    """
    Z = np.linspace(0.0, z_max, nz)
    columnas, x_out = [], None
    for i, z in enumerate(Z):
        U, _ = mpasm(U0, delta, LAMB, z, s=s, mag=mag, device=device)
        U = a_numpy(U)
        if x_out is None:
            n = U.shape[0]
            x_out = (np.arange(n) - n / 2) * delta * mag
        columnas.append(np.abs(U[U.shape[0] // 2]))
        if i % 20 == 0:
            liberar_memoria()
            print(f"    z = {z:7.1f} mm   ({i + 1}/{nz})", end="\r")
    liberar_memoria()
    print(" " * 40, end="\r")
    return Z, x_out, np.array(columnas).T          # (x, z)


def radio_medido(x_out, columna):
    """Radio 1/e leído del perfil: donde la amplitud cae a max/e.

    Se busca desde el centro hacia fuera y se interpola linealmente entre las
    dos muestras que cruzan el umbral.
    """
    centro = len(columna) // 2
    mitad = columna[centro:]
    umbral = mitad[0] / np.e
    debajo = np.nonzero(mitad < umbral)[0]
    if len(debajo) == 0:
        return np.nan
    j = debajo[0]
    if j == 0:
        return 0.0
    a, b = mitad[j - 1], mitad[j]
    t = (a - umbral) / (a - b)                     # 0 -> muestra j-1, 1 -> j
    return (x_out[centro + j - 1] * (1 - t) + x_out[centro + j] * t)


def figura_corte_axial(Z, x_out, mapa, destino):
    """Sección longitudinal x-z, con la predicción geométrica encima."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # normalizado COLUMNA A COLUMNA: la figura es sobre la anchura del haz, y
    # con normalización global la caída de amplitud (factor ~5 en 1200 mm)
    # oscurece el campo lejano hasta dejar el borde 1/e invisible. La caída va
    # aparte, en el pie.
    mapa_n = mapa / mapa.max(axis=0, keepdims=True)
    im = ax.imshow(mapa_n, origin="lower", aspect="auto", cmap="magma",
                   extent=[Z[0], Z[-1], x_out[0], x_out[-1]], vmin=0, vmax=1)

    geo = radio_geometrico(Z)
    ax.plot(Z, geo, lw=1.6, color="white", ls=(0, (5, 3)))
    ax.plot(Z, -geo, lw=1.6, color="white", ls=(0, (5, 3)))
    ax.annotate("radio 1/e geométrico   w0(R+z)/R",
                xy=(Z[int(0.72 * len(Z))], geo[int(0.72 * len(Z))]),
                xytext=(Z[-1] * 0.40, x_out[-1] * 0.70),
                fontsize=8.5, color="white", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="white", lw=0.9))

    _barra(fig, ax, im, "amplitud normalizada por columna")
    ax.set_xlabel("z [mm]   (dirección de propagación)")
    ax.set_ylabel("x [mm]")
    ax.set_title("Corte axial del haz  ·  sección longitudinal x-z",
                 fontsize=10.5, color=TINTA, pad=8)
    _limpiar(ax)
    caida = mapa[:, 0].max() / mapa[:, -1].max()
    fig.text(0.5, -0.02,
             f"el haz pasa de w0 = {W0:.0f} mm a {radio_geometrico(Z[-1]):.1f} mm "
             f"en {Z[-1]:.0f} mm  ·  {len(Z)} distancias propagadas con MPASM  ·  "
             f"la amplitud en el eje cae x{caida:.1f}",
             ha="center", fontsize=8, color=SUAVE)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ->", destino)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=512, help="lado de la malla")
    p.add_argument("--z", type=float, default=1200.0,
                   help="distancia de propagación en mm (la Fig. 3 usa 1200)")
    p.add_argument("--s", type=int, default=4, help="sobremuestreo de MPASM")
    p.add_argument("--mag", type=float, default=6.0,
                   help="ampliación de la malla de salida (referencia)")
    p.add_argument("--mag-frente", type=float, default=None,
                   help="fuerza la ampliación de la figura del frente; por "
                        "defecto se ajusta al recorte para que las franjas "
                        "queden bien muestreadas")
    p.add_argument("--nz", type=int, default=160,
                   help="distancias del barrido axial")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"))
    p.add_argument("--salida", type=pathlib.Path,
                   default=RAIZ / "resultados" / "escenario",
                   help="carpeta de salida (tres PNG)")
    args = p.parse_args(argv)
    args.salida.mkdir(parents=True, exist_ok=True)
    estilo()

    U0, delta, x = campo_entrada(args.n)
    radio = radio_geometrico(args.z)

    # cada figura pide un muestreo distinto porque responde a otra pregunta.
    # El frente se recorta al radio 1/e, asi que su ventana debe ajustarse a
    # ese recorte y no desperdiciar resolucion fuera: con mag=6 la ventana son
    # 30 mm de los que se muestran 10, y las franjas del borde quedan a 3
    # muestras por periodo, ilegibles.
    mag_frente = args.mag if args.mag_frente is None else args.mag_frente
    if args.mag_frente is None:
        mag_frente = 2.2 * radio / (args.n * delta)
    U, Kf = mpasm(U0, delta, LAMB, args.z, s=args.s, mag=mag_frente,
                  device=args.device)
    U = a_numpy(U)
    delta_out = delta * mag_frente
    x_out = (np.arange(U.shape[0]) - U.shape[0] / 2) * delta_out

    print(f"delta entrada {delta:.5f} mm (Tabla 1: 0.0098) | "
          f"delta salida {delta_out:.5f} mm | Kf = {Kf:.3f}")
    print(f"ventana entrada {L0:.2f} mm | ventana salida "
          f"{U.shape[0] * delta_out:.2f} mm | radio 1/e esperado {radio:.2f} mm")
    print(f"dispositivo: {'GPU' if gpu_disponible() and args.device != 'cpu' else 'CPU'}")

    print("\nFiguras:")
    figura_apertura(U0, x, args.salida / "apertura.png")
    figura_frente(U, x_out, args.z, radio, Kf, args.s, mag_frente,
                  args.salida / "frente_propagado.png")

    # el barrido usa su propio mag: la ventana debe cubrir el haz a TODO z, y
    # con holgura, o el radio 1/e final quedaría pegado al borde del mapa
    mag_ax = 2.6 * radio / (args.n * delta)
    print(f"\nBarrido axial: {args.nz} distancias, malla de salida x{mag_ax:.2f}")
    Z, x_ax, mapa = barrido_axial(U0, delta, args.z, args.nz, args.s, mag_ax,
                                  args.device)
    figura_corte_axial(Z, x_ax, mapa, args.salida / "corte_axial.png")

    # --- comprobación: el barrido debe seguir a la óptica geométrica ---------
    medido = np.array([radio_medido(x_ax, mapa[:, i]) for i in range(len(Z))])
    esperado = radio_geometrico(Z)
    valido = np.isfinite(medido) & (Z > 0)
    desv = np.abs(medido[valido] - esperado[valido]) / esperado[valido]
    print(f"\nRadio 1/e medido contra w0(R+z)/R sobre {valido.sum()} distancias:")
    print(f"  desviación mediana {np.median(desv) * 100:.2f} %  ·  "
          f"máxima {desv.max() * 100:.2f} %")
    print(f"  z = {Z[-1]:.0f} mm:  medido {medido[-1]:.3f} mm, "
          f"geométrico {esperado[-1]:.3f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
