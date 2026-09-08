"""Cómo evoluciona el coeficiente de compresión frecuencial K_f, Ec. (14).

Cuatro paneles en resultados/kf/evolucion_kf.png, uno por cada cosa que hay
que saber de K_f antes de fiarse de un MPASM. Los tres últimos son problemas
REALES que este repo encontró y corrigió; el panel los deja verse en vez de
tener que creerlos.

    1. K_f vs z          la evolución, y su forma cerrada
    2. K_f vs z firmado  el fallo del valor absoluto (tarea 63)
    3. paper vs código   la errata `*+` de la Ec. (14)
    4. K_f vs N          por qué K_f va POR EJE (tarea 61)
    5. razón vs s·N·λ    de qué depende el tamaño de esa errata
    6. dónde importa     y por qué en el DLHM real no importa

LOS PANELES 5 Y 6 EXISTEN POR UNA CONTRADICCIÓN APARENTE. Las diapositivas del
informe de avance 1 titulan «K_f sobreestimado hasta 4.8x», y el panel 3 de
esta misma figura da 0.77, o sea SUBestimado. Las dos cifras son correctas: la
razón entre las dos fórmulas es 1/(s·N·λ), que no es adimensional, así que su
valor -y su SIGNO respecto a 1- depende del montaje.

    diapo (N=512, λ=405 nm, s=1)        s·N·λ = 0.207 mm   razón 4.83
    Tabla 1 (N=512, λ=632.8 nm, s=4)    s·N·λ = 1.296 mm   razón 0.77
    holograma DLHM (N=3000, λ=532 nm)   s·N·λ = 1.596 mm   razón 0.63

Que el factor tenga unidades de 1/longitud ES la prueba de que la versión con
producto no puede ser correcta, y es un argumento más fuerte que cualquier
número concreto: una errata dimensionalmente consistente daría un factor fijo.

Y el panel 6 dice dónde muerde de verdad. Con la geometría DLHM del modelo de
referencia -L = 8 mm, z = 2 mm, o sea propagar 6.000 mm, sobre 3000 px de
1.85 µm- el umbral está en 19.3 mm: K_f vale 1 en las DOS fórmulas y la errata
no cambia absolutamente nada. Los 150 mm que la diapositiva etiqueta como «z
del montaje DLHM» son la distancia de las pruebas de onda plana, no una
distancia DLHM.

LA FORMA CERRADA, que es lo que hace legible todo lo demás. Por encima del
umbral, la Ec. (14) se reduce a

    K_f  ~=  (1/delta) * sqrt(lambda * z / (s*N))                        (A)

y el umbral en que K_f despega de 1 es exactamente

    z_umbral  =  delta^2 * s * N / lambda                                (B)

Las dos están comprobadas contra kf_auto() en este script: (A) coincide a 1e-5
ya en z = 1000 mm, y el umbral medido cae sobre (B). De (A) salen de un vistazo
las tres dependencias que el repo tenía anotadas por separado:

    K_f ~ sqrt(z)     comprimir es cosa de propagar lejos, no de la imagen
    K_f ~ 1/sqrt(N)   la que obliga a calcularlo por eje (tarea 61)
    K_f ~ 1/sqrt(s)   subir el sobremuestreo ALEJA la compresión, aunque
                      parezca lo contrario. Es lo que dicen los docstrings de
                      mpasm y de los mi_prueba_*, y (A) es su razón.

Y (B) dice algo que no estaba escrito en ninguna parte: z_umbral = s * z_lim,
donde z_lim = delta^2*N/lambda es el límite de reversibilidad de BL-ASM que
fija tests/test_propagadores.py. O sea que MPASM empieza a comprimir justo
donde BL-ASM deja de ser invertible, escalado por s. No son dos umbrales, es
el mismo: el punto en que el ancho de banda de la función de transferencia
deja de caber en el intervalo espectral que la malla puede muestrear.

POR QUÉ ESTE SCRIPT NO PROPAGA NADA. K_f es aritmética escalar: no hace falta
un campo para dibujarlo, y meterlo mezclaría el comportamiento de la fórmula
con el del propagador. Lo que se dibuja es la ESTIMACIÓN; lo que esa
estimación produce al propagar ya lo fijan tests/test_propagadores.py
(mpasm_sigue_al_gaussiano_en_su_regimen) y el barrido de scripts/exactitud.py.

Uso:
    Tesis_env/Scripts/python.exe -m scripts.kf_evolucion
    Tesis_env/Scripts/python.exe -m scripts.kf_evolucion --n 1024
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")            # sin ventana: el script guarda, no muestra
import matplotlib.pyplot as plt
import numpy as np

from CamposT.montaje import TABLA1
from CamposT.propagadores import kf_auto

#: raíz del repo, para que la salida no dependa del directorio de invocación
RAIZ = pathlib.Path(__file__).resolve().parent.parent

# --- escenario de la Tabla 1 -------------------------------------------------
# Los valores viven en CamposT.montaje.TABLA1, que es la única fuente.
LAMB = TABLA1.lamb          # mm, HeNe
DELTA = TABLA1.delta        # mm, paso de la malla (L0 / (N-1))
N_BASE = TABLA1.N           # puntos por lado

#: Sobremuestreos que se dibujan. s = 1 es MPASM sin sobremuestrear y s = 10
#: es el defecto de CamposT.propagadores.mpasm().
ESES = (1, 2, 4, 10)

#: Los tres escenarios que conviven en el trabajo, con los que el panel 5
#: sitúa la errata. (nombre, N, delta [mm], lambda [mm], s).
#:
#: El primero es el de las diapositivas del avance 1 -de ahí sale su «4.8x»-,
#: el segundo el de la Tabla 1 del paper, y el tercero el holograma DLHM de
#: referencia, que es el único de los tres con datos de verdad detrás.
ESCENARIOS = (
    ("diapos avance 1", 512, 3.45e-3, 405e-6, 1),
    ("Tabla 1 (paper)", 512, 5.0 / 511, 632.8e-6, 4),
    ("holograma DLHM", 3000, 1.85e-3, 532e-6, 1),
)

#: La geometría DLHM del modelo de referencia (referencia/carlos/DLHM-model).
#: z es fuente->muestra y L fuente->sensor, así que lo que se propaga es L - z.
DLHM_N, DLHM_DELTA, DLHM_LAMB = 3000, 1.85e-3, 532e-6
DLHM_L, DLHM_Z = 8.0, 2.0

TINTA, SUAVE, AZUL, TEJA = "#1a1a1a", "#8a8a8a", "#1f4e79", "#c1543a"
VERDE = "#2e6f4e"

# --- la lamina del avance 1 --------------------------------------------------
# Escenario y paleta de la figura que se presento el 31/08 (diapositiva 9 de
# CamiloR_Avance_1.pptx). Los colores estan MUESTREADOS del PNG original, no
# aproximados: asi la version corregida se puede intercambiar por la vieja en la
# presentacion sin que se note el cambio de estilo, que es de lo que se trata.
LAMINA_N, LAMINA_DELTA, LAMINA_LAMB, LAMINA_S = 512, 3.45e-3, 405e-6, 1
LAMINA_Z = 150.0            # la vertical de la figura original
NARANJA, TEAL = "#D97706", "#0E9B93"
REJILLA, EJE, GRIS = "#EDEEF1", "#C5C7CB", "#8C9098"


# ═══════════════════════════════════════════════════════════════════════════
#  LAS VARIANTES DE Kf
# ═══════════════════════════════════════════════════════════════════════════
#
# La buena es kf_auto(), que vive en el paquete. Aquí sólo se escriben las
# ROTAS, y cada una aísla UN fallo: kf_original() de scripts/retro_mpasm.py
# lleva los dos a la vez -es la transcripción literal del código publicado- y
# con los dos mezclados no se ve cuál cuesta qué.

def kf_sin_valor_absoluto(N, delta, lamb, z, s=1):
    """Ec. (14) en su forma correcta, pero aplicada a z tal cual.

    Es el fallo de la tarea 63, aislado. La Ec. (14) está escrita para z > 0;
    a z < 0 el radicando sale negativo, fmax es NaN o negativo, y el
    max(1.0, ...) devuelve 1 SIN AVISAR. Como CamposT.retropropagacion propaga
    siempre a -z, eso dejaba a MPASM corriendo como un FFT-ASM rellenado de
    ceros en TODA reconstrucción: RMS 2.3e-1 contra 1.7e-3 con el Kf correcto.

    El fallo es mudo por partida doble: no lanza, y devuelve un valor legal.
    """
    if z == 0:
        return 1.0
    Ns = s * N
    A = (Ns * lamb) ** 2
    B = (8 * Ns * lamb * z) ** 2
    with np.errstate(invalid="ignore"):
        fmax = np.sqrt(np.sqrt(A**2 + B) - A) / (4 * np.sqrt(2) * z * lamb)
        valor = (1 / (2 * delta)) / fmax
    return 1.0 if not (valor > 1) else float(valor)   # NaN -> 1, como el original


def razon_codigo_paper(N, lamb, s=1):
    """Cuánto encoge K_f la errata `*+`, en el régimen asintótico. -> factor.

    El código original escribe A**2 *+ (8*N*lamb*z)**2, que en Python es un
    PRODUCTO, donde la Ec. (14) impresa dice una SUMA. Lejos del umbral:

        paper:   sqrt(A^2 + B) -> sqrt(B)
        codigo:  sqrt(A^2 * B) -> A * sqrt(B)

    o sea que el radicando del código lleva un factor A de más, y K_f -que va
    como 1/sqrt(radicando)- queda dividido por sqrt(A) = s*N*lambda.

    LO QUE HACE PELIGROSA A ESTA ERRATA es que sqrt(A) NO es adimensional:
    tiene unidades de longitud, y por eso la versión con producto no puede ser
    correcta. Pero con los números de la Tabla 1 vale 1.296, así que el efecto
    es un K_f un 23 % más bajo: no revienta, no da NaN, no sale del rango
    plausible. Sólo comprime de menos, siempre, en la misma proporción.
    """
    return 1.0 / (s * N * lamb)


def umbral(N, delta, lamb, s=1):
    """z a partir de la cual K_f > 1, Ec. (B) de la cabecera. -> mm."""
    return delta**2 * s * N / lamb


def asintota(N, delta, lamb, z, s=1):
    """K_f por la forma cerrada, Ec. (A) de la cabecera."""
    return (1 / delta) * np.sqrt(lamb * np.abs(z) / (s * N))


# ═══════════════════════════════════════════════════════════════════════════
#  COMPROBACIONES  --  los números que la figura afirma, medidos
# ═══════════════════════════════════════════════════════════════════════════

def comprobar(N, delta, lamb):
    """Contrasta las formas cerradas contra kf_auto(). -> dict de resultados.

    Se corre siempre, antes de dibujar: una figura que ilustra una fórmula sin
    haberla contrastado no vale más que la fórmula.
    """
    peor_asintota = 0.0
    for s in ESES:
        for z in (1e3, 1e4, 1e5, 2e5):
            exacto = kf_auto(N, delta, lamb, z, s=s)
            cerrado = asintota(N, delta, lamb, z, s=s)
            peor_asintota = max(peor_asintota, abs(cerrado / exacto - 1))

    # el umbral: la z más pequeña de una rejilla fina en la que Kf despega
    peor_umbral = 0.0
    for s in ESES:
        zs = np.logspace(-1, 4, 20000)
        kfs = np.array([kf_auto(N, delta, lamb, z, s=s) for z in zs])
        medido = zs[np.argmax(kfs > 1.0)]
        peor_umbral = max(peor_umbral, abs(medido / umbral(N, delta, lamb, s) - 1))

    # la errata: razón medida contra 1/(s*N*lambda)
    peor_razon = 0.0
    for s in ESES:
        for z in (1e4, 1e5, 2e5):
            medida = (kf_auto(N, delta, lamb, z, s=s, formula="codigo")
                      / kf_auto(N, delta, lamb, z, s=s))
            peor_razon = max(peor_razon,
                             abs(medida / razon_codigo_paper(N, lamb, s) - 1))

    return {"asintota": peor_asintota, "umbral": peor_umbral,
            "razon": peor_razon}


# ═══════════════════════════════════════════════════════════════════════════
#  LOS CUATRO PANELES
# ═══════════════════════════════════════════════════════════════════════════

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


def panel_evolucion(ax, N, delta, lamb):
    """K_f vs z, un trazo por s, con la forma cerrada encima."""
    zs = np.logspace(0, 5.5, 400)
    colores = plt.cm.viridis(np.linspace(0.15, 0.8, len(ESES)))
    for s, color in zip(ESES, colores):
        kfs = [kf_auto(N, delta, lamb, z, s=s) for z in zs]
        ax.loglog(zs, kfs, color=color, lw=1.6, label=f"s = {s}")
        ax.loglog(zs, asintota(N, delta, lamb, zs, s=s), color=color, lw=0.9,
                  ls=(0, (2, 2)), alpha=0.9)
        ax.axvline(umbral(N, delta, lamb, s), color=color, lw=0.7,
                   ls=(0, (1, 3)), alpha=0.7)

    ax.axhline(1.0, color=TINTA, lw=0.8)
    ax.text(1.4, 1.06, "K_f = 1: sin comprimir", fontsize=7.5, color=TINTA)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("K_f")
    ax.set_title("1. La evolución: K_f crece como sqrt(z)", fontsize=10,
                 color=TINTA, pad=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.text(0.98, 0.04,
            "puntos: (1/δ)·√(λz/sN)\nvertical: z = δ²sN/λ",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=SUAVE)
    _limpiar(ax)


def panel_signo(ax, N, delta, lamb, s=4):
    """K_f a z negativa: el fallo del valor absoluto, aislado."""
    zs = np.linspace(-2e5, 2e5, 801)
    bien = [kf_auto(N, delta, lamb, z, s=s) for z in zs]
    mal = [kf_sin_valor_absoluto(N, delta, lamb, z, s=s) for z in zs]

    ax.plot(zs / 1e3, bien, color=AZUL, lw=1.8, label="con |z| (corregido)")
    ax.plot(zs / 1e3, mal, color=TEJA, lw=1.8, ls=(0, (4, 2)),
            label="sin |z| (original)")
    ax.axvspan(-200, 0, color=TEJA, alpha=0.06)
    ax.text(-100, ax.get_ylim()[1] * 0.55,
            "retropropagar\n(z < 0)", ha="center", fontsize=8, color=TEJA)
    ax.annotate("compresión APAGADA\nen toda reconstrucción",
                xy=(-120, 1.0), xytext=(-190, 9),
                fontsize=8, color=TEJA,
                arrowprops=dict(arrowstyle="->", color=TEJA, lw=0.9))
    ax.set_xlabel("z [10³ mm]")
    ax.set_ylabel("K_f")
    ax.set_title(f"2. El valor absoluto (tarea 63)  ·  s = {s}", fontsize=10,
                 color=TINTA, pad=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center")
    _limpiar(ax)


def panel_errata(ax, N, delta, lamb, s=4):
    """La Ec. (14) impresa (A²+B) contra la que corre el código (A²·B)."""
    zs = np.logspace(1.5, 5.5, 300)
    paper = np.array([kf_auto(N, delta, lamb, z, s=s) for z in zs])
    codigo = np.array([kf_auto(N, delta, lamb, z, s=s, formula="codigo")
                       for z in zs])

    ax.loglog(zs, paper, color=AZUL, lw=1.8, label="paper:  A² + B")
    ax.loglog(zs, codigo, color=TEJA, lw=1.8, ls=(0, (4, 2)),
              label="código: A² · B")
    ax.axhline(1.0, color=TINTA, lw=0.8)

    r = razon_codigo_paper(N, lamb, s)
    ax.text(0.97, 0.06,
            f"razón asintótica = 1/(s·N·λ) = {r:.4f}\n"
            f"o sea un {100 * (1 - r):.0f} % menos de compresión EN ESTE\n"
            f"escenario. En otro sale > 1: ver el panel 5",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=TEJA)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("K_f")
    ax.set_title("3. La errata `*+` de la Ec. (14)", fontsize=10,
                 color=TINTA, pad=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _limpiar(ax)


def panel_por_eje(ax, delta, lamb, z=12000.0, s=4):
    """K_f vs N: la dependencia 1/sqrt(N) que obliga a calcularlo por eje."""
    ns = np.unique(np.round(np.logspace(np.log10(32), np.log10(4096), 60))
                   ).astype(int)
    kfs = [kf_auto(int(n), delta, lamb, z, s=s) for n in ns]
    ax.loglog(ns, kfs, color=AZUL, lw=1.8)

    # la malla rectangular del hallazgo de la tarea 61
    m_corto, m_largo = 64, 128
    kf_corto = kf_auto(m_corto, delta, lamb, z, s=s)
    kf_largo = kf_auto(m_largo, delta, lamb, z, s=s)
    ax.plot([m_corto, m_largo], [kf_corto, kf_largo], "o", color=TEJA, ms=6,
            zorder=5)
    ax.annotate("", xy=(m_largo, kf_largo), xytext=(m_corto, kf_corto),
                arrowprops=dict(arrowstyle="<->", color=TEJA, lw=1.1))
    ax.text(m_largo * 1.25, (kf_corto * kf_largo) ** 0.5,
            f"malla {m_corto}×{m_largo}:\n"
            f"K_fy = {kf_corto:.2f}  (eje de {m_corto})\n"
            f"K_fx = {kf_largo:.2f}  (eje de {m_largo})\n"
            f"razón {kf_corto / kf_largo:.3f} = √{m_largo // m_corto}\n"
            f"con UN solo valor, el eje\ncorto queda submuestreado",
            fontsize=7.5, color=TEJA, va="center")

    ax.set_xlabel("N (puntos por eje)")
    ax.set_ylabel("K_f")
    ax.set_title(f"4. K_f va como 1/√N, y por eso POR EJE (tarea 61)\n"
                 f"z = {z:.0f} mm, s = {s}", fontsize=10, color=TINTA, pad=8)
    _limpiar(ax)


def panel_razon(ax):
    """La razón código/paper frente a s·N·λ: de qué depende el «4.8x».

    Es el panel que reconcilia la diapositiva 9 del avance 1 con el panel 3 de
    esta figura. La razón es 1/(s·N·λ), una hipérbola que cruza 1 en
    s·N·λ = 1 mm: a la izquierda la errata SOBREestima y a la derecha
    SUBestima. Ninguna de las dos cifras que circulan es un error; son la misma
    ley en dos montajes.
    """
    x = np.logspace(np.log10(0.06), np.log10(6), 400)
    ax.loglog(x, 1 / x, color=AZUL, lw=1.8)
    ax.axhline(1.0, color=TINTA, lw=0.9)
    ax.axvline(1.0, color=TINTA, lw=0.9, ls=(0, (4, 3)))

    ax.fill_between(x, 1 / x, 1.0, where=(x < 1), color=TEJA, alpha=0.07)
    ax.fill_between(x, 1 / x, 1.0, where=(x > 1), color=VERDE, alpha=0.07)
    ax.text(0.066, 2.3, "el código\nSOBREestima K_f", fontsize=8, color=TEJA)
    ax.text(5.4, 0.52, "el código\nSUBestima K_f", fontsize=8, color=VERDE,
            ha="right")

    # Posiciones explicitas: dos de los tres escenarios caen casi encima del
    # cruce, y cualquier desplazamiento proporcional los amontona ahi.
    donde = {"diapos avance 1": (0.42, 9.2),
             "Tabla 1 (paper)": (0.30, 0.42),
             "holograma DLHM": (2.9, 3.1)}
    for nombre, n, _d, l, ss in ESCENARIOS:
        snl = ss * n * l
        r = 1 / snl
        ax.plot([snl], [r], "o", color=TINTA, ms=6, zorder=5)
        ax.annotate(f"{nombre}\ns·N·λ = {snl:.3f} mm\nrazón = {r:.2f}",
                    xy=(snl, r), xytext=donde[nombre],
                    fontsize=7.5, color=TINTA, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=SUAVE, lw=0.7))

    ax.set_xlabel("s · N · λ  [mm]")
    ax.set_ylabel("K_f código / K_f paper")
    ax.set_title("5. De qué depende el tamaño de la errata:  1/(s·N·λ)\n"
                 "tiene UNIDADES de 1/longitud, y por eso cambia de signo",
                 fontsize=10, color=TINTA, pad=8)
    _limpiar(ax)


def panel_donde_importa(ax):
    """K_f en la geometría DLHM de verdad: la errata no llega a activarse.

    La diapositiva 9 dice «a la distancia de trabajo del montaje». Con la
    geometría del modelo de referencia esa distancia es L - z = 6.000 mm, y el
    umbral de compresión está en 19.3 mm: las dos fórmulas devuelven 1 y la
    errata no cambia nada. Donde sí muerde es dos órdenes más lejos.
    """
    zs = np.logspace(np.log10(0.5), np.log10(500), 400)
    paper = [kf_auto(DLHM_N, DLHM_DELTA, DLHM_LAMB, z) for z in zs]
    codigo = [kf_auto(DLHM_N, DLHM_DELTA, DLHM_LAMB, z, formula="codigo")
              for z in zs]
    ax.loglog(zs, paper, color=AZUL, lw=1.8, label="paper:  A² + B")
    ax.loglog(zs, codigo, color=TEJA, lw=1.8, ls=(0, (4, 2)),
              label="código: A² · B")

    zu = umbral(DLHM_N, DLHM_DELTA, DLHM_LAMB)
    ax.axvspan(zs[0], zu, color=VERDE, alpha=0.08)
    ax.axvline(zu, color=VERDE, lw=1.0, ls=(0, (3, 2)))
    ax.text(zu * 0.88, 3.4, f"umbral\n{zu:.1f} mm", fontsize=7.5,
            color=VERDE, ha="right")

    propagada = DLHM_L - DLHM_Z
    ax.plot([propagada], [1.0], "o", color=TINTA, ms=7, zorder=5)
    ax.annotate(f"DLHM real: se propaga L−z = {propagada:.3f} mm\n"
                f"K_f = 1 en las DOS fórmulas\n"
                f"la errata no cambia nada aquí",
                xy=(propagada, 1.0), xytext=(1.0, 2.3),
                fontsize=8, color=TINTA,
                arrowprops=dict(arrowstyle="->", color=TINTA, lw=0.9))

    ax.plot([150], [kf_auto(DLHM_N, DLHM_DELTA, DLHM_LAMB, 150)], "s",
            color=TEJA, ms=6, zorder=5)
    ax.text(430, kf_auto(DLHM_N, DLHM_DELTA, DLHM_LAMB, 150) * 1.10,
            "los 150 mm de la diapo:\nonda plana, no DLHM",
            fontsize=7.5, color=TEJA, ha="right")

    ax.set_xlabel("distancia propagada [mm]")
    ax.set_ylabel("K_f")
    ax.set_title(f"6. Dónde importa de verdad\n"
                 f"N = {DLHM_N}, δ = {DLHM_DELTA * 1e3:.2f} µm, "
                 f"λ = {DLHM_LAMB * 1e6:.0f} nm, s = 1",
                 fontsize=10, color=TINTA, pad=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _limpiar(ax)


# ═══════════════════════════════════════════════════════════════════════════

def figura_lamina(destino):
    """La figura de la diapositiva 9 del avance 1, con los rótulos corregidos.

    Las CURVAS no se tocan: estaban bien, y se comprobó punto por punto contra
    kf_auto() -1.97 en z->0, el codo en el umbral de 15.05 mm, 15.26 contra
    3.16 en z = 150, 21.55 contra 4.47 en z = 300-. Lo que se corrige son dos
    rótulos:

    EL TÍTULO decía «K_f sobreestimado hasta 4.8x», sin decir con qué. Se lee
    como una propiedad de la errata y es una propiedad de ESTE escenario: la
    razón entre las dos fórmulas es 1/(s·N·λ), que con N = 3000 y λ = 532 nm
    -el holograma DLHM de referencia- vale 0.63, o sea que subestima. El título
    nuevo lleva las condiciones y la ley.

    LA VERTICAL decía «z del montaje DLHM», y 150 mm no es una distancia DLHM:
    es la de las pruebas de onda plana de las diapositivas 3-6. El montaje
    propaga L - z = 6 mm, donde el umbral de compresión (19.3 mm con sus
    parámetros) ni siquiera se ha alcanzado y K_f vale 1 en las dos fórmulas.
    """
    N, d, l, ss = LAMINA_N, LAMINA_DELTA, LAMINA_LAMB, LAMINA_S
    zs = np.linspace(0.0, 300.0, 600)
    paper = [kf_auto(N, d, l, z, s=ss) for z in zs]
    codigo = [kf_auto(N, d, l, z, s=ss, formula="codigo") for z in zs]
    razon = kf_auto(N, d, l, LAMINA_Z, s=ss, formula="codigo") / \
        kf_auto(N, d, l, LAMINA_Z, s=ss)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    ax.grid(True, color=REJILLA, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(zs, codigo, color=NARANJA, lw=2.2, zorder=3,
            label="codigo publicado  (×)")
    ax.plot(zs, paper, color=TEAL, lw=2.2, zorder=3,
            label="Ec. (14) del paper  (+)")

    ax.axvline(LAMINA_Z, color=GRIS, lw=1.1, ls=(0, (1, 2)), zorder=2)
    # Corregido: 150 mm es la z de las pruebas de onda plana, no la del DLHM.
    # Va abajo a la derecha, que es el unico cuadrante libre: la curva verde no
    # pasa de 4.5 y la naranja esta arriba.
    ax.text(LAMINA_Z + 8, 2.9, "z de las pruebas\nde onda plana",
            fontsize=8.5, color=GRIS, va="top")

    ax.set_xlim(0, 300)
    ax.set_ylim(0, 22.8)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Distancia de propagacion  z  [mm]", fontsize=10)
    ax.set_ylabel("Coeficiente $K_f$", fontsize=10)
    ax.tick_params(labelsize=9.5, color=EJE, labelcolor="#3a3d42")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EJE)

    ax.set_title(f"$K_f$ sobreestimado {razon:.1f}× a z = {LAMINA_Z:.0f} mm",
                 fontsize=12.5, fontweight="bold", color="#20242a", pad=30)
    # Corregido: el titulo viejo no decia en que escenario, y sin eso se lee
    # como una propiedad de la errata. Dos lineas: la primera es el dato, la
    # segunda el argumento.
    ax.text(0.5, 1.088,
            f"N = {N}, δ = {d * 1e3:.2f} µm, λ = {l * 1e6:.0f} nm, s = {ss}",
            transform=ax.transAxes, ha="center", fontsize=8.6, color=GRIS)
    ax.text(0.5, 1.028,
            "la razón entre las dos fórmulas es 1/(s·N·λ):"
            " con otro montaje se invierte",
            transform=ax.transAxes, ha="center", fontsize=8.6, color=GRIS)

    ax.legend(fontsize=10, frameon=False, loc="upper left",
              handlelength=1.7, borderaxespad=0.7)

    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return destino


def figura_diapo(destino):
    """Los paneles 5 y 6 solos, apaisados: es lo que va a la lámina.

    La figura de seis paneles es material de trabajo. Para una diapositiva
    sirven estos dos, que son los que corrigen lo que decía el avance 1.
    """
    estilo()
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.9))
    panel_razon(ax[0])
    panel_donde_importa(ax[1])
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return destino


def figura(N, delta, lamb, destino):
    estilo()
    fig, ax = plt.subplots(2, 3, figsize=(16.8, 8.8))
    panel_evolucion(ax[0, 0], N, delta, lamb)
    panel_signo(ax[0, 1], N, delta, lamb)
    panel_por_eje(ax[0, 2], delta, lamb)
    panel_errata(ax[1, 0], N, delta, lamb)
    panel_razon(ax[1, 1])
    panel_donde_importa(ax[1, 2])
    fig.suptitle("Estimación de K_f, Ec. (14) de Zhao et al. (2020)",
                 fontsize=12.5, color=TINTA)
    fig.text(0.5, 0.005,
             f"paneles 1-4: escenario de la Tabla 1  ·  λ = {lamb * 1e6:.1f} nm"
             f"  ·  δ = {delta * 1e3:.2f} µm  ·  N = {N}"
             f"      |      paneles 5-6: ver su propio rótulo, "
             f"que de eso van",
             ha="center", fontsize=8, color=SUAVE)
    fig.tight_layout(rect=[0, 0.015, 1, 0.97])
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return destino


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=N_BASE,
                   help=f"puntos por lado (defecto {N_BASE}, la Tabla 1)")
    p.add_argument("--salida", type=pathlib.Path,
                   default=RAIZ / "resultados" / "kf" / "evolucion_kf.png")
    p.add_argument("--diapo", action="store_true",
                   help="escribe ademas los paneles 5 y 6 solos, para la lamina")
    p.add_argument("--lamina", action="store_true",
                   help="reproduce la figura de la diapositiva 9 del avance 1, "
                        "con los rotulos corregidos")
    args = p.parse_args(argv)

    N, delta, lamb = args.n, DELTA, LAMB
    print(f"K_f del escenario de la Tabla 1: lambda {lamb * 1e6:.1f} nm, "
          f"delta {delta * 1e3:.2f} um, N {N}\n")

    c = comprobar(N, delta, lamb)
    print("COMPROBACION de las formas cerradas contra kf_auto():")
    print(f"  K_f ~= (1/d)*sqrt(l*z/(s*N))      error relativo max "
          f"{c['asintota']:.2e}")
    print(f"  z_umbral = d^2*s*N/lambda         error relativo max "
          f"{c['umbral']:.2e}")
    print(f"  razon codigo/paper = 1/(s*N*l)    error relativo max "
          f"{c['razon']:.2e}\n")

    print(f"{'s':>3} {'z_umbral [mm]':>14} {'= s * z_lim(BLAS)':>19} "
          f"{'K_f(12000)':>11} {'K_f(80000)':>11}")
    z_lim = delta**2 * N / lamb
    for s in ESES:
        print(f"{s:3d} {umbral(N, delta, lamb, s):14.1f} {s * z_lim:19.1f} "
              f"{kf_auto(N, delta, lamb, 12000, s=s):11.4f} "
              f"{kf_auto(N, delta, lamb, 80000, s=s):11.4f}")

    print(f"\nA z < 0 SIN valor absoluto, K_f vale "
          f"{kf_sin_valor_absoluto(N, delta, lamb, -80000, s=4):.1f} "
          f"donde deberia valer "
          f"{kf_auto(N, delta, lamb, -80000, s=4):.4f}: la compresion se apaga.")

    print("\nLA ERRATA EN CADA ESCENARIO (razon codigo/paper = 1/(s*N*lambda)):")
    print(f"  {'escenario':<18} {'s*N*lambda':>11} {'razon':>7}   efecto")
    for nombre, n, _d, l, ss in ESCENARIOS:
        r = razon_codigo_paper(n, l, ss)
        print(f"  {nombre:<18} {ss * n * l:11.4f} {r:7.2f}   "
              f"{'SOBREestima' if r > 1 else 'SUBestima'}")

    zu_dlhm = umbral(DLHM_N, DLHM_DELTA, DLHM_LAMB)
    print(f"\nEn la geometria DLHM real se propaga L-z = {DLHM_L - DLHM_Z:.3f} mm "
          f"y el umbral esta en {zu_dlhm:.1f} mm:")
    print(f"  paper  {kf_auto(DLHM_N, DLHM_DELTA, DLHM_LAMB, DLHM_L - DLHM_Z):.4f}"
          f"   codigo {kf_auto(DLHM_N, DLHM_DELTA, DLHM_LAMB, DLHM_L - DLHM_Z, formula='codigo'):.4f}"
          f"   -> la errata no cambia NADA ahi.")

    print("\n->", figura(N, delta, lamb, args.salida))
    if args.diapo:
        print("->", figura_diapo(args.salida.with_name("diapo_kf.png")))
    if args.lamina:
        N, d, l, ss = LAMINA_N, LAMINA_DELTA, LAMINA_LAMB, LAMINA_S
        print(f"\nLamina del avance 1 (N={N}, delta={d*1e3:.2f} um, "
              f"lambda={l*1e6:.0f} nm, s={ss}):")
        print(f"  umbral {umbral(N, d, l, ss):.2f} mm   "
              f"razon a z={LAMINA_Z:.0f}: "
              f"{kf_auto(N,d,l,LAMINA_Z,s=ss,formula='codigo')/kf_auto(N,d,l,LAMINA_Z,s=ss):.2f}")
        print("->", figura_lamina(args.salida.with_name("lamina_kf_avance1.png")))


if __name__ == "__main__":
    main()
