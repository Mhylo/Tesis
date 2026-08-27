"""Ida y vuelta con un objeto de AMPLITUD Y FASE. Edita ALFA y dale a Run.

    U0 = A * exp(i * ALFA * A)

A es la imagen (transmitancia en [0,1]) y ALFA gradua cuanta fase lleva el
objeto. Absorbe donde A es pequena, y retarda proporcionalmente a lo que
transmite. Los dos extremos ya tenian nombre:

    ALFA = 0      ->  U0 = A          objeto de amplitud puro
    A = 1 fijo    ->  U0 = exp(i*d)   objeto de fase puro

Es lo mismo que CamposT.campos.load_field(mode="mixto", phase_depth=ALFA), que
esta probado en tests/test_objeto_mixto.py. Aqui va escrito en linea, en una
sola linea de main(), porque ALFA es justo el parametro que vas a querer tocar
y conviene tenerlo delante.

ALFA NECESITA UN OBJETO EN ESCALA DE GRISES
-------------------------------------------
Sobre un objeto BINARIO -y usaf_like solo toma 0.0 y 1.0- esta formula es
degenerada y ALFA no hace NADA:

    donde A = 0  ->  0 * exp(0)      = 0
    donde A = 1  ->  1 * exp(i*ALFA) = exp(i*ALFA)

o sea el campo entero multiplicado por una constante de modulo 1. Eso es una
fase GLOBAL: un cambio de origen de fases, que se cancela en |U|^2 y conmuta
con la propagacion. El script lo comprueba al arrancar y te avisa.

resultados/campos/entrada_gris.png es el USAF suavizado, 219 niveles, y sirve.

QUE HACE LA VUELTA, Y QUE NO
----------------------------
Retropropaga el CAMPO COMPLEJO que sale de la ida, no sqrt(|h|^2). Eso conserva
la fase, asi que la reconstruccion sale limpia y sin imagen gemela.

El precio, dicho para que no te sorprenda: esta prueba NO PUEDE FALLAR. La ida
multiplica por exp(+i*phi) y la vuelta por exp(-i*phi), asi que el resultado es
la identidad matematica y la correlacion sale 1.000 para cualquier ALFA, Z o
PAD. Sirve para ver que hace ALFA y para comprobar que el propagador no tiene
errores de signo. No dice nada sobre lo que veria un sensor: un sensor mide
|h|^2 y ahi la fase se pierde, que es otro experimento.

UNIDADES: milimetros para todo (LAMB, DELTA, Z).

    633 nm -> 633e-6 mm     3.45 um -> 3.45e-3 mm     10 mm -> 10.0
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ════════════════════════════════════════════════════════════════════════════
#  PARAMETROS  --  es lo unico que hay que editar
# ════════════════════════════════════════════════════════════════════════════

#: El OBJETO (transmitancia), no un holograma. En ESCALA DE GRISES, o ALFA no
#: hara nada (ver la cabecera).
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\campos\entrada.png"

#: PROFUNDIDAD DE FASE del objeto, en radianes:  U0 = A * exp(i * ALFA * A)
#:
#:    0        objeto de amplitud puro, fase 0 (lo que tenias antes)
#:    pi/2     retardo moderado
#:    pi       medio ciclo entre lo opaco y lo transparente
#:    > pi     OJO: la fase envuelve. np.angle() devuelve [-pi, pi], asi que a
#:             partir de aqui el panel de fase enseña saltos de 2*pi que son
#:             del display, no del objeto. Es el phase wrapping de las muestras
#:             gruesas, y es un problema real, no un artefacto del script.
ALFA = np.pi / 2

#: Longitud de onda [mm].
LAMB = 633e-6

#: Paso de pixel [mm].
DELTA = 3.45e-3

#: Distancia objeto <-> sensor [mm], POSITIVA. La ida va a +Z y la vuelta a -Z.
Z = 150.0

#: Relleno de ceros. La FFT convoluciona de forma circular: lo que sale por un
#: borde reentra por el opuesto. SIN esto el holograma tiene un 20% de error a
#: Z = 10 mm y un 186% a Z = 150 mm, medido. Con 2 el doblez queda fuera del
#: recorte. La ida y vuelta cierra igual con PAD = 1 -el doblez es reversible-,
#: asi que el error NO se ve en la reconstruccion: solo en el holograma.
PAD = 2

#: Carpeta donde guardar la figura, o None para solo mostrarla.
SALIDA = None


# ════════════════════════════════════════════════════════════════════════════
#  PROPAGADOR
# ════════════════════════════════════════════════════════════════════════════

def angularSpectrum(field, z, wavelength, dx, dy, scale_factor=1):
    """
    Propagación angular del frente de onda usando el espectro angular
    field: campo complejo
    z: distancia de propagación
    wavelength: longitud de onda
    dx, dy: pasos espaciales
    """
    # NO EDITAR: copiada tal cual de la implementacion de referencia.
    field = np.array(field)
    M, N = field.shape
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)

    kernel = np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2)) + 0j
    phase = np.exp(1j * z * scale_factor * 2 * np.pi * np.sqrt(kernel))

    tmp = field_spec * phase
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)

    return out


def con_relleno(U, pad):
    """Mete U en el centro de una malla `pad` veces mayor. -> (rel, i, j).

    El relleno se pone UNA VEZ y las dos propagaciones viven dentro de la malla
    grande; el recorte va solo al final, para pintar. Es importante y cuesta
    entenderlo, asi que aqui van los numeros medidos con este objeto a Z = 10:

        sin relleno                          err = 9.58e-16
        relleno + recorte ENTRE ida y vuelta err = 5.24e-02   <- roto
        relleno una vez, recorte al final    err = 8.78e-16

    Recortar en medio pone a cero el anillo de relleno. Ese anillo casi no
    lleva energia -menos del 0.05% del holograma-, pero anularlo es una
    DISCONTINUIDAD: retropropagar un campo truncado difracta desde ese borde y
    reparte el error por todo el plano. No es energia perdida, es un borde
    duro inventado.

    Retropropagar UNA sola vez -lo que hace retro_intensidad.py- si puede
    rellenar y recortar en la misma llamada: alli no hay una segunda
    propagacion despues que vea el borde.
    """
    M, N = U.shape
    if pad <= 1:
        return np.asarray(U, dtype=complex), 0, 0
    rel = np.zeros((M * pad, N * pad), dtype=complex)
    i, j = (rel.shape[0] - M) // 2, (rel.shape[1] - N) // 2
    rel[i:i + M, j:j + N] = U
    return rel, i, j


# ════════════════════════════════════════════════════════════════════════════
#  FIGURAS
# ════════════════════════════════════════════════════════════════════════════

def pinta_intensidad(ax, U, titulo):
    """|U|^2, normalizado por su maximo. Modulo cuadrado y nada mas."""
    I = np.abs(U) ** 2
    m = I.max()
    ax.imshow(I / m if m > 0 else I, cmap="gray", vmin=0, vmax=1)
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")


def pinta_fase(ax, U, titulo):
    """angle(U) con la OPACIDAD pesada por la amplitud.

    Sin el pesado esto es ilegible. Donde el campo vale ~1e-16, np.angle() de
    un negativo de 1e-17 devuelve pi, asi que el fondo sale como sal y pimienta
    a pi y tapa la estructura real. No es ruido fisico: es la fase del cero
    numerico. Pesar por |U| lo apaga y deja ver solo donde hay campo.
    """
    fase = np.angle(U)
    alfa = np.abs(U).astype(float)
    p = np.percentile(alfa, 99.5)
    alfa = np.clip(alfa / p, 0, 1) ** 0.45 if p > 0 else np.zeros_like(alfa)
    im = ax.imshow(fase, cmap="twilight", vmin=-np.pi, vmax=np.pi, alpha=alfa)
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")
    return im


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    if not pathlib.Path(RUTA).is_file():
        raise SystemExit(f"No encuentro la imagen en:\n    {RUTA}\n\n"
                         "Edita la constante RUTA al principio del archivo.")
    if Z <= 0:
        raise SystemExit(f"Z = {Z} pero es la distancia objeto-sensor y va "
                         f"POSITIVA: los signos los ponen la ida (+Z) y la "
                         f"vuelta (-Z).\nPon Z = {abs(Z)}.")

    A = np.asarray(Image.open(RUTA).convert("L"), dtype=np.float64) / 255.0
    M, N = A.shape

    # ---- EL OBJETO. Esta es la linea de ALFA. -------------------------------
    U0 =  A * np.exp(1j * ALFA * A)

    niveles = len(np.unique(A))
    print(f"objeto {RUTA}")
    print(f"  malla {M}x{N}, {niveles} niveles de gris")
    print(f"  ALFA = {ALFA:.4f} rad ({np.degrees(ALFA):.1f} deg)")
    print(f"  lambda {LAMB * 1e6:.1f} nm | delta {DELTA * 1e3:.3f} um | "
          f"ida +{Z:g} mm, vuelta {-Z:+g} mm | relleno x{PAD}")

    if niveles <= 2 and ALFA != 0:
        print(f"\n  AVISO: el objeto es BINARIO ({niveles} niveles), asi que "
              f"ALFA no va a hacer nada.\n  Con A en {{0, 1}} queda 0 donde "
              f"A = 0 y exp(i*ALFA) donde A = 1: una fase GLOBAL,\n  que se "
              f"cancela en |U|^2 y conmuta con la propagacion. Vas a ver la "
              f"misma\n  figura para cualquier ALFA. Usa una imagen en escala "
              f"de grises\n  (resultados/campos/entrada_gris.png sirve).")

    fase_obj = np.abs(np.angle(U0)).max()
    print(f"\n  fase del objeto: hasta {fase_obj:.3f} rad"
          + ("" if ALFA <= np.pi else
             f"  <- ALFA > pi: la fase ENVUELVE, el panel enseña saltos de 2pi"))

    # ---- ida y vuelta, con el CAMPO COMPLEJO --------------------------------
    # El relleno se pone UNA vez y las dos propagaciones ocurren dentro de la
    # malla grande. Recortar entre medias romperia la identidad: ver con_relleno().
    rel, i, j = con_relleno(U0, PAD)
    h_rel = angularSpectrum(rel,   +Z, LAMB, DELTA, DELTA)
    r_rel = angularSpectrum(h_rel, -Z, LAMB, DELTA, DELTA)

    corte = (slice(i, i + M), slice(j, j + N))
    U_h, U_r = h_rel[corte], r_rel[corte]

    err = np.abs(U_r - U0).max() / np.abs(U0).max()
    I_h = np.abs(U_h) ** 2
    print(f"  contraste del holograma |U|^2: {I_h.std() / I_h.mean():.4f}")
    print(f"  ida y vuelta del campo complejo: error max relativo = {err:.2e}"
          f"   <- es la identidad, tiene que ser ~1e-15")

    # ---- figura: 2x3, intensidad arriba y fase abajo ------------------------
    campos = [("objeto  U0 = A e^(i ALFA A)", U0),
              (f"ida: campo a +{Z:g} mm", U_h),
              (f"vuelta: campo a {-Z:+g} mm", U_r)]

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.6))
    for k, (titulo, U) in enumerate(campos):
        pinta_intensidad(ax[0, k], U, titulo)
        im = pinta_fase(ax[1, k], U, "")
    ax[0, 0].set_ylabel("intensidad  |U|^2", fontsize=10)
    ax[1, 0].set_ylabel("fase [rad]", fontsize=10)
    # el ylabel no se ve con axis("off"): lo ponemos como texto al margen
    for fila, etiqueta in ((0, "|U|$^2$"), (1, "fase")):
        ax[fila, 0].text(-0.04, 0.5, etiqueta, transform=ax[fila, 0].transAxes,
                         rotation=90, va="center", ha="right", fontsize=11)

    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015,
                      ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.ax.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
    cb.set_label("fase [rad] | opacidad = amplitud", fontsize=9)
    fig.suptitle(f"Objeto de amplitud y fase, ida y vuelta del campo complejo "
                 f"-- ALFA = {ALFA:.3f} rad, Z = {Z:g} mm", fontsize=12)

    if SALIDA is not None:
        destino = pathlib.Path(SALIDA)
        destino.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino / f"mi_prueba_alfa{ALFA:.3f}.png", dpi=150,
                    bbox_inches="tight")
        print(f"  -> {destino / f'mi_prueba_alfa{ALFA:.3f}.png'}")
    plt.show()


if __name__ == "__main__":
    main()
