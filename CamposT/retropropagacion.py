"""Retropropagación de un holograma: de la intensidad medida al objeto.

Es el camino inverso de pipeline.propagar(). Se parte de una imagen que el
usuario elige en cada corrida —el holograma, ya grabado— y se recupera el
plano del objeto propagando a distancia negativa con los tres propagadores del
paquete, para poder compararlos sobre el mismo dato.

Dos cosas que conviene tener claras antes de leer una reconstrucción:

Fase. Un sensor registra |U|², así que la fase del campo en el plano del
holograma se pierde en la medida. Retropropagar sqrt(I) recupera el objeto y,
superpuesta, su imagen gemela desenfocada. Este módulo no la suprime: enseña
la reconstrucción cruda. Quitarla (Gerchberg-Saxton, resta de fondo,
phase-shifting) es un problema aparte.

Signo. retropropagar() recibe distancias POSITIVAS, la separación
sensor→objeto tal como se mide en el montaje, y propaga internamente a −z.

Y aquí va el aviso que se deriva de lo anterior: sobre la intensidad, ese
signo NO SE PUEDE COMPROBAR. El campo de entrada sqrt(I) es real, y para
entrada real U(−z) = conj(U(+z)), luego |U(−z)|² = |U(+z)|² exactamente. Una
pila de foco calculada con el signo invertido sale idéntica, imagen por
imagen. Es la misma simetría que produce la imagen gemela. Por eso la
convención se fija por contrato en tests/test_retropropagacion.py, y no
mirando reconstrucciones: no hay nada que mirar. Donde sí se nota es en la
fase del campo devuelto, y en cuanto se encadena cualquier paso no real —la
corrección de fuente puntual de DLHM, un filtro complejo, una iteración de
recuperación de fase—, que es exactamente hacia donde va esto.

Qué NO hace: la corrección de fuente puntual divergente. Se asume iluminación
colimada (in-line clásico, onda plana). En DLHM con pinhole hay magnificación
M = L/z, y reconstruir exige reescalar el paso de píxel y usar una distancia
efectiva; eso entra como un cambio de coordenadas ANTES de llamar aquí, sin
tocar nada de este módulo.

Como la distancia de enfoque no se conoce de antemano, la unidad de trabajo es
un barrido: una pila de reconstrucciones sobre un rango de z, para mirar cuál
enfoca.

    python -m CamposT.retropropagacion holograma.png --z 10 60 --pasos 25
"""

import argparse
import pathlib

import numpy as np

from CamposT.campos import load_field
from CamposT.pipeline import guardar, intensidad, propagar
from CamposT.roi import anadir_argumentos, desde_argumentos, informe

#: los tres propagadores comparables sobre el mismo holograma
METODOS = ("fft", "blas", "mpasm")

#: parámetros que sólo entiende mpasm(). FFT-ASM y BL-ASM no los aceptan, así
#: que se filtran por método en vez de exigir al llamante que sepa cuál va con
#: cuál.
KW_MPASM = ("s", "Kf", "r", "mag", "formula")


# ------------------------------------------------------------------- barrido
def barrido_z(z, pasos=25):
    """Distancias de reconstrucción, en las mismas unidades que delta y lamb.

    Un solo valor (escalar o lista de uno) devuelve ese valor. Dos valores se
    interpretan como los extremos de un barrido de `pasos` distancias
    equiespaciadas. Más de dos es un error: para una lista arbitraria de
    distancias, pásasela directamente a retropropagar().
    """
    z = np.atleast_1d(np.asarray(z, dtype=float)).ravel()
    if z.size == 1:
        return z
    if z.size != 2:
        raise ValueError(
            f"barrido_z espera uno o dos valores, no {z.size}. Para una lista "
            "arbitraria de distancias, pásala directamente a retropropagar().")
    if pasos < 1:
        raise ValueError(f"pasos debe ser >= 1, no {pasos}")
    return np.linspace(z[0], z[1], pasos)


# ------------------------------------------------------------ retropropagación
def retropropagar(U_h, delta, lamb, zs, metodos=METODOS, pad=2, device="auto",
                  dtype=None, **kw):
    """Retropropaga un campo de holograma a cada distancia, con cada método.

    U_h es el campo en el plano del sensor: sqrt de la intensidad medida, que
    es lo que devuelve campos.load_field(..., mode='holograma').

    zs son distancias POSITIVAS sensor→objeto. Internamente se propaga a −z.

    Genera (metodo, z, campo, info), con z el valor positivo que se pidió.
    Itera por método y, dentro, por distancia, de modo que la pila de foco de
    cada propagador salga en orden. No acumula: el llamante decide qué hacer
    con cada campo antes de que llegue el siguiente, que en un barrido largo
    es la diferencia entre caber en memoria y no caber.

    Los kwargs de MPASM (s, Kf, r, mag, formula) se filtran por método: pasarle
    s=2 a fft_asm sería un TypeError, y obligar al llamante a llevar la cuenta
    de qué parámetro va con qué propagador es justo lo que esta función existe
    para evitar.
    """
    desconocidos = [m for m in metodos if m not in METODOS]
    if desconocidos:
        raise ValueError(f"métodos desconocidos: {desconocidos}. "
                         f"Disponibles: {list(METODOS)}")
    ajenos = [k for k in kw if k not in KW_MPASM]
    if ajenos:
        raise ValueError(f"parámetros no reconocidos: {ajenos}. "
                         f"Sólo MPASM acepta extras: {list(KW_MPASM)}")

    zs = np.atleast_1d(np.asarray(zs, dtype=float)).ravel()
    # El signo se comprueba aquí y no sólo en la CLI. Una distancia negativa
    # propagaría a +z, o sea hacia adelante, y sobre la intensidad eso no se
    # ve: para entrada real |U(-z)|² = |U(+z)|², que es justo la ambigüedad
    # que documenta el módulo. Sería un error mudo, del mismo tipo que el Kf
    # apagado a z < 0, y la suite no puede vigilar a quien llame a la función
    # sin pasar por la CLI.
    if np.any(zs <= 0):
        raise ValueError(
            f"las distancias son sensor-objeto, positivas: el signo lo pone la "
            f"retropropagación. Recibidas {zs[zs <= 0]}.")

    for metodo in metodos:
        extra = kw if metodo == "mpasm" else {}
        for z in zs:
            U, info = propagar(U_h, delta, lamb, -float(z), metodo=metodo,
                               pad=pad, device=device, dtype=dtype, **extra)
            yield metodo, float(z), U, info


def nombre_png(z):
    """Nombre del PNG de una distancia del barrido.

    Tres decimales, no el z0020.png entero de resultados/campos/: un barrido
    linspace da distancias no enteras y redondearlas haría que dos
    reconstrucciones distintas escribieran en el mismo archivo. El ancho fijo
    mantiene el orden alfabético igual al orden del barrido.
    """
    return f"z{z:08.3f}.png"


# --------------------------------------------------------------------- CLI
def _parser():
    p = argparse.ArgumentParser(
        prog="python -m CamposT.retropropagacion",
        description="Retropropaga un holograma con FFT-ASM, BL-ASM y MPASM.",
        epilog="Unidades: usa mm para todo (delta, lamb, z) o µm para todo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("holograma", type=pathlib.Path,
                   help="imagen del holograma (intensidad medida)")
    p.add_argument("--z", type=float, nargs="+", required=True,
                   metavar="Z",
                   help="distancia sensor-objeto, positiva. Un valor, o dos "
                        "para barrer entre ellos")
    p.add_argument("--pasos", type=int, default=25,
                   help="distancias del barrido, si --z trae dos valores")
    p.add_argument("--delta", type=float, default=3.45e-3,
                   help="paso de píxel del sensor [mm]")
    p.add_argument("--lamb", type=float, default=405e-6,
                   help="longitud de onda [mm]")
    p.add_argument("--metodos", nargs="+", default=list(METODOS),
                   choices=METODOS, help="propagadores a comparar")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"))
    p.add_argument("--N", type=int, default=None,
                   help="redimensionar el holograma a N×N antes de propagar")
    p.add_argument("--pad", type=int, default=2,
                   help="factor de relleno de ceros; 1 lo desactiva")
    p.add_argument("--s", type=int, default=1,
                   help="sobremuestreo de MPASM. El defecto de mpasm() es 10, "
                        "pero la matriz espectral es (s·N)² POR DISTANCIA: en "
                        "un barrido eso no cabe. Súbelo para un solo z")
    p.add_argument("--gamma", type=float, default=0.6,
                   help="gamma de la imagen guardada")
    p.add_argument("--invert", action="store_true",
                   help="invertir la imagen antes de tomar la raíz")
    p.add_argument("--salida", type=pathlib.Path, default=None,
                   help="carpeta destino [resultados/retropropagacion/<nombre>]")
    anadir_argumentos(p)
    return p


def main(argv=None):
    from CamposT.backend import liberar_memoria

    args = _parser().parse_args(argv)
    if not args.holograma.is_file():
        raise SystemExit(f"no existe el holograma: {args.holograma}")

    zs = barrido_z(args.z, args.pasos)
    if np.any(zs <= 0):
        raise SystemExit("--z son distancias sensor-objeto positivas; el signo "
                         "lo pone la retropropagación.")

    # Las salidas van siempre bajo la raíz del repo, se lance el módulo desde
    # donde se lance: una ruta relativa las dejaría en el directorio de
    # invocación.
    raiz = pathlib.Path(__file__).resolve().parent.parent
    destino = args.salida or (raiz / "resultados" / "retropropagacion"
                              / args.holograma.stem)

    U_h = load_field(args.holograma, N=args.N, mode="holograma",
                     invert=args.invert)
    forma = U_h.shape        # antes de recortar: la fraccion se mide sobre esto

    print(f"Holograma: {args.holograma}  {U_h.shape}")
    print(f"delta = {args.delta} mm, lamb = {args.lamb} mm, pad = {args.pad}")

    roi = desde_argumentos(args, U_h, args.holograma.name)
    if roi is not None:
        if args.roi_interactivo:
            print(f"\nROI elegida con el raton. Para repetirla:\n"
                  f"    {roi.como_argumento()}\n")
        # Recortar ANTES de informar. informe() no comprueba que la ROI quepa
        # -no es su trabajo- asi que con unas coordenadas malas imprimiria un
        # porcentaje perfectamente creible de una ventana que no existe, y solo
        # despues reventaria. Al reves, el error sale primero y limpio.
        try:
            U_h = roi.recortar(U_h)
        except ValueError as exc:
            # SystemExit limpio, como el resto de errores de usuario (archivo
            # que no existe, z no positiva). recortar() sigue lanzando
            # ValueError para quien la llame desde Python: solo esta frontera
            # de CLI lo convierte.
            raise SystemExit(str(exc)) from exc
        print(informe(roi, forma, zs, args.lamb, args.delta))

    print(f"{len(zs)} distancia(s) de {zs[0]:.3f} a {zs[-1]:.3f} mm "
          f"× {len(args.metodos)} método(s)")

    escritos = 0
    actual = None
    for metodo, z, U, info in retropropagar(
            U_h, args.delta, args.lamb, zs, metodos=args.metodos,
            pad=args.pad, device=args.device, s=args.s):
        if metodo != actual:
            actual = metodo
            carpeta = destino / metodo
            carpeta.mkdir(parents=True, exist_ok=True)
            print(f"\n  {metodo} ({info['device']}, {info['dtype']})")
        guardar(intensidad(U), carpeta / nombre_png(z), gamma=args.gamma)
        escritos += 1
        print(f"    z = {z:9.3f} mm  ->  {nombre_png(z)}")
        del U
        liberar_memoria()

    print(f"\n{escritos} reconstrucciones en {destino}, "
          f"una carpeta por propagador.")


if __name__ == "__main__":
    main()
