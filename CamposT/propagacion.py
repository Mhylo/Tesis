"""La ida: de una imagen a su campo propagado, desde la linea de comandos.

Es el hermano de retropropagacion.py y el espejo exacto de su CLI. La vuelta ya
tenia entrada de usuario y la ida no: `python -m CamposT.pipeline` es un
benchmark con usaf_like(512) cableado que no acepta una imagen, asi que hasta
ahora propagar la tuya hacia adelante solo se podia escribiendo Python.

    python -m CamposT.propagacion objeto.png --z 20 --modo transmitancia
    python -m CamposT.propagacion objeto.png --z 20 --roi 312 208 256 256

NO CONFUNDIR CON propagadores.py: alli estan los ALGORITMOS (MPASM, FFT-ASM,
BL-ASM); aqui esta la CORRIDA de ida de punta a punta -leer la imagen,
recortarla si hay ROI, propagarla a cada distancia, escribir los PNG-. Este
modulo no reimplementa ningun propagador: llama a pipeline.propagar().

EL SIGNO, simetrico y explicito: --z es POSITIVA en las dos CLIs. Aqui se
propaga a +z; en la vuelta el menos lo pone retropropagar(). Una distancia
negativa aqui seria pedirle a la ida que haga de vuelta, y hay un modulo para
eso.

POR QUE NO HAY UN GENERADOR propagar_barrido() como el retropropagar() de la
vuelta: aquel existe para poner el signo por dentro y filtrar los kwargs de
MPASM, o sea para impedir dos errores que aqui no se pueden cometer. Envolver
propagar() en otra capa que no decide nada seria una capa de mas.

UNIDADES: milímetros para todo.  633 nm -> 633e-6    3.45 um -> 3.45e-3
"""

import argparse
import pathlib

import numpy as np

from CamposT.campos import load_field
from CamposT.pipeline import guardar, intensidad, propagar
# barrido_z y nombre_png viven en el modulo de la vuelta porque se escribieron
# alli primero, pero no tienen direccion: un linspace de distancias y un nombre
# de archivo de ancho fijo. Se importan en vez de copiarse -copiarlas es como
# los tres retro_*.py llegaron a sumar 2715 lineas- y en vez de mudarlas, que
# obligaria a tocar tests/test_retropropagacion.py, que ya las prueba.
from CamposT.retropropagacion import METODOS, barrido_z, nombre_png
from CamposT.roi import anadir_argumentos, desde_argumentos, informe

#: los cinco modos de campos.load_field(). 'amplitud' es su defecto y el de aqui.
MODOS = ("amplitud", "fase", "mixto", "transmitancia", "holograma")


# --------------------------------------------------------------------- CLI
def _parser():
    p = argparse.ArgumentParser(
        prog="python -m CamposT.propagacion",
        description="Propaga una imagen hacia adelante con FFT-ASM, BL-ASM y "
                    "MPASM.",
        epilog="Unidades: usa mm para todo (delta, lamb, z) o µm para todo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("imagen", type=pathlib.Path,
                   help="imagen del objeto")
    p.add_argument("--z", type=float, nargs="+", required=True, metavar="Z",
                   help="distancia de propagacion, positiva. Un valor, o dos "
                        "para barrer entre ellos")
    p.add_argument("--pasos", type=int, default=25,
                   help="distancias del barrido, si --z trae dos valores")
    p.add_argument("--modo", default="amplitud", choices=MODOS,
                   help="como se interpreta la imagen: transmitancia de "
                        "amplitud, de fase, mixta, binarizada, o intensidad "
                        "medida")
    p.add_argument("--delta", type=float, default=3.45e-3,
                   help="paso de píxel del sensor [mm]")
    p.add_argument("--lamb", type=float, default=405e-6,
                   help="longitud de onda [mm]")
    p.add_argument("--metodos", nargs="+", default=list(METODOS),
                   choices=METODOS, help="propagadores a comparar")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"))
    p.add_argument("--N", type=int, default=None,
                   help="redimensionar la imagen a N×N antes de propagar")
    p.add_argument("--pad", type=int, default=2,
                   help="factor de relleno de ceros; 1 lo desactiva")
    p.add_argument("--s", type=int, default=1,
                   help="sobremuestreo de MPASM. El defecto de mpasm() es 10, "
                        "pero la matriz espectral es (s·N)² POR DISTANCIA: en "
                        "un barrido eso no cabe. Súbelo para un solo z")
    p.add_argument("--gamma", type=float, default=0.6,
                   help="gamma de la imagen guardada")
    p.add_argument("--invert", action="store_true",
                   help="invertir la imagen antes de construir el campo")
    p.add_argument("--salida", type=pathlib.Path, default=None,
                   help="carpeta destino [resultados/propagacion/<nombre>]")
    anadir_argumentos(p)
    return p


def main(argv=None):
    from CamposT.backend import liberar_memoria

    args = _parser().parse_args(argv)
    if not args.imagen.is_file():
        raise SystemExit(f"no existe la imagen: {args.imagen}")

    zs = barrido_z(args.z, args.pasos)
    if np.any(zs <= 0):
        raise SystemExit(
            "--z son distancias de propagacion positivas: este es el camino de "
            "ida. Para el de vuelta, python -m CamposT.retropropagacion.")

    # Las salidas van siempre bajo la raíz del repo, se lance el módulo desde
    # donde se lance: una ruta relativa las dejaría en el directorio de
    # invocación.
    raiz = pathlib.Path(__file__).resolve().parent.parent
    destino = args.salida or (raiz / "resultados" / "propagacion"
                              / args.imagen.stem)

    U0 = load_field(args.imagen, N=args.N, mode=args.modo, invert=args.invert)
    forma = U0.shape        # antes de recortar: la fraccion se mide sobre esto

    print(f"Imagen: {args.imagen}  {U0.shape}  (modo {args.modo})")
    print(f"delta = {args.delta} mm, lamb = {args.lamb} mm, pad = {args.pad}")

    roi = desde_argumentos(args, U0, args.imagen.name)
    if roi is not None:
        if args.roi_interactivo:
            print(f"\nROI elegida con el raton. Para repetirla:\n"
                  f"    {roi.como_argumento()}\n")
        # Recortar ANTES de informar, por la misma razon que en la vuelta: con
        # unas coordenadas malas, informe() imprimiria un porcentaje creible de
        # una ventana que no existe y solo despues reventaria recortar().
        try:
            U0 = roi.recortar(U0)
        except ValueError as exc:
            # SystemExit limpio, igual que retropropagacion.py: recortar()
            # sigue lanzando ValueError para quien la llame desde Python, solo
            # esta frontera de CLI lo convierte.
            raise SystemExit(str(exc)) from exc
        print(informe(roi, forma, zs, args.lamb, args.delta))

    print(f"{len(zs)} distancia(s) de {zs[0]:.3f} a {zs[-1]:.3f} mm "
          f"× {len(args.metodos)} método(s)")

    escritos = 0
    for metodo in args.metodos:
        # s es de MPASM. propagar() lo ignoraria en los otros dos en vez de
        # fallar, y un parametro que se traga en silencio es peor que uno que
        # revienta: se filtra aqui, como hace retropropagar().
        extra = dict(s=args.s) if metodo == "mpasm" else {}
        carpeta = destino / metodo
        carpeta.mkdir(parents=True, exist_ok=True)
        cabecera = False
        for z in zs:
            U, info = propagar(U0, args.delta, args.lamb, float(z),
                               metodo=metodo, pad=args.pad,
                               device=args.device, **extra)
            if not cabecera:
                print(f"\n  {metodo} ({info['device']}, {info['dtype']})")
                cabecera = True
            guardar(intensidad(U), carpeta / nombre_png(z), gamma=args.gamma)
            escritos += 1
            print(f"    z = {z:9.3f} mm  ->  {nombre_png(z)}")
            del U
            liberar_memoria()

    print(f"\n{escritos} campos en {destino}, una carpeta por propagador.")


if __name__ == "__main__":
    main()
