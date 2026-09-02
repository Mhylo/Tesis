"""Recortar una ventana de la imagen y propagar sólo eso.

La ROI es un RECORTE PREVIO al propagador: `Roi.recortar(U)` devuelve un array
más chico y después se llama a `pipeline.propagar()` como siempre. De ahí sale
que sirva igual para la ida y para la vuelta sin que haya dos códigos que
puedan divergir, y que no haya que tocar ni una línea de propagadores.py.

QUÉ CUESTA RECORTAR. El muestreo a delta acota el ángulo de difracción que la
malla puede representar: sin(theta) = lambda / (2*delta). La luz que sale de UN
punto del objeto llega al otro plano repartida sobre un disco de radio

    r = z * tan(theta) / delta   píxeles

que con 633 nm, 3.45 um y z = 20 mm son 534 px. Si la ventana que recortas es
más chica que ese disco, estás tirando parte del cono de cada punto.

El recorte es SECO: no se recorta un margen de guarda alrededor. Se decidió así
a propósito (D2 del spec) porque el propósito de la ROI es coste, y con r = 534
px una ROI de 256 px pediría una malla de 1324 px de lado, o sea más de lo que
se intentaba evitar. Un margen que casi siempre es mayor que la imagen no es
una optimización.

Por eso radio_del_cono() es INFORMACIÓN, NO UNA GUARDA: informe() lo imprime
siempre y avisa cuando la ventana se queda corta, y se recorta igual. Cuánto
cuesta está medido en tests/test_roi.py, no descrito con adjetivos.

EL MISMO NÚMERO SE LEE DISTINTO EN CADA SENTIDO:

    en la VUELTA (holograma -> objeto)  dice cuánta RESOLUCIÓN pierdes: el cono
        de cada punto del objeto que cae en la ventana está recortado.

    en la IDA (objeto -> sensor)  dice cuánto --pad necesitas: es el radio
        sobre el que se va a extender la luz de cada punto de tu ROI, y si la
        malla rellenada no lo cubre, la convolución circular de la FFT lo
        devuelve por el borde opuesto.

OJO AL ORDEN. Las coordenadas son píxeles de la imagen YA CARGADA, o sea
después de que --N la redimensione: cargar -> recortar -> propagar. Con el
ratón esto no se puede equivocar, porque el selector enseña esa misma imagen.
A mano sí.

UNIDADES: milímetros para todo.  633 nm -> 633e-6    3.45 um -> 3.45e-3
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["Roi", "radio_del_cono", "informe", "elegir", "anadir_argumentos",
           "desde_argumentos"]


@dataclass(frozen=True)
class Roi:
    """Ventana rectangular en píxeles de la imagen ya cargada.

    Inmutable como Tabla1: una ROI que cambia a mitad de corrida es un recorte
    que no se puede anotar, y anotarlo es la mitad del punto de esta clase (la
    otra mitad es como_argumento()).
    """

    x0: int
    y0: int
    ancho: int
    alto: int

    def __post_init__(self):
        for campo in ("x0", "y0", "ancho", "alto"):
            v = getattr(self, campo)
            # bool es subclase de int y Roi(True, ...) no significa nada
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise TypeError(
                    f"{campo} = {v!r}: una ROI se mide en pixeles enteros. "
                    f"Redondea tu: int(np.floor(...)) para que la ventana quepa "
                    f"dentro de lo que marcaste, int(np.ceil(...)) para que lo "
                    f"contenga. Esa decision no es de este constructor.")
            # los enteros de NumPy valen -argparse y elegir() los producen- pero
            # se guardan como int para que el repr se lea
            object.__setattr__(self, campo, int(v))

        if self.x0 < 0 or self.y0 < 0:
            raise ValueError(
                f"el origen de la ROI es ({self.x0}, {self.y0}) y es negativo. "
                f"Son indices de la malla: empiezan en (0, 0), esquina superior "
                f"izquierda.")
        if self.ancho < 2 or self.alto < 2:
            raise ValueError(
                f"la ROI mide {self.ancho}x{self.alto} px y no da para propagar "
                f"nada. Hacen falta 2 px por lado como minimo.")

    def recortar(self, U):
        """La ventana de U, de forma (alto, ancho).

        REVIENTA si no cabe, en vez de ajustarse al borde. Una ventana ajustada
        tendria la forma que pediste y el contenido de otra region: un
        resultado creible y falso, que es justo el fallo que este repo trata
        igual en SinMedir y en _comprobar_ventana().
        """
        U = np.asarray(U)
        if U.ndim != 2:
            raise ValueError(
                f"recortar espera un campo 2D (M, N), y esto tiene forma "
                f"{U.shape}.")
        M, N = U.shape
        if self.y0 + self.alto > M or self.x0 + self.ancho > N:
            raise ValueError(
                f"la ROI de {self} se sale de una malla {M}x{N}: llega hasta la "
                f"columna {self.x0 + self.ancho} y la fila {self.y0 + self.alto}."
                f" No se ajusta al borde a proposito. Corrige las coordenadas, o "
                f"comprueba si --N redimensiono la imagen: la ROI se mide sobre "
                f"la imagen YA redimensionada.")
        return U[self.y0:self.y0 + self.alto, self.x0:self.x0 + self.ancho]

    def como_argumento(self):
        """La linea que repite este recorte manana.

        Es la reproducibilidad entera del modulo: se pega en el cronograma, en
        el pie de una figura o en el siguiente comando.
        """
        return f"--roi {self.x0} {self.y0} {self.ancho} {self.alto}"

    def __str__(self):
        return f"{self.ancho}x{self.alto} px en ({self.x0}, {self.y0})"


# ─────────────────────────────────────────────────────────── el cono, y su precio

def radio_del_cono(z, lamb, delta):
    """Radio en PIXELES sobre el que un punto reparte su luz a distancia z.

    El muestreo a delta acota la frecuencia espacial representable a
    1/(2*delta), o sea el angulo de difraccion a sin(theta) = lambda/(2*delta).
    Mas alla de ahi la malla no puede describir la onda, propague quien
    propague. La luz de un punto llega al otro plano sobre un disco de radio
    z*tan(theta), y en pixeles eso es z*tan(theta)/delta.

    Es el numero que dice cuanto cuesta recortar. Va con |z|: retropropagar a
    -z reparte la luz sobre el mismo disco.

    Con lambda >= 2*delta no hay angulo propagante que la malla represente: la
    raiz se hace imaginaria y se devuelve inf, que es la respuesta correcta -no
    hay ventana suficientemente grande-.
    """
    sin_max = lamb / (2 * delta)
    if sin_max >= 1:
        return np.inf
    return abs(z) * (sin_max / np.sqrt(1 - sin_max**2)) / delta


def informe(roi, forma, zs, lamb, delta):
    """Texto de varias lineas: que se recorta y que cuesta. NO imprime.

    Devolver texto en vez de imprimirlo es lo que permite comprobar en una
    prueba que el aviso sale cuando toca y no sale cuando no, sin capturar
    stdout. Las CLIs lo imprimen.

    `forma` es (M, N) ANTES de recortar: la fraccion de la imagen no significa
    nada contra la imagen ya recortada.

    `zs` acepta un escalar o una secuencia. En un barrido se informa del cono
    en los DOS EXTREMOS, porque r crece con z y un solo numero mentiria.
    """
    M, N = forma
    zs = np.atleast_1d(np.asarray(zs, dtype=float)).ravel()
    extremos = sorted({float(zs.min()), float(zs.max())})
    semilado = min(roi.ancho, roi.alto) / 2

    lineas = [f"  recorte: {roi.como_argumento()}",
              f"    {roi.ancho}x{roi.alto} de {N}x{M}  "
              f"({roi.ancho * roi.alto / (M * N):.1%} de la imagen)"]

    corto = False
    for z in extremos:
        r = radio_del_cono(z, lamb, delta)
        corto = corto or semilado < r
        lineas.append(f"    cono a z = {z:g}: radio {r:.0f} px")
    lineas.append(f"    (sin theta = lambda/(2 delta) = {lamb / (2 * delta):.4f}"
                  f"; semilado de la ventana {semilado:.0f} px)")

    if corto:
        lineas += [
            "    AVISO: la ventana es mas chica que el cono de un punto.",
            "      en la VUELTA: tiras parte del cono de cada punto, y la "
            "reconstruccion sale con menos",
            "        resolucion y con anillos en los bordes.",
            "      en la IDA: sube --pad hasta que la malla rellenada cubra ese "
            "radio, o el cono",
            "        reentra por el borde opuesto (la FFT convoluciona en "
            "circulo).",
            "      Se recorta igual: es tu decision, no una guarda.",
        ]
    return "\n".join(lineas)
