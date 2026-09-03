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

        Devuelve una COPIA, no una vista de U: una vista mantendria vivo el
        array entero de origen a traves de su .base mientras dure la corrida,
        y el proposito de la ROI es justo que la propagacion quepa en memoria.
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
        return np.ascontiguousarray(
            U[self.y0:self.y0 + self.alto, self.x0:self.x0 + self.ancho])

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
    lineas.append(f"    (lambda/(2 delta) = {lamb / (2 * delta):.4f}"
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


# ──────────────────────────────────────────────────────────── seleccion a raton

def elegir(I, titulo=""):
    """Abre I, deja arrastrar un rectangulo y devuelve la Roi.

    Bloquea hasta que cierres la ventana. Si la cierras sin haber arrastrado
    nada, aborta: recortar el plano entero en silencio despues de haber pedido
    una ventana seria hacer otra cosa distinta de la que se pidio.

    Las coordenadas salen en pixeles enteros y recortadas a la malla, con floor
    en el origen y ceil en el extremo, de modo que la ventana devuelta CONTIENE
    lo que arrastraste en vez de quedarse por dentro.

    NO IMPRIME. Devuelve la Roi y quien la use decide que hacer con ella; las
    dos CLIs imprimen roi.como_argumento() para que puedas repetir el recorte.

    matplotlib se importa AQUI DENTRO, no arriba: `import CamposT.roi` no debe
    exigir un backend grafico. Lo vigila
    tests/test_roi.py::test_importar_roi_no_arrastra_matplotlib.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    I = np.asarray(I)
    if I.ndim != 2:
        raise ValueError(f"elegir espera una imagen 2D, no {I.shape}.")
    M, N = I.shape
    caja = {}

    def al_soltar(inicio, final):
        caja["xy"] = (inicio.xdata, inicio.ydata, final.xdata, final.ydata)

    alto_fig = 8.0 * M / N
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, min(alto_fig, 9.0))))
    ax.imshow(I, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{titulo}\nArrastra la ventana y cierra esta figura",
                 fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    estilo = dict(facecolor="none", edgecolor="red", linewidth=1.5)
    try:
        selector = RectangleSelector(ax, al_soltar, useblit=True, button=[1],
                                     interactive=True, props=estilo)
    except TypeError:
        # matplotlib < 3.5 llamaba rectprops a lo que ahora es props
        selector = RectangleSelector(ax, al_soltar, useblit=True, button=[1],
                                     interactive=True, rectprops=estilo)
    # el selector tiene que sobrevivir a esta funcion mientras la figura este
    # abierta: sin una referencia viva, el recolector se lo lleva y el raton
    # deja de hacer nada
    ax._selector_roi = selector

    plt.show()

    if "xy" not in caja or any(v is None for v in caja["xy"]):
        raise SystemExit(
            "Pediste elegir la ventana con el raton y no arrastraste ninguna.\n"
            "Vuelve a lanzarlo y arrastra sobre la imagen, o pasa las "
            "coordenadas a mano con --roi X0 Y0 ANCHO ALTO.")

    xa, ya, xb, yb = caja["xy"]
    x0 = int(np.clip(np.floor(min(xa, xb)), 0, N - 1))
    y0 = int(np.clip(np.floor(min(ya, yb)), 0, M - 1))
    x1 = int(np.clip(np.ceil(max(xa, xb)), x0 + 1, N))
    y1 = int(np.clip(np.ceil(max(ya, yb)), y0 + 1, M))

    if (x1 - x0) < 2 or (y1 - y0) < 2:
        raise SystemExit(
            f"La ventana que arrastraste es de {x1 - x0}x{y1 - y0} pixeles y no "
            f"da para propagar nada. Arrastra un rectangulo de verdad.")

    return Roi(x0, y0, x1 - x0, y1 - y0)


def _vista(U):
    """La imagen que se le ensena al raton, normalizada a [0, 1].

    Es |U|^2 normalizada, la vista de siempre. PERO un objeto de fase pura
    -campos.load_field(mode='fase')- tiene |U| == 1 en todo el plano, asi que
    esa intensidad sale constante y la imagen se veria blanca: no hay nada que
    arrastrar. Ahi se ensena np.angle(U) en su lugar, porque para un objeto de
    fase la fase ES donde esta el objeto.

    isclose() y no ==: exp(i*t) da |U|^2 = 1 en aritmetica exacta, pero con
    ruido de redondeo de punto flotante (~1e-16), asi que == casi nunca es
    cierto y dejaria pasar la rama de la intensidad igual.
    """
    U = np.asarray(U)
    I = np.abs(U) ** 2
    if not np.isclose(I.min(), I.max()):
        return I / I.max()
    fase = np.angle(U)
    lo, hi = fase.min(), fase.max()
    return ((fase - lo) / (hi - lo) if hi > lo
            else np.zeros_like(fase, dtype=float))


# ─────────────────────────────────────────────────────────── enganche a argparse

def anadir_argumentos(parser):
    """Anade --roi y --roi-interactivo, excluyentes, a un parser.

    Las dos CLIs -la ida y la vuelta- comparten estos dos argumentos y su texto
    de ayuda. Definirlos por separado en cada una es como divergen dos ayudas
    que deberian decir exactamente lo mismo.
    """
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--roi", type=int, nargs=4, default=None,
                   metavar=("X0", "Y0", "ANCHO", "ALTO"),
                   help="ventana a recortar, en pixeles de la imagen YA "
                        "cargada, o sea DESPUES de --N. El recorte es seco: "
                        "se avisa de lo que cuesta y se recorta igual")
    g.add_argument("--roi-interactivo", action="store_true",
                   help="arrastra la ventana con el raton y cierra la figura. "
                        "Al elegirla se imprime la linea --roi que la repite")
    return parser


def desde_argumentos(args, U=None, titulo=""):
    """La Roi que piden los argumentos, o None si no se pidio ninguna.

    U es el campo que se le ensena al raton, y solo hace falta por el camino
    interactivo: _vista(U) aqui evita que las dos CLIs repitan esa cuenta y
    que una de las dos la cambie.
    """
    if args.roi is not None:
        return Roi(*args.roi)
    if not getattr(args, "roi_interactivo", False):
        return None
    if U is None:
        raise ValueError(
            "--roi-interactivo necesita el campo para ensenartelo, y no se le "
            "paso ninguno.")
    return elegir(_vista(U), titulo)
