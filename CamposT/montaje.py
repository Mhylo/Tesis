"""Los números del montaje real y los de la Tabla 1 del paper, separados.

    TABLA1     escenario de Zhao 2020. Completo y fijo: son valores publicados
               y ninguna medida de laboratorio los cambia.
    MONTAJE    el banco DLHM del laboratorio. Se rellena con lo que se mida
               (tarea 17 del cronograma). Lo que aún no se ha medido no vale un
               número por defecto: vale SIN_MEDIR.

POR QUÉ ESTÁN SEPARADOS. Son dos escenarios distintos y mezclarlos ya pasó: al
01/09 el repo tenía TRES longitudes de onda conviviendo —632.8 nm en
figura_escenario.py y exactitud.py, que es la Tabla 1 y está bien; 633 nm en
los cuatro retro_*.py; y 405 nm en comparacion.py y mi_holograma.py—. Entre 633
y 405 nm hay un 56 %, y lambda entra en z_lim, en Kf, en el radio del cono de
difracción y en la distancia a la que enfoca una reconstrucción.

POR QUÉ SIN_MEDIR NO ES None NI UN VALOR PLAUSIBLE. Un valor plausible se
propaga en silencio y devuelve un resultado creíble que hay que tirar entero
cuando aparecen los números de verdad. SIN_MEDIR revienta en cuanto alguien lo
mete en una cuenta, y dice qué magnitud falta y cómo se obtiene. Es la misma
decisión que _comprobar_ventana() en propagadores.py, que aborta en vez de
devolver copias periódicas superpuestas, y que el endurecimiento de sam()
contra campos degenerados: fallar fuerte antes que devolver un número que
parece bueno.

Cómo se rellena: docs/hoja_parametros_montaje.md es el checklist de laboratorio
que produce estos valores.

UNIDADES: milímetros para todo.  633 nm -> 633e-6    3.45 um -> 3.45e-3
"""

from dataclasses import dataclass, fields

__all__ = ["ParametroSinMedir", "SinMedir", "Tabla1", "Montaje",
           "TABLA1", "MONTAJE", "pendientes", "informe"]


class ParametroSinMedir(RuntimeError):
    """Se metió en una cuenta un parámetro del montaje que aún no se ha medido."""


class SinMedir:
    """Ocupa el sitio de un parámetro y revienta en cuanto alguien lo usa.

    No es None: None se cuela en un `or` y en un `if`, y algunas cuentas lo
    aceptan. Esto no se deja usar en ninguna operación aritmética, ni convertir
    a float, ni absorber por NumPy.
    """

    __slots__ = ("nombre", "unidad", "como")

    def __init__(self, nombre, unidad, como):
        self.nombre, self.unidad, self.como = nombre, unidad, como

    def _reventar(self, *_a, **_k):
        raise ParametroSinMedir(
            f"{self.nombre} [{self.unidad}] no se ha medido todavía "
            f"(tarea 17 del cronograma). Se obtiene así: {self.como}. "
            f"Rellénalo en CamposT/montaje.py, en MONTAJE."
        )

    # Toda via de escape hacia una cuenta queda cerrada.
    __add__ = __radd__ = __sub__ = __rsub__ = _reventar
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _reventar
    __pow__ = __rpow__ = __neg__ = __abs__ = _reventar
    __float__ = __int__ = __index__ = _reventar
    __array__ = _reventar               # para que NumPy tampoco lo cuele

    def __bool__(self):
        return False                    # 'if not MONTAJE.lamb:' funciona

    def __repr__(self):
        return f"<SIN MEDIR: {self.nombre} [{self.unidad}]>"


# ═══════════════════════════════════════════════════════════════════════════
#  Tabla 1 del paper (Zhao 2020). NO SE TOCA: son los valores publicados.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Tabla1:
    """Escenario de la Tabla 1: gaussiana por frente esférico divergente.

    La lente de f = -300 mm aplicada sobre la cintura deja el haz con R = 300 mm
    y la cintura intacta, o sea un pinhole virtual: lo que se modela es el
    frente esférico divergente, no la lente (tarea 59).
    """

    lamb: float = 632.8e-6      #: HeNe, exacto. No es 633e-6.
    L0: float = 5.0             #: ventana de entrada
    W0: float = 1.0             #: radio de cintura 1/e
    R: float = 300.0            #: radio del frente esférico
    N: int = 512                #: puntos por lado de la malla

    @property
    def delta(self):
        """Paso de la malla. Sale de la ventana y los puntos, no se mide."""
        return self.L0 / (self.N - 1)


#: Las siete distancias de la tabla del paper. La curva de exactitud.py añade
#: una rejilla logarítmica entre ellas: siete puntos sobre siete órdenes de
#: magnitud no dibujan una curva, dibujan una poligonal.
Z_TABLA = (500, 2000, 6000, 12000, 30000, 80000, 200000)


# ═══════════════════════════════════════════════════════════════════════════
#  Montaje DLHM del laboratorio. AQUI se escriben las medidas de la tarea 17.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Montaje:
    """El banco real. Sustituye cada SinMedir por el número que midas.

    Los dos pasos de píxel van por separado a propósito: la tarea 61 corrigió
    Kf para calcularlo POR EJE, porque va como ~1/sqrt(N) y con un solo valor
    el eje corto quedaba submuestreado en silencio —en una malla 64x128 el
    error medido fue de 5x a 63x mayor—. Los sensores de DLHM rara vez son
    cuadrados, así que aquí nunca se asume que lo sean.
    """

    # --- fuente ---
    lamb: float = SinMedir(
        "lambda del láser", "mm",
        "se lee en el láser o su hoja de datos; HeNe son 632.8e-6, no 633e-6")
    laser: str = SinMedir("modelo del láser", "-", "se lee en el equipo")
    pinhole: float = SinMedir(
        "diámetro del pinhole", "mm",
        "se lee en la montura; de aquí sale NA si no viene especificada")

    # --- sensor ---
    delta_x: float = SinMedir(
        "paso de píxel horizontal", "mm",
        "hoja de datos del sensor; 3.45 um se escribe 3.45e-3")
    delta_y: float = SinMedir(
        "paso de píxel vertical", "mm",
        "hoja de datos del sensor; anótalo aunque parezca igual al horizontal")
    px_x: int = SinMedir("ancho del sensor", "px", "hoja de datos del sensor")
    px_y: int = SinMedir("alto del sensor", "px", "hoja de datos del sensor")
    sensor: str = SinMedir("modelo del sensor", "-", "se lee en la cámara")
    bits: int = SinMedir("profundidad de bits", "-", "ajustes de la cámara")

    # --- geometría DLHM ---
    L: float = SinMedir(
        "distancia fuente -> sensor", "mm", "se mide en el banco")
    z: float = SinMedir(
        "distancia fuente -> muestra", "mm", "se mide en el banco")
    z_min: float = SinMedir(
        "z mínima alcanzable", "mm",
        "recorrido de la platina; define el borde del barrido de la tarea 29")
    z_max: float = SinMedir(
        "z máxima alcanzable", "mm",
        "recorrido de la platina; define el borde del barrido de la tarea 29")
    NA: float = SinMedir(
        "apertura numérica", "-",
        "especificada, o de la geometría del pinhole y la distancia")

    # --- adquisición ---
    gamma: bool = SinMedir(
        "¿la cámara aplica gamma?", "-",
        "ajustes de la cámara; si no consta, captura la misma escena a dos "
        "exposiciones conocidas: si es lineal, el nivel medio se duplica")

    @property
    def magnificacion(self):
        """M = L/z, la magnificación geométrica de la fuente puntual.

        Es la forma que fija la tarea 26. La DISTANCIA EFECTIVA no está aquí a
        propósito: sale del desarrollo de la tarea 24, que aún no está escrito,
        y ponerle una fórmula ahora sería inventarla.
        """
        return self.L / self.z

    @property
    def cuadrado(self):
        """Si el sensor es cuadrado. Si no, Kf va por eje (tarea 61)."""
        return self.delta_x == self.delta_y and self.px_x == self.px_y


TABLA1 = Tabla1()

#: Al 01/09 está entero sin medir: es exactamente lo que persigue la tarea 17.
MONTAJE = Montaje()


def pendientes(escenario=MONTAJE):
    """Los parámetros que siguen sin medir, en orden de declaración."""
    return tuple(f.name for f in fields(escenario)
                 if isinstance(getattr(escenario, f.name), SinMedir))


def informe(escenario=MONTAJE):
    """Texto de una línea por parámetro: lo medido y lo que falta."""
    lineas = []
    for f in fields(escenario):
        v = getattr(escenario, f.name)
        lineas.append(f"  {f.name:<12} {'PENDIENTE' if isinstance(v, SinMedir) else v}")
    faltan = len(pendientes(escenario))
    cabeza = (f"{escenario.__class__.__name__}: {faltan} parámetro(s) sin medir"
              if faltan else f"{escenario.__class__.__name__}: completo")
    return "\n".join([cabeza, *lineas])


if __name__ == "__main__":
    print(informe(TABLA1), end="\n\n")
    print(informe(MONTAJE))
