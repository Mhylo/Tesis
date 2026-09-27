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
    #: 528 nm, dados por Carlos el 09/09/2026. NO es ninguno de los que ya
    #: había: ni las 632.8 de la Tabla 1, ni los 633 de los retro_*.py, ni los
    #: 405 de comparacion.py -ni el 532e-9 que usan sus propios main_dlhm.py y
    #: reconstruction_dlhm.py-. Entre 528 y 532 hay un 0.76 %; parece nada, pero
    #: lambda entra en la fase del propagador, y esos scripts suyos son la
    #: referencia contra la que se valida. Si algo no cuadra al 1 %, mirar aquí.
    lamb: float = 528e-6
    laser: str = SinMedir("modelo del láser", "-", "se lee en el equipo")
    #: 5 um de DIAMETRO. La apertura es circular: confirmado el 09/09/2026, y
    #: eso cierra la duda que dejaba la lectura de Carlos, que venía como
    #: "5 x 5 um" y podía ser una apertura cuadrada. Importa porque la NA que
    #: salga de esta geometría es UNA SOLA, no una por eje: el cono de
    #: iluminación tiene simetría de revolución. (El sensor sigue sin tenerla:
    #: delta_x/delta_y y px_x/px_y van por separado por la tarea 61, y eso no
    #: cambia.)
    #:
    #: Cota que ya se puede calcular, y que NO es la NA del sistema: el primer
    #: cero de Airy de una apertura circular abre el cono a sin(theta) =
    #: 1.22*lamb/d = 0.1288 -7.40 grados-, o sea 2.60 mm de mancha iluminada a
    #: L = 10 mm, que con pasos de 1.83 um son 1420 px de diámetro. Sirve de
    #: comprobación cruzada en cuanto lleguen px_x/px_y: si el sensor es más
    #: ancho que eso, la NA del sistema la fija el pinhole; si es más estrecho,
    #: la fija el sensor. Por eso NA sigue en SinMedir: le falta el sensor.
    pinhole: float = 5e-3

    # --- sensor ---
    #: 1.83 um. Tampoco es el 1.85e-6 que traen main_dlhm.py, main_dlhm.m,
    #: reconstruction_dlhm.py y simulation_reconstruction_asm_dlhm.py.
    delta_x: float = 1.83e-3
    #: Carlos dio UN solo tamaño de píxel, así que aquí va el mismo valor. Esto
    #: es una lectura, no una medida: la hoja de laboratorio pide delta_y aparte
    #: justamente porque Kf se calcula por eje desde la tarea 61 y un eje
    #: submuestreado no avisa. Si la hoja de datos del sensor los distingue,
    #: este es el número que hay que corregir.
    delta_y: float = 1.83e-3
    px_x: int = SinMedir("ancho del sensor", "px", "hoja de datos del sensor")
    px_y: int = SinMedir("alto del sensor", "px", "hoja de datos del sensor")
    sensor: str = SinMedir("modelo del sensor", "-", "se lee en la cámara")
    bits: int = SinMedir("profundidad de bits", "-", "ajustes de la cámara")

    # --- geometría DLHM ---
    #: 10.0 mm, y ESTE NUMERO NO SE MIDIO: SE SUMO. Carlos dio las dos
    #: distancias del banco por tramos -fuente->muestra 3 mm, muestra->sensor
    #: 7 mm- y L es la total, o sea 3 + 7. Consecuencias: (1) la incertidumbre
    #: de L es la de los DOS tramos sumada, no la de una regla; (2) si alguna
    #: vez se remide, se remiden los tramos y se vuelve a sumar aquí, porque el
    #: tramo muestra->sensor no tiene campo propio en esta clase.
    L: float = 10.0
    #: 3.0 mm, fuente (pinhole) -> muestra. Este sí es un tramo medido.
    #: M = L/z = 3.33. Encaja en el orden de sus dos scripts, que no concuerdan
    #: entre sí: main_dlhm.py va con L = 8, z = 2 (M = 4) y
    #: reconstruction_dlhm.py con L = 11, z = 4.95 (M = 2.22).
    z: float = 3.0
    z_min: float = SinMedir(
        "z mínima alcanzable", "mm",
        "recorrido de la platina; define el borde del barrido de la tarea 29")
    z_max: float = SinMedir(
        "z máxima alcanzable", "mm",
        "recorrido de la platina; define el borde del barrido de la tarea 29")
    NA: float = SinMedir(
        "apertura numérica", "-",
        "especificada, o la menor de dos: la del cono del pinhole -circular, "
        "1.22*lamb/pinhole = 0.1288- y la que abarca el sensor desde L. La "
        "segunda necesita px_x/px_y, que siguen sin medir")

    # --- adquisición ---
    gamma: bool = SinMedir(
        "¿la cámara aplica gamma?", "-",
        "ajustes de la cámara; si no consta, captura la misma escena a dos "
        "exposiciones conocidas: si es lineal, el nivel medio se duplica")

    @property
    def magnificacion(self):
        """M = L/z, la magnificación geométrica de la fuente puntual.

        Es la forma que fija la tarea 26. La DISTANCIA EFECTIVA no está aquí a
        propósito, y ya no hace falta: la tarea 24 la dio por el escalado de
        Fresnel, z_eq = z(L-z)/L en la malla delta/M, pero eso es paraxial y
        con detalle fino se equivoca (0.89 de correlación en el eje, 0.63 en
        el borde del sensor). CamposT/dlhm.py reconstruye sin ella, con z y L
        directamente y la esférica exacta; distancia_equivalente() queda ahí
        solo para el escalado de comparación.
        """
        return self.L / self.z

    @property
    def cuadrado(self):
        """Si el sensor es cuadrado. Si no, Kf va por eje (tarea 61)."""
        return self.delta_x == self.delta_y and self.px_x == self.px_y


TABLA1 = Tabla1()

#: Al 09/09 tiene seis de quince, todos dados por Carlos: lambda, pinhole, los
#: dos pasos de píxel y la geometría L/z. Los nueve que faltan son el sensor
#: entero (modelo, píxeles por lado, bits), el recorrido de la platina
#: (z_min/z_max, que es lo que dibuja el barrido de la tarea 29), la NA, el
#: modelo del láser y la gamma. Eso sigue siendo la tarea 17.
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
