"""La malla del barrido del objetivo 2 (tarea 29), con los números del banco.

El objetivo 2 pide los rangos de distancia, apertura numérica, s y K_f en los
que cada propagador mantiene una precisión aceptable bajo el muestreo de DLHM.
Este módulo fija ESA malla y nada más: qué puntos se calculan, por qué esos y
cuánto cuestan. La tarea 30 la recorre y calcula el error en cada punto.

LO QUE SALE DEL BANCO (CamposT/montaje.py, dados por Carlos el 09/09/2026):
lambda = 528 nm, pinhole circular de 5 um, delta = 1.83 um, L = 10 mm y
z = 3 mm. L queda FIJA: es la del banco, y con L fija la magnificación
M = L/z y la NA quedan como los dos ejes geométricos.

EL PROBLEMA EN CADA PUNTO. Con fuente puntual, retropropagar el holograma a la
muestra equivale, en el escalado de Fresnel de la tarea 24, a propagar una onda
plana una distancia z_eq = z(L - z)/L sobre la malla delta/M. Es el único
camino por el que FFT-ASM, BLAS y MPASM se aplican a DLHM tal cual (la tarea
26 los conecta ahí), así que su error de MUESTREO se mide sobre ese problema:
N x N puntos de paso delta·z/L a la distancia z_eq. El error de FORMULACIÓN
del escalado (paraxial: 0.89 de correlación con detalle fino, frente al MPASM
esférico exacto de CamposT/dlhm.py) ya está medido y es otro resultado; aquí
no se mezcla.

LOS EJES Y POR QUÉ.

    z     de 1.5 a 8 mm cada 0.5 (14 valores, el 3.0 del banco incluido).
          Abajo, z_min_muestreo() = L·lambda/(2·delta) = 1.443 mm: por debajo,
          el píxel retroproyectado delta/M es menor que lambda/2 y achicarlo
          no da información (la limita la NA, no el píxel). Arriba, z = 8 deja
          M = 1.25; más cerca del sensor DLHM deja de magnificar. El recorrido
          real de la platina (z_min, z_max) sigue sin medir: cuando llegue será
          un subrectángulo de esta malla, sin repetir nada.

    NA    0.03, 0.06, 0.09, 0.1288, 0.17, 0.21 y 0.25, la NA de una ventana
          cuadrada de N x N píxeles vista desde la fuente: sin(atan(N·delta /
          (2L))). Con delta y L fijas, barrer NA ES barrer N (n_de_na()). El
          0.1288 es el primer cero de Airy del pinhole, 1.22·lambda/d: la NA
          que ilumina un pinhole de 5 um ideal. Se pasa de ahí a propósito: los
          hologramas de referencia caen mucho más despacio que ese Airy (por
          eso la fuente por defecto de mi_prueba_dlhm es la de Lopera) y la NA
          real del banco sigue sin medir. El tope 0.25 (N = 2822) lo pone la
          memoria: MPASM con s = 3 pide 1.79 GB, y la RTX 3050 tiene 4.

    s     1, 2 y 3. Con s = 4 el punto más grande pide 2.62 GB, demasiado
          cerca de los 3.2 libres.

    K_f   0.5, 1 y 2 veces el K_f automático de la Ec. (14) (kf_auto). Lo que
          se pregunta es si el automático es el bueno, así que se mide a cada
          lado. Un punto con s·K_f < 1 viola r·mag <= s·K_f (tarea 63): la
          salida saldría con copias periódicas superpuestas y mpasm() aborta.
          Esos puntos quedan en la malla marcados como fuera por construcción
          y no se calculan: son parte del mapa, la región prohibida.

EL BANCO EN LA MALLA. En el punto del banco (z = 3, NA del pinhole, N = 1420)
la ventana de FFT-ASM cubre el 38 % de la banda que el problema necesita
(fraccion_banda_fft = 0.38) y K_f automático vale 1.65 con s = 1. Esa
fracción cruza 1 entre z = 5 y 5.5 mm con la NA del pinhole, entre 3.5 y 4
con NA 0.25, y con NA 0.03 no llega a cruzarlo (0.82 en z = 8). La malla
captura la frontera donde FFT-ASM deja de valer, que es justo lo que el
objetivo 2 tiene que ubicar.

    Tesis_env/Scripts/python.exe -m CamposT.barrido      # la tabla por punto
"""

from dataclasses import dataclass

import numpy as np

from CamposT.dlhm import distancia_equivalente, magnificacion
from CamposT.montaje import MONTAJE
from CamposT.pipeline import diagnostico
from CamposT.propagadores import kf_auto, memoria_mpasm

__all__ = ["n_de_na", "na_de_n", "z_min_muestreo", "na_pinhole", "Malla", "MALLA"]

#: Lo que se le permite pedir a la GPU en un punto del barrido. La RTX 3050
#: tiene 4 GB y deja ~3.2 libres con el escritorio; 2.5 GB deja margen para el
#: campo de entrada, la referencia y la métrica.
PRESUPUESTO_GPU = 2.5e9


def n_de_na(na, L, delta):
    """El lado N (par) de la ventana cuadrada que ve la fuente con apertura na."""
    if not 0 < na < 1:
        raise ValueError(f"na = {na!r}: va en (0, 1)")
    return 2 * int(round(L * np.tan(np.arcsin(na)) / delta))


def na_de_n(n, L, delta):
    """La NA de una ventana de n píxeles de lado: sin(atan(n·delta / (2L)))."""
    return float(np.sin(np.arctan(n * delta / (2 * L))))


def z_min_muestreo(L, lamb, delta):
    """z por debajo de la cual delta/M < lambda/2: L·lambda/(2·delta)."""
    return L * lamb / (2 * delta)


def na_pinhole(lamb, diametro):
    """sin(theta) del primer cero de Airy de una apertura circular."""
    return 1.22 * lamb / diametro


@dataclass(frozen=True)
class Malla:
    """Los ejes del barrido y lo fijo. Todo en mm."""

    lamb: float
    delta: float
    L: float
    z_banco: float
    na_banco: float
    z: tuple
    na: tuple
    s: tuple = (1, 2, 3)
    kf_factor: tuple = (0.5, 1.0, 2.0)

    def __post_init__(self):
        if min(self.z) < z_min_muestreo(self.L, self.lamb, self.delta):
            raise ValueError("hay z por debajo de z_min_muestreo(): delta/M < lambda/2")
        if max(self.z) >= self.L:
            raise ValueError("z tiene que quedar entre la fuente y el sensor (z < L)")

    @property
    def n(self):
        """El lado de la ventana de cada NA, en el mismo orden."""
        return tuple(n_de_na(a, self.L, self.delta) for a in self.na)

    def problema(self, z, n):
        """El problema de onda plana equivalente del punto (z, n), con lo que
        conviene saber de él antes de propagar."""
        M = magnificacion(z, self.L)
        paso = self.delta / M
        zeq = distancia_equivalente(z, self.L)
        d = diagnostico(n, paso, self.lamb, zeq)
        return {
            "z": z, "N": n, "NA": na_de_n(n, self.L, self.delta),
            "M": M, "paso": paso, "z_eq": zeq,
            "fraccion_banda_fft": d["fraccion_banda_util_FFT"],
            "kf_auto": {s: kf_auto(n, paso, self.lamb, zeq, s=s) for s in self.s},
        }

    def puntos(self):
        """Cada cálculo del barrido, como dict. 'valido' es False en los puntos
        con s·K_f < 1, que mpasm() rechaza: van al mapa, pero no se calculan."""
        for z in self.z:
            for n in self.n:
                p = self.problema(z, n)
                base = {"z": z, "N": n, "z_eq": p["z_eq"], "paso": p["paso"]}
                for metodo in ("fft_asm", "blas"):
                    yield {**base, "metodo": metodo, "s": None, "kf": None,
                           "valido": True}
                for s in self.s:
                    for f in self.kf_factor:
                        kf = f * p["kf_auto"][s]
                        yield {**base, "metodo": "mpasm", "s": s, "kf": kf,
                               "kf_factor": f, "valido": s * kf >= 1}

    def memoria_pico(self):
        """Bytes de pico del punto más caro: MPASM con el N y el s mayores."""
        n, s = max(self.n), max(self.s)
        return memoria_mpasm(n, n, s=s)

    def resumen(self):
        """Una línea por (z, NA): el problema y el K_f automático por s."""
        cab = (f"{'z':>5} {'N':>5} {'NA':>7} {'M':>6} {'paso um':>8} {'z_eq':>6} "
               f"{'banda FFT':>9}  " + "  ".join(f"Kf s={s}" for s in self.s))
        lineas = [cab]
        for z in self.z:
            for na, n in zip(self.na, self.n):
                p = self.problema(z, n)
                marca = " <- banco" if (z, na) == (self.z_banco, self.na_banco) else ""
                lineas.append(
                    f"{z:5.1f} {n:5d} {p['NA']:7.4f} {p['M']:6.2f} {p['paso'] * 1e3:8.3f} "
                    f"{p['z_eq']:6.3f} {p['fraccion_banda_fft']:9.2f}  "
                    + "  ".join(f"{p['kf_auto'][s]:6.2f}" for s in self.s) + marca)
        todos = list(self.puntos())
        validos = sum(p["valido"] for p in todos)
        lineas.append(
            f"\n{len(self.z)} z x {len(self.na)} NA = {len(self.z) * len(self.na)} "
            f"problemas; {len(todos)} cálculos, {validos} válidos y "
            f"{len(todos) - validos} fuera por construcción (s·K_f < 1). "
            f"Pico de memoria: {self.memoria_pico() / 1e9:.2f} GB.")
        return "\n".join(lineas)


def _malla_del_montaje(montaje=MONTAJE):
    """La malla con lambda, delta, L y z del banco. delta es delta_x: la
    ventana es cuadrada, y si el sensor real no lo es, sus dos NA son dos
    puntos de este mismo mapa. La NA del banco es la del pinhole mientras la
    real siga sin medir."""
    na_pin = round(na_pinhole(montaje.lamb, montaje.pinhole), 4)
    return Malla(
        lamb=montaje.lamb, delta=montaje.delta_x, L=montaje.L, z_banco=montaje.z,
        na_banco=na_pin,
        z=tuple(float(v) for v in np.arange(1.5, 8.01, 0.5)),
        na=(0.03, 0.06, 0.09, na_pin, 0.17, 0.21, 0.25),
    )


#: La malla de la tarea 29.
MALLA = _malla_del_montaje()


if __name__ == "__main__":
    print(MALLA.resumen())
