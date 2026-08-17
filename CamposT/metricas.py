"""Como se compara un campo calculado con su referencia.

    sam            Simulation accuracy metric del paper, en decibelios.
                   Mas alto es mejor. Va sobre intensidades y lleva un factor
                   de escala ajustado, asi que mide la FORMA del campo, no su
                   nivel ni su fase.
    rms_amplitud   error RMS entre amplitudes normalizadas. Mas bajo es mejor.
                   Es la medida de paridad GPU/CPU, donde no hay una referencia
                   "verdadera" que justifique ajustar la escala.

Las dos convenciones son opuestas: confundirlas invierte la lectura de
cualquier grafica. Por eso viven juntas y con los nombres separados.

La Ec. (16) del paper esta impresa con una errata; ver alfa_sam().
"""

import numpy as np

from CamposT.backend import a_numpy


# ------------------------------------------------------------------- métricas
def _intensidad(U):
    """|U|² en float64 y en CPU, venga el campo de donde venga.

    Las Ecs. (15)-(16) van sobre intensidades. La reducción se hace siempre en
    doble precisión: es una suma sobre todo el plano, donde complex64
    acumularía error sin ahorrar nada apreciable.
    """
    return np.abs(a_numpy(U)).astype(np.float64) ** 2


def alfa_sam(U, ref, formula="corregida"):
    """Factor de escala α que mejor ajusta la referencia al campo medido.

    La Ec. (16) de Zhao et al. (2020) está impresa como

        α = ∫∫ I·I*_ref dxdy / ∫∫ I_ref dxdy                        (16)

    y ese denominador es una errata: tiene que ser ∫∫I_ref². Tres razones,
    cualquiera de ellas concluyente:

    1. Dimensiones. α multiplica a I_ref en la Ec. (15), luego es
       adimensional. Con ∫I_ref el cociente tiene unidades de intensidad;
       con ∫I_ref² es adimensional.
    2. Caso trivial. Un campo idéntico a la referencia debe dar α = 1. Con
       la impresa da el valor medio de la intensidad (0.081 en el gaussiano
       de la Tabla 1 a z = 12000 mm), y de ahí SAM = 0.365 dB en vez de ∞.
    3. Poder de discriminación. Con la impresa el SAM queda por debajo de
       2.5 dB para todos los métodos y todas las distancias, así que no
       podría generar la Figura 4 del propio paper; con la corregida separa
       MPASM de FFT-ASM en 27 dB a z = 12000 mm.

    Es el ajuste por mínimos cuadrados: el α que minimiza ∫|I − α·I_ref|².
    El conjugado de la Ec. (16) es decorativo, porque I e I_ref son reales.

    formula="literal" reproduce la Ec. (16) tal como está impresa, misma
    convención que kf_auto(..., formula="codigo") para la errata de la
    Ec. (14). Es para poder contrastar contra la Figura 4 publicada, no para
    usarla.
    """
    I_ref = _intensidad(ref)
    den = np.sum(I_ref) if formula == "literal" else np.sum(I_ref * I_ref)
    return float(np.sum(_intensidad(U) * I_ref) / den)


def sam(U, ref, formula="corregida"):
    """Simulation accuracy metric en decibelios, Ec. (15). Más alto = mejor.

        SAM = 10·lg[ ∫∫ I dxdy / ∫∫ |I − α·I_ref| dxdy ]

    El dxdy se cancela entre numerador y denominador, así que la métrica no
    depende del paso de píxel y no hace falta pasárselo.

    Devuelve +inf cuando el residuo se anula (campo idéntico a la referencia
    salvo escala). Acepta arrays de CPU o de GPU.

    Lo que NO mide: como α es un factor de escala ajustado, SAM es ciego a un
    error de normalización global. Mide la forma del campo, no su nivel, y
    sólo sobre intensidades: dos campos con la misma intensidad y distinta
    fase dan el mismo SAM. Para la fase hay que comparar el campo complejo.

    Sobre formula="literal", ver alfa_sam().
    """
    I = _intensidad(U)
    alfa = alfa_sam(U, ref, formula=formula)
    residuo = float(np.sum(np.abs(I - alfa * _intensidad(ref))))
    if residuo == 0.0:
        return np.inf
    return float(10 * np.log10(np.sum(I) / residuo))


def rms_amplitud(U, ref):
    """Error RMS entre amplitudes normalizadas por su máximo. Más bajo = mejor.

    No es SAM y no es la métrica del paper: es la medida de paridad que usa
    scripts/comparacion.py para contrastar el mismo campo calculado en GPU y
    en CPU. Ahí no hay una referencia "verdadera" que justifique el ajuste de
    escala de la Ec. (16), y lo que interesa es la discrepancia bruta.
    """
    a = np.abs(a_numpy(U)).astype(np.float64)
    b = np.abs(a_numpy(ref)).astype(np.float64)
    return float(np.sqrt(np.mean((a / a.max() - b / b.max()) ** 2)))
