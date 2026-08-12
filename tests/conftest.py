"""Configuración común de la suite.

La misma prueba se ejecuta en CPU (complex128) y, si hay CUDA, en GPU
(complex64). Es el punto del Objetivo 1: verificar que el cambio de
dispositivo y de precisión no cambia el resultado más allá de lo que impone
la mantisa, no que haya dos implementaciones que casualmente coinciden.
"""

import numpy as np
import pytest

from CamposT.backend import gpu_disponible, liberar_memoria
from CamposT.propagadores import gauss_beam

# --- parámetros del campo de prueba ------------------------------------------
# Tabla 1 del paper (haz gaussiano, sin lente) con N reducido: la matriz
# espectral de MPASM es (sN)², así que N=64 con s=4 son 256² y la suite corre
# en segundos incluso en la tarjeta de 4 GB.
N = 64
L0 = 5.0            # mm, lado del plano de entrada
W0 = 1.0            # mm, cintura del haz
LAMB = 632.8e-6     # mm, HeNe
DELTA = L0 / (N - 1)

#: z de prueba en mm. zR = pi w0^2 / lamb = 4965 mm, así que la lista cubre
#: campo cercano, la zona de Rayleigh y campo lejano.
Z_CERCANO = (500, 2000)
Z_LEJANO = (12000, 80000)
Z_TODOS = Z_CERCANO + Z_LEJANO


def _tolerancia(device):
    """Cota de error relativo admisible según la mantisa del dtype de trabajo.

    complex128 tiene eps = 2.2e-16 y complex64 eps = 1.2e-7. Sobre una DFT de
    N=64 el error se acumula como ~sqrt(N)·eps, así que se deja un margen de
    unos tres órdenes de magnitud sobre eso. No son números ajustados hasta
    que la prueba pase: si una prueba necesita más margen que este, lo que
    falla es el propagador, no la tolerancia.
    """
    return 1e-12 if device == "cpu" else 1e-5


@pytest.fixture(params=["cpu"] + (["gpu"] if gpu_disponible() else []))
def device(request):
    """Cada prueba que pida este fixture se ejecuta una vez por dispositivo."""
    yield request.param
    if request.param == "gpu":
        liberar_memoria()


@pytest.fixture
def tol(device):
    return _tolerancia(device)


@pytest.fixture
def campo():
    """Haz gaussiano de entrada, en CPU. Los propagadores lo suben solos."""
    U0, _, _ = gauss_beam(N, DELTA, W0, device="cpu")
    return U0


@pytest.fixture
def malla():
    """(X, Y) del plano de entrada, para las referencias analíticas."""
    _, X, Y = gauss_beam(N, DELTA, W0, device="cpu")
    return X, Y


# --- utilidades de comparación -----------------------------------------------
def error_relativo(a, b):
    """Error máximo entre dos campos, normalizado por la amplitud de b.

    Se compara el campo complejo completo, no sólo la amplitud: un propagador
    puede acertar el módulo y equivocar la fase, y para holografía la fase es
    justamente lo que interesa.
    """
    a = np.asarray(_bajar(a))
    b = np.asarray(_bajar(b))
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def energia(U):
    """Suma de |U|², en float64 sea cual sea el dtype del campo."""
    return float(np.sum(np.abs(np.asarray(_bajar(U), dtype=np.complex128)) ** 2))


def _bajar(U):
    from CamposT.backend import a_numpy
    return a_numpy(U)
