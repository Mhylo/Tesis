"""Pruebas de scripts/cruzada_dlhm.py: ida y vuelta DLHM con onda esferica,
cadena de Carlos (ASM) y MPASM esferico (CamposT.dlhm), cruzadas.

En CPU y N = 256 para que corran en segundos; la geometria es la del banco de
Carlos: 528 nm, 1.83 um, L = 10 mm, z = 3 mm.
"""

import numpy as np
import pytest

from CamposT.metricas import sam
from scripts import cruzada_dlhm as cd

N, Z, L, LAMB, DELTA = 256, 3.0, 10.0, 528e-6, 1.83e-3


@pytest.fixture(scope="module")
def objeto():
    """Una mancha lisa: con un sensor de 256 px la apertura en la muestra es
    ~0.03 y resuelve ~16 um; barras de 3 um no se reconstruyen con ningun
    metodo y la prueba mediria la NA, no el propagador."""
    y, x = np.mgrid[:N, :N] - N / 2
    return (1 - 0.5 * np.exp(-(x**2 + y**2) / (2 * 15.0**2))).astype(np.complex128)


def centro(a, frac):
    n = a.shape[0]
    m = int(n * frac) // 2
    return a[n // 2 - m:n // 2 + m, n // 2 - m:n // 2 + m]


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(np.abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real))


@pytest.fixture(scope="module")
def idas(objeto):
    return (cd.ida_carlos(objeto, Z, L, LAMB, DELTA, N, device="cpu"),
            cd.ida_mpasm(objeto, Z, L, LAMB, DELTA, N, device="cpu"))


def test_ida_carlos_sin_objeto_es_uno():
    """Sin objeto el holograma es 1: la referencia va analitica en el sensor y
    solo se propaga la perturbacion. Propagar la iluminacion entera, recortada
    a la ventana, la reinyecta por los bordes (en el centro |h - 1| llegaba a 5.7)."""
    h = cd.ida_carlos(np.ones((N, N), complex), Z, L, LAMB, DELTA, N, device="cpu")
    assert np.abs(h - 1).max() < 1e-12


def test_ida_vuelta_carlos(objeto, idas):
    """No cierra exacta como FFT->FFT: la vuelta divide por la referencia
    recortada en el sensor y propagada (U0), que no es la esferica de la ida.
    Lo que queda es el limite de su cadena con este sensor."""
    t = cd.vuelta_carlos(idas[0], Z, L, LAMB, DELTA, device="cpu")
    assert sam(centro(t, 0.5), centro(objeto, 0.5)) > 20


def test_ida_espejo_concuerda_con_exacta(idas):
    """La parte paraxial coincide: si el signo de la propagacion de Carlos
    (e^{-ikz}) estuviera al reves, la ida espejo retropropagaria y no se
    pareceria a la exacta."""
    assert corr(centro(idas[0] - 1, 0.25), centro(idas[1] - 1, 0.25)) > 0.99


def test_vuelta_mpasm_de_ida_mpasm(objeto, idas):
    t = cd.vuelta_mpasm(idas[1], Z, L, LAMB, DELTA, device="cpu")
    assert sam(centro(t, 0.5), centro(objeto, 0.5)) > 30
