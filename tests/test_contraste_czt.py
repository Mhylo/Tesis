"""Pruebas de scripts/contraste_czt.py: el MPASM calculado con la CZT tiene
que dar lo mismo que con sus matrices.

En CPU, complex128 y N = 64 para que corran en segundos.
"""

import numpy as np
import pytest

from CamposT.propagadores import kf_auto, mpasm
from scripts import contraste_czt as cc

N, DELTA, LAMB = 64, 5.0 / 64, 632.8e-6


@pytest.fixture(scope="module")
def haz():
    x = (np.arange(N) - N / 2) * DELTA
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / 1.0**2).astype(np.complex128)


def test_dft_czt_es_la_matriz():
    """Una fila de la DFT del MPASM, con mallas no centradas y los dos signos."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(3, 40)) + 1j * rng.normal(size=(3, 40))
    e = 0.3 + 0.07 * np.arange(40)
    s = -2.1 + 0.013 * np.arange(97)
    for signo in (-1, +1):
        directa = u @ np.exp(signo * 2j * np.pi * np.outer(e, s))
        assert cc.relativa(cc.dft_czt(u, e, s, signo, axis=1), directa) < 1e-12


@pytest.mark.parametrize("z, s", [(500.0, 1), (12e3, 4), (200e3, 2)])
def test_mpasm_czt_coincide_con_mpasm(haz, z, s):
    Kf = kf_auto(N, DELTA, LAMB, z, s=s)
    U, _ = mpasm(haz, DELTA, LAMB, z, s=s, Kf=Kf, device="cpu", dtype=np.complex128)
    assert cc.relativa(cc.mpasm_czt(haz, DELTA, LAMB, z, s, Kf), np.asarray(U)) < 1e-9


def test_detecta_otra_distancia(haz):
    """La prueba de arriba tiene que poder fallar: a otra z no coinciden. (Un
    K_f un 10 % distinto no sirve: con el haz liso los dos cálculos siguen bien
    muestreados y difieren solo 3e-4.)"""
    z, s = 12e3, 2
    Kf = kf_auto(N, DELTA, LAMB, z, s=s)
    U, _ = mpasm(haz, DELTA, LAMB, z, s=s, Kf=Kf, device="cpu", dtype=np.complex128)
    assert cc.relativa(cc.mpasm_czt(haz, DELTA, LAMB, 2 * z, s, Kf), np.asarray(U)) > 1e-2
