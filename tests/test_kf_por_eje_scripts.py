"""mpasm_bloques de los scripts calcula Kf por eje, igual que la libreria.

REGRESION. retro_mpasm.py y mi_prueba_mpasm.py usaban min(Kfy, Kfx) -el Kf
del eje LARGO- en los dos ejes, y en malla rectangular el eje corto se
aliasaba sin avisar. CamposT.propagadores.mpasm() ya lo tenia arreglado; las
copias de los scripts no.
"""

import importlib

import numpy as np
import pytest

from CamposT.propagadores import kf_auto, mpasm

MODULOS = ("scripts.retro_mpasm", "scripts.mi_prueba_mpasm")
M, N, DELTA, LAMB, Z = 64, 128, 3.45e-3, 405e-6, -150.0


@pytest.fixture(scope="module")
def campo():
    rng = np.random.default_rng(0)
    return rng.random((M, N)) + 1j * rng.random((M, N))


@pytest.mark.parametrize("nombre", MODULOS)
@pytest.mark.parametrize("s", [1, 2])
def test_mpasm_bloques_coincide_con_la_libreria_en_malla_rectangular(nombre, s, campo):
    m = importlib.import_module(nombre)
    U, kf = m.mpasm_bloques(campo, Z, LAMB, DELTA, s=s, xp=np, dtype=np.complex128)
    ref, kf_ref = mpasm(campo, DELTA, LAMB, Z, s=s, device="cpu", dtype=np.complex128)
    assert kf == kf_ref
    assert np.max(np.abs(U - ref)) / np.max(np.abs(ref)) < 1e-12


@pytest.mark.parametrize("nombre", MODULOS)
def test_mpasm_bloques_devuelve_un_kf_por_eje(nombre, campo):
    m = importlib.import_module(nombre)
    _, kf = m.mpasm_bloques(campo, Z, LAMB, DELTA, s=1, xp=np, dtype=np.complex128)
    assert kf == (kf_auto(M, DELTA, LAMB, Z), kf_auto(N, DELTA, LAMB, Z))
    assert kf[0] > kf[1]                   # el eje corto comprime mas


@pytest.mark.parametrize("nombre", MODULOS)
def test_mpasm_bloques_acepta_kf_escalar_y_pareja(nombre, campo):
    m = importlib.import_module(nombre)
    _, escalar = m.mpasm_bloques(campo, Z, LAMB, DELTA, Kf=2.0, xp=np)
    _, pareja = m.mpasm_bloques(campo, Z, LAMB, DELTA, Kf=(3.0, 2.5), xp=np)
    assert escalar == 2.0 and pareja == (3.0, 2.5)
