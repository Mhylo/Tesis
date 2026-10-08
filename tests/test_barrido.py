"""Verificación de CamposT.barrido: la malla del objetivo 2 (tarea 29).

Lo que se fija, y por qué:

- NA <-> N ida y vuelta, y N siempre par: con delta y L fijas, barrer NA es
  barrer N, y una conversión corrida desplaza todo el mapa.
- La malla sale del banco: lambda, delta, L, z = 3 y la NA del pinhole están
  en ella, y ningún z cae por debajo del límite de muestreo.
- La marca "valido" coincide con lo que mpasm() rechaza de verdad: si no, el
  barrido calcularía puntos que abortan, o se saltaría puntos buenos.
- Los números que cita el docstring (0.38 de banda FFT en el banco, el cruce
  entre 5 y 5.5 mm, el pico de 1.79 GB): si el montaje o kf_auto cambian, el
  texto deja de ser cierto y esto lo dice.

Todo es aritmética escalar: corre en milisegundos y sin GPU.
"""

import pytest

from CamposT import barrido
from CamposT.barrido import MALLA, Malla, n_de_na, na_de_n
from CamposT.montaje import MONTAJE
from CamposT.propagadores import _comprobar_ventana


@pytest.mark.parametrize("na", [0.03, 0.1288, 0.25])
def test_na_de_n_deshace_n_de_na(na):
    n = n_de_na(na, MONTAJE.L, MONTAJE.delta_x)
    assert n % 2 == 0
    # medio píxel por lado es lo más que el redondeo a par puede correr la NA
    paso_na = na_de_n(n + 2, MONTAJE.L, MONTAJE.delta_x) - na_de_n(n, MONTAJE.L, MONTAJE.delta_x)
    assert abs(na_de_n(n, MONTAJE.L, MONTAJE.delta_x) - na) <= paso_na


def test_n_de_na_rechaza_na_fuera_de_0_1():
    with pytest.raises(ValueError):
        n_de_na(0.0, 10.0, 1.83e-3)
    with pytest.raises(ValueError):
        n_de_na(1.0, 10.0, 1.83e-3)


def test_la_malla_es_la_del_banco():
    assert (MALLA.lamb, MALLA.delta, MALLA.L) == (MONTAJE.lamb, MONTAJE.delta_x, MONTAJE.L)
    assert MALLA.z_banco in MALLA.z
    assert MALLA.na_banco in MALLA.na
    assert MALLA.na_banco == pytest.approx(1.22 * MONTAJE.lamb / MONTAJE.pinhole, abs=1e-4)


def test_ningun_z_baja_del_limite_de_muestreo():
    for z in MALLA.z:
        paso = MALLA.delta * z / MALLA.L
        assert paso >= MALLA.lamb / 2


def test_la_malla_rechaza_z_imposibles():
    comun = dict(lamb=MALLA.lamb, delta=MALLA.delta, L=MALLA.L, z_banco=3.0,
                 na_banco=0.1288, na=(0.1,))
    with pytest.raises(ValueError, match="z_min_muestreo"):
        Malla(z=(1.0, 3.0), **comun)
    with pytest.raises(ValueError, match="z < L"):
        Malla(z=(3.0, 10.0), **comun)


def test_valido_coincide_con_lo_que_mpasm_rechaza():
    vistos = {True: 0, False: 0}
    for p in MALLA.puntos():
        if p["metodo"] != "mpasm":
            continue
        vistos[p["valido"]] += 1
        if p["valido"]:
            _comprobar_ventana(1, 1.0, p["s"], p["kf"], p["kf"])
        else:
            with pytest.raises(ValueError):
                _comprobar_ventana(1, 1.0, p["s"], p["kf"], p["kf"])
    assert vistos[True] and vistos[False]


def test_cuenta_de_puntos():
    todos = list(MALLA.puntos())
    problemas = len(MALLA.z) * len(MALLA.na)
    assert problemas == 98
    assert len(todos) == problemas * (2 + len(MALLA.s) * len(MALLA.kf_factor))
    assert sum(not p["valido"] for p in todos) == 78


def test_el_banco_esta_submuestreado_para_fft_asm():
    n = MALLA.n[MALLA.na.index(MALLA.na_banco)]
    p = MALLA.problema(MALLA.z_banco, n)
    assert n == 1420
    assert p["fraccion_banda_fft"] == pytest.approx(0.38, abs=0.005)
    assert p["kf_auto"][1] == pytest.approx(1.65, abs=0.005)


@pytest.mark.parametrize("na, z_antes, z_despues", [(0.1288, 5.0, 5.5), (0.25, 3.5, 4.0)])
def test_donde_fft_asm_cruza_la_banda(na, z_antes, z_despues):
    n = MALLA.n[MALLA.na.index(na)]
    assert MALLA.problema(z_antes, n)["fraccion_banda_fft"] < 1
    assert MALLA.problema(z_despues, n)["fraccion_banda_fft"] > 1


def test_con_na_minima_fft_asm_no_cruza():
    n = MALLA.n[0]
    assert all(MALLA.problema(z, n)["fraccion_banda_fft"] < 1 for z in MALLA.z)
    assert MALLA.problema(8.0, n)["fraccion_banda_fft"] == pytest.approx(0.82, abs=0.005)


def test_el_pico_de_memoria_cabe_en_la_gpu():
    pico = MALLA.memoria_pico()
    assert pico == pytest.approx(1.79e9, rel=0.01)
    assert pico <= barrido.PRESUPUESTO_GPU


def test_el_resumen_marca_el_banco_una_vez():
    assert MALLA.resumen().count("<- banco") == 1
