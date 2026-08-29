"""El máximo de la curva de nitidez, sólo cuando el barrido acota el foco.

Los scripts de `scripts/retro_*.py` imprimen dónde enfoca cada vuelta
tomando el argmax de la curva de nitidez. El argmax siempre existe: si el
barrido no contiene el foco, devuelve el extremo más alto y el script lo
publica con dos decimales como si fuera una medida.

CUÁNDO PASA ESO, Y POR QUÉ NO ES UN BUG DE LA MÉTRICA

Al acercarse al plano del holograma, la reconstrucción tiende al holograma
mismo, que es un patrón de franjas densas: nitidez máxima. Es una tendencia
monótona hacia z -> 0, no un pico de foco. Cuando el pico verdadero es débil
-la vuelta B, con la imagen gemela encima- esa tendencia gana, y el argmax se
va al extremo corto del barrido.

LA SEPARACIÓN, MEDIDA

Sobre 24 casos (dos propagadores x dos vueltas x dos rangos de barrido x tres
distancias), los 4 que erraban tenían el argmax en un extremo y los 20 que
acertaban no. Ninguna excepción en ninguno de los dos sentidos.

La prominencia del pico -(max - mediana) / desviación- NO separa: los aciertos
bajan a 0.9 sigma y los fallos suben a 1.9. Por eso la guarda es geométrica y
no estadística, que era lo que parecía a primera vista.
"""

import matplotlib
matplotlib.use("Agg")           # antes de importar los scripts, que traen pyplot

import importlib

import numpy as np
import pytest

MODULOS = ("scripts.retro_mpasm", "scripts.retro_fft_angular",
           "scripts.retro_blas", "scripts.retro_holograma")


@pytest.fixture(params=MODULOS)
def mod(request):
    return importlib.import_module(request.param)


@pytest.fixture
def zs():
    """Las 25 distancias de un barrido 0.4-1.6 alrededor de Z = 50 mm."""
    return np.linspace(0.4, 1.6, 25) * 50.0


def campana(zs, centro, ancho=6.0):
    return np.exp(-((zs - centro) / ancho) ** 2)


# --------------------------------------------------------- el caso que vale
def test_devuelve_la_distancia_del_maximo_cuando_esta_dentro(mod, zs):
    """Un pico limpio en mitad del barrido se reporta tal cual."""
    assert mod.pico_de_foco(zs, campana(zs, 50.0)) == pytest.approx(50.0, abs=1.3)


# ------------------------------------------------- los casos que no valen
def test_devuelve_None_si_el_maximo_cae_al_principio(mod, zs):
    """Es la tendencia hacia el plano del holograma, no un foco.

    Sin esta guarda el script imprimiria 'vuelta B enfoca en z = 20.00 mm'.
    """
    monotona = np.linspace(1.0, 0.2, len(zs))
    assert mod.pico_de_foco(zs, monotona) is None


def test_devuelve_None_si_el_maximo_cae_al_final(mod, zs):
    """El otro extremo: el barrido se queda corto por el lado largo."""
    assert mod.pico_de_foco(zs, np.linspace(0.2, 1.0, len(zs))) is None


def test_un_pico_justo_dentro_del_margen_si_se_acepta(mod, zs):
    """La guarda es un margen del 15 %, no 'que no sea el primero'.

    Con 25 puntos eso son los indices 0-3 y 21-24. El indice 4 es valido: si
    la guarda fuese solo el extremo exacto, dejaria pasar los casos medidos
    que erraban en el indice 3.
    """
    curva = np.zeros(len(zs)); curva[4] = 1.0
    assert mod.pico_de_foco(zs, curva) == pytest.approx(zs[4])
    curva = np.zeros(len(zs)); curva[3] = 1.0
    assert mod.pico_de_foco(zs, curva) is None


def test_una_curva_plana_no_da_foco(mod, zs):
    """Sin estructura no hay nada que localizar; el argmax seria el indice 0."""
    assert mod.pico_de_foco(zs, np.ones(len(zs))) is None


# ------------------------------------------------- las tres copias, a la vez
def test_las_copias_dan_el_mismo_resultado(zs):
    """El riesgo de la arquitectura autonoma: que alguien lo arregle en uno."""
    mods = [importlib.import_module(m) for m in MODULOS]
    for curva in (campana(zs, 50.0), np.linspace(1.0, 0.2, len(zs)),
                  campana(zs, 62.0)):
        valores = [m.pico_de_foco(zs, curva) for m in mods]
        assert valores == [valores[0]] * len(valores)
