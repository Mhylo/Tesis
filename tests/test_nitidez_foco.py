"""nitidez() tiene que crecer cuando la luz se concentra. Ver por qué no lo hacía.

Los scripts de `scripts/retro_*.py` llevan su propia copia de nitidez(),
que es la métrica con la que el barrido decide dónde enfoca la
retropropagación. Estas pruebas son el único sitio donde se miran a la vez.

EL FALLO QUE MOTIVA EL FICHERO

nitidez() normalizaba por el MÁXIMO antes de medir la energía del gradiente:

    I = I / I.max()

El máximo no es una constante del problema: es una cantidad que depende del
foco. Al enfocar, la luz se concentra y el máximo sube, así que dividir por él
cancela justo el efecto que se quiere medir. Sobre una gaussiana de energía
fija, estrechar el pico de 30 a 5 píxeles deja la métrica en x0.96 -DECRECE-,
mientras que normalizando por la media da x1247.

Con la métrica ciega, el `vuelta B enfoca en z = ...` del barrido era el argmax
de un ruido: acertaba el 25 % de las veces y erraba hasta un 15 %. Normalizando
por la media acierta el 100 % con un error medio del 0.3 %, medido sobre siete
distancias y cuatro resoluciones de barrido.

POR QUÉ LA MEDIA Y NO OTRA COSA

La media es la energía por píxel, y la propagación la conserva salvo lo que se
va por los bordes. Es un normalizador estable con z, que es exactamente lo que
el máximo no es. Sigue haciendo falta normalizar algo: sin ello la métrica
mediría el brillo del plano y no su estructura.
"""

import matplotlib
matplotlib.use("Agg")           # antes de importar los scripts, que traen pyplot

import importlib

import numpy as np
import pytest

#: Los tres scripts de ida y vuelta, mas el de retropropagacion sola. La
#: prueba corre entera sobre cada uno.
MODULOS = ("scripts.retro_mpasm", "scripts.retro_fft_angular",
           "scripts.retro_blas", "scripts.retro_holograma")


@pytest.fixture(params=MODULOS)
def mod(request):
    return importlib.import_module(request.param)


def gaussiana(w, L=256):
    """Pico de anchura w y energía total 1. w pequeño = enfocado.

    Energía fija a propósito: así la única diferencia entre dos casos es cómo
    de concentrada está la luz, que es lo que la métrica debe ver.
    """
    y, x = np.mgrid[0:L, 0:L]
    I = np.exp(-((x - L / 2) ** 2 + (y - L / 2) ** 2) / w ** 2)
    return I / I.sum()


# ------------------------------------------------------- el fallo, en directo
def test_nitidez_crece_al_concentrarse_la_luz(mod):
    """Lo mínimo que se le pide a una métrica de foco, y lo que no cumplía.

    Con la normalización por el máximo esto salía plano (x0.96 de w=30 a w=5):
    la métrica no distinguía un plano enfocado de uno desenfocado.
    """
    anchos = (30.0, 20.0, 12.0, 8.0, 5.0)
    valores = [mod.nitidez(gaussiana(w)) for w in anchos]
    assert valores == sorted(valores), (
        f"nitidez no crece al estrechar el pico: {valores}")
    assert valores[-1] > 100 * valores[0]


def test_nitidez_distingue_un_pico_de_un_plano_uniforme(mod):
    """Un campo uniforme no tiene estructura: su nitidez es 0."""
    assert mod.nitidez(np.ones((64, 64))) == pytest.approx(0.0, abs=1e-12)
    assert mod.nitidez(gaussiana(8.0)) > 1.0


# ------------------------------------------------------------- invariancias
def test_nitidez_no_depende_del_brillo_absoluto(mod):
    """Escalar la intensidad no cambia la nitidez: mide estructura, no brillo.

    Es lo que hace comparables dos distancias del barrido, a las que no llega
    la misma energía.
    """
    I = gaussiana(10.0)
    assert mod.nitidez(1e-6 * I) == pytest.approx(mod.nitidez(I), rel=1e-9)
    assert mod.nitidez(4.2e5 * I) == pytest.approx(mod.nitidez(I), rel=1e-9)


def test_un_campo_nulo_no_revienta(mod):
    """I.mean() es 0 y dividir daría NaN. Un plano negro no tiene estructura."""
    assert mod.nitidez(np.zeros((32, 32))) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------- las tres copias, a la vez
def test_las_copias_dan_el_mismo_numero():
    """El riesgo de la arquitectura autónoma: que alguien lo arregle en uno."""
    I = gaussiana(9.0)
    mods = [importlib.import_module(m) for m in MODULOS]
    valores = [m.nitidez(I) for m in mods]
    assert valores == pytest.approx([valores[0]] * len(valores), abs=0.0)
