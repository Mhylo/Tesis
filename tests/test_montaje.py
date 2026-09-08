"""Verificación de CamposT.montaje.

Lo que se protege aquí no es una cuenta, es una DECISIÓN: que un parámetro sin
medir no valga un número. La tentación de poner un valor plausible «mientras
tanto» es exactamente el fallo que el módulo existe para impedir, así que hay
una prueba que la detecta.
"""

import numpy as np
import pytest

from CamposT.montaje import (MONTAJE, TABLA1, Montaje, ParametroSinMedir,
                             SinMedir, Z_TABLA, informe, pendientes)


# ── Tabla 1: valores publicados, no se tocan ────────────────────────────────

def test_tabla1_son_los_valores_del_paper():
    assert TABLA1.lamb == 632.8e-6      # HeNe exacto, NO 633e-6
    assert (TABLA1.L0, TABLA1.W0, TABLA1.R, TABLA1.N) == (5.0, 1.0, 300.0, 512)


def test_tabla1_delta_sale_de_la_ventana_y_los_puntos():
    assert TABLA1.delta == TABLA1.L0 / (TABLA1.N - 1)


def test_tabla1_no_tiene_nada_pendiente():
    assert pendientes(TABLA1) == ()


def test_las_siete_distancias_de_la_tabla():
    assert Z_TABLA == (500, 2000, 6000, 12000, 30000, 80000, 200000)


def test_tabla1_es_inmutable():
    with pytest.raises(Exception):
        TABLA1.lamb = 633e-6


# ── El centinela: toda vía hacia una cuenta está cerrada ────────────────────

@pytest.mark.parametrize("operacion", [
    pytest.param(lambda v: v * 2, id="multiplicar"),
    pytest.param(lambda v: 2 * v, id="multiplicar-reflejado"),
    pytest.param(lambda v: v + 1, id="sumar"),
    pytest.param(lambda v: 1 - v, id="restar-reflejado"),
    pytest.param(lambda v: 1 / v, id="dividir-reflejado"),
    pytest.param(lambda v: v ** 2, id="potencia"),
    pytest.param(lambda v: -v, id="negar"),
    pytest.param(lambda v: abs(v), id="valor-absoluto"),
    pytest.param(lambda v: float(v), id="float"),
    pytest.param(lambda v: int(v), id="int"),
    pytest.param(lambda v: np.array([v], dtype=float), id="numpy"),
])
def test_usar_un_parametro_sin_medir_revienta(operacion):
    with pytest.raises(ParametroSinMedir):
        operacion(MONTAJE.lamb)


def test_el_mensaje_dice_que_falta_y_como_se_obtiene():
    with pytest.raises(ParametroSinMedir) as e:
        MONTAJE.delta_y * 1
    texto = str(e.value)
    assert "paso de píxel vertical" in texto
    assert "tarea 17" in texto
    assert "CamposT/montaje.py" in texto


def test_una_propiedad_derivada_tambien_revienta():
    # magnificacion = L/z, y las dos están sin medir.
    with pytest.raises(ParametroSinMedir):
        MONTAJE.magnificacion


def test_el_centinela_es_falsy_pero_no_es_None():
    assert not MONTAJE.lamb
    assert MONTAJE.lamb is not None
    assert isinstance(MONTAJE.lamb, SinMedir)


def test_repr_no_revienta():
    # Poder imprimirlo es justo lo que hace útil el marcador al depurar.
    assert "SIN MEDIR" in repr(MONTAJE.lamb)


# ── Regresión: la decisión de diseño, no la implementación ──────────────────

def test_ningun_parametro_del_montaje_tiene_un_valor_plausible_por_defecto():
    """Si esto falla, alguien rellenó un hueco con un numero «mientras tanto».

    Rellenar MONTAJE con medidas reales es el objetivo, pero entonces la
    tarea 17 está cerrada y esta prueba se actualiza a propósito, no de paso.
    """
    assert len(pendientes(Montaje())) == len(Montaje.__dataclass_fields__)


def test_informe_distingue_completo_de_pendiente():
    assert "completo" in informe(TABLA1)
    assert "sin medir" in informe(MONTAJE)


def test_los_dos_pasos_de_pixel_van_por_separado():
    """Tarea 61: Kf se calcula por eje, así que el montaje no asume cuadrado."""
    campos = Montaje.__dataclass_fields__
    assert "delta_x" in campos and "delta_y" in campos
    assert "px_x" in campos and "px_y" in campos
