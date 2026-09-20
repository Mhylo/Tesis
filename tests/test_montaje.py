"""Verificación de CamposT.montaje.

Lo que se protege aquí no es una cuenta, es una DECISIÓN: que un parámetro sin
medir no valga un número. La tentación de poner un valor plausible «mientras
tanto» es exactamente el fallo que el módulo existe para impedir, así que hay
una prueba que la detecta.
"""

from dataclasses import fields

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

#: Cobaya del centinela. ANTES estas pruebas tiraban de MONTAJE.lamb, y el
#: 09/09 dejaron de valer en cuanto lambda pasó a ser un número medido: se
#: estaba probando la CLASE a través de un campo que estaba destinado a
#: rellenarse. Aquí se prueba la clase contra una instancia propia, que no
#: caduca, y más abajo se comprueba aparte que MONTAJE sigue usándola.
COBAYA = SinMedir("magnitud de prueba", "mm", "no se obtiene, es de mentira")


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
        operacion(COBAYA)


def test_lo_que_sigue_pendiente_en_el_montaje_es_el_centinela_de_verdad():
    """Que MONTAJE use SinMedir, y no otra cosa, para lo que aún no tiene."""
    for nombre in pendientes(MONTAJE):
        valor = getattr(MONTAJE, nombre)
        assert isinstance(valor, SinMedir), nombre
        with pytest.raises(ParametroSinMedir):
            valor * 1


def test_el_mensaje_dice_que_falta_y_como_se_obtiene():
    # z_max sigue pendiente al 09/09: es el recorrido de la platina, y no se
    # lee en ninguna hoja de datos, hay que ir al banco.
    with pytest.raises(ParametroSinMedir) as e:
        MONTAJE.z_max * 1
    texto = str(e.value)
    assert "z máxima alcanzable" in texto
    assert "tarea 17" in texto
    assert "CamposT/montaje.py" in texto


def test_la_magnificacion_sale_de_los_tramos_que_dio_carlos():
    """M = L/z con L = 10 (3 + 7, sumado) y z = 3."""
    assert MONTAJE.magnificacion == pytest.approx(10.0 / 3.0)


def test_una_propiedad_derivada_revienta_si_le_falta_un_dato():
    """Lo que protegía la prueba vieja: la cuenta derivada no inventa nada.

    Ya no se puede comprobar sobre MONTAJE, porque L y z están medidas. Se
    comprueba sobre un montaje al que se le quita L a propósito.
    """
    sin_L = Montaje(L=SinMedir("distancia fuente -> sensor", "mm",
                               "se mide en el banco"))
    with pytest.raises(ParametroSinMedir):
        sin_L.magnificacion


def test_el_centinela_es_falsy_pero_no_es_None():
    assert not MONTAJE.NA
    assert MONTAJE.NA is not None
    assert isinstance(MONTAJE.NA, SinMedir)


def test_repr_no_revienta():
    # Poder imprimirlo es justo lo que hace útil el marcador al depurar.
    assert "SIN MEDIR" in repr(MONTAJE.NA)


# ── Regresión: la decisión de diseño, no la implementación ──────────────────

#: Los seis que Carlos dio el 09/09/2026, con su valor exacto en mm. Cualquier
#: otro campo con número es un valor «mientras tanto», que es justo lo que este
#: módulo existe para impedir.
DADOS_POR_CARLOS = {
    "lamb": 528e-6,
    "pinhole": 5e-3,
    "delta_x": 1.83e-3,
    "delta_y": 1.83e-3,
    "L": 10.0,          # 3 + 7: sumada de los dos tramos, no medida de una vez
    "z": 3.0,
}


def test_el_montaje_solo_tiene_los_numeros_que_alguien_dio():
    """Si esto falla, o llegaron medidas nuevas, o alguien rellenó un hueco.

    Las dos cosas son legítimas, pero se hacen a propósito: si son medidas
    nuevas, se añaden aquí con su valor y se anota de dónde salieron. Lo que
    esta prueba impide es que un número plausible entre de paso y se propague
    en silencio -que es exactamente el fallo del que nace montaje.py-.
    """
    con_numero = {f.name: getattr(MONTAJE, f.name)
                  for f in fields(MONTAJE)
                  if not isinstance(getattr(MONTAJE, f.name), SinMedir)}
    assert con_numero == pytest.approx(DADOS_POR_CARLOS)


def test_la_clase_no_trae_ningun_valor_por_defecto_de_mas():
    """Una instancia recién hecha no sabe más que MONTAJE: son los mismos."""
    assert pendientes(Montaje()) == pendientes(MONTAJE)


def test_lo_que_falta_al_09_09_es_el_sensor_la_platina_y_la_adquisicion():
    """Deja escrito el estado de la tarea 17: nueve de quince sin medir.

    No es decoración: es lo que hay que pedir en la próxima visita al banco, y
    tenerlo en una prueba obliga a actualizarlo cuando llegue.
    """
    assert pendientes(MONTAJE) == (
        "laser",                            # para citarlo
        "px_x", "px_y", "sensor", "bits",   # el sensor entero
        "z_min", "z_max",                   # recorrido: el barrido de la t. 29
        "NA",                               # el límite de la tarea 40
        "gamma",                            # la trampa de la tarea 69
    )


def test_informe_distingue_completo_de_pendiente():
    assert "completo" in informe(TABLA1)
    assert "sin medir" in informe(MONTAJE)


def test_los_dos_pasos_de_pixel_van_por_separado():
    """Tarea 61: Kf se calcula por eje, así que el montaje no asume cuadrado."""
    campos = Montaje.__dataclass_fields__
    assert "delta_x" in campos and "delta_y" in campos
    assert "px_x" in campos and "px_y" in campos
