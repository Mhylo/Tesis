"""Verificación del target USAF sintético y su conversión a resolución.

De aquí sale una cifra de portada de la tesis: la resolución lateral efectiva
en pares de líneas por milímetro (tarea 40, semana 12). Un factor 2 en esa
conversión no se nota a ojo en ninguna figura y falsea el resultado principal,
así que la geometría dibujada y la que declara la API tienen que contrastarse
contra la imagen, no darse por supuestas.

Convenio USAF 1951: un PAR DE LÍNEAS es una barra más un hueco. Con barras y
huecos del mismo ancho b, el periodo espacial es 2b y

    lp/mm = 1 / (periodo · delta) = 1 / (2·b·delta)
"""

import numpy as np
import pytest

from CamposT.campos import (anchos_barra_usaf, lineas_por_mm_usaf,
                            periodos_usaf, usaf_like)


def periodo_medido(t, columna):
    """Periodo real (barra + hueco) leído de la imagen, en píxeles."""
    cambios = np.where(np.diff(t[:, columna]) != 0)[0]
    anchos = np.diff(cambios)
    assert len(anchos) >= 2, "la columna no cruza suficientes barras"
    return int(anchos[0] + anchos[1])


def test_el_periodo_declarado_es_el_que_esta_dibujado():
    """Contrasta la API contra los píxeles. Es la prueba que delata el factor 2:
    periodos_usaf() devolvía el ancho de barra, no el periodo."""
    N = 256
    t = usaf_like(N)
    # el elemento 0 ocupa la primera celda; su grupo de barras horizontales
    # queda a la izquierda del bloque
    assert periodo_medido(t, 30) == periodos_usaf(N)[0]


def test_el_ancho_de_barra_es_la_mitad_del_periodo():
    """Barras y huecos del mismo ancho: es como se dibuja y como es el target
    real. Si dejaran de serlo, la conversión a lp/mm dejaría de valer."""
    N = 256
    t = usaf_like(N)
    cambios = np.where(np.diff(t[:, 30]) != 0)[0]
    barra, hueco = np.diff(cambios)[:2]
    assert barra == hueco == anchos_barra_usaf(N)[0]
    assert periodos_usaf(N)[0] == 2 * anchos_barra_usaf(N)[0]


def test_lineas_por_mm_cuenta_el_par_completo_no_la_barra():
    """lp/mm = 1/(periodo·delta), no 1/(barra·delta). Con barras de b píxeles
    el par de líneas mide 2b, así que confundirlos duplica la resolución
    reportada."""
    N, delta = 256, 3.45e-3
    b = anchos_barra_usaf(N)[0]
    assert lineas_por_mm_usaf(delta, N)[0] == pytest.approx(1 / (2 * b * delta))


def test_los_elementos_decrecen_como_el_target_real():
    """El periodo cae por 2^(1/6) de un elemento al siguiente, que es la
    progresión del USAF 1951 dentro de un grupo."""
    periodos = periodos_usaf(512, n_elementos=6)
    razones = [a / b for a, b in zip(periodos, periodos[1:])]
    assert all(1.0 < r < 1.35 for r in razones), razones


def test_la_resolucion_no_decrece_con_el_elemento():
    """Elementos más finos, no menos lp/mm. No se exige crecimiento estricto:
    los anchos de barra son enteros, así que dos elementos consecutivos pueden
    redondear al mismo valor."""
    lp = lineas_por_mm_usaf(3.45e-3, 512)
    assert all(a <= b for a, b in zip(lp, lp[1:]))
    assert lp[-1] > lp[0], "la progresión tiene que llegar a alguna parte"


def test_los_elementos_finos_se_saturan_en_el_limite_de_muestreo():
    """A N pequeño los últimos elementos colapsan todos a barras de 2 px.

    Con 2 píxeles por barra se está justo en Nyquist: esos elementos no son
    distinguibles entre sí y NO sirven para reportar resolución. Queda fijado
    aquí porque la cifra de la tarea 40 depende de elegir un elemento que el
    muestreo resuelva de verdad.
    """
    barras = anchos_barra_usaf(256)
    assert barras[-1] == 2
    assert barras.count(2) > 1, "a N=256 varios elementos saturan en 2 px"
    assert anchos_barra_usaf(512).count(2) == 0, "a N=512 ninguno satura"


def columna_del_elemento(N, i, n_elementos=12, cols=3):
    """Columna que cruza las barras horizontales del elemento i, y el rango de
    filas que ocupan. Reproduce la disposición de usaf_like."""
    b = anchos_barra_usaf(N, n_elementos, cols)[i]
    rows = int(np.ceil(n_elementos / cols))
    cell_w, cell_h = N // cols, N // rows
    r, c = divmod(i, cols)
    y0 = r * cell_h + (cell_h - 5 * b) // 2
    x0 = c * cell_w + (cell_w - 11 * b) // 2
    return x0 + b, slice(y0, y0 + 5 * b)


@pytest.mark.parametrize("N", (256, 512))
@pytest.mark.parametrize("elemento", (0, 3, 7))
def test_la_geometria_declarada_coincide_con_la_dibujada(N, elemento):
    """usaf_like dibuja y anchos_barra_usaf mide. Si divergieran, todas las
    cifras de resolución quedarían mal sin que fallara nada visible.

    Se comprueban varios elementos, no sólo el primero: una progresión
    equivocada coincide con la buena en el elemento 0 por construcción.
    """
    t = usaf_like(N)
    columna, filas = columna_del_elemento(N, elemento)
    assert periodo_medido(t[filas, :], columna) == periodos_usaf(N)[elemento]
