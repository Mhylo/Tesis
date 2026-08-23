"""Verificación de la orquestación: relleno, recorte, despacho y escritura.

pipeline.py es la puerta de entrada del paquete —es lo que llaman los scripts,
la CLI de retropropagación y el propio README— y hasta ahora se comprobaba
sólo de refilón, a través de las pruebas de los propagadores y de la
retropropagación. Lo que se verifica aquí no es la física, que ya tiene sus
suites, sino el contrato de la capa que la envuelve: que el campo que sale
tiene el tamaño que se prometió, que el relleno de ceros va donde debe, que
info dice lo que de verdad se usó, y que escribir un PNG no inventa píxeles.
"""

import numpy as np
import pytest
from conftest import DELTA, LAMB, N, error_relativo
from PIL import Image

from CamposT.backend import a_numpy, dtype_por_defecto, get_xp
from CamposT.pipeline import (crop_center, diagnostico, guardar, intensidad,
                              pad_field, propagar)

Z = 500.0        # mm, dentro del campo cercano de la malla de conftest


# --- relleno y recorte -------------------------------------------------------
def test_pad_field_deja_el_campo_centrado_y_ceros_alrededor():
    """El relleno existe para separar el campo de los bordes de la convolución
    circular: si no queda centrado, no cumple su función."""
    U0 = np.arange(16, dtype=complex).reshape(4, 4) + 1.0
    Up = pad_field(U0, 2)

    assert Up.shape == (8, 8)
    assert np.array_equal(Up[2:6, 2:6], U0)
    borde = np.ones((8, 8), dtype=bool)
    borde[2:6, 2:6] = False
    assert np.all(Up[borde] == 0)


def test_pad_field_rellena_en_el_dispositivo_del_campo(device):
    """El relleno se construye donde vive el campo.

    Rellenar siempre en CPU obligaba a bajar el campo de la tarjeta y volver a
    subirlo para no hacer nada más que ponerle ceros alrededor. Lo que se exige
    aquí es lo único que importa para la corrección: que el resultado no
    dependa de dónde se haya hecho.
    """
    xp, dev = get_xp(device)
    U0 = np.arange(16, dtype=complex).reshape(4, 4) + 1.0

    Up = pad_field(xp.asarray(U0), 2, xp)

    assert type(Up) is type(xp.asarray(U0)), "el relleno cambió de dispositivo"
    assert np.array_equal(a_numpy(Up), pad_field(U0, 2))


def test_crop_center_deshace_el_relleno():
    """Recortar lo rellenado tiene que devolver exactamente lo de partida, o el
    campo saldría desplazado medio píxel y nadie lo notaría."""
    U0 = np.arange(36, dtype=complex).reshape(6, 6)
    assert np.array_equal(crop_center(pad_field(U0, 3), 6, 6), U0)


# --- despacho ----------------------------------------------------------------
@pytest.mark.parametrize("metodo", ["fft", "blas", "mpasm"])
def test_propagar_devuelve_la_malla_de_entrada(metodo, campo, device):
    """Con la malla de salida sin tocar, el campo vuelve del tamaño que entró:
    el relleno es un detalle interno y no debe asomar en la salida."""
    kw = {"s": 1} if metodo == "mpasm" else {}
    Uz, _ = propagar(campo, DELTA, LAMB, Z, metodo=metodo, pad=2,
                     device=device, **kw)
    assert Uz.shape == (N, N)


def test_propagar_no_recorta_cuando_la_malla_de_salida_cambia(campo, device):
    """Con r > 1 la salida es OTRA malla, no la de entrada rellenada: recortarla
    al tamaño de entrada tiraría justo la parte que se pidió calcular."""
    Uz, _ = propagar(campo, DELTA, LAMB, Z, metodo="mpasm", pad=2, r=2, s=2,
                     device=device)
    assert Uz.shape == (2 * 2 * N, 2 * 2 * N)   # r=2 sobre la malla rellenada


def test_propagar_informa_de_lo_que_uso(campo, device):
    """info no es decorativo: es lo que acaba en el nombre de los archivos y en
    las tablas del documento, así que tiene que decir lo que se usó de verdad."""
    Uz, info = propagar(campo, DELTA, LAMB, Z, metodo="mpasm", pad=2, s=2,
                        device=device)

    assert info["device"] == device
    assert info["dtype"] == np.dtype(dtype_por_defecto(device)).name
    assert info["s"] == 2
    assert info["Kf"] >= 1.0, "Kf comprime, nunca expande"


def test_propagar_con_metodo_desconocido_falla(campo):
    """Un método mal escrito tiene que fallar, no caer en un defecto silencioso."""
    with pytest.raises(ValueError):
        propagar(campo, DELTA, LAMB, Z, metodo="kreuzer", device="cpu")


def test_propagar_sin_relleno_es_el_propagador_desnudo(campo, device, tol):
    """pad=1 no debe añadir nada: es la vía de escape para comparar contra un
    propagador de fuera, que no rellena."""
    from CamposT.propagadores import fft_asm

    Uz, _ = propagar(campo, DELTA, LAMB, Z, metodo="fft", pad=1, device=device)
    directo = fft_asm(campo, DELTA, LAMB, Z, device=device)
    assert error_relativo(Uz, directo) == 0.0


# --- intensidad y escritura --------------------------------------------------
def test_intensidad_normaliza_solo_si_se_le_pide():
    """Sin normalizar es lo que hace falta para comparar dos campos entre sí;
    normalizada, lo que hace falta para mirarla."""
    U = np.array([[1 + 1j, 2 + 0j]], dtype=complex)
    assert np.allclose(a_numpy(intensidad(U, normalizar=False)), [[2.0, 4.0]])
    assert np.allclose(a_numpy(intensidad(U)), [[0.5, 1.0]])


def test_guardar_un_campo_nulo_escribe_un_negro(tmp_path):
    """Regresión: normalizar por el máximo daba 0/0 con un campo idénticamente
    nulo, o sea NaN por todo el array y un PNG de basura, sin error ni aviso.
    Un negro es un resultado legítimo -toda la señal fuera de la ventana- y
    hay que poder verlo como tal."""
    png = tmp_path / "nulo.png"
    guardar(np.zeros((8, 8)), png)

    assert np.all(np.asarray(Image.open(png)) == 0)


def test_guardar_lleva_el_maximo_a_blanco(tmp_path):
    """La imagen se normaliza al máximo del campo, así que el píxel más
    brillante es 255 pase lo que pase con la escala absoluta."""
    png = tmp_path / "rampa.png"
    I = np.linspace(0, 7, 8).reshape(1, 8) * 1e-9   # escala irrelevante

    guardar(I, png)

    A = np.asarray(Image.open(png))
    assert A.max() == 255 and A[0, 0] == 0


# --- criterios ---------------------------------------------------------------
def test_diagnostico_ve_estrecharse_la_banda_util_de_fft():
    """Es el número que decide si hace falta MPASM: la fracción de banda que
    FFT-ASM muestrea bien cae al alejarse, y Kf sube para compensarlo."""
    cerca = diagnostico(N, DELTA, LAMB, 500)
    lejos = diagnostico(N, DELTA, LAMB, 80000)

    assert lejos["fraccion_banda_util_FFT"] < cerca["fraccion_banda_util_FFT"]
    assert lejos["Kf_sugerido"] > cerca["Kf_sugerido"]
    assert 0 < lejos["NA_maxima_registrable"] <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
