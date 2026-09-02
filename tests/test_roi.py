"""Verificación del recorte de una ventana de la imagen.

Lo que se protege aquí no es aritmética de índices, son dos DECISIONES:

  · Una ROI que se sale de la malla revienta en vez de ajustarse al borde.
    Ajustar en silencio devolvería una ventana con la forma pedida y el
    contenido equivocado: un resultado creíble y falso. Misma familia que
    SinMedir en montaje.py y _comprobar_ventana() en propagadores.py.

  · El recorte es SECO, sin margen de guarda, y eso cuesta. El coste está
    medido en test_recortar_y_propagar_no_es_propagar_y_recortar, que es la
    prueba que justifica el resto del módulo.
"""

import numpy as np
import pytest

from CamposT.roi import Roi


# --- geometría ---------------------------------------------------------------
def test_recortar_devuelve_exactamente_la_ventana_pedida():
    """La forma es (alto, ancho): fila primero, como cualquier array 2D."""
    U = np.arange(64, dtype=float).reshape(8, 8)
    recorte = Roi(x0=2, y0=3, ancho=4, alto=2).recortar(U)
    assert recorte.shape == (2, 4)
    assert np.array_equal(recorte, U[3:5, 2:6])


def test_salirse_de_la_malla_revienta_en_vez_de_ajustarse():
    """Ajustar al borde devolvería otra ventana con la forma correcta."""
    U = np.zeros((8, 8))
    with pytest.raises(ValueError, match="se sale"):
        Roi(6, 0, 4, 4).recortar(U)
    with pytest.raises(ValueError, match="se sale"):
        Roi(0, 6, 4, 4).recortar(U)


def test_recortar_un_array_que_no_es_2d_falla():
    with pytest.raises(ValueError, match="2D"):
        Roi(0, 0, 2, 2).recortar(np.zeros((4, 4, 3)))


@pytest.mark.parametrize("ancho,alto", [(1, 4), (4, 1), (0, 4), (4, 0)])
def test_una_ventana_de_menos_de_dos_pixeles_revienta(ancho, alto):
    with pytest.raises(ValueError, match="no da para propagar"):
        Roi(0, 0, ancho, alto)


def test_un_origen_negativo_revienta():
    with pytest.raises(ValueError, match="negativo"):
        Roi(-1, 0, 4, 4)


def test_coordenadas_no_enteras_revientan():
    """Redondear por su cuenta decidiría si la ventana contiene lo que
    marcaste o se queda por dentro, y esa decisión no es del constructor."""
    with pytest.raises(TypeError, match="enteros"):
        Roi(0.5, 0, 4, 4)


def test_una_coordenada_de_numpy_vale_y_se_guarda_como_int():
    """elegir() y argparse pueden entregar enteros de NumPy; no son un error,
    pero dentro se guardan como int para que el repr sea legible."""
    roi = Roi(np.int64(2), np.int32(3), 4, 5)
    assert (roi.x0, roi.y0) == (2, 3)
    assert type(roi.x0) is int


def test_la_roi_es_inmutable():
    """Como TABLA1: una ROI que cambia a mitad de corrida es un recorte que no
    se puede anotar."""
    roi = Roi(0, 0, 4, 4)
    with pytest.raises(Exception):
        roi.x0 = 5


# --- reproducibilidad --------------------------------------------------------
def test_como_argumento_da_la_linea_que_repite_el_recorte():
    assert Roi(312, 208, 256, 256).como_argumento() == "--roi 312 208 256 256"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
