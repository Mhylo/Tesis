"""Lectura de los hologramas de Carlos.

DATOS es la carpeta con datos_amp/ y datos_phase/ descomprimidos. Se puede
cambiar con la variable de entorno HOLOGRAMAS_CARLOS.
"""

import os
import pathlib

import numpy as np
from PIL import Image

DATOS = pathlib.Path(os.environ.get(
    "HOLOGRAMAS_CARLOS", pathlib.Path.home() / "Downloads" / "hologramas_carlos"))

#: conjunto -> subcarpeta
CARPETA = {"amp": "datos_amp", "phase": "datos_phase"}

#: el foco de un objeto de amplitud es el MAXIMO de contraste de |t|; el de
#: uno de fase, el MINIMO (en el foco |t| = 1)
SIGNO = {"amp": "max", "phase": "min"}


def ruta(conjunto, k, captura, datos=None):
    return pathlib.Path(datos or DATOS) / CARPETA[conjunto] / f"z{int(k)}" / f"{captura}.jpg"


def leer(conjunto, k, captura, datos=None):
    """-> (holograma, ref) en float64, las cuentas de la camara tal cual."""
    h = np.asarray(Image.open(ruta(conjunto, k, captura, datos)), np.float64)
    r = np.asarray(Image.open(ruta(conjunto, k, "ref", datos)), np.float64)
    return h, r
