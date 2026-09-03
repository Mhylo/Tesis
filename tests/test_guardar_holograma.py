"""guardar_holograma() esta duplicada en los tres scripts retro_* A PROPOSITO
(decision D2 de docs/superpowers/specs/2026-09-03-extraer-holograma-design.md):
son el contraste INDEPENDIENTE de CamposT, y por eso no pueden importar el
paquete -si lo hicieran, coincidir con el dejaria de significar algo-.

El precio de esa independencia es que nada obliga a las tres copias a seguir
diciendo lo mismo. `tests/test_nitidez_foco.py::test_las_copias_dan_el_mismo_numero`
ya guarda esa esquina para nitidez(); esta prueba hace lo mismo para
guardar_holograma(), que es la copia mas nueva y la unica que escribe archivos
que otras herramientas -scripts/retro_holograma.py, sobre todo- consumen
despues. El riesgo es el mismo que documenta esa otra prueba: que alguien
arregle o cambie el formato en un script y se le olvide en los otros dos, y
el .txt/.png/.npy de uno deje de significar lo mismo que el de sus hermanos.
"""

import matplotlib
matplotlib.use("Agg")           # antes de importar los scripts, que traen pyplot

import importlib
import inspect

MODULOS = ("scripts.retro_mpasm", "scripts.retro_fft_angular", "scripts.retro_blas")


def test_las_copias_de_guardar_holograma_dan_el_mismo_codigo():
    """El riesgo de la arquitectura autonoma: que alguien lo arregle en uno."""
    fuentes = [inspect.getsource(importlib.import_module(m).guardar_holograma)
               for m in MODULOS]
    assert fuentes[0] == fuentes[1] == fuentes[2], (
        "guardar_holograma() diverge entre los tres scripts retro_*")
