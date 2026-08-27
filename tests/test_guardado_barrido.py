"""El guardado de la pila de foco de los tres scripts de retropropagación.

Los scripts de `scripts/retro_*.py` son autónomos a propósito: cada uno lleva
su propia copia de los ayudantes, sin importar CamposT, para servir de
contraste independiente del paquete. `nombre_png` y `guardar_png` son las dos
copias nuevas, y estas pruebas son el único sitio donde se miran a la vez.

Aquí hay CUATRO copias, no tres: `CamposT.retropropagacion.nombre_png` escribe
la pila de foco del paquete con el mismo formato de nombre. Si una de las
cuatro deriva, dos barridos del mismo objeto dejan de ordenarse igual y las
carpetas dejan de poder compararse imagen contra imagen, que es para lo único
que existen.

QUÉ SE COMPRUEBA, Y POR QUÉ CADA COSA

    nombre_png    Tres decimales y ancho fijo. Un `linspace` da distancias no
                  enteras: `z{z:04.0f}` —el formato de resultados/campos/—
                  haría que 49.6 y 50.2 escribieran los dos en z0050.png, y la
                  segunda reconstrucción pisaría a la primera sin aviso. El
                  ancho fijo es lo que mantiene el orden alfabético igual al
                  orden del barrido.

    guardar_png   Normaliza por el máximo, así que mide contraste y no brillo
                  absoluto: dos distancias del barrido son comparables aunque
                  la energía que llega a cada plano no sea la misma.

LA TRAMPA QUE JUSTIFICA UNA DE LAS PRUEBAS

`I / I.max()` con `I` idénticamente nula es 0/0: NaN por todo el array, y
`(NaN * 255).astype(uint8)` es basura, sin excepción y sin aviso. Es el fallo
que `CamposT.pipeline.guardar` ya documenta y arregla. Un plano negro es un
resultado legítimo —pasa al reconstruir un campo que se anuló— y hay que poder
verlo como tal en vez de recibir ruido.
"""

import matplotlib
matplotlib.use("Agg")           # antes de importar los scripts, que traen pyplot

import importlib

import numpy as np
import pytest
from PIL import Image

from CamposT.retropropagacion import nombre_png as nombre_png_paquete

#: Los tres scripts de ida y vuelta. La prueba corre entera sobre cada uno.
# scripts.retro_fft_angular ya no esta: se reescribio como un script minimo
# (angularSpectrum + figura) y estas funciones compartidas se fueron con el.
# Quedan las dos copias que si las tienen.
MODULOS = ("scripts.retro_mpasm", "scripts.retro_blas")


@pytest.fixture(params=MODULOS)
def mod(request):
    """Cada prueba se ejecuta una vez por script de retropropagación."""
    return importlib.import_module(request.param)


@pytest.fixture
def intensidad():
    """Mapa de intensidad con estructura, no ruido plano.

    Con un array constante cualquier normalización daría lo mismo y la prueba
    del máximo no distinguiría una implementación correcta de una que divide
    por la media.
    """
    y, x = np.mgrid[0:32, 0:48]
    return (np.sin(x / 4.0) ** 2 * np.cos(y / 6.0) ** 2).astype(float)


# ------------------------------------------------------------------ nombre_png
def test_nombre_png_lleva_tres_decimales_y_ancho_fijo(mod):
    """El formato es z{z:08.3f}.png. Ver el docstring del módulo."""
    assert mod.nombre_png(50.0) == "z0050.000.png"
    assert mod.nombre_png(7.25) == "z0007.250.png"


def test_nombre_png_no_colisiona_sobre_las_distancias_de_un_barrido(mod):
    """El motivo de los tres decimales: un linspace no da distancias enteras.

    Con el formato entero de resultados/campos/ este barrido produciría
    nombres repetidos y las reconstrucciones se pisarían en silencio.
    """
    zs = np.linspace(0.4, 1.6, 25) * 50.0
    nombres = [mod.nombre_png(z) for z in zs]
    assert len(set(nombres)) == len(zs)


def test_nombre_png_ordena_alfabeticamente_igual_que_el_barrido(mod):
    """El ancho fijo es lo que hace que `ls` enseñe la pila en orden."""
    zs = np.linspace(5.0, 150.0, 40)
    nombres = [mod.nombre_png(z) for z in zs]
    assert nombres == sorted(nombres)


# ----------------------------------------------------------------- guardar_png
def test_un_campo_nulo_escribe_un_png_negro_y_no_ruido(mod, tmp_path):
    """0/0 daba NaN por todo el array. Ver el docstring del módulo."""
    destino = tmp_path / "nulo.png"
    mod.guardar_png(np.zeros((16, 16)), destino)
    assert np.array_equal(np.asarray(Image.open(destino)),
                          np.zeros((16, 16), dtype=np.uint8))


def test_guardar_png_normaliza_por_el_maximo(mod, intensidad, tmp_path):
    """Escalar la intensidad no cambia la imagen: mide contraste, no brillo."""
    uno, otro = tmp_path / "uno.png", tmp_path / "otro.png"
    mod.guardar_png(intensidad, uno)
    mod.guardar_png(37.5 * intensidad, otro)
    assert uno.read_bytes() == otro.read_bytes()


def test_guardar_png_escribe_la_intensidad_sin_gamma(mod, intensidad, tmp_path):
    """|U|² / max, LINEAL y sin gamma.

    Tenía `gamma = 0.5` con la justificación de parecerse a los paneles de las
    figuras. Pero los paneles aplicaban la misma gamma, y

        (|U|² / max)^0.5  =  |U| / sqrt(max)

    o sea que ninguna de las dos enseñaba intensidad: las dos enseñaban
    AMPLITUD, con la etiqueta |U|² puesta. Ahora las dos son |U|² y se siguen
    pareciendo entre sí, que era el requisito de verdad.

    La división por el máximo se queda: es lineal, no cambia la relación entre
    valores, y un PNG de 8 bits no admite floats.
    """
    destino = tmp_path / "intensidad.png"
    mod.guardar_png(intensidad, destino)
    esperado = (intensidad / intensidad.max() * 255).astype(np.uint8)
    assert np.array_equal(np.asarray(Image.open(destino)), esperado)


def test_guardar_png_no_escribe_la_amplitud(mod, intensidad, tmp_path):
    """Y dicho al revés, que es lo que impide que la gamma vuelva sola.

    La prueba de arriba fija lo que SÍ se escribe; ésta fija lo que NO. Sin
    ella, alguien que reintrodujera `** 0.5` tendría que ver fallar una prueba
    cuyo nombre habla de intensidad, y podría pensar que la que sobra es la
    prueba.
    """
    destino = tmp_path / "no_amplitud.png"
    mod.guardar_png(intensidad, destino)
    amplitud = ((intensidad / intensidad.max()) ** 0.5 * 255).astype(np.uint8)
    assert not np.array_equal(np.asarray(Image.open(destino)), amplitud)


def test_guardar_png_crea_la_carpeta_que_falte(mod, intensidad, tmp_path):
    """El barrido escribe en resultados/reconstruccion/<objeto>/<metodo>/<A|B>/,
    que no existe antes de la primera corrida."""
    destino = tmp_path / "reconstruccion" / "obj" / "blas" / "A" / "z0050.000.png"
    mod.guardar_png(intensidad, destino)
    assert destino.is_file()


# ------------------------------------------------- las copias, todas a la vez
def test_nombre_png_coincide_en_las_cuatro_copias():
    """Tres scripts más el del paquete. Ver el docstring del módulo."""
    zs = np.linspace(0.4, 1.6, 25) * 50.0
    mods = [importlib.import_module(m) for m in MODULOS]
    for z in zs:
        valores = [m.nombre_png(z) for m in mods] + [nombre_png_paquete(z)]
        assert valores == [valores[0]] * len(valores)


def test_los_tres_guardar_png_dan_los_mismos_bytes(intensidad, tmp_path):
    """El riesgo real de la arquitectura autónoma: alguien arregla la gamma en
    uno de los tres y los otros dos siguen escribiendo la vieja."""
    mods = [importlib.import_module(m) for m in MODULOS]
    salidas = []
    for k, m in enumerate(mods):
        destino = tmp_path / f"{k}.png"
        m.guardar_png(intensidad, destino)
        salidas.append(destino.read_bytes())
    assert salidas == [salidas[0]] * len(salidas)


# ------------------------------------------------------------ carpeta_barrido
def test_carpeta_barrido_separa_objeto_metodo_y_vuelta(mod):
    """resultados/reconstruccion/<objeto>/<metodo>/<A|B>/, bajo la raíz del repo.

    El <objeto> no es decoración: sin él, correr con entrada.png y después con
    BenchmarkTarget.png a la misma Z escribe los mismos nombres en la misma
    carpeta y la segunda pila pisa a la primera en silencio.
    """
    carpeta = mod.carpeta_barrido("entrada", "blas", "A")
    assert carpeta.parts[-5:] == ("resultados", "reconstruccion", "entrada",
                                  "blas", "A")


def test_carpeta_barrido_no_depende_del_directorio_de_invocacion(mod):
    """Las salidas van bajo la raíz del repo se lance el script desde donde se
    lance: una ruta relativa las dejaría en el directorio de invocación."""
    carpeta = mod.carpeta_barrido("obj", "fft", "B")
    assert carpeta.is_absolute()
    assert (carpeta.parents[4] / "CamposT").is_dir()


def test_las_tres_carpeta_barrido_dan_la_misma_ruta():
    """Que las tres copias no se separen en dónde escriben."""
    mods = [importlib.import_module(m) for m in MODULOS]
    rutas = [m.carpeta_barrido("obj", "mpasm", "B") for m in mods]
    assert rutas == [rutas[0]] * len(rutas)
