"""Los ayudantes de fase de los tres scripts de retropropagación.

Los scripts de scripts/retro_*.py son autónomos a propósito: cada uno lleva su
propia copia de cargar_objeto, propagar, nitidez y parecido, sin importar
CamposT, para servir de contraste independiente del paquete. El precio de esa
decisión es que tres copias pueden divergir sin que nadie se entere, y estas
pruebas son el único sitio donde las tres se miran a la vez.

QUÉ SE COMPRUEBA, Y POR QUÉ CADA COSA

    sin_piston   Propagar multiplica el campo por una fase global -el exp(ikz)
                 y lo que arrastre la normalización-. Esa constante no es un
                 error de reconstrucción: es la elección de origen de fases.
                 Compararla contra el objeto sin quitarla mide una constante
                 irrelevante y estropea cualquier número que se publique.

    rms_fase     Error RMS de fase contra el objeto, sobre la máscara de
                 objeto brillante. El objeto es real y positivo, o sea fase 0,
                 así que su fase SÍ es un dato conocido con el que contrastar.

LA TRAMPA QUE JUSTIFICA LA MITAD DEL FICHERO

El pistón NO puede calcularse promediando ángulos. angle() devuelve valores en
(-pi, pi], así que un campo cuya fase real sea ~pi devuelve unos píxeles en
+3.14 y otros en -3.14, y la media aritmética de eso es ~0: el pistón sale
nulo justo cuando vale pi, y la fase se queda entera dentro del error.

La forma correcta es el ángulo del fasor MEDIO, sum(U), que no conoce la
discontinuidad porque nunca corta el círculo. test_el_piston_se_quita_aunque
_la_fase_caiga_sobre_la_discontinuidad es esa prueba, y es la que separa una
implementación correcta de una que funciona en los ejemplos amables.

Referencia del caso aleatorio: la RMS de una fase uniforme en (-pi, pi] es
pi/sqrt(3) = 1.8138 rad = 103.9 grados. Es el "cero de información" contra el
que se leen los números de las vueltas A y B.
"""

import matplotlib
matplotlib.use("Agg")           # antes de importar los scripts, que traen pyplot

import importlib

import numpy as np
import pytest

#: Los tres scripts de ida y vuelta. La prueba corre entera sobre cada uno.
MODULOS = ("scripts.retro_mpasm", "scripts.retro_fft_angular",
           "scripts.retro_blas")

#: RMS de una fase uniforme en (-pi, pi]. Ver el docstring.
RMS_UNIFORME = np.pi / np.sqrt(3)


@pytest.fixture(params=MODULOS)
def mod(request):
    """Cada prueba se ejecuta una vez por script de retropropagación."""
    return importlib.import_module(request.param)


@pytest.fixture
def objeto():
    """Campo real y positivo, o sea de fase 0, y su máscara de brillo.

    Real y positivo es lo que es un objeto de transmitancia: por eso su fase
    es conocida y sirve de referencia.
    """
    rng = np.random.default_rng(20260824)
    a = rng.uniform(0.4, 1.0, (48, 64))
    return a.astype(np.complex128), a > 0.5


# ------------------------------------------------------------------ sin_piston
def test_sin_piston_aplana_una_fase_global(mod, objeto):
    """Un campo con fase constante queda en fase 0 al quitarle el pistón."""
    U0, mask = objeto
    fase = mod.sin_piston(U0 * np.exp(1j * 2.0), mask)
    assert np.max(np.abs(fase[mask])) < 1e-9


def test_el_piston_se_quita_aunque_la_fase_caiga_sobre_la_discontinuidad(
        mod, objeto):
    """El caso pi: promediar ángulos da 0 donde el pistón vale pi.

    Con la fase repartida a los dos lados del corte de rama, angle() devuelve
    unos píxeles en +pi y otros en -pi. Una implementación que promedie esos
    ángulos saca pistón ~0 y deja la fase entera sin quitar: el max quedaría
    cerca de pi en vez de cerca del ruido que se metió.
    """
    U0, mask = objeto
    rng = np.random.default_rng(7)
    ruido = rng.uniform(-0.02, 0.02, U0.shape)
    fase = mod.sin_piston(U0 * np.exp(1j * (np.pi + ruido)), mask)
    assert np.max(np.abs(fase[mask])) < 0.05


# -------------------------------------------------------------------- rms_fase
def test_el_campo_identico_no_tiene_error_de_fase(mod, objeto):
    """rms_fase de un campo consigo mismo es 0, no un residuo pequeño."""
    U0, mask = objeto
    assert mod.rms_fase(U0, U0, mask) == pytest.approx(0.0, abs=1e-12)


def test_una_fase_global_no_cuenta_como_error(mod, objeto):
    """Es el punto de quitar el pistón: la constante no es un error.

    Sin esta invariancia, el mismo campo reconstruido daría números distintos
    según a qué distancia se propagase, que es exactamente lo que NO queremos
    medir.
    """
    U0, mask = objeto
    assert mod.rms_fase(U0 * np.exp(1j * 2.5), U0, mask) == pytest.approx(
        0.0, abs=1e-12)


def test_la_fase_aleatoria_da_el_rms_de_la_uniforme(mod, objeto):
    """El techo de la métrica: sin información de fase se llega a pi/sqrt(3).

    Fija la escala con la que se leen los resultados de las dos vueltas: un
    número muy por debajo de esto significa que la fase volvió de verdad.
    """
    U0, mask = objeto
    rng = np.random.default_rng(11)
    fase = rng.uniform(-np.pi, np.pi, U0.shape)
    assert mod.rms_fase(np.abs(U0) * np.exp(1j * fase), U0, mask) == (
        pytest.approx(RMS_UNIFORME, rel=0.10))


def test_el_error_de_fase_no_depende_de_la_amplitud(mod, objeto):
    """rms_fase mide fase, no brillo: escalar el campo no la mueve.

    Separa esta métrica de parecido(), que sí normaliza amplitudes. Las dos
    conviven en el mismo script y confundirlas invierte conclusiones.
    """
    U0, mask = objeto
    rng = np.random.default_rng(3)
    U = U0 * np.exp(1j * rng.uniform(-0.3, 0.3, U0.shape))
    assert mod.rms_fase(1e-4 * U, U0, mask) == pytest.approx(
        mod.rms_fase(U, U0, mask), rel=1e-9)


# ------------------------------------------------- las tres copias, a la vez
def test_las_tres_copias_dan_el_mismo_numero(objeto):
    """Los tres scripts duplican estos ayudantes; que no se separen.

    Es el riesgo real de la arquitectura autónoma: alguien arregla la fórmula
    en uno de los tres y los otros dos siguen publicando el número viejo.
    """
    U0, mask = objeto
    rng = np.random.default_rng(5)
    U = U0 * np.exp(1j * rng.uniform(-1.0, 1.0, U0.shape))

    mods = [importlib.import_module(m) for m in MODULOS]
    valores = [m.rms_fase(U, U0, mask) for m in mods]
    assert valores == pytest.approx([valores[0]] * len(valores), abs=0.0)

    fases = [m.sin_piston(U, mask) for m in mods]
    for otra in fases[1:]:
        assert np.array_equal(fases[0], otra)
