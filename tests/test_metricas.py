"""Verificación de la métrica SAM (simulation accuracy metric).

Tarea 14 del cronograma (semana 6). SAM es la métrica con la que el paper
construye su Figura 4, así que reproducirla es condición para reproducir la
figura; una métrica "parecida" produce una figura que no se puede contrastar
contra la publicada.

Definición, Ecs. (15)-(16) de Zhao et al., Opt. Lett. 45, 5937 (2020):

    SAM = 10·lg[ ∫∫ I dxdy / ∫∫ |I − α·I_ref| dxdy ]        (15)
    α   = ∫∫ I·I*_ref dxdy / ∫∫ I_ref dxdy                   (16)

ERRATA EN LA EC. (16). El denominador impreso es ∫I_ref; tiene que ser
∫I_ref². Tres argumentos, cualquiera de ellos suficiente:

- Dimensiones: α multiplica a I_ref, así que es adimensional. ∫I·I_ref/∫I_ref
  tiene unidades de intensidad. Con ∫I_ref² sí es adimensional.
- Caso trivial: con la fórmula impresa, un campo idéntico a la referencia da
  α = 0.081 y SAM = 0.365 dB en vez de α = 1 y SAM = ∞.
- Poder de discriminación: la impresa deja el SAM clavado bajo 2.5 dB para
  todos los métodos y todas las distancias, así que no podría producir la
  Figura 4 del propio paper. La corregida separa MPASM de FFT-ASM por 27 dB
  a z = 12000 mm.

Es la segunda errata del paper, después del A²·B de la Ec. (14) en su código.
Se implementa la corregida; sam(..., formula="literal") reproduce la impresa,
misma convención que kf_auto(..., formula="codigo").

Tres consecuencias de la definición que estas pruebas fijan:

1. Va sobre INTENSIDADES, no amplitudes, y el resultado está en decibelios:
   más alto es mejor. La métrica que había antes en propagadores.py era un
   RMS de amplitudes normalizadas, donde más bajo es mejor. Son escalas
   opuestas, así que confundirlas invierte la lectura de cualquier gráfica.

2. El dxdy se cancela: Ec. (15) es un cociente de dos integrales con la misma
   medida. Por eso sam() no recibe el paso de píxel.

3. α es un factor de escala ajustado, así que SAM no puede detectar un error
   de normalización global. Lo que sí detecta es la forma del campo. La
   comprobación de la escala absoluta es otra prueba, no esta.
"""

import numpy as np
import pytest

from conftest import DELTA, LAMB, N, W0

from CamposT.metricas import alfa_sam, sam
from CamposT.propagadores import fft_asm, mpasm
from CamposT.referencias import gauss_analytic


def campo_de_intensidad(I):
    """Campo cuya intensidad |U|² es I. Las Ecs. (15)-(16) se escriben sobre
    intensidades, pero los propagadores devuelven campos complejos."""
    return np.sqrt(I).astype(complex)


#: SAM y α del caso `mitad_y_mitad`, calculados a mano (ver el fixture)
ALFA_ESPERADA = 14 / 5
SAM_ESPERADO = 10 * np.log10(8 / 1.2)
ALFA_LITERAL_ESPERADA = 14 / 3


@pytest.fixture
def mitad_y_mitad():
    """Caso con SAM y α calculables a mano.

    La referencia NO es constante, a propósito: con I_ref uniforme las dos
    versiones de la Ec. (16) coinciden y el caso no distinguiría la impresa
    de la corregida.

        I_ref = 1 en media imagen,  2 en la otra media
        I     = 2 en media imagen,  6 en la otra media

        α   = ∫I·I_ref / ∫I_ref² = (2·1 + 6·2)/(1² + 2²) = 14/5 = 2.8
        ∫|I − α·I_ref| ∝ |2 − 2.8| + |6 − 5.6| = 0.8 + 0.4 = 1.2
        ∫I            ∝ 2 + 6 = 8
        SAM = 10·lg(8/1.2) = 8.2391 dB

    Con el denominador impreso, α valdría (2·1 + 6·2)/(1 + 2) = 14/3.
    """
    n = 32
    I, I_ref = np.full((n, n), 2.0), np.ones((n, n))
    I[n // 2:, :], I_ref[n // 2:, :] = 6.0, 2.0
    return campo_de_intensidad(I), campo_de_intensidad(I_ref)


# ------------------------------------------------------- la fórmula del paper
def test_sam_reproduce_el_valor_calculado_a_mano(mitad_y_mitad):
    U, ref = mitad_y_mitad
    assert sam(U, ref) == pytest.approx(SAM_ESPERADO, abs=1e-9)


def test_alfa_es_el_ajuste_por_minimos_cuadrados(mitad_y_mitad):
    U, ref = mitad_y_mitad
    assert alfa_sam(U, ref) == pytest.approx(ALFA_ESPERADA, abs=1e-9)


def test_alfa_literal_reproduce_la_ecuacion_16_tal_como_esta_impresa(mitad_y_mitad):
    """La errata queda accesible, no borrada: si alguien contrasta contra la
    Figura 4 publicada y no cuadra, tiene que poder probar las dos."""
    U, ref = mitad_y_mitad
    assert alfa_sam(U, ref, formula="literal") == pytest.approx(
        ALFA_LITERAL_ESPERADA, abs=1e-9)


def test_alfa_vale_uno_cuando_el_campo_coincide_con_la_referencia(mitad_y_mitad):
    """El caso trivial que delata la errata: α es el factor que mejor ajusta
    la referencia al campo, así que comparar algo consigo mismo da 1."""
    _, ref = mitad_y_mitad
    assert alfa_sam(ref, ref) == pytest.approx(1.0, abs=1e-12)
    assert alfa_sam(ref, ref, formula="literal") != pytest.approx(1.0, abs=1e-3)


def test_sam_de_un_campo_consigo_mismo_es_infinito():
    """El residuo de la Ec. (15) se anula y el cociente diverge. Es el caso
    que aparece en cuanto alguien compara un campo con él mismo, así que no
    puede ser un ZeroDivisionError ni un nan."""
    U = campo_de_intensidad(np.random.default_rng(0).random((16, 16)) + 0.5)
    assert sam(U, U) == np.inf


# ------------------------------------------------- qué mide y qué no mide SAM
def test_sam_es_invariante_a_la_escala_del_campo(mitad_y_mitad):
    """Para eso está α: la Ec. (16) reescala la referencia al campo medido,
    así que multiplicar el campo por una constante no cambia el resultado."""
    U, ref = mitad_y_mitad
    assert sam(1000 * U, ref) == pytest.approx(sam(U, ref), rel=1e-9)


def test_sam_es_invariante_a_la_escala_de_la_referencia(mitad_y_mitad):
    """Con α de mínimos cuadrados la métrica no depende de en qué unidades
    venga la referencia, que es lo que se espera de una medida de exactitud.
    Con la Ec. (16) impresa esto no se cumple: otra señal de la errata."""
    U, ref = mitad_y_mitad
    assert sam(U, 2 * ref) == pytest.approx(sam(U, ref), rel=1e-9)
    assert sam(U, 2 * ref, formula="literal") != pytest.approx(
        sam(U, ref, formula="literal"), rel=1e-3)


def test_sam_baja_cuando_el_campo_se_degrada():
    """Más alto = mejor. Es la convención opuesta a la del RMS que había
    antes, y de ella depende cómo se lee la Figura 4."""
    rng = np.random.default_rng(1)
    ref = campo_de_intensidad(np.ones((32, 32)))
    base = np.ones((32, 32))
    sams = [sam(campo_de_intensidad(base + ruido * rng.standard_normal((32, 32))),
                ref)
            for ruido in (0.01, 0.05, 0.20)]
    assert sams[0] > sams[1] > sams[2]


# ------------------------------------------------------ uso físico: Figura 4
@pytest.mark.parametrize("z", (12000, 80000))
def test_mpasm_saca_mejor_sam_que_fft_asm_donde_fft_asm_aliasa(z, campo, malla):
    """La comprobación que justifica MPASM, ahora en la métrica del paper.

    Es el contenido de la Figura 4: a z grande FFT-ASM alías porque el haz
    no cabe en la ventana, y MPASM con sobremuestreo mantiene la exactitud.
    """
    X, Y = malla
    ref = gauss_analytic(X, Y, W0, LAMB, z)
    sam_mpasm = sam(mpasm(campo, DELTA, LAMB, z, s=4, device="cpu")[0], ref)
    sam_fft = sam(fft_asm(campo, DELTA, LAMB, z, device="cpu"), ref)
    assert sam_mpasm > sam_fft + 10.0, (
        f"z={z}: MPASM {sam_mpasm:.1f} dB vs FFT-ASM {sam_fft:.1f} dB")


@pytest.mark.gpu
def test_sam_acepta_campos_que_viven_en_la_gpu(campo):
    """La métrica se evalúa en CPU y en float64: es una reducción sobre todo
    el plano, donde complex64 acumularía error sin ganar nada."""
    from CamposT.backend import gpu_disponible
    if not gpu_disponible():
        pytest.skip("sin GPU")
    U_gpu, _ = mpasm(campo, DELTA, LAMB, 12000, s=2, device="gpu")
    U_cpu, _ = mpasm(campo, DELTA, LAMB, 12000, s=2, device="cpu")
    ref = gauss_analytic(*np.meshgrid((np.arange(N) - N / 2) * DELTA,
                                      (np.arange(N) - N / 2) * DELTA),
                         W0, LAMB, 12000)
    assert sam(U_gpu, ref) == pytest.approx(sam(U_cpu, ref), rel=1e-3)
