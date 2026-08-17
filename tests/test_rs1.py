"""Verificación de la Rayleigh-Sommerfeld I como referencia numérica.

Tarea 12 del cronograma (semana 6). El paper calcula su I_ref con la integral
R-S y dice, sin más detalle: "To get the calculation result of the R−S integral
in short time, we reduce the 2-D calculation to 1-D".

Qué significa esa reducción
---------------------------
La R-S I completa, sin aproximaciones,

    U(P) = -(1/2π) ∬ U₀(Q) · (ik + 1/r) · (e^{ikr}/r) · (z/r) dS

cuesta O(N⁴): para cada punto del plano de salida hay que integrar sobre todo
el plano de entrada. Medido sobre rayleigh1Free de pyLHM, la malla 512×512 de
la Tabla 1 costaría entre una y dos horas.

Si el campo de entrada tiene simetría circular —el gaussiano de la Tabla 1 la
tiene— el integrando sólo depende de ρ, ρ' y del ángulo entre ellos, así que
en polares la integral se reduce a

    U(ρ) = -(1/2π) ∫ U₀(ρ') ρ' [ ∫₀^{2π} (ik + 1/r)(e^{ikr}/r)(z/r) dφ ] dρ'
    r = √(ρ² + ρ'² − 2ρρ'·cos φ + z²)

y cuesta O(N_ρ·N_ρ'·N_φ). No es una aproximación: es la misma integral escrita
en las coordenadas que respetan la simetría del problema. Por eso la prueba
que manda es la primera: contrastar contra la fuerza bruta 2-D de pyLHM, que
es código de terceros ya publicado y no comparte una sola línea con esta.

Por qué importa que sea R-S y no el gaussiano analítico
-------------------------------------------------------
La solución analítica del haz gaussiano (gauss_analytic) es paraxial. La R-S
no lo es. Como referencia, la R-S es la más fuerte de las dos, y la diferencia
entre ambas mide exactamente dónde deja de valer la aproximación paraxial.

Límite conocido: la reducción integra sobre el disco inscrito en la ventana,
no sobre el cuadrado completo. Para un campo confinado en ese disco la
diferencia es nula; si no lo está, rs1_radial avisa.
"""

import numpy as np
import pytest

from conftest import DELTA, L0, LAMB, N, W0

from CamposT.referencias import (gauss_analytic, gauss_beam, n_rho_auto,
                                 rs1_radial)


#: A w0 = 0.6 mm en una ventana de 5 mm el campo en el borde del disco
#: inscrito vale exp(-(2.5/0.6)²) = 3e-8, así que el disco y el cuadrado
#: contienen lo mismo y el contraste contra pyLHM (que integra sobre el
#: cuadrado) no arrastra sesgo por las esquinas. Y con zR = 1787 mm el haz
#: sigue cabiendo en la ventana a z = 4000 mm (w = 1.47 mm), así que tampoco
#: hay truncamiento que confundir con error de cuadratura.
W0_ESTRECHO = 0.6


def campo_estrecho(n, delta):
    x = (np.arange(n) - n / 2) * delta
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / W0_ESTRECHO**2).astype(complex), X, Y


def pylhm_rs1(U0, delta, lamb, z):
    """R-S I por fuerza bruta 2-D, de pyLHM. Devuelve None si no está el repo."""
    import os
    import sys
    raiz = os.path.join("referencia", "carlos", "DLHM-processing-tools-main",
                        "DLHM-processing-tools-main")
    if not os.path.isdir(raiz):
        return None
    if raiz not in sys.path:
        sys.path.insert(0, raiz)
    import pyLHM.myfunctions as LHM
    n = U0.shape[0]
    U, _ = LHM.reconstruct().rayleigh1Free(z, U0, lamb, [delta, delta],
                                           [delta, delta], (n, n))
    return U


# --------------------------------------------- contra una implementación ajena
def test_concuerda_con_la_rs_por_fuerza_bruta_de_pylhm():
    """Valida la FORMULACIÓN contra código de terceros ya publicado.

    pyLHM integra sobre el cuadrado completo, punto de salida a punto de
    salida, sin usar la simetría: no comparte una línea con esta
    implementación. Un error en el núcleo, en el factor de oblicuidad, en el
    prefactor -1/2π o en el signo del exponente daría una discrepancia de
    orden 1. Que queden en el 2 % dice que la formulación es la misma.

    NO es una prueba de exactitud, y la tolerancia no puede apretarse: el
    residuo es el error de discretización de pyLHM, no el de esta función.
    Sus nodos de cuadratura SON los píxeles de entrada, así que no puede
    refinarse; la versión radial sí, y converge dos órdenes por debajo (ver
    test_converge_al_refinar_la_cuadratura_radial). La exactitud se mide
    contra la solución analítica, no contra otra cuadratura.
    """
    n, delta, z = 64, L0 / 63, 4000.0
    U0, X, Y = campo_estrecho(n, delta)
    esperado = pylhm_rs1(U0, delta, LAMB, z)
    if esperado is None:
        pytest.skip("referencia/carlos/ no está; ver README")
    obtenido = rs1_radial(U0, delta, LAMB, z, device="cpu")

    # la cota no es un número elegido: es el error que comete la propia pyLHM,
    # medido aquí mismo contra la solución analítica
    ref = gauss_analytic(X, Y, W0_ESTRECHO, LAMB, z)

    def error_contra_analitica(U):
        a = np.abs(U)
        return float(np.max(np.abs(a / a.max() - ref / ref.max())))

    err_pylhm = error_contra_analitica(esperado)
    err_propio = error_contra_analitica(obtenido)
    discrepancia = np.max(np.abs(obtenido - esperado)) / np.max(np.abs(esperado))

    assert discrepancia < 3 * err_pylhm, (
        f"discrepancia {discrepancia:.2e} demasiado grande para explicarse "
        f"por la discretización de pyLHM ({err_pylhm:.2e}): apunta a una "
        f"diferencia de formulación, no de cuadratura")
    assert err_propio < err_pylhm / 5, (
        f"la reducción radial debería ser la más exacta de las dos: "
        f"propio {err_propio:.2e} vs pyLHM {err_pylhm:.2e}")


# ------------------------------------------------- propiedades de la reducción
def test_la_salida_conserva_la_simetria_circular():
    """Entra un campo con simetría de revolución, sale otro. Si la cuadratura
    angular estuviera mal montada, esto se rompe antes que nada."""
    n, delta = 48, L0 / 47
    U0, X, Y = campo_estrecho(n, delta)
    U = rs1_radial(U0, delta, LAMB, 2000.0, device="cpu")
    # la malla x = (arange(N) - N/2)*delta no es simétrica respecto al origen,
    # así que el espejo NO es U[::-1]; lo que sí debe cumplirse es que el campo
    # sea función únicamente del radio
    R = np.hypot(X, Y).ravel()
    _, inv = np.unique(np.round(R / delta, 9), return_inverse=True)
    Uf = U.ravel()
    media = (np.bincount(inv, weights=Uf.real)
             + 1j * np.bincount(inv, weights=Uf.imag)) / np.bincount(inv)
    assert np.max(np.abs(Uf - media[inv])) < 1e-10 * np.max(np.abs(Uf))


def test_rechaza_campos_sin_simetria_circular():
    """La reducción sólo vale bajo simetría de revolución. Un campo sin ella
    daría un resultado plausible y silenciosamente falso, así que se rechaza
    en la puerta en vez de devolver basura."""
    n, delta = 32, L0 / 31
    U0, X, Y = campo_estrecho(n, delta)
    U0_asimetrico = U0 * np.exp(-((X - 1.0) ** 2) / 4.0)
    with pytest.raises(ValueError, match="simetría circular"):
        rs1_radial(U0_asimetrico, delta, LAMB, 2000.0, device="cpu")


def test_converge_al_refinar_la_cuadratura_radial():
    """El refinamiento sucesivo tiene que dar cambios cada vez menores.

    Es la prueba de convergencia propiamente dicha, y la que justifica que
    n_rho_auto refine por encima del paso de píxel cuando la fase lo exige:
    la R-S suma en el espacio un núcleo que oscila como e^{ikr}, y el
    muestreo del sensor no tiene por qué resolverlo.
    """
    n, delta, z = 64, L0 / 63, 4000.0
    U0, _, _ = campo_estrecho(n, delta)
    campos = [rs1_radial(U0, delta, LAMB, z, n_rho_in=nr, n_phi=2048,
                         device="cpu")
              for nr in (32, 64, 128, 256)]
    cambios = [np.max(np.abs(b - a)) / np.max(np.abs(b))
               for a, b in zip(campos, campos[1:])]
    assert cambios[0] > cambios[1] > cambios[2], f"no converge: {cambios}"
    assert cambios[-1] < 1e-3, f"cambio final {cambios[-1]:.2e}"


def test_refina_la_cuadratura_cuando_el_paso_de_pixel_no_resuelve_la_fase():
    """A z corto el muestreo del sensor NO basta para la R-S.

    A z = 500 mm y N = 32, el paso de fase entre píxeles vecinos es de ~8 rad,
    muy por encima de Nyquist. n_rho_auto tiene que refinar por encima del
    píxel, y hacerlo tiene que acercar el resultado al convergido. Sin esta
    prueba la rama de refinamiento no se ejercita nunca: a z grande el paso de
    píxel ya cumple y max(n_pixel, ...) se queda con n_pixel.
    """
    n, delta, z = 32, L0 / 31, 500.0
    n_pixel = n // 2
    assert n_rho_auto(n_pixel * delta, delta, LAMB, z) > 4 * n_pixel, (
        "a esta distancia el criterio de fase debe mandar sobre el píxel")

    U0, _, _ = campo_estrecho(n, delta)
    convergido = rs1_radial(U0, delta, LAMB, z, n_rho_in=800, n_phi=4096,
                            device="cpu")

    def error(U):
        return float(np.max(np.abs(U - convergido)) / np.max(np.abs(convergido)))

    con_refinamiento = error(rs1_radial(U0, delta, LAMB, z, device="cpu"))
    sin_refinamiento = error(rs1_radial(U0, delta, LAMB, z, n_rho_in=n_pixel,
                                        device="cpu"))
    assert con_refinamiento < sin_refinamiento / 5, (
        f"refinar no mejoró: con {con_refinamiento:.2e}, sin "
        f"{sin_refinamiento:.2e}")


def test_converge_al_refinar_la_cuadratura_angular():
    """n_phi se elige solo a partir de la excursión de fase en φ. Doblarlo no
    debe cambiar el resultado: si lo cambia, el automático se queda corto."""
    n, delta = 64, L0 / 63
    U0, _, _ = campo_estrecho(n, delta)
    base = rs1_radial(U0, delta, LAMB, 4000.0, device="cpu")
    fino = rs1_radial(U0, delta, LAMB, 4000.0, n_phi=4096, device="cpu")
    assert np.max(np.abs(base - fino)) / np.max(np.abs(fino)) < 1e-4


# ------------------------------------------ contra la solución paraxial exacta
@pytest.mark.parametrize("z", (2500.0, 4000.0))
def test_sigue_al_gaussiano_analitico_donde_la_paraxial_vale(z):
    """A z grande frente a la cintura, R-S y la solución paraxial del haz
    gaussiano tienen que coincidir. Es la comprobación cruzada entre las dos
    referencias independientes del Objetivo 1."""
    n, delta = 128, L0 / 127
    U0, X, Y = campo_estrecho(n, delta)
    U = rs1_radial(U0, delta, LAMB, z, n_rho_in=128, device="cpu")
    ref = gauss_analytic(X, Y, W0_ESTRECHO, LAMB, z)
    amp = np.abs(U)
    error = np.sqrt(np.mean((amp / amp.max() - ref / ref.max()) ** 2))
    assert error < 1e-3, f"z={z}: RMS {error:.2e}"


@pytest.mark.gpu
def test_cpu_y_gpu_dan_el_mismo_campo():
    """Misma política que el resto del paquete: el cambio de dispositivo y de
    precisión no puede mover el resultado más de lo que impone complex64."""
    from CamposT.backend import gpu_disponible
    if not gpu_disponible():
        pytest.skip("sin GPU")
    n, delta = 32, L0 / 31
    U0, _, _ = campo_estrecho(n, delta)
    U_cpu = rs1_radial(U0, delta, LAMB, 4000.0, device="cpu")
    U_gpu = rs1_radial(U0, delta, LAMB, 4000.0, device="gpu")
    from CamposT.backend import a_numpy
    error = np.max(np.abs(a_numpy(U_gpu) - U_cpu)) / np.max(np.abs(U_cpu))
    assert error < 1e-5


# --------------------------------------------------------- el núcleo, aparte
def test_el_nucleo_tiende_a_su_forma_de_campo_lejano_cuando_kr_es_grande():
    """El término 1/r de (ik + 1/r) pesa 1/(kr) frente a ik.

    Se prueba aquí y no a través de la cuadratura porque en el único régimen
    donde el término es apreciable —r del orden de λ— el error de
    discretización de cualquier cuadratura lo tapa: medido contra la fuerza
    bruta de pyLHM a z = 2λ, quitar el término mueve el resultado 8e-4 sobre
    una discrepancia de fondo de 5.6e-2.
    """
    from CamposT.referencias import nucleo_rs1
    k = 2 * np.pi / LAMB
    for r, cota in ((1e6 * LAMB, 1e-6), (1e3 * LAMB, 1e-3)):
        completo = nucleo_rs1(np.array([r]), k, r)
        lejano = 1j * k * np.exp(1j * k * r) * (r / r**2)
        assert abs(completo[0] - lejano) / abs(lejano) < cota * 1.01


def test_el_termino_1_sobre_r_del_nucleo_cuenta_cuando_r_es_como_lambda():
    """La contrapartida de la prueba anterior: en campo cercano el término no
    es despreciable, y es lo que hace que esta sea la R-S exacta y no su
    aproximación de campo lejano."""
    from CamposT.referencias import nucleo_rs1
    k = 2 * np.pi / LAMB
    r = 2 * LAMB
    completo = nucleo_rs1(np.array([r]), k, r)
    lejano = 1j * k * np.exp(1j * k * r) * (r / r**2)
    assert abs(completo[0] - lejano) / abs(lejano) > 0.05
