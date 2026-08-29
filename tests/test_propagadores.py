"""Verificación de los propagadores: identidad en z=0, energía y reversibilidad.

Tarea 9 del cronograma (semana 5). Estas tres propiedades no requieren una
solución analítica: se cumplen o no por construcción del método, así que
distinguen un propagador correcto de uno que "da algo parecido".

Sobre qué se le exige a cada método
-----------------------------------
FFT-ASM es unitario por construcción: la FFT lo es y |H| = 1 salvo en las
ondas evanescentes, que en la malla de prueba no existen (lamb/(2·delta) =
0.004 << 1). De ahí que se le exija conservación de energía y reversibilidad
*exactas*.

BLAS y MPASM pierden energía a propósito. BLAS aplica una máscara de banda
limitada y MPASM comprime el intervalo espectral por Kf > 1; en ambos casos
se descarta espectro, y descartar espectro es irreversible. Exigirles
conservación de energía a todo z sería exigirles que no hagan lo que hacen.
Lo que sí se les exige es no *ganar* energía nunca, y ser exactos en el
régimen donde su aproximación está inactiva (Kf = 1, o z por debajo del
límite de banda).
"""

import numpy as np
import pytest

from conftest import (DELTA, L0, LAMB, N, W0, Z_CERCANO, Z_TODOS,
                      energia, error_relativo)

from CamposT.propagadores import (blas, fft_asm, frecuencias_fft, kf_auto,
                                  mpasm, transfer_function)
from CamposT.referencias import gauss_analytic, gauss_beam


def solo_campo(resultado):
    """mpasm devuelve (campo, Kf); fft_asm y blas sólo el campo."""
    return resultado[0] if isinstance(resultado, tuple) else resultado


def z_limite_blas(n, delta, lamb):
    """Distancia a la que la máscara de BLAS empieza a recortar espectro.

    El límite de banda es flim = 1/(lamb·sqrt((2z/(delta·N))² + 1)) y la malla
    llega hasta fmax = 1/(2·delta). Igualando ambos y despejando z:

        z_lim = delta·N / (2·lamb·fmax) = delta²·N / lamb

    Por debajo, la máscara no descarta nada y BLAS coincide con FFT-ASM; por
    encima, descarta, y lo descartado no vuelve.
    """
    return delta ** 2 * n / lamb


def cintura(w0, lamb, z):
    """Radio del haz gaussiano tras propagar z. zR = pi·w0²/lamb."""
    return w0 * np.sqrt(1 + (z / (np.pi * w0 ** 2 / lamb)) ** 2)


#: los tres propagadores bajo la firma común, con Kf y s neutralizados en
#: MPASM para que las tres funciones sean comparables término a término
PROPAGADORES = {
    "mpasm": lambda U, z, **kw: mpasm(U, DELTA, LAMB, z, s=1, Kf=1.0, **kw)[0],
    "fft_asm": lambda U, z, **kw: fft_asm(U, DELTA, LAMB, z, **kw),
    "blas": lambda U, z, **kw: blas(U, DELTA, LAMB, z, **kw),
}


# ---------------------------------------------------------------- caso z = 0
@pytest.mark.parametrize("nombre", list(PROPAGADORES))
def test_z_cero_devuelve_el_campo_de_entrada(nombre, campo, device, tol):
    """Propagar una distancia nula no puede cambiar el campo."""
    U = PROPAGADORES[nombre](campo, 0, device=device)
    assert error_relativo(U, campo) < tol


@pytest.mark.parametrize("s", [1, 2, 4])
def test_mpasm_en_z_cero_es_identidad_para_cualquier_sobremuestreo(s, campo):
    """El sobremuestreo cambia el coste, no el resultado en z = 0.

    Comprueba que la DFT matricial sobremuestreada y su inversa son un par
    exacto: si la normalización (s²·M·N·Kf²) estuviera mal, este es el primer
    sitio donde se vería.
    """
    U, Kf = mpasm(campo, DELTA, LAMB, 0, s=s, device="cpu")
    assert Kf == 1.0
    assert error_relativo(U, campo) < 1e-12


# ------------------------------------------------------------------- energía
@pytest.mark.parametrize("z", Z_TODOS)
def test_fft_asm_conserva_la_energia(z, campo, device, tol):
    """FFT-ASM es unitario: la FFT lo es y |H| = 1 en toda la malla."""
    U = fft_asm(campo, DELTA, LAMB, z, device=device)
    assert abs(energia(U) / energia(campo) - 1) < tol


@pytest.mark.parametrize("z", Z_TODOS)
def test_mpasm_conserva_la_energia_sin_compresion(z, campo, device, tol):
    """Con s = Kf = 1, MPASM es FFT-ASM escrito como producto matricial."""
    U, _ = mpasm(campo, DELTA, LAMB, z, s=1, Kf=1.0, device=device)
    assert abs(energia(U) / energia(campo) - 1) < tol


@pytest.mark.parametrize("nombre", list(PROPAGADORES))
@pytest.mark.parametrize("z", Z_TODOS)
def test_ningun_propagador_gana_energia(nombre, z, campo, device, tol):
    """Propagar en espacio libre no crea energía. Perderla es admisible
    (filtro de banda, compresión frecuencial); ganarla nunca lo es, y sería
    el síntoma de una normalización equivocada."""
    U = PROPAGADORES[nombre](campo, z, device=device)
    assert energia(U) / energia(campo) <= 1 + tol


def test_blas_pierde_energia_de_forma_monotona(campo):
    """El límite de banda de BLAS se estrecha al crecer z, así que la energía
    que deja pasar sólo puede decrecer. Un repunte indicaría que la máscara
    se está calculando con el signo o el eje cambiados."""
    E0 = energia(campo)
    fracciones = [energia(blas(campo, DELTA, LAMB, z, device="cpu")) / E0
                  for z in (0, 100, 500, 2000, 6000, 12000, 30000, 80000)]
    for anterior, siguiente in zip(fracciones, fracciones[1:]):
        assert siguiente <= anterior + 1e-12
    assert fracciones[-1] < 0.5, "a z grande el filtro debería recortar de verdad"


# ------------------------------------------------------------- reversibilidad
@pytest.mark.parametrize("z", Z_TODOS)
def test_fft_asm_es_reversible(z, campo, device, tol):
    """H(-z) = conj(H(z)) sin ondas evanescentes, luego propagar z y volver
    -z devuelve el campo original."""
    ida = fft_asm(campo, DELTA, LAMB, z, device=device)
    vuelta = fft_asm(ida, DELTA, LAMB, -z, device=device)
    assert error_relativo(vuelta, campo) < tol


@pytest.mark.parametrize("z", Z_TODOS)
def test_mpasm_es_reversible_sin_compresion(z, campo, device, tol):
    ida, _ = mpasm(campo, DELTA, LAMB, z, s=1, Kf=1.0, device=device)
    vuelta, _ = mpasm(ida, DELTA, LAMB, -z, s=1, Kf=1.0, device=device)
    assert error_relativo(vuelta, campo) < tol


@pytest.mark.parametrize("fraccion", [0.15, 0.5, 0.9])
def test_blas_es_reversible_mientras_el_filtro_no_actua(fraccion, campo, device, tol):
    """Por debajo de z_lim la máscara no descarta nada y BLAS coincide con
    FFT-ASM. El z se toma como fracción del límite en vez de a ojo, porque
    depende de la malla: con N=64 y L0=5 mm son 637 mm, y un z fijo de 2000
    caería ya en el otro régimen."""
    z = fraccion * z_limite_blas(campo.shape[0], DELTA, LAMB)
    ida = blas(campo, DELTA, LAMB, z, device=device)
    vuelta = blas(ida, DELTA, LAMB, -z, device=device)
    assert error_relativo(vuelta, campo) < tol


@pytest.mark.parametrize("fraccion", [1.1, 3.0, 20.0])
def test_blas_no_es_reversible_pasado_su_limite_de_banda(fraccion, campo):
    """Contrapartida de la prueba anterior, y comprobación de que z_lim marca
    de verdad el cruce: apenas un 10 % por encima, la vuelta ya no reconstruye
    el campo. Queda escrito como comportamiento esperado, no como un fallo por
    descubrir."""
    z = fraccion * z_limite_blas(campo.shape[0], DELTA, LAMB)
    ida = blas(campo, DELTA, LAMB, z, device="cpu")
    vuelta = blas(ida, DELTA, LAMB, -z, device="cpu")
    assert error_relativo(vuelta, campo) > 1e-6


# --------------------------------------------- MPASM en su régimen propio
# Las pruebas anteriores fijan s = Kf = 1 para poder comparar los tres métodos
# término a término, pero ese ajuste desactiva justamente lo que MPASM aporta.
# Lo que sigue lo ejercita con sobremuestreo y compresión activos.

@pytest.mark.parametrize("z", Z_TODOS)
def test_mpasm_con_compresion_no_gana_energia(z, campo, device, tol):
    """Con Kf > 1 la normalización lleva un factor Kf² (dos veces, ida en la
    DFT directa). Si ese factor se cayera, el campo saldría escalado por Kf²
    y la energía por Kf⁴: a z = 80000, Kf = 5.6, casi mil veces. Es la única
    prueba de la suite sensible a la escala absoluta en este régimen."""
    U, Kf = mpasm(campo, DELTA, LAMB, z, s=4, device=device)
    assert energia(U) / energia(campo) <= 1 + tol


def test_mpasm_activa_la_compresion_a_z_grande(campo):
    """Salvaguarda de la prueba anterior: si kf_auto dejara de comprimir, esos
    casos pasarían a ejercitar el régimen trivial sin que nadie se entere."""
    assert kf_auto(N, DELTA, LAMB, 500, s=4) == 1.0
    assert kf_auto(N, DELTA, LAMB, 80000, s=4) > 5


@pytest.mark.parametrize("z", [12000, 80000])
def test_mpasm_acierta_donde_fft_asm_aliasa(z, campo, malla):
    """El resultado que justifica el método. A estos z el haz ya no cabe en la
    ventana y FFT-ASM devuelve un campo irreconocible (RMS ~0.16 y ~0.88
    contra la solución analítica), mientras que MPASM con s = 4 y Kf
    automático sigue al gaussiano exacto con RMS < 1e-3 sobre la misma malla
    de entrada y sin ampliarla."""
    X, Y = malla
    assert cintura(W0, LAMB, z) > L0 / 2

    U_mpasm, Kf = mpasm(campo, DELTA, LAMB, z, s=4, device="cpu")
    U_fft = fft_asm(campo, DELTA, LAMB, z, device="cpu")

    assert Kf > 1, "el caso debe caer en el régimen comprimido"
    assert _rms_contra_analitico(U_mpasm, X, Y, z) < 1e-3
    assert _rms_contra_analitico(U_fft, X, Y, z) > 0.1


# ------------------------------------------------- equivalencia entre métodos
@pytest.mark.parametrize("z", Z_TODOS)
def test_mpasm_sin_sobremuestreo_coincide_con_fft_asm(z, campo, device, tol):
    """Lo que promete el docstring de mpasm(): con s = Kf = r = mag = 1 es
    FFT-ASM. Si esta prueba cae, los dos métodos han dejado de ser
    comparables y la tabla de tiempos del Objetivo 1 pierde sentido."""
    U_mpasm, _ = mpasm(campo, DELTA, LAMB, z, s=1, Kf=1.0, r=1, mag=1.0,
                       device=device)
    U_fft = fft_asm(campo, DELTA, LAMB, z, device=device)
    assert error_relativo(U_mpasm, U_fft) < tol


@pytest.mark.gpu
@pytest.mark.parametrize("nombre", list(PROPAGADORES))
@pytest.mark.parametrize("z", Z_TODOS)
def test_gpu_y_cpu_dan_el_mismo_campo(nombre, z, campo):
    """El mismo código en los dos dispositivos. La diferencia admisible es la
    de complex64 frente a complex128, no la de dos algoritmos distintos: si
    la política de fases en float64 de backend.py fallara, el error aquí se
    dispararía varios órdenes de magnitud."""
    pytest.importorskip("cupy")
    from CamposT.backend import gpu_disponible
    if not gpu_disponible():
        pytest.skip("sin GPU CUDA")

    U_cpu = PROPAGADORES[nombre](campo, z, device="cpu")
    U_gpu = PROPAGADORES[nombre](campo, z, device="gpu", dtype=np.complex64)
    assert error_relativo(U_gpu, U_cpu) < 1e-5


# ----------------------------------------------------------- malla de salida
def test_r_y_mag_controlan_la_malla_de_salida(campo):
    """r fija el número de puntos de salida y mag el paso, Ecs. (6)-(8).
    Es lo que permite reconstruir un plano más fino sin repropagar.

    El caso r = 2 va con s = 2 porque duplicar los puntos con mag = 1 duplica
    la ventana, y la ventana tiene que caber en el periodo del espectro:
    r·mag <= s·Kf. Con s = 1 esto devolvía la extensión periódica del campo
    en lugar del campo, y esta prueba no se enteraba porque sólo miraba la
    forma del array.
    """
    N_entrada = campo.shape[0]
    U, _ = mpasm(campo, DELTA, LAMB, 2000, s=2, Kf=1.0, r=2, device="cpu")
    assert U.shape == (2 * N_entrada, 2 * N_entrada)

    U, _ = mpasm(campo, DELTA, LAMB, 2000, s=1, Kf=1.0, mag=0.5, device="cpu")
    assert U.shape == (N_entrada, N_entrada)


# ------------------------------------------ ventana de salida contra periodo
@pytest.mark.parametrize("s, r, mag", [(1, 1, 2.0), (1, 2, 1.0), (4, 2, 4.0)])
def test_mpasm_aborta_si_la_ventana_no_cabe_en_el_periodo(s, r, mag, campo):
    """r·mag > s·Kf significa que la malla de salida abarca más de un periodo
    espacial del espectro, y el campo sale con copias de sí mismo encima.

    Antes no fallaba ni avisaba: devolvía un array de la forma pedida con el
    contenido equivocado (medido contra el gaussiano analítico, error
    relativo 1.0). Un fallo ruidoso es lo único aceptable aquí, porque el
    síntoma —un halo periódico en el borde— es fácil de confundir con
    difracción de verdad.
    """
    with pytest.raises(ValueError, match="periodo del espectro"):
        mpasm(campo, DELTA, LAMB, 2000, s=s, Kf=1.0, r=r, mag=mag, device="cpu")


def test_la_ventana_justa_es_legal_y_correcta(campo, malla):
    """El límite r·mag = s·Kf no es conservador: justo ahí la ventana cubre
    exactamente un periodo, no hay solape, y el campo sigue siendo el bueno.

    Se compara con el doble de holgura: si el límite estuviera mal puesto, los
    dos errores no coincidirían.
    """
    z = 2000
    justa, _ = mpasm(campo, DELTA, LAMB, z, s=2, Kf=1.0, mag=2.0, device="cpu")
    holgada, _ = mpasm(campo, DELTA, LAMB, z, s=4, Kf=1.0, mag=2.0, device="cpu")

    def rms(U):
        n = U.shape[0]
        xo = (np.arange(n) - n / 2) * DELTA * 2.0
        XO, YO = np.meshgrid(xo, xo)
        return _rms_contra_analitico(U, XO, YO, z)

    assert rms(justa) < 1e-3
    assert rms(justa) == pytest.approx(rms(holgada), rel=0.2)


# -------------------------------------------------------------------- kf_auto
def _gauss_no_cuadrado(M, N, delta, w0):
    """Gaussiano centrado en una malla M x N. gauss_beam sólo hace cuadradas."""
    x = (np.arange(N) - N / 2) * delta
    y = (np.arange(M) - M / 2) * delta
    X, Y = np.meshgrid(x, y)
    return np.exp(-(X**2 + Y**2) / w0**2).astype(complex), X, Y


def test_kf_se_calcula_por_eje_en_campos_no_cuadrados():
    """La Tabla 1 del paper lista K_fx y K_fy por separado.

    Kf sale como ~1/sqrt(N), así que el eje corto necesita MÁS compresión que
    el largo. Usar el de un solo eje para los dos submuestrea el corto en
    silencio: ni avisa ni falla, sólo alía. Y los sensores de DLHM no son
    cuadrados (1024x1280, 1200x1920), así que esto aparece en cuanto se
    procesen hologramas reales.
    """
    M, N, z = 32, 64, 12000
    U0, _, _ = _gauss_no_cuadrado(M, N, DELTA, W0)
    _, Kf = mpasm(U0, DELTA, LAMB, z, s=2, device="cpu")
    assert isinstance(Kf, tuple), "en malla no cuadrada Kf debe venir por eje"
    Kfy, Kfx = Kf
    assert Kfy == pytest.approx(kf_auto(M, DELTA, LAMB, z, s=2))
    assert Kfx == pytest.approx(kf_auto(N, DELTA, LAMB, z, s=2))
    assert Kfy > Kfx, "el eje corto necesita más compresión"


def test_en_malla_cuadrada_kf_sigue_siendo_un_escalar():
    """Los dos ejes coinciden, así que devolver la pareja sólo estorbaría."""
    U0, _, _ = _gauss_no_cuadrado(48, 48, DELTA, W0)
    _, Kf = mpasm(U0, DELTA, LAMB, 12000, s=2, device="cpu")
    assert isinstance(Kf, float)


def test_kf_por_eje_evita_el_aliasing_del_eje_corto():
    """La prueba física del arreglo, con un target USAF en malla rectangular.

    Hace falta contenido de alta frecuencia para que el aliasing aparezca: con
    un gaussiano suave las dos variantes dan lo mismo, porque no hay nada
    cerca del límite de banda que se pueda plegar. Por eso aquí se propaga un
    target de barras y no el haz de las demás pruebas.

    Referencia: el mismo campo con s alto, donde el muestreo en frecuencia es
    lo bastante fino como para que la compresión apenas actúe.
    """
    from CamposT.campos import usaf_like

    M, N, z, s = 64, 128, 150.0, 4
    delta, lamb = 3.45e-3, 405e-6          # sensor y láser del montaje DLHM
    U0 = usaf_like(N)[(N - M) // 2:(N + M) // 2, :].astype(complex)

    referencia, _ = mpasm(U0, delta, lamb, z, s=16, device="cpu")

    def error(U):
        a, b = np.abs(np.asarray(U)), np.abs(np.asarray(referencia))
        return float(np.sqrt(np.mean((a / a.max() - b / b.max()) ** 2)))

    por_eje, _ = mpasm(U0, delta, lamb, z, s=s, device="cpu")
    unico, _ = mpasm(U0, delta, lamb, z, s=s,
                     Kf=kf_auto(N, delta, lamb, z, s=s), device="cpu")

    assert error(por_eje) < error(unico) / 5, (
        f"por eje {error(por_eje):.2e} vs Kf único {error(unico):.2e}")


def test_transponer_el_campo_transpone_el_resultado():
    """Invariancia que no necesita ninguna referencia externa.

    Propagar U0 y propagar U0.T tienen que dar resultados transpuestos entre
    sí, porque el problema es el mismo con los ejes cambiados de nombre. Si
    K_fy y K_fx se aplicaran al eje equivocado, o si un solo Kf se usara para
    los dos, esta igualdad se rompe: el eje corto y el largo dejarían de
    tratarse como lo que son.

    Hace falta precisamente porque las pruebas que comparan contra un campo
    de referencia calculado con el mismo mpasm no lo verían: el error estaría
    también en la referencia.
    """
    M, N, z = 32, 64, 12000
    U0, _, _ = _gauss_no_cuadrado(M, N, DELTA, W0)
    directo, _ = mpasm(U0, DELTA, LAMB, z, s=2, device="cpu")
    transpuesto, _ = mpasm(np.ascontiguousarray(U0.T), DELTA, LAMB, z, s=2,
                           device="cpu")
    assert error_relativo(directo, np.asarray(transpuesto).T) < 1e-12


def test_la_amplitud_absoluta_es_correcta_en_malla_no_cuadrada():
    """Contrasta la ESCALA, no la forma.

    Todas las demás comparaciones normalizan por el máximo, así que un error
    global de normalización —por ejemplo Kfx² donde debería ir Kfx·Kfy— pasa
    inadvertido. Aquí se compara la amplitud absoluta contra el valor exacto
    del haz gaussiano, en un régimen donde el haz cabe holgadamente en la
    ventana corta y el truncamiento no falsea el máximo.
    """
    M, N, w0, z = 32, 64, 0.25, 1000
    U0, X, Y = _gauss_no_cuadrado(M, N, DELTA, w0)
    U, _ = mpasm(U0, DELTA, LAMB, z, s=4, device="cpu")
    esperado = gauss_analytic(X, Y, w0, LAMB, z).max()
    assert float(np.abs(np.asarray(U)).max()) == pytest.approx(esperado, rel=1e-3)


@pytest.mark.parametrize("z", Z_TODOS)
def test_kf_no_depende_del_signo_de_z(z):
    """La compresión la fija el ancho de banda de H, y H(-z) = conj(H(z)):
    mismo ancho, mismo Kf.

    Aplicando la Ec. (14) tal cual a z < 0 salía fmax < 0 y el max(1.0, ...)
    devolvía 1, o sea compresión apagada. Como retropropagacion.py propaga
    siempre a -z, eso dejaba a MPASM sin lo único que lo distingue de FFT-ASM
    en TODA reconstrucción, y en silencio.
    """
    assert kf_auto(N, DELTA, LAMB, -z, s=4) == kf_auto(N, DELTA, LAMB, z, s=4)


def test_mpasm_retropropaga_igual_de_bien_que_propaga():
    """La consecuencia física de lo anterior, sin referencia externa.

    Para cualquier propagador de espacio libre  U(-z) = conj(P(+z){conj(U)}),
    porque H(-z) = conj(H(z)) y conjugar el campo conjuga la transformada. Eso
    da una referencia para z < 0 construida por el camino z > 0, que es donde
    la Ec. (14) está escrita y donde se sabe que Kf funciona.

    Se propaga un target de barras, no el gaussiano de las demás pruebas:
    hace falta contenido de alta frecuencia para que la falta de compresión
    alíe. Con Kf apagado el error medido era 5.4, con Kf correcto es 1.1e-2.
    """
    from CamposT.campos import usaf_like

    n, delta, lamb, z, s = 128, 3.45e-3, 633e-6, 400.0, 4
    U0 = usaf_like(n).astype(complex)

    referencia = np.conj(np.asarray(
        mpasm(np.conj(U0), delta, lamb, z, s=s, device="cpu")[0]))
    vuelta, Kf = mpasm(U0, delta, lamb, -z, s=s, device="cpu")

    assert Kf > 1, "el caso debe caer en el régimen comprimido"
    assert error_relativo(vuelta, referencia) < 0.1


def test_kf_es_uno_en_z_cero():
    assert kf_auto(64, DELTA, LAMB, 0) == 1.0


@pytest.mark.parametrize("z", Z_TODOS)
def test_kf_nunca_baja_de_uno(z):
    """Kf < 1 significaría expandir el intervalo espectral en vez de
    comprimirlo, que no es lo que la Ec. (14) describe."""
    assert kf_auto(64, DELTA, LAMB, z) >= 1.0


def test_kf_crece_con_z():
    valores = [kf_auto(64, DELTA, LAMB, z) for z in (500, 2000, 12000, 80000)]
    assert valores == sorted(valores)


def test_la_formula_del_codigo_original_difiere_de_la_del_paper():
    """El código publicado escribe A**2 * B donde la Ec. (14) dice A**2 + B.
    Esta prueba fija la discrepancia por escrito: si alguien "arregla"
    kf_auto igualando las dos ramas, aquí se entera."""
    z = 12000
    del_paper = kf_auto(64, DELTA, LAMB, z, s=10, formula="paper")
    del_codigo = kf_auto(64, DELTA, LAMB, z, s=10, formula="codigo")
    assert del_paper != del_codigo


# ------------------------------------------------- función de transferencia
def test_la_transferencia_tiene_modulo_uno_en_las_ondas_propagantes():
    """|H| = 1 es la razón de que FFT-ASM sea unitario; se comprueba aparte
    para que un fallo señale la causa y no sólo el síntoma."""
    fx = np.linspace(-0.5, 0.5, 65) / LAMB * 0.9   # dentro del círculo |lamb·f| < 1
    H = transfer_function(fx, fx, LAMB, 2000, np, np.complex128)
    assert np.allclose(np.abs(H), 1.0, atol=1e-12)


def test_la_transferencia_anula_las_ondas_evanescentes():
    """Fuera del círculo lamb²(fx²+fy²) > 1 la onda no propaga y H debe ser
    exactamente cero, no un número grande."""
    delta_fino = 1e-4                    # mm; 1/(2·delta) > 1/lamb
    N_fino = 64
    fx = (np.arange(N_fino) - N_fino / 2) / (delta_fino * N_fino)
    H = transfer_function(fx, fx, LAMB, 2000, np, np.complex128)
    FX, FY = np.meshgrid(fx, fx)
    evanescentes = (LAMB * FX) ** 2 + (LAMB * FY) ** 2 >= 1.0
    assert evanescentes.any(), "la malla de prueba no contiene evanescentes"
    assert np.all(H[evanescentes] == 0)


# ------------------------------------------------- referencia analítica
def _rms_contra_analitico(U, X, Y, z):
    ref = gauss_analytic(X, Y, W0, LAMB, z)
    amplitud = np.abs(np.asarray(U))
    return float(np.sqrt(np.mean((amplitud / amplitud.max() - ref / ref.max()) ** 2)))


@pytest.mark.parametrize("z", Z_CERCANO)
def test_fft_asm_sigue_al_gaussiano_analitico(z, campo, malla):
    """Contraste contra la solución exacta del haz gaussiano. No es una
    propiedad estructural como las anteriores sino exactitud física, y es el
    puente hacia la tarea 13 (curvas de error).

    Sólo se exige mientras el haz propagado cabe en la ventana; el caso
    contrario es la prueba siguiente.
    """
    X, Y = malla
    assert cintura(W0, LAMB, z) < L0 / 2, "el haz debe caber en la ventana"
    assert _rms_contra_analitico(fft_asm(campo, DELTA, LAMB, z, device="cpu"),
                                 X, Y, z) < 0.01


def test_fft_asm_se_degrada_cuando_el_haz_no_cabe_en_la_ventana():
    """El talón de Aquiles de FFT-ASM, y la razón de ser de MPASM y BLAS.

    A z = 12000 mm el haz tiene wz = 2.6 mm y la ventana sólo llega a 2.5 mm:
    lo que sale por un borde reentra por el opuesto (la FFT es circular) y la
    amplitud deja de parecerse a la analítica. La prueba fija ambos lados —
    falla con la ventana estrecha, acierta con una ancha — para que quede
    demostrado que la causa es el tamaño de ventana y no el propagador.
    """
    z = 12000
    assert cintura(W0, LAMB, z) > L0 / 2

    U_estrecha, X, Y = gauss_beam(N, DELTA, W0, device="cpu")
    rms_estrecha = _rms_contra_analitico(
        fft_asm(U_estrecha, DELTA, LAMB, z, device="cpu"), X, Y, z)

    delta_ancha = (8 * L0) / (N - 1)
    U_ancha, X_a, Y_a = gauss_beam(N, delta_ancha, W0, device="cpu")
    rms_ancha = _rms_contra_analitico(
        fft_asm(U_ancha, delta_ancha, LAMB, z, device="cpu"), X_a, Y_a, z)

    assert rms_estrecha > 0.1, "con la ventana justa debería aliasar"
    assert rms_ancha < 0.01, "con ventana holgada debería seguir al analítico"


# ------------------------------------ rejilla de frecuencias y malla impar
def rejilla_vieja(n, delta):
    """La expresión que usaban fft_asm() y blas() antes de frecuencias_fft().

    Sólo coincide con la correcta cuando n es par. Se conserva aquí, y no en
    el paquete, porque su único uso legítimo es demostrar la diferencia.
    """
    return (np.arange(n) - n / 2) / (delta * n)


@pytest.mark.parametrize("n", [63, 64, 65, 66])
def test_la_rejilla_de_frecuencias_centra_la_continua(n):
    """fftshift deja la continua en el índice n//2, sea n par o impar.

    La rejilla tiene que valer 0 exactamente ahí, o H se evalúa en la
    frecuencia equivocada. Y tiene que ir creciendo, porque blas() localiza su
    máscara de banda con searchsorted.
    """
    fx = frecuencias_fft(n, DELTA)
    assert fx[n // 2] == 0.0
    assert np.all(np.diff(fx) > 0)
    assert np.allclose(np.diff(fx), 1 / (DELTA * n))


@pytest.mark.parametrize("n", [63, 65])
def test_la_rejilla_vieja_se_corre_medio_paso_en_malla_impar(n):
    """El fallo, aislado de todo lo demás: medio intervalo de frecuencia."""
    d = frecuencias_fft(n, DELTA) - rejilla_vieja(n, DELTA)
    assert np.allclose(d, 0.5 / (DELTA * n))
    assert rejilla_vieja(n, DELTA)[n // 2] != 0.0


@pytest.mark.parametrize("n", [32, 64, 66, 128])
def test_en_malla_par_las_dos_rejillas_son_la_misma(n):
    """Por qué arreglar el caso impar no puede mover ningún resultado par."""
    assert np.allclose(frecuencias_fft(n, DELTA), rejilla_vieja(n, DELTA),
                       rtol=1e-14, atol=1e-14)


def test_en_malla_par_el_cambio_de_rejilla_no_mueve_el_campo(monkeypatch):
    """Lo anterior, pero sobre el campo propagado y no sobre la rejilla."""
    import CamposT.propagadores as P
    U0, _, _ = gauss_beam(N, DELTA, W0, device="cpu")          # N es par
    nuevo = fft_asm(U0, DELTA, LAMB, 2000.0, device="cpu")
    monkeypatch.setattr(P, "frecuencias_fft", rejilla_vieja)
    viejo = fft_asm(U0, DELTA, LAMB, 2000.0, device="cpu")
    assert error_relativo(nuevo, viejo) < 1e-14


@pytest.mark.parametrize("nombre", ["fft_asm", "blas"])
@pytest.mark.parametrize("n", [63, 65])
def test_fft_asm_y_blas_siguen_al_analitico_en_malla_impar(nombre, n, monkeypatch):
    """La regresión de verdad: exactitud física con lado impar.

    Con la rejilla vieja el campo salía desplazado lambda·z·(0.5/(delta·n))/delta
    píxeles —0.39 px con los números de conftest— y el RMS contra el gaussiano
    analítico subía de 5.5e-4 a 4.2e-2. La prueba fija los dos lados para que
    quede demostrado que la causa es la rejilla y no otra cosa.

    mpasm() no aparece aquí: construye su propia rejilla junto con las
    matrices de la DFT, sin fftshift de por medio, y nunca tuvo el fallo.
    """
    import CamposT.propagadores as P
    z = 2000
    U0, X, Y = gauss_beam(n, DELTA, W0, device="cpu")
    assert cintura(W0, LAMB, z) < n * DELTA / 2, "el haz debe caber"

    assert _rms_contra_analitico(
        PROPAGADORES[nombre](U0, z, device="cpu"), X, Y, z) < 0.01

    monkeypatch.setattr(P, "frecuencias_fft", rejilla_vieja)
    assert _rms_contra_analitico(
        PROPAGADORES[nombre](U0, z, device="cpu"), X, Y, z) > 0.01


def test_la_reversibilidad_no_veia_el_fallo_de_malla_impar(monkeypatch):
    """Por qué vivió tanto: la ida y vuelta sale exacta con rejilla o sin ella.

    El medio paso se aplica en la ida y otra vez en la vuelta, con el mismo
    signo en la rejilla y el opuesto en z, así que se cancela. Ninguna de las
    pruebas de reversibilidad de este fichero podía verlo, y por eso hizo
    falta contrastar contra una solución analítica.
    """
    import CamposT.propagadores as P
    monkeypatch.setattr(P, "frecuencias_fft", rejilla_vieja)
    U0, _, _ = gauss_beam(65, DELTA, W0, device="cpu")
    ida = fft_asm(U0, DELTA, LAMB, 500.0, device="cpu")
    vuelta = fft_asm(ida, DELTA, LAMB, -500.0, device="cpu")
    assert error_relativo(vuelta, U0) < 1e-12


if __name__ == "__main__":
    # Ejecutar este fichero con `python tests/test_propagadores.py` (o con el
    # botón de Run del editor) lanza pytest sobre él. Sin esto, Python lo
    # importaría, no encontraría ningún `main` y terminaría sin hacer nada:
    # las pruebas las recolecta y ejecuta pytest, no el intérprete.
    raise SystemExit(pytest.main([__file__, "-v"]))
