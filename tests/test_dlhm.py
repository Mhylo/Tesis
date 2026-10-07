"""Verificación de CamposT.dlhm: MPASM con iluminación esférica exacta.

Lo que se fija, y por qué:

- La forma cerrada de ventana_chirp (integrales de Fresnel) contra la integral
  que la define, hecha por cuadratura: es la pieza nueva y todo lo demás
  descansa en ella.
- La ida contra Rayleigh-Sommerfeld directo, en el eje y fuera de él, con un
  objeto de detalle fino (~1 um). Es la prueba de que la esférica y el ASM
  están bien puestos sin aproximación paraxial: el escalado de la tarea 24 no
  la pasa con este objeto.
- La vuelta recupera el objeto de su propio holograma complejo, y lo pone
  donde está: con el objeto y el sensor descentrados, y distinto en y que en
  x, para que un eje cruzado o un signo cambiado no pase.
- En paraxial, la vuelta exacta coincide con el escalado: los dos son el
  mismo modelo donde el escalado vale.
- Las guardas: la ventana que no cabe en el periodo, la z fuera del rango del
  Reconstructor y las geometrías imposibles abortan en vez de devolver algo.

Todo con los números del montaje (CamposT/montaje.py) y mallas pequeñas: la
suite entera de este fichero corre en segundos en CPU.
"""

import numpy as np
import pytest

from CamposT import dlhm
from CamposT.backend import a_numpy, gpu_disponible

LAMB, DELTA, Z, L = 528e-6, 1.83e-3, 3.0, 10.0
M, D = L / Z, L - Z
K = 2 * np.pi / LAMB
DO = DELTA / M                      # paso del objeto: el píxel proyectado


def manchas(n, paso, centros, sigma):
    """Absorbente (0..0.5) hecho de gaussianas: compacto, suave y de banda
    ~1/(2 pi sigma) conocida. Devuelve t = 1 - a."""
    g = (np.arange(n) - n / 2) * paso
    X, Y = np.meshgrid(g, g)
    a = sum(np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2)) for cy, cx in centros)
    return 1 - 0.5 * a


def rs_directo(t, paso, centro_obj, Ys, Xs, zm=Z, Lm=L, emision=None):
    """Rayleigh-Sommerfeld I directo (Lopera, Ec. 1), campo de t - 1 sobre la
    malla (Ys, Xs) del sensor, dividido por la esférica en el sensor. zm y Lm,
    los del montaje salvo que se pidan otros. emision = a(sin theta): pesa cada
    punto del objeto con la amplitud que la fuente le manda."""
    n = t.shape[0]
    g = (np.arange(n) - n / 2) * paso
    xs = (g + centro_obj[1])[None, :].repeat(n, 0).ravel()
    ys = (g + centro_obj[0])[:, None].repeat(n, 1).ravel()
    fuente = (t - 1).ravel()
    m = np.abs(fuente) > 1e-12
    xs, ys, fuente = xs[m], ys[m], fuente[m]
    rho = np.sqrt(xs**2 + ys**2 + zm**2)
    fuente = fuente * np.exp(1j * K * rho) / rho * paso**2
    if emision is not None:
        fuente = fuente * emision(np.hypot(xs, ys) / rho)
    XX, YY = np.meshgrid(Xs, Ys)
    U = np.zeros(XX.size, complex)
    dm = Lm - zm
    for i in range(0, xs.size, 256):
        dx = XX.ravel()[None, :] - xs[i:i + 256, None]
        dy = YY.ravel()[None, :] - ys[i:i + 256, None]
        r = np.sqrt(dx * dx + dy * dy + dm * dm)
        U += fuente[i:i + 256] @ (dm / (2 * np.pi * r * r) * (1 / r - 1j * K) * np.exp(1j * K * r))
    r = np.sqrt(XX**2 + YY**2 + Lm**2)
    return U.reshape(XX.shape) / (np.exp(1j * K * r) / r)


def correlacion(u, v):
    u, v = np.ravel(u), np.ravel(v)
    return abs(np.vdot(u, v)) / np.sqrt(np.vdot(u, u).real * np.vdot(v, v).real)


def error_relativo(u, v):
    return np.linalg.norm(u - v) / np.linalg.norm(v)


# ------------------------------------------------------------- ventana_chirp
def test_ventana_chirp_coincide_con_su_definicion_por_cuadratura():
    """A[n, m] = paso int_{-B}^{B} e^{-i 2 pi g x_n} P^(f_m - g) dg, con
    P^(nu) = sqrt(i lamb R) e^{-i pi lamb R nu²} la transformada del chirp."""
    paso, R = DELTA, L
    B = 1 / (2 * paso)
    pos = np.array([-0.9, -0.2, 0.0, 0.31, 1.4])
    f = np.array([-400.0, -120.0, 0.0, 95.0, 270.0, 520.0, 800.0])
    from scipy.integrate import simpson
    A = dlhm.ventana_chirp(pos, f, paso, LAMB, R)
    g = np.linspace(-B, B, 400001)
    for n, x in enumerate(pos):
        for m, fm in enumerate(f):
            integrando = (np.exp(-2j * np.pi * g * x) * np.sqrt(1j * LAMB * R)
                          * np.exp(-1j * np.pi * LAMB * R * (fm - g) ** 2))
            ref = paso * simpson(integrando, x=g)
            assert abs(A[n, m] - ref) < 1e-9 * paso, (x, fm, A[n, m], ref)


def test_ventana_chirp_sin_curvatura_es_la_dft_recortada_a_la_banda():
    """Con R enorme el chirp desaparece: dentro de la banda A es la fila de la
    DFT de Riemann, paso·e^{-i 2 pi f x}; fuera, cero."""
    paso = DELTA
    B = 1 / (2 * paso)
    pos = np.linspace(-0.5, 0.5, 7)
    dentro = np.linspace(-0.9 * B, 0.9 * B, 11)
    fuera = np.array([-1.5 * B, 1.3 * B])
    A = dlhm.ventana_chirp(pos, np.r_[dentro, fuera], paso, LAMB, 1e15)
    dft = paso * np.exp(-2j * np.pi * np.outer(pos, dentro))
    assert np.abs(A[:, :dentro.size] - dft).max() < 1e-6 * paso
    assert np.abs(A[:, dentro.size:]).max() < 1e-6 * paso


def test_ventana_chirp_rechaza_un_chirp_convergente():
    with pytest.raises(ValueError, match="R > 0"):
        dlhm.ventana_chirp([0.0], [0.0], DELTA, LAMB, -1.0)


# ------------------------------------------------------------------ la fuente
#: Primer cero de 2 J1(x)/x: x = 3.8317..., o sea s = 1.2197 lamb/d.
X_CERO = 3.8317059702075125


def test_fuente_pinhole_es_el_airy_de_amplitud():
    """a(0) = 1, el primer cero donde dice la teoría y 2 J1(x)/x en puntos
    sueltos, con el signo: entre el primer cero y el segundo es negativa."""
    from scipy.special import j1
    d = 5e-3
    a = dlhm.fuente_pinhole(d, LAMB)
    assert a(np.zeros(1))[0] == 1.0
    s0 = X_CERO * LAMB / (np.pi * d)
    assert s0 / (LAMB / d) == pytest.approx(1.2197, abs=1e-4)
    assert abs(a(np.array([s0]))[0]) < 1e-12
    s = np.array([0.01, 0.05, 0.1, 0.2, 0.3])
    x = np.pi * d * s / LAMB
    np.testing.assert_allclose(a(s), 2 * j1(x) / x, rtol=1e-12)
    assert a(np.array([0.2]))[0] < 0


def test_fuente_pinhole_rechaza_un_diametro_no_positivo():
    with pytest.raises(ValueError, match="diametro"):
        dlhm.fuente_pinhole(0.0, LAMB)


def _envolvente_de_su_dlhm(forma, paso, na):
    """La envolvente PS de dlhm() de Lopera et al. 2024, escrita como en su
    código pero sobre la malla del paquete: exp(-2r/(L tan(asin NA))) con r la
    distancia de la fuente al píxel, menos su mínimo y entre su máximo."""
    Ys = dlhm.coordenadas(forma[0], paso)
    Xs = dlhm.coordenadas(forma[1], paso)
    u, v = np.meshgrid(Xs, Ys)
    r = np.sqrt(u**2 + v**2 + L**2)
    PS = np.exp(-2 * r / (L * np.tan(np.arcsin(na))))
    PS = PS - np.min(PS)
    return PS / np.max(PS)


def test_fuente_lopera_es_la_envolvente_de_su_dlhm():
    """|a|² en el sensor es su PS, en el sensor del banco (3000x4000 px de
    1.83 um) muestreado cada 100 px: con s_max, la esquina más lejana, sale su
    resta del mínimo; L se cancela."""
    forma, paso = (30, 40), 100 * DELTA
    rho = np.hypot(forma[0] / 2 * paso, forma[1] / 2 * paso)
    s_max = rho / np.hypot(rho, L)
    assert s_max == pytest.approx(0.416, abs=1e-3)
    a = dlhm.fuente_lopera(0.1, s_max)
    np.testing.assert_allclose(dlhm.iluminacion(forma, paso, L, a),
                               _envolvente_de_su_dlhm(forma, paso, 0.1), rtol=0, atol=1e-12)


def test_fuente_lopera_sin_s_max_es_la_exponencial_pura():
    """Sin la resta del mínimo: a(0) = 1, cae a 1/e² en intensidad donde
    sec(theta) - 1 = tan(asin NA) -sin(theta) = 0.4175 con NA = 0.1, no en
    0.1: la NA de su fórmula no es el semiángulo del cono- y no llega a cero."""
    a = dlhm.fuente_lopera(0.1)
    assert a(np.zeros(1))[0] == 1.0
    cos = 1 / (1 + np.tan(np.arcsin(0.1)))
    s = np.sqrt(1 - cos**2)
    assert s == pytest.approx(0.4175, abs=1e-4)
    assert a(np.array([s]))[0] ** 2 == pytest.approx(np.exp(-2), rel=1e-12)
    assert a(np.array([0.9]))[0] > 0


def test_fuente_lopera_con_s_max_llega_a_cero_y_no_pasa():
    a = dlhm.fuente_lopera(0.1, 0.416)
    assert a(np.zeros(1))[0] == 1.0
    assert abs(a(np.array([0.416]))[0]) < 1e-12
    np.testing.assert_array_equal(a(np.array([0.45, 0.8])), 0.0)
    s = np.linspace(0, 0.416, 200)
    assert np.all(np.diff(a(s)) < 0)


@pytest.mark.parametrize("na, s_max", [(0.0, None), (1.0, None), (-0.1, None),
                                       (0.1, 0.0), (0.1, 1.0)])
def test_fuente_lopera_rechaza_na_o_s_max_fuera_de_0_1(na, s_max):
    with pytest.raises(ValueError):
        dlhm.fuente_lopera(na, s_max)


def test_iluminacion_es_el_modulo_al_cuadrado_de_a_en_el_sensor():
    a = dlhm.fuente_pinhole(5e-3, LAMB)
    X0 = (0.7, 1.9)
    Ys = (np.arange(16) - 8) * DELTA + X0[0]
    Xs = (np.arange(24) - 12) * DELTA + X0[1]
    rho = np.hypot(Xs[None, :], Ys[:, None])
    esperado = a(rho / np.sqrt(rho**2 + L**2)) ** 2
    np.testing.assert_allclose(dlhm.iluminacion((16, 24), DELTA, L, a, centro=X0),
                               esperado, rtol=1e-12)
    np.testing.assert_array_equal(dlhm.iluminacion((16, 24), DELTA, L, None),
                                  np.ones((16, 24)))


@pytest.mark.parametrize("fuente, mensaje", [
    (lambda s: 0.5 * np.ones_like(s), r"a\(0\) = 1"),
    (lambda s: np.ones(3), "forma"),
    (lambda s: np.where(s > 1e-3, np.nan, 1.0), "finit")])
def test_una_fuente_mal_hecha_aborta(fuente, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        dlhm.iluminacion((16, 16), DELTA, L, fuente, centro=(0.7, 1.9))


# ------------------------------------------------------------------ la ida
@pytest.mark.parametrize("X0", [(0.0, 0.0), (0.7, 1.9)])
def test_ida_coincide_con_rayleigh_sommerfeld(X0):
    """Objeto de manchas de 1 um (banda ~500 mm^-1, detalle que el escalado
    paraxial no reproduce) y un trozo de 24x24 px del sensor alrededor de su
    proyección; fuera del eje, a 11.5 grados. El trozo va en `centro`, así que
    esto prueba también la esférica con coordenadas absolutas."""
    centro_obj = (X0[0] / M, X0[1] / M)
    t = manchas(40, DO, [(0.0, -3e-3), (4e-3, 4e-3), (-4e-3, 2e-3)], 1e-3)
    n_s = 24
    V = dlhm.holograma(t, DO, LAMB, Z, L, (n_s, n_s), 4 * DELTA, centro_objeto=centro_obj,
                       centro=X0, complejo=True, device="cpu")
    Ys = (np.arange(n_s) - n_s / 2) * 4 * DELTA + X0[0]
    Xs = (np.arange(n_s) - n_s / 2) * 4 * DELTA + X0[1]
    ref = rs_directo(t, DO, centro_obj, Ys, Xs)
    assert error_relativo(a_numpy(V), ref) < 1e-4


def test_la_ida_con_un_objeto_muestreado_por_debajo_de_lambda_medios():
    """REGRESION. El soporte de la ida recortaba la frecuencia del OTRO eje a
    +-0.999/lambda y no a la banda de ese eje, que es lo que hay en la rejilla.
    Con un objeto muestreado por debajo de lambda/2 su banda de Nyquist pasa de
    1/lambda: la esquina (fx, fy) salia evanescente, el aterrizaje infinito y
    la rejilla de frecuencias pedia memoria sin techo. Es lo que pasaba con el
    BenchmarkTarget entero, 3000x4000 px a delta*z/L = 0.55 um (86.7 GB solo
    para la rejilla); el cuadrado de 3000 no lo notaba porque su esquina aun
    se propagaba. Aqui, 0.25 um con lambda = 528 nm."""
    p = 0.25e-3
    t = manchas(48, p, [(0.0, -2e-3), (2e-3, 2e-3)], 1e-3)
    n_s = 24
    V = dlhm.holograma(t, p, LAMB, Z, L, (n_s, n_s), 4 * DELTA, complejo=True, device="cpu")
    Ys = Xs = (np.arange(n_s) - n_s / 2) * 4 * DELTA
    ref = rs_directo(t, p, (0.0, 0.0), Ys, Xs)
    assert error_relativo(a_numpy(V), ref) < 1e-4


def test_la_ida_con_el_sensor_ancho_frente_a_la_distancia():
    """REGRESION. Con el sensor ancho frente a d = L - z, las bandas de los dos
    ejes -validas cada una- llegan juntas a mas de 1/lambda: la esquina
    (fx, fy) es evanescente, su aterrizaje infinito y el periodo tambien. Es lo
    que impedia propagar el BenchmarkTarget a planos intermedios (d de 0.1 a
    4 mm; 14425 GB con d = 0.1) y simular la configuracion 3 de Lopera et al.
    2024 (d = 0.6 mm). Aqui d = 0.5 mm y un sensor de 0.64 mm de lado, que pide
    rayos de hasta 43 grados."""
    p, zc, Lc = 0.25e-3, 3.0, 3.5
    t = manchas(64, p, [(0.0, -2e-3), (3e-3, 2e-3)], 1e-3)
    n_s, paso_s = 32, 20e-3
    V = dlhm.holograma(t, p, LAMB, zc, Lc, (n_s, n_s), paso_s, complejo=True, device="cpu")
    Ys = Xs = (np.arange(n_s) - n_s / 2) * paso_s
    ref = rs_directo(t, p, (0.0, 0.0), Ys, Xs, zm=zc, Lm=Lc)
    assert error_relativo(a_numpy(V), ref) < 1e-4


def test_holograma_de_contraste_es_el_modulo_de_uno_mas_V():
    t = manchas(32, DO, [(0.0, 0.0)], 3e-3)
    kw = dict(device="cpu")
    V = dlhm.holograma(t, DO, LAMB, Z, L, (32, 32), DELTA, complejo=True, **kw)
    c = dlhm.holograma(t, DO, LAMB, Z, L, (32, 32), DELTA, **kw)
    np.testing.assert_allclose(c, np.abs(1 + V) ** 2, rtol=1e-12)


@pytest.mark.parametrize("X0", [(0.0, 0.0), (0.7, 1.9)])
def test_la_ida_con_fuente_coincide_con_rayleigh_sommerfeld(X0):
    """El Airy del pinhole de 5 um dentro de la ida, contra RS directo con la
    misma emisión. Fuera del eje el objeto cae a sin(theta) = 0.199, pasado el
    primer cero (0.1288): a vale -0.099 y un signo o una coordenada mal
    puestos no pasan (la ida SIN fuente contra esta RS da 11.1, medido).
    Medido con la implementación: 1.1e-5 en el eje y 4.6e-6 fuera."""
    a = dlhm.fuente_pinhole(5e-3, LAMB)
    centro_obj = (X0[0] / M, X0[1] / M)
    t = manchas(40, DO, [(0.0, -3e-3), (4e-3, 4e-3), (-4e-3, 2e-3)], 1e-3)
    n_s = 24
    V = dlhm.holograma(t, DO, LAMB, Z, L, (n_s, n_s), 4 * DELTA, centro_objeto=centro_obj,
                       centro=X0, complejo=True, fuente=a, device="cpu")
    Ys = (np.arange(n_s) - n_s / 2) * 4 * DELTA + X0[0]
    Xs = (np.arange(n_s) - n_s / 2) * 4 * DELTA + X0[1]
    ref = rs_directo(t, DO, centro_obj, Ys, Xs, emision=a)
    assert error_relativo(a_numpy(V), ref) < 1e-4


def test_con_fuente_la_intensidad_es_el_modulo_de_a_mas_V():
    """La referencia en el sensor ya no es 1 sino a_L: I/|S_L|² = |a_L + V|².
    Y sin objeto (t = 1, V = 0) lo que queda es exactamente iluminacion()."""
    a = dlhm.fuente_pinhole(5e-3, LAMB)
    X0 = (0.7, 1.9)
    t = manchas(32, DO, [(0.0, 0.0)], 3e-3)
    kw = dict(centro_objeto=(X0[0] / M, X0[1] / M), centro=X0, fuente=a, device="cpu")
    V = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (32, 32), DELTA, complejo=True, **kw))
    c = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (32, 32), DELTA, **kw))
    Ys = (np.arange(32) - 16) * DELTA + X0[0]
    Xs = (np.arange(32) - 16) * DELTA + X0[1]
    rho = np.hypot(Xs[None, :], Ys[:, None])
    a_L = a(rho / np.sqrt(rho**2 + L**2))
    np.testing.assert_allclose(c, np.abs(a_L + V) ** 2, rtol=1e-12)
    vacio = a_numpy(dlhm.holograma(np.ones((32, 32)), DO, LAMB, Z, L, (32, 32), DELTA, **kw))
    np.testing.assert_allclose(vacio, dlhm.iluminacion((32, 32), DELTA, L, a, centro=X0),
                               rtol=1e-12)


def test_una_fuente_constante_da_lo_mismo_que_ninguna():
    t = manchas(32, DO, [(0.0, 0.0)], 3e-3)
    uno = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (32, 32), DELTA,
                                 fuente=lambda s: np.ones_like(s), device="cpu"))
    nada = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (32, 32), DELTA, device="cpu"))
    np.testing.assert_allclose(uno, nada, rtol=1e-12)


def test_la_ida_con_una_fuente_mal_hecha_aborta():
    t = manchas(16, DO, [(0.0, 0.0)], 3e-3)
    with pytest.raises(ValueError, match=r"a\(0\) = 1"):
        dlhm.holograma(t, DO, LAMB, Z, L, (16, 16), DELTA,
                       fuente=lambda s: 2 * np.ones_like(s), device="cpu")


# ----------------------------------------------------------------- la vuelta
def _holograma_prueba(n=256, centro=(0.0, 0.0), centros=((0.0, 0.0),), sigma=12e-3):
    """Holograma complejo (sin gemela) de un objeto grueso, que cabe entero en
    la NA del sensor, para que la vuelta se pueda comparar con el objeto."""
    centro_obj = (centro[0] / M, centro[1] / M)
    t = manchas(n, DO, centros, sigma)
    V = dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, centro_objeto=centro_obj,
                       centro=centro, complejo=True, device="cpu")
    return t, 1 + a_numpy(V)


def test_la_vuelta_recupera_el_objeto_de_su_holograma():
    t, c = _holograma_prueba()
    tr = a_numpy(dlhm.reconstruir(c, DELTA, LAMB, Z, L, device="cpu"))
    assert correlacion(tr - 1, t - 1) > 0.999


def test_la_vuelta_pone_el_objeto_donde_esta():
    """Mancha única descentrada, distinta en y que en x, con el sensor también
    descentrado: el centroide de |t - 1| tiene que caer donde estaba la mancha.
    Un eje cruzado, un signo cambiado o el centro mal aplicado lo mueven."""
    centro = (-0.12, 0.2)
    mancha = (0.018, -0.03)                           # respecto del centro del objeto
    t, c = _holograma_prueba(centro=centro, centros=(mancha,), sigma=6e-3)
    tr = a_numpy(dlhm.reconstruir(c, DELTA, LAMB, Z, L, centro=centro, device="cpu"))
    # centroide de |t - 1|² en un entorno del máximo: sobre la ventana entera,
    # el fondo que deja el borde del sensor tira del centroide hacia el centro
    w = np.abs(tr - 1) ** 2
    i, j = np.unravel_index(np.argmax(w), w.shape)
    h = 40
    w = w[i - h:i + h, j - h:j + h]
    n = tr.shape[0]
    g = (np.arange(n) - n / 2) * DO
    yc = (w.sum(1) @ g[i - h:i + h]) / w.sum() + centro[0] / M
    xc = (w.sum(0) @ g[j - h:j + h]) / w.sum() + centro[1] / M
    # 0.2 px: un eje cruzado, un signo o el centro mal aplicado mueven la
    # mancha decenas de micras. Lo que queda (0.05 px medido) es el recorte de
    # NA asimétrico de un sensor descentrado, que no es simétrico alrededor
    # de la mancha.
    assert abs(yc - (centro[0] / M + mancha[0])) < 0.2 * DO
    assert abs(xc - (centro[1] / M + mancha[1])) < 0.2 * DO


def test_la_vuelta_funciona_donde_la_referencia_no_cabe_en_el_pixel():
    """Un trozo de sensor a 2 mm del eje: ahí la esférica de referencia tiene
    frecuencia local de 330 a 420 mm^-1 y el píxel solo muestrea hasta 273, así
    que multiplicar por ella en la malla del sensor ya no representa el campo.
    La vuelta tiene que seguir recuperando el objeto, y lo hace.

    OJO, lo que esta prueba NO fija: que las ventanas de ventana_chirp hagan
    falta. Sin ellas (la DFT de Riemann de la esférica muestreada) cada
    componente reaparece desplazada 1/delta, pero con un objeto grueso esas
    réplicas caen fuera de la banda de la rejilla y el resultado es el mismo
    (medido: 1.00000 las dos, con trozos de hasta 1024 px). Solo entran cuando
    el holograma tiene contenido cerca del Nyquist del píxel y la ventana del
    objeto es comparable a d·lamb/delta ~ 2 mm, o sea con el sensor entero:
    ahí la versión sin ventanas cae a 0.899 frente a 0.969
    (scripts/verif_dlhm_esferico.py). Eso es demasiado caro para la suite; las
    ventanas las fijan las dos pruebas de ventana_chirp de arriba."""
    centro = (0.0, 2.0)
    assert (centro[1] - 128 * DELTA) / (LAMB * L) > 1 / (2 * DELTA) * 1.1
    t, c = _holograma_prueba(n=256, centro=centro, centros=((0.01, -0.02), (-0.02, 0.015)))
    tr = a_numpy(dlhm.reconstruir(c, DELTA, LAMB, Z, L, centro=centro, device="cpu"))
    assert correlacion(tr - 1, t - 1) > 0.999


def test_en_paraxial_la_vuelta_exacta_es_el_escalado():
    """Objeto grueso cerca del eje: la fase que desprecia el escalado es
    despreciable, y los dos métodos tienen que dar lo mismo."""
    t, c = _holograma_prueba(n=256, sigma=15e-3)
    te = a_numpy(dlhm.reconstruir(c, DELTA, LAMB, Z, L, device="cpu"))
    ts = a_numpy(dlhm.reconstruir_escalado(c, DELTA, LAMB, Z, L, device="cpu"))
    assert correlacion(te - 1, ts - 1) > 0.999


def test_el_reconstructor_reutiliza_el_espectro_en_un_barrido():
    """Construido para un rango de z, campo(z) da lo mismo que reconstruir(z)
    en el foco, y el foco es el máximo de la correlación con el objeto."""
    t, c = _holograma_prueba()
    R = dlhm.Reconstructor(c, DELTA, LAMB, L, (2.9, 3.1), device="cpu")
    solo = a_numpy(dlhm.reconstruir(c, DELTA, LAMB, Z, L, device="cpu"))
    en_foco = a_numpy(R.campo(Z))
    assert error_relativo(en_foco - 1, solo - 1) < 1e-3
    # la malla de la muestra cambia con z (delta z/L): se compara en la misma
    # con paso fijo, la del foco
    corr = {zz: correlacion(a_numpy(R.campo(zz, paso=DO)) - 1, t - 1) for zz in (2.95, Z, 3.05)}
    assert corr[Z] > max(corr[2.95], corr[3.05])


def test_fondo_divide_el_holograma():
    t, c = _holograma_prueba(n=128)
    I_ref = np.full(c.shape, 2.5)
    a = a_numpy(dlhm.reconstruir(c * I_ref, DELTA, LAMB, Z, L, fondo=I_ref, device="cpu"))
    b = a_numpy(dlhm.reconstruir(c, DELTA, LAMB, Z, L, device="cpu"))
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_fondo_media_normaliza_un_holograma_crudo():
    """Un holograma en cuentas de 8 bits, con fondo='media': da lo mismo que su
    versión de contraste dividida por la misma media. Y la resta va en coma
    flotante: en uint8, 0 - 1 daría 255."""
    I = np.abs(1 + 0.3 * np.exp(1j * np.linspace(0, 20, 64 * 64))).reshape(64, 64) ** 2
    crudo = np.round(100 * I).astype(np.uint8)
    crudo[0, 0] = 0
    a = a_numpy(dlhm.reconstruir(crudo, DELTA, LAMB, Z, L, fondo="media", device="cpu"))
    b = a_numpy(dlhm.reconstruir(crudo.astype(float) / crudo.mean(), DELTA, LAMB, Z, L,
                                 device="cpu"))
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_un_holograma_crudo_sin_fondo_aborta():
    """Sin fondo se da por hecho que el holograma es de contraste. En cuentas
    de cámara no lo es, y reconstruirlo así lleva la referencia dentro."""
    with pytest.raises(ValueError, match="de contraste"):
        dlhm.reconstruir(np.full((32, 32), 120.0), DELTA, LAMB, Z, L, device="cpu")


def test_el_piso_anula_el_contraste_donde_el_fondo_es_oscuro():
    """Bajo sqrt(piso·max|fondo|) no se divide: c1 = 0, 'sin información',
    como lo que cae fuera del sensor. Un fondo con ceros no da NaN."""
    fondo = np.ones((32, 32))
    fondo[:, :8] = 1e-3
    fondo[0, 0] = 0.0
    c1, raiz, raiz_min = dlhm.contraste_amplitud(1.2 * fondo, fondo, piso=0.05)
    assert np.isfinite(c1).all()
    np.testing.assert_array_equal(c1[:, :8], 0.0)
    np.testing.assert_allclose(c1[:, 8:], 0.2, rtol=1e-12)
    np.testing.assert_array_equal(raiz, np.sqrt(fondo))
    assert raiz_min == pytest.approx(np.sqrt(0.05))


def test_sin_piso_el_contraste_por_amplitud_es_holo_menos_fondo_entre_su_raiz():
    rng = np.random.default_rng(3)
    fondo = 0.5 + rng.random((16, 16))
    holo = 0.5 + rng.random((16, 16))
    c1, raiz, raiz_min = dlhm.contraste_amplitud(holo, fondo, piso=0.0)
    np.testing.assert_allclose(c1, (holo - fondo) / np.sqrt(fondo), rtol=1e-12)
    assert raiz_min == 0.0


@pytest.mark.parametrize("fondo", [None, "media", 2.0])
def test_el_piso_sin_imagen_de_fondo_aborta(fondo):
    with pytest.raises(ValueError, match="piso"):
        dlhm.Reconstructor(np.ones((8, 8)), DELTA, LAMB, L, Z, fondo=fondo, piso=0.05,
                           device="cpu")


@pytest.mark.parametrize("piso", [-0.1, 1.0])
def test_un_piso_fuera_de_rango_aborta(piso):
    with pytest.raises(ValueError, match="piso"):
        dlhm.contraste_amplitud(np.ones((8, 8)), np.ones((8, 8)), piso=piso)


@pytest.mark.parametrize("holo, fondo, mensaje", [
    (np.ones((8, 8)), np.zeros((8, 8)), "finito"),
    (np.ones((8, 8)), np.full((8, 8), np.nan), "finito"),
    (np.ones((8, 8)), np.ones((8, 6)), "fondo es"),
    (np.ones((8, 8), complex), np.linspace(1, 2, 64).reshape(8, 8), "intensidad")])
def test_un_fondo_imagen_mal_hecho_aborta(holo, fondo, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        dlhm.contraste_amplitud(holo, fondo, piso=0.05)


def test_con_fondo_imagen_la_salida_es_la_vuelta_de_c1_entre_la_raiz_proyectada():
    """1 + B{c1}/sqrt(fondo) proyectado, con B la vuelta de siempre: se arma a
    mano con un Reconstructor SIN fondo sobre 1 + c1. En la malla por defecto
    la proyección es píxel a píxel."""
    n = 64
    a = dlhm.fuente_pinhole(20e-3, LAMB)
    t = manchas(n, DO, [(0.0, 0.0), (0.01, -0.01)], 3e-3)
    holo = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, fuente=a, device="cpu"))
    fondo = dlhm.iluminacion((n, n), DELTA, L, a)
    R = dlhm.Reconstructor(holo, DELTA, LAMB, L, Z, fondo=fondo, piso=0.05, device="cpu")
    c1, raiz, raiz_min = dlhm.contraste_amplitud(holo, fondo, piso=0.05)
    B = a_numpy(dlhm.Reconstructor(1 + c1, DELTA, LAMB, L, Z, device="cpu").campo(Z)) - 1
    luz = raiz >= raiz_min
    esperado = 1 + np.where(luz, B / np.where(luz, raiz, 1), 0)
    np.testing.assert_allclose(a_numpy(R.campo(Z)), esperado, rtol=1e-10, atol=1e-12)


def test_el_fondo_proyectado_es_la_amplitud_de_la_fuente_en_la_muestra():
    """sqrt(iluminacion()) llevado a la muestra por (y, x)·L/z -el mismo rayo-
    es |a(s_z)|: en la malla por defecto y en otra más fina y desplazada."""
    n = 64
    a = dlhm.fuente_pinhole(20e-3, LAMB)
    fondo = dlhm.iluminacion((n, n), DELTA, L, a)
    R = dlhm.Reconstructor(fondo.copy(), DELTA, LAMB, L, Z, fondo=fondo, device="cpu")
    for paso, m, centro in ((DO, n, (0.0, 0.0)), (DO / 2, 48, (3 * DO, -5 * DO))):
        y = dlhm.coordenadas(m, paso, centro[0])
        x = dlhm.coordenadas(m, paso, centro[1])
        rho = np.hypot(x[None, :], y[:, None])
        esperado = np.abs(a(rho / np.sqrt(rho**2 + Z**2)))
        np.testing.assert_allclose(R._raiz_proyectada(y, x, Z), esperado, rtol=1e-3)


def test_bajo_el_piso_la_salida_vale_uno_y_un_fondo_con_ceros_no_da_nan():
    """Fuera del disco, sin piso, la salida se multiplica por 1/sqrt de un fondo
    casi nulo: medido |t - 1| hasta 288 (0.54 dentro). Con piso vale 1."""
    n = 256
    a = dlhm.fuente_pinhole(20e-3, LAMB)
    t = manchas(n, DO, [(0.0, 0.0), (0.03, -0.02)], 6e-3)
    fondo = dlhm.iluminacion((n, n), DELTA, L, a)
    fondo[:, :4] = 0.0
    holo = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, fuente=a, device="cpu"))
    tr = a_numpy(dlhm.reconstruir(holo, DELTA, LAMB, Z, L, fondo=fondo, piso=0.05,
                                  device="cpu"))
    assert np.isfinite(tr).all()
    np.testing.assert_array_equal(tr[fondo < 0.05 * fondo.max()], 1.0)


def test_ida_y_vuelta_con_el_airy_en_el_centro_del_disco():
    """Holograma de INTENSIDAD con la emisión del pinhole, reconstruido
    dividiendo por amplitud con piso. Pinhole de 20 um: el disco (|a|² >= 0.05,
    138 px de radio) cubre la malla de 256 salvo las esquinas. En el centro del
    disco (|a|² >= 0.5) rinde lo mismo que sin fuente; dividiendo por la
    media, no. Medido: 0.888 con fuente, 0.893 sin fuente, 0.313 dividiendo
    por la media. Hacia el borde de la luz la división de la salida amplifica
    la gemela (0.656 en |a|² >= 0.05, contra 0.817 sin fuente): por eso se mide
    en el centro."""
    n = 256
    a = dlhm.fuente_pinhole(20e-3, LAMB)
    t = manchas(n, DO, [(0.0, 0.0), (0.03, -0.02), (-0.025, 0.03), (0.02, 0.035)], 6e-3)
    fondo = dlhm.iluminacion((n, n), DELTA, L, a)
    centro = fondo >= 0.5
    sin = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, device="cpu"))
    con = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, fuente=a, device="cpu"))

    def corr(holo, **kw):
        tr = a_numpy(dlhm.reconstruir(holo, DELTA, LAMB, Z, L, device="cpu", **kw))
        return correlacion((tr - 1)[centro], (t - 1)[centro])

    c_sin, c_con = corr(sin), corr(con, fondo=fondo, piso=0.05)
    assert c_con >= c_sin - 0.02, (c_con, c_sin)
    assert corr(con, fondo="media") < 0.5


# -------------------------------------------------------------------- guardas
def test_una_ventana_que_no_cabe_en_el_periodo_aborta():
    _, c = _holograma_prueba(n=64)
    R = dlhm.Reconstructor(c, DELTA, LAMB, L, Z, device="cpu")
    with pytest.raises(ValueError, match="copias periódicas"):
        R.campo(Z, forma=(64, 64 * 40))


def test_z_fuera_del_rango_del_reconstructor_aborta():
    _, c = _holograma_prueba(n=64)
    R = dlhm.Reconstructor(c, DELTA, LAMB, L, (2.9, 3.1), device="cpu")
    with pytest.raises(ValueError, match="fuera del rango"):
        R.campo(3.5)


@pytest.mark.parametrize("z", [0.0, L, 12.0])
def test_geometria_imposible_aborta(z):
    with pytest.raises(ValueError, match="0 < z < L"):
        dlhm.Reconstructor(np.ones((16, 16)), DELTA, LAMB, L, z, device="cpu")


def test_el_escalado_exige_un_solo_paso():
    with pytest.raises(ValueError, match="un solo"):
        dlhm.reconstruir_escalado(np.ones((16, 16)), (DELTA, 2 * DELTA), LAMB, Z, L,
                                  device="cpu")


def test_el_escalado_rechaza_un_fondo_que_varia_y_acepta_uno_constante():
    """El escalado divide por el fondo (D4): con un fondo imagen que varia no
    es lo que hace reconstruir() (normaliza por amplitud), asi que aborta y
    manda a reconstruir()/Reconstructor. Un fondo array CONSTANTE es lo mismo
    que un escalar: da lo mismo que sin fondo."""
    _, c = _holograma_prueba(n=64)
    varia = 1 + 0.2 * np.linspace(0, 1, c.size).reshape(c.shape)
    with pytest.raises(ValueError, match="reconstruir"):
        dlhm.reconstruir_escalado(c * varia, DELTA, LAMB, Z, L, fondo=varia, device="cpu")
    sin = a_numpy(dlhm.reconstruir_escalado(c, DELTA, LAMB, Z, L, device="cpu"))
    const = a_numpy(dlhm.reconstruir_escalado(c, DELTA, LAMB, Z, L,
                                              fondo=np.ones_like(c), device="cpu"))
    assert error_relativo(const - 1, sin - 1) < 1e-12


# ------------------------------------------------------------------ CPU / GPU
@pytest.mark.gpu
@pytest.mark.skipif(not gpu_disponible(), reason="sin GPU CUDA")
def test_gpu_da_lo_mismo_que_cpu():
    """complex64 en GPU frente a complex128 en CPU: la diferencia es la de la
    mantisa, no otra implementación."""
    t, c = _holograma_prueba(n=128)
    cpu = a_numpy(dlhm.reconstruir(c, DELTA, LAMB, Z, L, device="cpu"))
    gpu = a_numpy(dlhm.reconstruir(c, DELTA, LAMB, Z, L, device="gpu"))
    assert error_relativo(gpu - 1, cpu - 1) < 1e-4
    Vc = dlhm.holograma(t, DO, LAMB, Z, L, (128, 128), DELTA, complejo=True, device="cpu")
    Vg = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (128, 128), DELTA, complejo=True, device="gpu"))
    assert error_relativo(Vg, Vc) < 1e-4


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_disponible(), reason="sin GPU CUDA")
def test_gpu_da_lo_mismo_que_cpu_con_fuente():
    """La fuente se evalúa en la CPU en float64 y sube ya evaluada: en la GPU
    solo cambia la mantisa. Medido: 1.6e-6."""
    a = dlhm.fuente_pinhole(5e-3, LAMB)
    X0 = (0.7, 1.9)
    t = manchas(64, DO, [(0.0, 0.0), (0.01, -0.01)], 3e-3)
    kw = dict(centro_objeto=(X0[0] / M, X0[1] / M), centro=X0, fuente=a)
    cpu = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (64, 64), DELTA, device="cpu", **kw))
    gpu = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (64, 64), DELTA, device="gpu", **kw))
    fondo = dlhm.iluminacion((64, 64), DELTA, L, a, centro=X0)
    assert error_relativo(gpu - fondo, cpu - fondo) < 1e-4


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_disponible(), reason="sin GPU CUDA")
def test_gpu_da_lo_mismo_que_cpu_con_fondo_imagen():
    n = 64
    a = dlhm.fuente_pinhole(20e-3, LAMB)
    t = manchas(n, DO, [(0.0, 0.0), (0.01, -0.01)], 3e-3)
    holo = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, fuente=a, device="cpu"))
    fondo = dlhm.iluminacion((n, n), DELTA, L, a)
    kw = dict(fondo=fondo, piso=0.05)
    cpu = a_numpy(dlhm.reconstruir(holo, DELTA, LAMB, Z, L, device="cpu", **kw))
    gpu = a_numpy(dlhm.reconstruir(holo, DELTA, LAMB, Z, L, device="gpu", **kw))
    assert error_relativo(gpu - 1, cpu - 1) < 1e-4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
