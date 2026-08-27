"""El modo 'mixto' de load_field: U0 = t * exp(i*phase_depth*t).

Un objeto que ABSORBE y RETARDA a la vez, con la fase acoplada a la amplitud y
phase_depth graduando cuanta hay. Los dos extremos ya existian:

    phase_depth = 0        ->  U0 = t              es 'amplitud'
    t identicamente 1      ->  U0 = exp(i*d)       es 'fase' con amplitud plana

LA TRAMPA QUE ESTAS PRUEBAS FIJAN

Sobre un objeto BINARIO -y usaf_like lo es, solo toma 0.0 y 1.0- el modo es
degenerado y phase_depth no hace NADA:

    donde t = 0  ->  0 * exp(0)     = 0
    donde t = 1  ->  1 * exp(i*d)   = exp(i*d)

o sea que el campo entero queda multiplicado por la constante exp(i*d), de
modulo 1. Eso es una fase GLOBAL: un cambio de origen de fases. Se cancela en
|U|^2, conmuta con la propagacion, y el holograma que sale es identico para
todo d. Medido barriendo d de 0 a 2*pi sobre usaf_like: contraste del
holograma 2.129 y correlacion de la reconstruccion 0.725, las dos constantes
hasta el ultimo decimal.

El modo necesita un objeto en ESCALA DE GRISES para significar algo. No es un
defecto que haya que arreglar: es lo que dice la formula. Queda escrito aqui
para que no haya que redescubrirlo mirando figuras que no cambian.
"""

import numpy as np
import pytest

from CamposT.campos import load_field, usaf_like


@pytest.fixture
def gris(tmp_path):
    """PNG en escala de grises, y su transmitancia esperada.

    Una rampa: 256 niveles distintos, que es justo lo que usaf_like no tiene.
    Los valores del PNG son enteros 0..255, asi que t = v/255 se recupera
    exacto y las pruebas pueden comparar contra la formula sin tolerancia de
    cuantizacion.
    """
    from PIL import Image
    v = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    ruta = tmp_path / "rampa.png"
    Image.fromarray(v, mode="L").save(ruta)
    return ruta, v.astype(float) / 255.0


def test_mixto_es_la_formula(gris):
    """U0 = t * exp(i*d*t), literalmente."""
    ruta, t = gris
    d = 1.7
    U = load_field(ruta, mode="mixto", phase_depth=d)
    assert np.allclose(U, t * np.exp(1j * d * t))


def test_con_phase_depth_cero_el_mixto_es_el_modo_amplitud(gris):
    """d = 0 tiene que devolver exactamente 'amplitud', no algo parecido.

    Es la continuidad del parametro: si el caso limite no coincide, el modo
    esta introduciendo algo de su cosecha.
    """
    ruta, _ = gris
    assert np.array_equal(load_field(ruta, mode="mixto", phase_depth=0.0),
                          load_field(ruta, mode="amplitud"))


def test_sobre_grises_phase_depth_cambia_el_campo(gris):
    """Con niveles intermedios, la fase deja de ser global: hay estructura.

    La comprobacion no es "los campos difieren" -difieren tambien por una fase
    global-, sino que la DIFERENCIA DE FASE entre dos puntos con distinta
    transmitancia cambia con d. Eso es lo que sobrevive a quitar el piston y lo
    unico que la propagacion puede convertir en contraste.
    """
    ruta, t = gris
    # Los dos puntos salen de la rampa, no de un valor escrito a mano: la
    # columna 64 es 64/255 = 0.25098, no 0.25. Comparar contra 0.25 haria que
    # la prueba midiera mi aritmetica en vez de la de load_field.
    i, j = 64, 191
    salto = t[0, j] - t[0, i]
    for d in (0.0, 1.0, 2.0):
        U = load_field(ruta, mode="mixto", phase_depth=d)
        delta_fase = np.angle(U[0, j]) - np.angle(U[0, i])
        assert delta_fase == pytest.approx(d * salto, abs=1e-9)


def test_sobre_binario_phase_depth_es_solo_una_fase_global(tmp_path):
    """LA TRAMPA. Sobre usaf_like, subir phase_depth no cambia nada medible.

    Se comprueba contra |U|^2 y contra la diferencia de fase dentro del
    soporte, que son las dos cosas que un experimento podria ver. La fase
    global si cambia -exp(i*d) es distinto para cada d- pero eso es el origen
    de fases, no el objeto.
    """
    from PIL import Image
    t = usaf_like(128)
    assert set(np.unique(t)) == {0.0, 1.0}, "usaf_like dejo de ser binario"
    ruta = tmp_path / "usaf.png"
    Image.fromarray((t * 255).astype(np.uint8), mode="L").save(ruta)

    U0 = load_field(ruta, mode="mixto", phase_depth=0.0)
    soporte = np.abs(U0) > 0.5
    for d in (0.5, np.pi, 2 * np.pi):
        U = load_field(ruta, mode="mixto", phase_depth=d)
        # el modulo es identico
        assert np.allclose(np.abs(U), np.abs(U0))
        # y la fase es CONSTANTE en todo el soporte: no hay estructura
        assert np.ptp(np.angle(U[soporte])) == pytest.approx(0.0, abs=1e-12)


def test_mixto_no_rompe_los_modos_que_ya_existian(gris):
    """Los cuatro modos anteriores siguen dando lo mismo."""
    ruta, t = gris
    assert np.allclose(load_field(ruta, mode="amplitud"), t)
    assert np.allclose(load_field(ruta, mode="fase", phase_depth=np.pi),
                       np.exp(1j * np.pi * t))
    assert np.allclose(load_field(ruta, mode="transmitancia"), (t > 0.5))
    assert np.allclose(load_field(ruta, mode="holograma"), np.sqrt(t))
    with pytest.raises(ValueError):
        load_field(ruta, mode="no_existe")
