"""Verificación de la retropropagación de hologramas.

Lo que se demuestra aquí es que retropropagar a la distancia correcta
RE-LOCALIZA el objeto: la señal que en el holograma está repartida por todo el
plano en anillos de difracción vuelve a concentrarse donde estaba el objeto.
Ésa es la propiedad que no depende de ninguna referencia analítica y que
distingue una reconstrucción de un filtro que suaviza.

Por qué el signo NO se verifica mirando la reconstrucción
---------------------------------------------------------
Un holograma medido es real y no negativo, así que el campo de entrada
U_h = sqrt(I) es real. Y para un campo de entrada real,

    U(−z) = conj(U(+z))     ->     |U(−z)|² = |U(+z)|²

exactamente (medido aquí: 3.5e-16). Propagar el holograma en el sentido
equivocado da la MISMA intensidad, no una peor. Es la ambigüedad de la imagen
gemela, y es la razón de que la convención de signo tenga que fijarse por
contrato contra pipeline.propagar() —lo que hace
test_retropropagar_es_propagar_a_z_negativo— y no por inspección de una
imagen: sobre la intensidad no hay nada que inspeccionar.

Donde el signo sí se nota es en la FASE del campo reconstruido, y en cuanto se
encadena cualquier paso que no sea real: la corrección de fuente puntual de
DLHM, un filtro complejo, o una iteración de recuperación de fase.

El escenario
------------
Partícula opaca pequeña sobre fondo transparente: la geometría in-line
clásica, donde el fondo hace de onda de referencia. Con N=128, delta=3.45 µm y
lamb=405 nm, a z=16 mm la difracción se extiende sqrt(lamb·z) = 80 µm ≈ 23
píxeles, casi tres veces el objeto, así que el holograma no se le parece en
nada y la reconstrucción tiene todo que recuperar.
"""

import numpy as np
import pytest
from conftest import error_relativo
from PIL import Image

from CamposT.campos import load_field
from CamposT.pipeline import intensidad, propagar
from CamposT.retropropagacion import (METODOS, barrido_z, nombre_png,
                                      retropropagar)

# --- escenario in-line -------------------------------------------------------
N = 128
DELTA = 3.45e-3      # mm, paso de píxel del sensor
LAMB = 405e-6        # mm
Z = 16.0             # mm, separación sensor-objeto
LADO = 8             # px, lado de la partícula


def particula(n=N, lado=LADO):
    """Objeto de prueba: cuadrado opaco centrado sobre fondo transparente."""
    t = np.ones((n, n))
    i = (n - lado) // 2
    t[i:i + lado, i:i + lado] = 0.0
    return t


def holograma_sintetico(t, z=Z, device="cpu"):
    """Intensidad que registraría el sensor a distancia z del objeto.

    Es el dato de partida real: sólo módulo, la fase se pierde en la medida.
    """
    U, _ = propagar(t.astype(complex), DELTA, LAMB, z, metodo="fft", pad=1,
                    device=device)
    return intensidad(U, normalizar=False)


def concentracion(I, t):
    """Fracción del contraste que cae dentro de la huella del objeto.

    Se mide sobre |I/Ī − 1|, el apartamiento del fondo uniforme: en el
    holograma ese apartamiento son los anillos de difracción repartidos por
    todo el plano, y en una reconstrucción correcta vuelve a estar donde
    estaba el objeto. La huella ocupa el 0.39 % del plano, así que ése es el
    valor de una imagen sin ninguna estructura: cualquier cifra por encima
    mide localización real.

    No se compara contra el objeto término a término a propósito. La fase
    perdida en la medida deja la imagen gemela superpuesta, así que la
    reconstrucción NUNCA es el objeto y una métrica de parecido mezclaría dos
    cosas: si la retropropagación funciona, y cuánto estorba la gemela. Ésta
    sólo mide la primera.
    """
    I = np.asarray(_cpu(I), dtype=np.float64)
    D = np.abs(I / I.mean() - 1.0)
    return float(D[t == 0].sum() / D.sum())


# --- ida y vuelta ------------------------------------------------------------
def test_retropropagar_relocaliza_el_objeto(device):
    """Del objeto al holograma y del holograma al objeto.

    A z=16 mm el holograma tiene el 0.10 % de su contraste sobre la huella del
    objeto —es decir, nada: la huella misma es el 0.39 % del plano, así que la
    señal está completamente deslocalizada—. Retropropagarlo devuelve ~7.5 %,
    unas veinte veces la huella. Ese salto es lo que hace la retropropagación
    y lo que ningún filtro de suavizado produce.
    """
    t = particula()
    I_hol = holograma_sintetico(t, device=device)
    U_h = np.sqrt(np.asarray(_cpu(I_hol), dtype=np.float64)).astype(complex)

    (_, _, U_rec, _), = retropropagar(U_h, DELTA, LAMB, [Z], metodos=["fft"],
                                      pad=1, device=device)

    antes = concentracion(I_hol, t)
    despues = concentracion(intensidad(U_rec, normalizar=False), t)
    huella = float((t == 0).mean())

    assert despues > 10 * huella, (
        f"la reconstrucción no localiza el objeto: {despues:.4f} de contraste "
        f"sobre una huella de {huella:.4f} del plano")
    assert despues > 10 * antes, (
        f"la reconstrucción no concentra más que el holograma crudo "
        f"({despues:.4f} frente a {antes:.4f}): revisa la distancia")


def test_un_holograma_real_no_distingue_el_signo(device, tol):
    """La imagen gemela, medida.

    U_h = sqrt(I) es real, y para entrada real U(−z) = conj(U(+z)), así que
    las dos intensidades coinciden bit a bit. Propagar el holograma en el
    sentido equivocado NO produce una reconstrucción peor: produce la misma.

    Esta prueba existe para que quede escrito en la suite, no como propiedad
    deseable, sino como el límite del método: por eso la convención de signo se
    fija por contrato contra pipeline (test siguiente) y no mirando imágenes, y
    por eso la reconstrucción cruda siempre trae la gemela encima.
    """
    U_h = np.sqrt(particula()).astype(complex)

    (_, _, U_menos, _), = retropropagar(U_h, DELTA, LAMB, [Z], metodos=["fft"],
                                        pad=1, device=device)
    U_mas, _ = propagar(U_h, DELTA, LAMB, +Z, metodo="fft", pad=1,
                        device=device)

    assert error_relativo(U_menos, np.conj(_cpu(U_mas))) < tol
    I_menos = np.asarray(_cpu(intensidad(U_menos, normalizar=False)))
    I_mas = np.asarray(_cpu(intensidad(U_mas, normalizar=False)))
    assert np.allclose(I_menos, I_mas, rtol=tol, atol=tol * float(I_mas.max()))


# --- contrato de signo -------------------------------------------------------
@pytest.mark.parametrize("metodo", list(METODOS))
def test_retropropagar_es_propagar_a_z_negativo(metodo, device):
    """Fija la convención: distancias positivas entran, −z se propaga.

    Va por contrato y no por comentario porque es una decisión de interfaz que
    cualquier refactor puede invertir sin que nada más se rompa.
    """
    U_h = np.sqrt(particula()).astype(complex)
    extra = {"s": 1} if metodo == "mpasm" else {}

    (_, z_dado, U_retro, _), = retropropagar(U_h, DELTA, LAMB, [Z],
                                             metodos=[metodo], pad=1,
                                             device=device, **extra)
    U_ref, _ = propagar(U_h, DELTA, LAMB, -Z, metodo=metodo, pad=1,
                        device=device, **extra)

    assert z_dado == Z, "el z devuelto es el positivo que se pidió"
    assert error_relativo(U_retro, U_ref) == 0.0


# --- carga del holograma -----------------------------------------------------
def test_modo_holograma_toma_la_raiz(tmp_path):
    """La imagen es intensidad, así que el campo es su raíz.

    Tomarla como amplitud —lo que hace mode='amplitud'— eleva el campo al
    cuadrado; no falla, sólo reconstruye mal.
    """
    png = tmp_path / "h.png"
    rampa = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    Image.fromarray(rampa).save(png)

    amplitud = load_field(png, mode="amplitud")
    hol = load_field(png, mode="holograma")

    assert np.allclose(hol.real, np.sqrt(amplitud.real))
    assert np.all(hol.imag == 0)


def test_modo_desconocido_falla(tmp_path):
    """Un modo mal escrito tiene que fallar, no caer en un defecto silencioso."""
    png = tmp_path / "h.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(png)
    with pytest.raises(ValueError, match="mode desconocido"):
        load_field(png, mode="hologramaa")


# --- barrido -----------------------------------------------------------------
def test_barrido_de_un_solo_valor():
    assert barrido_z(20) == pytest.approx([20.0])
    assert barrido_z([20]) == pytest.approx([20.0])


def test_barrido_entre_dos_valores():
    zs = barrido_z([10, 60], pasos=6)
    assert zs == pytest.approx([10, 20, 30, 40, 50, 60])


def test_barrido_de_tres_valores_falla():
    """Ambiguo: ni extremos ni lista. Mejor fallar que adivinar."""
    with pytest.raises(ValueError, match="uno o dos valores"):
        barrido_z([10, 20, 30])


def test_los_nombres_del_barrido_no_colisionan():
    """Un linspace da distancias no enteras; redondearlas a entero haría que
    dos reconstrucciones distintas escribieran en el mismo archivo."""
    nombres = [nombre_png(z) for z in barrido_z([10, 11], pasos=25)]
    assert len(set(nombres)) == 25
    assert nombres == sorted(nombres), "el orden alfabético sigue al barrido"


# --- filtrado de parámetros --------------------------------------------------
def test_s_no_llega_a_los_propagadores_que_no_lo_aceptan():
    """s es de MPASM. Que llegue a fft_asm sería un TypeError, y evitar que el
    llamante lleve la cuenta de qué parámetro va con qué método es justo lo que
    retropropagar() existe para hacer."""
    U_h = np.sqrt(particula(32, 4)).astype(complex)
    salidas = list(retropropagar(U_h, DELTA, LAMB, [Z], pad=1, device="cpu",
                                 s=2))
    assert [m for m, _, _, _ in salidas] == list(METODOS)


def test_metodo_desconocido_falla():
    U_h = np.ones((8, 8), dtype=complex)
    with pytest.raises(ValueError, match="métodos desconocidos"):
        list(retropropagar(U_h, DELTA, LAMB, [Z], metodos=["kreuzer"]))


def test_parametro_ajeno_falla():
    """Un kwarg mal escrito se tragaría en silencio si sólo se filtrara por
    método: 'ss=2' no llegaría a nadie y MPASM correría con s=10."""
    U_h = np.ones((8, 8), dtype=complex)
    with pytest.raises(ValueError, match="no reconocidos"):
        list(retropropagar(U_h, DELTA, LAMB, [Z], ss=2))


def _cpu(a):
    from CamposT.backend import a_numpy
    return a_numpy(a)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
