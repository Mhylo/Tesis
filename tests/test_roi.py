"""Verificación del recorte de una ventana de la imagen.

Lo que se protege aquí no es aritmética de índices, son dos DECISIONES:

  · Una ROI que se sale de la malla revienta en vez de ajustarse al borde.
    Ajustar en silencio devolvería una ventana con la forma pedida y el
    contenido equivocado: un resultado creíble y falso. Misma familia que
    SinMedir en montaje.py y _comprobar_ventana() en propagadores.py.

  · El recorte es SECO, sin margen de guarda, y eso cuesta. El coste está
    medido en test_recortar_y_propagar_no_es_propagar_y_recortar, que es la
    prueba que justifica el resto del módulo.
"""

import numpy as np
import pytest

from CamposT.pipeline import propagar
from CamposT.roi import Roi, informe, radio_del_cono

# escenario in-line, el mismo de tests/test_retropropagacion.py
DELTA = 3.45e-3      # mm, paso de pixel del sensor
LAMB = 405e-6        # mm
Z = 16.0             # mm


# --- geometría ---------------------------------------------------------------
def test_recortar_devuelve_exactamente_la_ventana_pedida():
    """La forma es (alto, ancho): fila primero, como cualquier array 2D."""
    U = np.arange(64, dtype=float).reshape(8, 8)
    recorte = Roi(x0=2, y0=3, ancho=4, alto=2).recortar(U)
    assert recorte.shape == (2, 4)
    assert np.array_equal(recorte, U[3:5, 2:6])


def test_salirse_de_la_malla_revienta_en_vez_de_ajustarse():
    """Ajustar al borde devolvería otra ventana con la forma correcta."""
    U = np.zeros((8, 8))
    with pytest.raises(ValueError, match="se sale"):
        Roi(6, 0, 4, 4).recortar(U)
    with pytest.raises(ValueError, match="se sale"):
        Roi(0, 6, 4, 4).recortar(U)


def test_recortar_un_array_que_no_es_2d_falla():
    with pytest.raises(ValueError, match="2D"):
        Roi(0, 0, 2, 2).recortar(np.zeros((4, 4, 3)))


@pytest.mark.parametrize("ancho,alto", [(1, 4), (4, 1), (0, 4), (4, 0)])
def test_una_ventana_de_menos_de_dos_pixeles_revienta(ancho, alto):
    with pytest.raises(ValueError, match="no da para propagar"):
        Roi(0, 0, ancho, alto)


def test_un_origen_negativo_revienta():
    with pytest.raises(ValueError, match="negativo"):
        Roi(-1, 0, 4, 4)


def test_coordenadas_no_enteras_revientan():
    """Redondear por su cuenta decidiría si la ventana contiene lo que
    marcaste o se queda por dentro, y esa decisión no es del constructor."""
    with pytest.raises(TypeError, match="enteros"):
        Roi(0.5, 0, 4, 4)


def test_una_coordenada_de_numpy_vale_y_se_guarda_como_int():
    """elegir() y argparse pueden entregar enteros de NumPy; no son un error,
    pero dentro se guardan como int para que el repr sea legible."""
    roi = Roi(np.int64(2), np.int32(3), 4, 5)
    assert (roi.x0, roi.y0) == (2, 3)
    assert type(roi.x0) is int


def test_la_roi_es_inmutable():
    """Como TABLA1: una ROI que cambia a mitad de corrida es un recorte que no
    se puede anotar."""
    roi = Roi(0, 0, 4, 4)
    with pytest.raises(Exception):
        roi.x0 = 5


# --- reproducibilidad --------------------------------------------------------
def test_como_argumento_da_la_linea_que_repite_el_recorte():
    assert Roi(312, 208, 256, 256).como_argumento() == "--roi 312 208 256 256"


def test_str_da_la_forma_legible_que_va_en_los_titulos():
    """Formato congelado: la Task 6 lo mete en el suptitle de la figura."""
    assert str(Roi(312, 208, 256, 256)) == "256x256 px en (312, 208)"


# --- el cono de difraccion ---------------------------------------------------
def test_sin_angulo_propagante_el_cono_es_infinito():
    """Con lambda >= 2*delta no hay onda propagante que la malla represente.
    inf es la respuesta correcta: no hay ventana suficientemente grande."""
    assert radio_del_cono(20, lamb=2 * 3.45e-3, delta=3.45e-3) == np.inf
    assert radio_del_cono(20, lamb=1e-2, delta=3.45e-3) == np.inf


def test_el_cono_crece_con_la_distancia():
    radios = [radio_del_cono(z, 633e-6, 3.45e-3) for z in (20, 60, 150)]
    assert radios == sorted(radios)
    # el numero que cita el docstring del modulo, y que hay que corregir si
    # alguna vez cambia la formula
    assert radios[0] == pytest.approx(534.1, abs=0.5)


def test_el_cono_no_distingue_el_signo():
    """Va con |z|: retropropagar a -z reparte la luz sobre el mismo disco."""
    assert (radio_del_cono(-20, 633e-6, 3.45e-3)
            == radio_del_cono(20, 633e-6, 3.45e-3))


# --- el informe --------------------------------------------------------------
def test_el_informe_avisa_cuando_la_ventana_se_queda_corta():
    texto = informe(Roi(0, 0, 256, 256), (1024, 1024), 20, 633e-6, 3.45e-3)
    assert "AVISO" in texto
    assert "534" in texto, "el radio del cono tiene que salir con un numero"
    assert "--roi 0 0 256 256" in texto


def test_el_informe_no_avisa_cuando_la_ventana_cubre_el_cono():
    texto = informe(Roi(0, 0, 1400, 1400), (2048, 2048), 20, 633e-6, 3.45e-3)
    assert "AVISO" not in texto


def test_el_informe_da_el_cono_en_los_dos_extremos_del_barrido():
    """r crece con z: en un barrido, un solo numero miente."""
    texto = informe(Roi(0, 0, 256, 256), (1024, 1024), [20, 150],
                    633e-6, 3.45e-3)
    assert "534" in texto and "4006" in texto


def test_el_informe_de_una_sola_distancia_no_la_repite():
    texto = informe(Roi(0, 0, 256, 256), (1024, 1024), 20, 633e-6, 3.45e-3)
    assert texto.count("cono a z") == 1


# --- lo que cuesta el recorte seco -------------------------------------------
def test_recortar_y_propagar_no_es_propagar_y_recortar():
    """LA PRUEBA QUE JUSTIFICA EL MODULO: el coste de D2, medido.

    Recortar y propagar el trozo NO da lo mismo que propagar entero y recortar
    el resultado, porque la ventana tira la luz que venia de fuera y ademas
    crea dos bordes duros que el plano entero no tenia.

    Medido en este escenario: error maximo relativo 0.2099 y rms relativo
    0.1017 con FFT-ASM (con BL-ASM, 0.0903 y 0.0391: la mascara de banda
    limitada tambien recorta parte de lo que aqui se pierde). El umbral se
    deja holgado en 0.05 porque lo que la prueba fija es que la diferencia
    EXISTE y es de este orden, no un valor exacto.

    Es deliberado que no haya una asercion de que el error se concentre en el
    borde: se comprobo, y NO es asi. En este escenario el maximo cae en el
    nucleo (0.2099) y el marco de 8 px da la mitad (0.1212), porque el objeto
    esta en el centro y es donde el campo varia mas. Los anillos en el borde
    son un rasgo visual de la reconstruccion, no donde vive el error maximo.

    Si algun dia se implementa el margen de guarda, esta es la prueba que dira
    si funciono.
    """
    t = np.ones((128, 128))
    t[56:72, 56:72] = 0.0                      # particula opaca centrada
    U = t.astype(complex)
    roi = Roi(32, 32, 64, 64)

    antes, _ = propagar(roi.recortar(U), DELTA, LAMB, Z, metodo="fft",
                        pad=1, device="cpu")
    entero, _ = propagar(U, DELTA, LAMB, Z, metodo="fft", pad=1, device="cpu")
    despues = roi.recortar(entero)

    assert antes.shape == despues.shape == (64, 64)
    error = np.max(np.abs(antes - despues)) / np.max(np.abs(despues))
    assert error > 0.05, (
        f"error relativo {error:.4f}: recortar y propagar deberia diferir de "
        f"propagar y recortar. Si esto baja a cero, alguien metio un margen de "
        f"guarda y hay que actualizar D2 del spec.")


# --- matplotlib perezoso -----------------------------------------------------
def test_importar_roi_no_arrastra_matplotlib():
    """Un modulo del paquete que al importarse exige un backend grafico deja de
    poder usarse desde un test, un servidor sin pantalla o un cuaderno. Es la
    misma politica que hace que CamposT/__init__.py no importe nada, para no
    arrastrar CuPy.

    Va en un subproceso porque en la sesion de pytest matplotlib ya esta
    importado por otras pruebas: preguntarle a sys.modules aqui no probaria
    nada.
    """
    import subprocess
    import sys

    codigo = ("import sys, CamposT.roi; "
              "sys.exit(1 if 'matplotlib' in sys.modules else 0)")
    hecho = subprocess.run([sys.executable, "-c", codigo],
                           capture_output=True, text=True)
    assert hecho.returncode == 0, (
        f"import CamposT.roi arrastro matplotlib.\n{hecho.stderr}")


# --- _vista(): la imagen que se le ensena al raton ---------------------------
def test_vista_usa_la_fase_cuando_la_intensidad_no_tiene_contraste():
    """Un objeto de fase pura tiene |U| == 1 en todo el plano: la intensidad
    sale constante y blanca, sin nada que arrastrar con el raton. _vista()
    tiene que caer en la fase, que ahi si varia."""
    from CamposT.roi import _vista

    m, n = 30, 40
    t = np.linspace(0, 2 * np.pi, m * n).reshape(m, n)   # no constante
    U = np.exp(1j * t)                                    # objeto de fase pura

    I = np.abs(U) ** 2
    assert I.max() - I.min() < 1e-9, (
        "un objeto de fase pura tiene que dar |U| ~= 1")

    vista = _vista(U)
    assert vista.shape == U.shape
    assert vista.min() >= 0.0 and vista.max() <= 1.0
    assert vista.std() > 0.05, (
        "tiene que haber contraste real, no una imagen plana")


def test_vista_usa_la_intensidad_para_un_objeto_de_amplitud():
    """Con contraste real en |U|^2, _vista() no toca la fase: sigue siendo la
    normalizacion de siempre."""
    from CamposT.roi import _vista

    t = np.linspace(0.1, 1.0, 30 * 40).reshape(30, 40)
    U = t.astype(complex)
    I = np.abs(U) ** 2

    assert np.allclose(_vista(U), I / I.max())


# --- las dos CLIs comparten los mismos argumentos ----------------------------
def test_como_argumento_hace_round_trip_por_el_parser_real():
    """Un formato que solo se comprueba contra si mismo no garantiza nada: se
    parsea con el parser DE VERDAD de la CLI y tiene que salir la misma Roi."""
    from CamposT.retropropagacion import _parser

    original = Roi(312, 208, 256, 256)
    args = _parser().parse_args(
        ["h.png", "--z", "20", *original.como_argumento().split()])
    assert Roi(*args.roi) == original


def test_roi_y_roi_interactivo_juntos_son_un_error():
    from CamposT.retropropagacion import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(["h.png", "--z", "20", "--roi", "0", "0", "4", "4",
                              "--roi-interactivo"])


def test_sin_roi_no_hay_roi():
    from CamposT.retropropagacion import _parser
    from CamposT.roi import desde_argumentos

    args = _parser().parse_args(["h.png", "--z", "20"])
    assert desde_argumentos(args) is None


# --- la CLI de la vuelta -----------------------------------------------------
def _png_de_prueba(ruta, n=64):
    """Un holograma diminuto: particula opaca sobre fondo claro."""
    from PIL import Image

    t = np.full((n, n), 255, dtype=np.uint8)
    t[n // 2 - 4:n // 2 + 4, n // 2 - 4:n // 2 + 4] = 0
    Image.fromarray(t).save(ruta)
    return ruta


def test_la_vuelta_recorta_y_el_png_sale_del_tamano_de_la_roi(tmp_path):
    """La ventana va RECTANGULAR a proposito: con una cuadrada, cruzar los ejes
    en algun punto de la cadena daria exactamente el mismo tamano y la prueba
    no se enteraria. PIL da size = (ancho, alto)."""
    from PIL import Image

    from CamposT.retropropagacion import main

    holo = _png_de_prueba(tmp_path / "h.png", n=64)
    salida = tmp_path / "out"
    main([str(holo), "--z", "16", "--delta", "3.45e-3", "--lamb", "405e-6",
          "--metodos", "fft", "--device", "cpu", "--pad", "1",
          "--roi", "16", "8", "32", "24", "--salida", str(salida)])

    escritos = sorted((salida / "fft").glob("*.png"))
    assert len(escritos) == 1
    assert Image.open(escritos[0]).size == (32, 24)


def test_la_roi_se_mide_sobre_la_imagen_YA_redimensionada(tmp_path):
    """--N redimensiona antes de recortar. Una ROI de 32 px sobre una imagen
    reducida a 32x32 es la imagen entera; sobre el archivo de 64x64 seria un
    cuarto. El orden es cargar -> recortar -> propagar."""
    from PIL import Image

    from CamposT.retropropagacion import main

    holo = _png_de_prueba(tmp_path / "h.png", n=64)
    salida = tmp_path / "out"
    main([str(holo), "--z", "16", "--delta", "3.45e-3", "--lamb", "405e-6",
          "--metodos", "fft", "--device", "cpu", "--pad", "1", "--N", "32",
          "--roi", "0", "0", "32", "32", "--salida", str(salida)])

    assert Image.open(next((salida / "fft").glob("*.png"))).size == (32, 32)


def test_una_roi_que_se_sale_aborta_la_corrida(tmp_path):
    """La CLI sale con SystemExit limpio, no con el traceback de un
    ValueError: mismo trato que el resto de errores de usuario."""
    from CamposT.retropropagacion import main

    holo = _png_de_prueba(tmp_path / "h.png", n=64)
    with pytest.raises(SystemExit, match="se sale"):
        main([str(holo), "--z", "16", "--device", "cpu",
              "--roi", "40", "0", "32", "32", "--salida", str(tmp_path / "o")])


# --- la CLI de la ida --------------------------------------------------------
def test_la_ida_recorta_y_el_png_sale_del_tamano_de_la_roi(tmp_path):
    """Rectangular por la misma razon que en la vuelta: una ventana cuadrada no
    delataria unos ejes cruzados."""
    from PIL import Image

    from CamposT.propagacion import main

    obj = _png_de_prueba(tmp_path / "obj.png", n=64)
    salida = tmp_path / "out"
    main([str(obj), "--z", "16", "--delta", "3.45e-3", "--lamb", "405e-6",
          "--metodos", "fft", "--device", "cpu", "--pad", "1",
          "--roi", "16", "8", "32", "24", "--salida", str(salida)])

    escritos = sorted((salida / "fft").glob("*.png"))
    assert len(escritos) == 1
    assert Image.open(escritos[0]).size == (32, 24)


def test_la_ida_barre_distancias_como_la_vuelta(tmp_path):
    from CamposT.propagacion import main

    obj = _png_de_prueba(tmp_path / "obj.png", n=32)
    salida = tmp_path / "out"
    main([str(obj), "--z", "10", "20", "--pasos", "3", "--delta", "3.45e-3",
          "--lamb", "405e-6", "--metodos", "fft", "--device", "cpu",
          "--pad", "1", "--salida", str(salida)])
    assert len(list((salida / "fft").glob("*.png"))) == 3


@pytest.mark.parametrize("z_malo", ["-16", "0"])
def test_la_ida_no_acepta_distancias_no_positivas(tmp_path, z_malo):
    """La ida propaga a +z y la vuelta a -z. Una distancia negativa aqui es
    pedirle a la ida que haga de vuelta, y hay un modulo para eso."""
    from CamposT.propagacion import main

    obj = _png_de_prueba(tmp_path / "obj.png", n=32)
    with pytest.raises(SystemExit, match="positivas"):
        main([str(obj), "--z", z_malo, "--device", "cpu",
              "--salida", str(tmp_path / "o")])


def test_una_roi_que_se_sale_aborta_la_corrida_en_la_ida(tmp_path):
    """Mismo contrato que en la vuelta: SystemExit limpio, no un traceback."""
    from CamposT.propagacion import main

    obj = _png_de_prueba(tmp_path / "obj.png", n=64)
    with pytest.raises(SystemExit, match="se sale"):
        main([str(obj), "--z", "16", "--device", "cpu",
              "--roi", "40", "0", "32", "32", "--salida", str(tmp_path / "o")])


def test_el_modo_por_defecto_de_la_ida_es_amplitud():
    from CamposT.propagacion import _parser

    assert _parser().parse_args(["o.png", "--z", "16"]).modo == "amplitud"


def test_roi_interactivo_sin_campo_que_ensenar_revienta():
    """Hueco que dejo la Task 4: desde_argumentos() no puede abrir el selector
    sin la imagen, y esa guarda no la alcanzaba ninguna prueba. Las dos CLIs
    siempre le pasan un campo, asi que desde ellas es inalcanzable; esta
    prueba es la unica forma de que se compruebe, y ahora hay dos CLIs
    apoyadas en ella."""
    from CamposT.retropropagacion import _parser
    from CamposT.roi import desde_argumentos

    args = _parser().parse_args(["h.png", "--z", "20", "--roi-interactivo"])
    with pytest.raises(ValueError, match="necesita el campo"):
        desde_argumentos(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
