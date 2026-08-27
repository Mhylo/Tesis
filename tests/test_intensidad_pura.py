"""Qué sale de la retropropagación cuando la intensidad se deja PURA.

`pipeline.intensidad` calcula |U|², pero por defecto lo divide por su máximo, y
`guardar` le aplica encima una gamma antes de escribir el PNG. Los tres
scripts de `scripts/retro_*.py` guardan con gamma 0.5 y sus figuras pintan
`(im / im.max()) ** 0.5`. O sea que ninguna imagen que sale hoy del barrido es
|U|²: son |U|.

Estas pruebas fijan qué hace cada capa por separado, y qué queda si se quitan
todas. No proponen cambiarlo: describen lo que hay, para que la diferencia
entre "la reconstrucción es así" y "la estamos pintando así" quede escrita en
algún sitio y no haya que volver a deducirla mirando figuras.

LA IDENTIDAD QUE LO RESUME

    (|U|² / max)^0.5  =  |U| / sqrt(max)

Una gamma de 0.5 sobre una intensidad ES la amplitud. No es una corrección de
visualización pequeña: es cambiar la magnitud física que se enseña.

EL OBJETO CONTRA EL QUE SE MIDE

`usaf_like` es binario, con 8.7 % de píxeles a 1 y el resto a 0 (con los
parámetros de aquí). Ese porcentaje es la vara: una reconstrucción fiel
debería tener su energía sobre más o menos ese soporte, y cuánto se aleja de
él dice cuánto se ha repartido por el plano —anillos, imagen gemela, DC—.
"""

import pathlib

import matplotlib
matplotlib.use("Agg")           # sin backend interactivo: la suite no abre ventanas
import matplotlib.pyplot as plt
import numpy as np
import pytest

from CamposT.campos import usaf_like
from CamposT.pipeline import intensidad, propagar

#: 512 y no menos: estas pruebas escriben figuras para mirar, y a 256 las
#: barras de los elementos pequenos del target caen por debajo de un par de
#: pixeles y no se distingue si el borrado lo hace la reconstruccion o la
#: malla. Cuesta ~1 s propagar, una sola vez para todo el modulo.
N = 512
DELTA = 3.45e-3         # mm
LAMB = 633e-6           # mm
Z = 50.0                # mm

#: Donde van las figuras. Bajo la raiz del repo, no relativo al directorio
#: desde el que se lance pytest.
SALIDA = (pathlib.Path(__file__).resolve().parent.parent
          / "resultados" / "intensidad_pura")


@pytest.fixture(scope="module")
def campos():
    """Objeto, holograma y las dos vueltas. Se propaga una vez para todas.

    Vuelta A parte del campo complejo del holograma; vuelta B de sqrt(|U|²),
    que es lo único que da un sensor. La diferencia entre las dos es la fase
    perdida en la medida.
    """
    t = usaf_like(N)
    U0 = t.astype(complex)
    kw = dict(metodo="fft", pad=2, device="cpu", dtype=np.complex128)
    U_h, _ = propagar(U0, DELTA, LAMB, +Z, **kw)
    I_h = np.abs(U_h) ** 2
    U_a, _ = propagar(U_h, DELTA, LAMB, -Z, **kw)
    U_b, _ = propagar(np.sqrt(I_h).astype(complex), DELTA, LAMB, -Z, **kw)
    return {"objeto": U0, "A": U_a, "B": U_b}


def soporte(I, umbral=1 / 255):
    """Fracción del plano que sobrevive a 8 bits: I >= umbral · max.

    Es la pregunta operativa, no una métrica abstracta: lo que caiga por
    debajo de 1/255 del máximo se escribe como 0 en el PNG y desaparece de la
    figura. Mide cuánto del plano se ve, no cuánta energía hay.
    """
    return float((I / I.max() >= umbral).mean())


# ------------------------------------------------------- qué hace cada capa
def test_intensidad_sin_normalizar_es_exactamente_el_modulo_cuadrado(campos):
    """La forma pura, bit a bit: sin escalas, sin gamma, sin recortes."""
    U = campos["A"]
    assert np.array_equal(intensidad(U, normalizar=False), np.abs(U) ** 2)


def test_normalizar_por_defecto_borra_la_escala_absoluta(campos):
    """El defecto divide por el máximo, así que dos planos del barrido salen
    con el mismo blanco aunque les llegue energía muy distinta.

    Es lo que hace comparables las imágenes entre sí, y a la vez lo que impide
    leer en ellas cuánta luz hay realmente en cada plano.
    """
    U = campos["A"]
    pura = intensidad(U, normalizar=False)
    # el maximo absoluto depende del objeto y de N, y no es 1: la
    # retropropagacion concentra energia por encima del maximo del objeto
    assert pura.max() > 1.0
    assert intensidad(U).max() == pytest.approx(1.0)
    assert np.array_equal(intensidad(U), pura / pura.max())


def test_la_gamma_de_media_convierte_la_intensidad_en_amplitud(campos):
    """(|U|² / max)^0.5 = |U| / sqrt(max). Ver el docstring del módulo.

    Es la identidad que explica por qué las figuras del barrido no enseñan lo
    que su título dice: con gamma 0.5 sobre |U|², lo pintado es |U|.
    """
    U = campos["A"]
    con_gamma = intensidad(U) ** 0.5
    amplitud = np.abs(U) / np.sqrt((np.abs(U) ** 2).max())
    assert np.allclose(con_gamma, amplitud, rtol=1e-12, atol=1e-15)


# ------------------------------------------ qué sale con la intensidad pura
def test_la_intensidad_pura_de_la_vuelta_A_se_concentra_donde_el_objeto(campos):
    """Sin gamma, la vuelta A cae cerca del soporte real del objeto.

    El objeto ocupa ~8.7 % del plano. En |U|² puro la reconstrucción visible
    ocupa ~22 %: las barras más el anillado de la banda perdida. Con gamma 0.5
    sube a ~76 %, o sea que TRES CUARTAS PARTES de lo que se ve en el panel de
    la vuelta A es fondo levantado por la gamma, no señal.
    """
    obj = soporte(intensidad(campos["objeto"], normalizar=False))
    pura = soporte(intensidad(campos["A"], normalizar=False))
    con_gamma = soporte(intensidad(campos["A"], normalizar=False) ** 0.5)

    assert obj == pytest.approx(0.087, abs=0.02)
    assert pura < 0.35
    assert con_gamma > 2.5 * pura


def test_la_intensidad_pura_de_la_vuelta_B_no_tiene_fondo_oscuro(campos):
    """La gemela llena el plano, y eso no es un efecto de la gamma.

    Ya en |U|² puro la vuelta B ocupa >85 % del plano, contra el ~22 % de la
    vuelta A. Quitar la gamma hace la vuelta A mucho más limpia y a la B casi
    nada: su fondo no es una elección de pintado, es la imagen gemela.
    """
    pura_a = soporte(intensidad(campos["A"], normalizar=False))
    pura_b = soporte(intensidad(campos["B"], normalizar=False))
    assert pura_b > 0.85
    assert pura_b > 3 * pura_a


def test_el_suelo_de_la_vuelta_B_esta_dos_ordenes_por_encima_del_de_la_A(campos):
    """La mediana normalizada: dónde está el píxel típico del plano.

    En la vuelta A es ~2e-4 del máximo -plano oscuro con estructura brillante-.
    En la B es ~4e-2, doscientas veces más alto. Ése es el nivel de la gemela,
    y es contra ese fondo contra el que compite cualquier métrica de foco.
    """
    med = lambda U: float(np.median(intensidad(U)))
    assert med(campos["A"]) < 1e-3
    assert med(campos["B"]) > 1e-2
    assert med(campos["B"]) > 50 * med(campos["A"])


# ------------------------------------------------------------------ figuras
@pytest.fixture(scope="module")
def salida():
    """Carpeta de las figuras, creada al vuelo.

    Estas pruebas escriben a resultados/ y no a tmp_path a proposito: la
    diferencia entre |U|^2 y |U| es la clase de cosa que hay que MIRAR, y una
    figura en un directorio temporal que pytest borra al terminar no se mira.
    """
    SALIDA.mkdir(parents=True, exist_ok=True)
    return SALIDA


def _png(nombre, I, salida):
    """Escribe un mapa normalizado por su maximo, sin gamma, a 8 bits."""
    from PIL import Image
    A = np.asarray(I, dtype=np.float64)
    m = A.max()
    A = np.clip(A / m, 0, 1) if m > 0 else np.zeros_like(A)
    destino = salida / nombre
    Image.fromarray((A * 255).astype(np.uint8)).save(destino)
    return destino


def test_escribe_las_intensidades_puras_a_tamano_completo(campos, salida):
    """Un PNG por vuelta y por tratamiento, sin reescalar, para poder ampliar.

    Los cuatro juntos son la comparacion: A pura contra A con gamma es lo que
    ensena cuanto fondo levanta el pintado; B pura contra B con gamma, que en
    la vuelta B el fondo no lo levanta nadie porque ya estaba.
    """
    escritos = []
    for vuelta in ("A", "B"):
        I = intensidad(campos[vuelta], normalizar=False)
        escritos.append(_png(f"{vuelta}_pura.png", I, salida))
        escritos.append(_png(f"{vuelta}_gamma05.png", I ** 0.5, salida))

    for destino in escritos:
        assert destino.is_file()
        assert destino.stat().st_size > 1000        # no es un PNG vacio


def test_escribe_la_figura_de_comparacion(campos, salida):
    """La figura de dos filas por tres columnas: objeto, pura, gamma.

    El porcentaje del titulo de cada panel es la fraccion del plano que
    sobrevive a 8 bits, que es lo que decide si un pixel se ve o sale negro.
    Es el numero que hace legible la figura: sin el, las dos columnas de la
    derecha parecen dos ajustes de brillo del mismo dato.
    """
    I_obj = intensidad(campos["objeto"], normalizar=False)
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 9))
    for k, vuelta in enumerate(("A", "B")):
        I = intensidad(campos[vuelta], normalizar=False)
        paneles = [(f"objeto |U0|^2  ({100 * soporte(I_obj):.0f} % visible)", I_obj),
                   (f"|U|^2 PURA  ({100 * soporte(I):.0f} % visible)", I),
                   (f"gamma 0.5 = |U|  ({100 * soporte(I ** 0.5):.0f} % visible)",
                    I ** 0.5)]
        for j, (titulo, im) in enumerate(paneles):
            im = np.asarray(im, float)
            ax[k, j].imshow(im / im.max(), cmap="gray", vmin=0, vmax=1)
            ax[k, j].set_title(titulo, fontsize=9.5)
            ax[k, j].axis("off")
        etiqueta = "campo complejo" if vuelta == "A" else "de la intensidad"
        ax[k, 1].set_title(f"vuelta {vuelta} ({etiqueta}) -- " + paneles[1][0],
                           fontsize=9.5)
    fig.suptitle(f"Intensidad pura contra lo que pintan las figuras del "
                 f"barrido   (z = {Z:g} mm, N = {N})", fontsize=11)
    fig.tight_layout()
    destino = salida / "comparacion.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)

    assert destino.is_file()
    assert destino.stat().st_size > 10_000
