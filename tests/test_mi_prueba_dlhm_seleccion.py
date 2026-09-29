"""La seleccion de foco de barrido_foco en scripts/mi_prueba_dlhm.py.

Todo en CPU y en mallas de 128x160 px, con la lambda y el delta del montaje.
Para que una malla tan chica tenga la NA del banco, L y z se reducen a 0.8 y
0.24 mm (M = 3.33, la del montaje): el sensor de 128 px abarca asi 0.21 de NA
desde la muestra. Con L = 10 mm esa malla veria 0.017 y el objeto saldria
borrado a cualquier z.

Lo que se fija (D1-D12 de
docs/superpowers/specs/2026-09-28-mi-prueba-dlhm-seleccion-design.md):

- D5  las metricas copiadas dan los mismos numeros que las de barrido_foco,
      y tienen el mismo cuerpo;
- D4  con OBJETO = "fase" la Tamura cambia de signo y Pearson va en valor
      absoluto, y la Tamura invertida enfoca en Z_REAL;
- D2  la ventana del motor exacto es el trozo de la reconstruccion entera, y
      los dos motores entregan la ventana alineada con la referencia;
- D3  el motor de Carlos exige un recorte cuadrado -el ROI cuando
      CARLOS_RECORTE es None- y divide por |U0|^2;
- el barrido grueso + fino encuentra Z_REAL con los dos motores;
- D6  Pearson o SSIM sin referencia abortan; D2 un rango que se sale de
      0 < z < L con la guarda del fino tambien;
- y main() de punta a punta: simula, guarda el holograma y lo vuelve a leer.
"""

import ast
import functools
import pathlib

import matplotlib
matplotlib.use("Agg")           # antes de importar el script, que trae pyplot

import cv2 as cv
import numpy as np
import pytest
from skimage.metrics import structural_similarity

from CamposT import dlhm
from scripts import mi_prueba_dlhm as mp

RAIZ = pathlib.Path(__file__).resolve().parents[1]
BARRIDO_FOCO = RAIZ / "scripts" / "barrido_foco.py"

#: Las que viajan copiadas desde barrido_foco.py.
COPIADAS = ("tamura", "pearson", "ssim", "anchura_media", "margen_del_pico")

LAMB, DELTA = 528e-6, 1.83e-3
L, Z = 0.8, 0.24
FORMA = (128, 160)
#: (fila, columna, alto, ancho): centrada y cuadrada, vale para los dos motores
VENTANA = (32, 48, 64, 64)
#: paso grueso de los barridos de prueba, 5 um
PASO = 0.005


def hay_gpu():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


sin_gpu = pytest.mark.skipif(not hay_gpu(), reason="no hay CUDA en esta maquina")


@pytest.fixture
def globales():
    """Deja como estaban las globales de las metricas: fijar_referencia las pisa."""
    antes = (mp.xp, mp.REF, mp.REF_CPU, mp.REF_RANGO)
    yield
    mp.xp, mp.REF, mp.REF_CPU, mp.REF_RANGO = antes


# ── D5: las metricas son las de barrido_foco ────────────────────────────────

def de_barrido_foco(ref):
    """Las COPIADAS sacadas de la FUENTE de barrido_foco.py, con sus globales.

    Con ast y no importando: barrido_foco.py corre su barrido entero al
    importarse. Se compilan solo esas cinco definiciones, en un espacio con las
    globales que leen.
    """
    arbol = ast.parse(BARRIDO_FOCO.read_text(encoding="utf-8"))
    defs = [n for n in arbol.body
            if isinstance(n, ast.FunctionDef) and n.name in COPIADAS]
    assert sorted(d.name for d in defs) == sorted(COPIADAS)
    espacio = {"np": np, "xp": np, "structural_similarity": structural_similarity,
               "REF": ref, "REF_CPU": ref, "REF_RANGO": float(ref.max() - ref.min())}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(BARRIDO_FOCO), "exec"),
         espacio)
    return espacio


def test_las_metricas_copiadas_dan_los_mismos_numeros(globales):
    """Mismos datos, mismos numeros, bit a bit: son la misma cuenta.

    Entran tambien los casos degenerados -un mapa constante, uno nulo- porque
    son los que recorren las ramas que devuelven 0.0 en vez de un NaN.
    """
    rng = np.random.default_rng(0)
    ref = rng.random((48, 64))
    bf = de_barrido_foco(ref)
    mp.fijar_referencia(ref, np)
    for A in (rng.random((48, 64)), 3 * ref + 1, -ref, np.ones((48, 64)),
              np.zeros((48, 64))):
        for n in ("tamura", "pearson", "ssim"):
            assert getattr(mp, n)(A, A) == bf[n](A, A), n
    zs = np.linspace(2.5, 3.5, 101)
    for c in (np.exp(-((zs - 3.0) / 0.05) ** 2) + 0.01 * rng.random(101),
              np.linspace(0.0, 1.0, 101), np.ones(101)):
        np.testing.assert_array_equal(mp.anchura_media(zs, c),
                                      bf["anchura_media"](zs, c))
        for ancho in (0.1, 1.0):
            z_pico = zs[int(np.argmax(c))]
            np.testing.assert_array_equal(mp.margen_del_pico(zs, c, z_pico, ancho),
                                          bf["margen_del_pico"](zs, c, z_pico, ancho))


def _cuerpos(ruta):
    """{nombre: firma y cuerpo sin el docstring} de las COPIADAS de un archivo."""
    arbol = ast.parse(pathlib.Path(ruta).read_text(encoding="utf-8"))
    cuerpos = {}
    for n in arbol.body:
        if isinstance(n, ast.FunctionDef) and n.name in COPIADAS:
            cuerpo = n.body
            if (isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant)
                    and isinstance(cuerpo[0].value.value, str)):
                cuerpo = cuerpo[1:]
            cuerpos[n.name] = (ast.dump(n.args)
                               + ast.dump(ast.Module(body=cuerpo, type_ignores=[])))
    return cuerpos


def test_las_copias_tienen_el_mismo_cuerpo():
    """Los docstrings pueden hablar de DLHM; el codigo no puede cambiar.

    Cubre las ramas que los datos de la prueba de arriba no pisen. Si falla,
    alguien toco una de las dos: se cambian las dos o ninguna.
    """
    assert _cuerpos(mp.__file__) == _cuerpos(BARRIDO_FOCO)


# ── D4: el sentido de OBJETO ────────────────────────────────────────────────

def test_en_fase_la_tamura_cambia_de_signo_y_pearson_va_en_valor_absoluto(globales):
    rng = np.random.default_rng(1)
    ref = rng.random((32, 32))
    mp.fijar_referencia(ref, np)
    A = 2 - ref                                    # anticorrelada con la referencia
    assert mp.puntuar("tamura", A, A, "fase") == -mp.tamura(A, A)
    assert mp.puntuar("tamura", A, A, "amplitud") == mp.tamura(A, A)
    assert mp.puntuar("pearson", A, A, "amplitud") == pytest.approx(-1.0)
    assert mp.puntuar("pearson", A, A, "fase") == pytest.approx(1.0)
    assert mp.puntuar("ssim", A, A, "fase") == mp.ssim(A, A)


def test_en_fase_tamura_mide_la_amplitud_y_las_de_referencia_la_fase():
    assert mp.clase_de("tamura", "fase") == "amplitud"
    assert mp.clase_de("pearson", "fase") == mp.clase_de("ssim", "fase") == "fase"
    assert {mp.clase_de(n, "amplitud") for n in mp.FUNCION} == {"amplitud"}
    t = np.exp(1j * np.array([[0.5, -1.0]])) * 2
    np.testing.assert_allclose(mp.mapa(t, "amplitud", np), [[2.0, 2.0]])
    np.testing.assert_allclose(mp.mapa(t, "fase", np), [[0.5, -1.0]])
    assert mp.mapa(t.astype(np.complex64), "fase", np).dtype == np.float64


# ── el objeto y sus hologramas ──────────────────────────────────────────────

def objeto_de_prueba(forma=FORMA):
    """a en [0, 1]: barras de 3, 4 y 6 px, tres puntos y una L, alrededor del
    centro y lejos del borde, con un paso bajo suave para que sea de banda
    limitada. Lo que se reconstruye bien con la NA de estas mallas."""
    M, N = forma
    a = np.zeros(forma)
    cy, cx = M // 2, N // 2
    for k, w in enumerate((3, 4, 6)):
        x0 = cx - 30 + k * 20
        for j in range(3):
            a[cy - 25:cy - 5, x0 + 2 * j * w:x0 + 2 * j * w + w] = 1
    yy, xx = np.ogrid[:M, :N]
    for dy, dx in ((10, -20), (12, -8), (15, 5)):
        a[(yy - cy - dy) ** 2 + (xx - cx - dx) ** 2 <= 9] = 1
    a[cy + 5:cy + 25, cx + 15:cx + 19] = 1
    a[cy + 21:cy + 25, cx + 15:cx + 30] = 1
    fy, fx = np.fft.fftfreq(M)[:, None], np.fft.fftfreq(N)[None, :]
    a = np.fft.ifft2(np.fft.fft2(a) * np.exp(-(fy ** 2 + fx ** 2) / (2 * 0.25 ** 2))).real
    return np.clip(a, 0, 1)


@functools.lru_cache(maxsize=None)
def holograma(tipo, campo):
    """(t, holograma) del objeto de prueba a Z. tipo "amplitud": t = 1 - 0.8a;
    "fase": t = exp(i*pi/2*a). campo "complejo" (1 + V) o "intensidad"."""
    a = objeto_de_prueba()
    t = 1 - 0.8 * a if tipo == "amplitud" else np.exp(1j * np.pi / 2 * a)
    V = dlhm.a_numpy(dlhm.holograma(t, DELTA * Z / L, LAMB, Z, L, FORMA, DELTA,
                                    complejo=True, device="cpu"))
    return t, (1 + V if campo == "complejo" else np.abs(1 + V) ** 2)


def motor_de(nombre, c, rango=(Z - 0.06, Z + 0.06)):
    if nombre == "exacto":
        return mp.MotorExacto(c, LAMB, DELTA, L, rango, device="cpu")
    return mp.MotorCarlos(c, VENTANA, LAMB, DELTA, L)


# ── D2 y D3: la ventana de cada motor ───────────────────────────────────────

def test_la_ventana_descentrada_del_exacto_es_el_trozo_de_la_entera():
    """D2: una ventana (fila, columna, alto, ancho) de la malla M x N se pide
    con centro = (f0 + alto/2 - M/2, c0 + ancho/2 - N/2) * delta*z/L, y es el
    trozo correspondiente de la reconstruccion entera. Descentrada, rectangular
    y en una malla rectangular: un eje cruzado no pasa. Medido: 2.6e-14."""
    _, c = holograma("amplitud", "intensidad")
    m = motor_de("exacto", c)
    entera = m.campo(Z, m.ventana_maxima)
    assert entera.shape == FORMA
    f0, c0, alto, ancho = 10, 70, 50, 70
    trozo = m.campo(Z, (f0, c0, alto, ancho))
    assert trozo.shape == (alto, ancho)
    assert np.abs(trozo - entera[f0:f0 + alto, c0:c0 + ancho]).max() < 1e-10


@pytest.mark.parametrize("nombre, minimo", [("exacto", 0.98), ("carlos", 0.75)])
def test_los_dos_motores_entregan_la_ventana_alineada_con_la_referencia(nombre, minimo):
    """D2 y D3: en Z_REAL la ventana de cada motor se parece a la referencia en
    su sitio, y bastante menos corrida 2 px. Holograma complejo, sin gemela.

    Medido: exacto 0.991 (0.496 corrida); Carlos 0.842 (0.598). El de Carlos
    queda por debajo porque solo ve su recorte -menos NA- y porque su malla
    remuestreada es mas gruesa que la del ROI. SIN dividir por |U0|^2 daba 0.21:
    ver MotorCarlos.
    """
    t, c = holograma("amplitud", "complejo")
    f0, c0, alto, ancho = VENTANA
    g = np.abs(t)[f0:f0 + alto, c0:c0 + ancho]
    A = np.abs(motor_de(nombre, c).campo(Z, VENTANA))
    assert A.shape == (alto, ancho)
    en_su_sitio = mp.correlacion(A, g)
    corrida = mp.correlacion(A[2:, 2:], g[:-2, :-2])
    assert en_su_sitio > minimo
    assert en_su_sitio > corrida + 0.15


@pytest.mark.parametrize("sobremuestreo", [None, 1])
def test_la_onda_de_referencia_es_la_de_dentro_de_reconstruir(sobremuestreo):
    """reconstruir(unos) = U0*conj(U0) = |U0|^2: onda_de_referencia() repite la
    malla, el paso y las funciones de reconstruir(), en las dos ramas."""
    unos = np.ones((64, 64))
    Rec = mp.reconstruir(unos, Z, LAMB, DELTA, L, sobremuestreo=sobremuestreo)
    U0 = mp.onda_de_referencia(unos.shape, Z, LAMB, DELTA, L, sobremuestreo=sobremuestreo)
    assert Rec.shape == U0.shape
    assert np.abs(Rec - np.abs(U0) ** 2).max() < 1e-12 * np.abs(Rec).max()


def test_el_motor_de_carlos_exige_un_roi_cuadrado():
    """reconstruir() usa el mismo paso, delta*P/N, en los dos ejes: con un
    recorte rectangular la esferica sale con otra curvatura en x que en y."""
    _, c = holograma("amplitud", "intensidad")
    with pytest.raises(SystemExit) as fallo:
        mp.MotorCarlos(c, (32, 48, 64, 80), LAMB, DELTA, L)
    assert "CUADRADO" in str(fallo.value)


def test_el_motor_de_carlos_solo_reconstruye_su_recorte():
    _, c = holograma("amplitud", "intensidad")
    m = motor_de("carlos", c)
    assert m.ventana_maxima == VENTANA
    with pytest.raises(ValueError):
        m.campo(Z, (0, 0, 64, 64))


def test_interpolar_centrado_es_exacto_en_lo_lineal():
    """Una rampa lineal por eje se interpola sin error, y la malla nueva que se
    sale de la vieja aborta en vez de extrapolar."""
    n, p = 40, 1.0
    x = (np.arange(n) - n / 2) * p
    A = (2 * x[:, None] + 3 * x[None, :]) * (1 + 1j)
    B = mp.interpolar_centrado(A, p, (25, 30), 0.7)
    y2 = (np.arange(25) - 25 / 2) * 0.7
    x2 = (np.arange(30) - 30 / 2) * 0.7
    np.testing.assert_allclose(B, (2 * y2[:, None] + 3 * x2[None, :]) * (1 + 1j), atol=1e-12)
    with pytest.raises(ValueError):
        mp.interpolar_centrado(A, p, (60, 60), 1.0)


# ── D4: la Tamura invertida enfoca un objeto de fase ────────────────────────

def test_en_fase_la_tamura_invertida_enfoca_en_z_real(globales):
    """Un objeto de fase pura tiene |t| = 1 en el foco: contraste de amplitud
    MINIMO. Con el signo cambiado, el maximo cae en Z. Holograma complejo y
    motor exacto: medido -0.241 en Z frente a -0.40 en los vecinos. (Con el de
    Carlos y un recorte de 64 px no pasa: el borde del recorte domina el
    contraste de amplitud. Ver la cabecera del script.)"""
    _, c = holograma("fase", "complejo")
    mp.fijar_referencia(None, np)
    zs = np.linspace(Z - 0.05, Z + 0.05, 21)
    curvas = mp.barrido(motor_de("exacto", c), VENTANA, zs, ("tamura",), "fase")
    assert zs[int(np.argmax(curvas["tamura"]))] == pytest.approx(Z, abs=1e-9)


# ── el barrido grueso + fino ────────────────────────────────────────────────

@pytest.mark.parametrize("desfase", [0.0, 0.0023])
@pytest.mark.parametrize("nombre, tolerancia", [("exacto", PASO / 2), ("carlos", PASO)])
def test_el_barrido_grueso_y_fino_encuentra_z_real(nombre, tolerancia, desfase, globales):
    """Holograma de INTENSIDAD, las tres metricas, grueso de 5 um y fino de 1 um.

    desfase 0.0023 deja Z fuera de la rejilla gruesa: es el fino el que tiene
    que encontrarlo. La tolerancia no es un paso fino: el error lo pone la
    gemela, no el paso. Medido: el exacto a +1.3 um como mucho, el de Carlos a
    -2.7 um (su pico es mas ancho y plano: ve menos NA). Medio paso grueso y
    un paso grueso son la cota de que el fino sirve.
    """
    t, c = holograma("amplitud", "intensidad")
    f0, c0, alto, ancho = VENTANA
    zs = np.linspace(Z + desfase - 0.05, Z + desfase + 0.05, 21)
    m = motor_de(nombre, c, (zs[0] - PASO, zs[-1] + PASO))
    mp.fijar_referencia(mp.mapa(t[f0:f0 + alto, c0:c0 + ancho], "amplitud", np), np)
    metricas = ("tamura", "pearson", "ssim")
    curvas, picos, finos, z_foco = mp.enfocar(m, VENTANA, zs, 11, metricas, "amplitud",
                                              en_cpu={"ssim"})
    assert set(finos) == set(picos.values())
    for n in metricas:
        assert len(curvas[n]) == len(zs)
        assert abs(z_foco[n] - Z) < tolerancia, (n, z_foco[n])


def test_repartir_las_metricas_entre_cpu_y_dispositivo_no_cambia_ninguna_curva(globales):
    t, c = holograma("amplitud", "intensidad")
    f0, c0, alto, ancho = VENTANA
    m = motor_de("exacto", c)
    mp.fijar_referencia(mp.mapa(t[f0:f0 + alto, c0:c0 + ancho], "amplitud", np), np)
    zs = np.linspace(Z - 0.01, Z + 0.01, 5)
    metricas = ("tamura", "pearson", "ssim")
    reloj = {}
    a = mp.barrido(m, VENTANA, zs, metricas, "amplitud", en_cpu=set(), reloj=reloj)
    b = mp.barrido(m, VENTANA, zs, metricas, "amplitud", en_cpu={"ssim"}, reloj=reloj)
    for n in metricas:
        np.testing.assert_array_equal(a[n], b[n])
    assert set(reloj) == {"reconstruir", *metricas}


@sin_gpu
def test_solapar_no_cambia_ningun_numero(globales):
    """Encadenar la GPU con la CPU mueve el reloj, no las curvas: bit a bit."""
    import cupy as cp
    t, c = holograma("amplitud", "intensidad")
    f0, c0, alto, ancho = VENTANA
    m = mp.MotorExacto(c, LAMB, DELTA, L, (Z - 0.06, Z + 0.06), device="gpu",
                       dtype=np.complex64)
    mp.fijar_referencia(mp.mapa(t[f0:f0 + alto, c0:c0 + ancho], "amplitud", np), cp)
    zs = np.linspace(Z - 0.01, Z + 0.01, 5)
    metricas = ("tamura", "pearson", "ssim")
    reloj = {}
    a = mp.barrido(m, VENTANA, zs, metricas, "amplitud", en_cpu={"ssim"}, solapar=False)
    b = mp.barrido(m, VENTANA, zs, metricas, "amplitud", en_cpu={"ssim"}, solapar=True,
                   reloj=reloj)
    for n in metricas:
        np.testing.assert_array_equal(a[n], b[n])
    assert "espera GPU" in reloj


# ── D6 y D2: lo que aborta antes de reconstruir ─────────────────────────────

@pytest.mark.parametrize("metricas", [("pearson",), ("tamura", "ssim")])
def test_pedir_pearson_o_ssim_sin_referencia_aborta(metricas):
    with pytest.raises(SystemExit) as fallo:
        mp.comprobar_configuracion(metricas, "png", "amplitud", "exacto",
                                   "intensidad", "holo.png", None, None)
    assert '("tamura",)' in str(fallo.value)


def test_solo_tamura_sin_referencia_vale():
    mp.comprobar_configuracion(("tamura",), "png", "amplitud", "exacto",
                               "intensidad", "holo.png", None, None)


@pytest.mark.parametrize("cambio", [
    dict(metricas=()), dict(metricas=("nitidez",)), dict(formato="tif"),
    dict(objeto="intensidad"), dict(motor="rapido"), dict(campo="fase"),
    dict(referencia=None), dict(z_real=None)])
def test_una_configuracion_invalida_aborta(cambio):
    """Con RUTA = None: simular necesita el objeto y la z."""
    kw = dict(metricas=("tamura",), formato="png", objeto="amplitud", motor="exacto",
              campo="intensidad", ruta=None, referencia="objeto.png", z_real=3.0)
    kw.update(cambio)
    with pytest.raises(SystemExit):
        mp.comprobar_configuracion(**kw)


def test_el_rango_lleva_la_guarda_del_fino():
    zs, paso, (lo, hi) = mp.rango_del_barrido(0.2, 0.3, 11, L)
    assert paso == pytest.approx(0.01)
    assert (lo, hi) == (zs[0] - paso, zs[-1] + paso)


@pytest.mark.parametrize("z_min, z_max, pasos", [
    (0.005, 0.1, 11),       # la guarda baja de z = 0
    (0.5, 0.795, 11),       # la guarda pasa de L = 0.8
    (0.3, 0.2, 11),         # al reves
    (0.2, 0.3, 1)])         # sin paso
def test_un_rango_que_se_sale_de_la_muestra_aborta(z_min, z_max, pasos):
    with pytest.raises(SystemExit):
        mp.rango_del_barrido(z_min, z_max, pasos, L)


# ── la region medida ────────────────────────────────────────────────────────

def test_sin_roi_se_mide_el_cuarto_central_del_lado_corto():
    """D9: BORDE_PX = None es 3*min(M, N)//8 por lado."""
    assert mp.region_medida((3000, 4000), None, None) == ((1125, 1125, 750, 1750), None)
    assert mp.region_medida((3000, 4000), None, 1000)[0] == (1000, 1000, 1000, 2000)


def test_un_roi_de_tupla_se_da_en_x_y_y_se_devuelve_en_filas():
    ventana, roi = mp.region_medida((3000, 4000), (100, 200, 1000, 750), None)
    assert ventana == (200, 100, 750, 1000)      # (fila, columna, alto, ancho)
    assert (roi.x0, roi.y0, roi.ancho, roi.alto) == (100, 200, 1000, 750)


@pytest.mark.parametrize("roi, borde", [
    (None, 480), (None, -1), ((3900, 0, 200, 200), None), ((0, 0, 50, 50), None),
    ((1, 2, 3), None)])
def test_una_region_que_no_sirve_aborta(roi, borde):
    with pytest.raises(SystemExit):
        mp.region_medida((1000, 4000), roi, borde)


# ── main() de punta a punta ─────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path, monkeypatch, globales):
    """main() sobre el objeto de prueba, en la CPU y escribiendo en tmp_path.

    La referencia es un PNG de 8 bits con el fondo negro y el objeto blanco,
    como el BenchmarkTarget: con INVERTIR = True, t = 1 - img."""
    ruta = tmp_path / "objeto.png"
    cv.imwrite(str(ruta), np.round(255 * objeto_de_prueba()).astype(np.uint8))
    valores = dict(
        MOSTRAR=False, GUARDAR_FIGURA=True, GUARDAR_BARRIDO=False,
        METRICAS=("tamura", "pearson", "ssim"), EN_CPU={"ssim"}, SOLAPAR=True,
        RUTA=None, FONDO=None, REFERENCIA=ruta, OBJETO="amplitud", INVERTIR=True,
        Z_REAL=Z, CAMPO="intensidad", LAMB=LAMB, DELTA=DELTA, L=L,
        Z_MIN=Z - 0.05, Z_MAX=Z + 0.05, PASOS=21, PASOS_FINO=11,
        ROI=(48, 32, 64, 64), BORDE_PX=None, MOTOR="exacto", DISPOSITIVO="cpu",
        DTYPE=None, SOBREMUESTREO=None, SIMULADOR="exacto", NA_FUENTE=0.1,
        FUENTE=None, PISO=0.05, CARLOS_RECORTE=96,
        RESULTADOS=tmp_path / "salida")
    for k, v in valores.items():
        monkeypatch.setattr(mp, k, v)
    return tmp_path / "salida" / "sim_objeto_z0.24mm_intensidad_exacto_amplitud_roi48-32-64x64"


def test_main_simula_guarda_y_encuentra_el_foco(config, monkeypatch):
    z_foco = mp.main()
    for f in ("sim_objeto_z0.24mm_intensidad.npy", "curvas.npz",
              "barrido_metricas.png", "campos.png"):
        assert (config / f).exists(), f
    assert all(abs(z - Z) < PASO / 2 for z in z_foco.values()), z_foco
    # lo guardado se vuelve a leer sin simular, y cae en la misma carpeta
    monkeypatch.setattr(mp, "RUTA", config / "sim_objeto_z0.24mm_intensidad.npy")
    z_foco_2 = mp.main()
    assert all(abs(z_foco_2[n] - z_foco[n]) <= 2 * PASO / 10 for n in z_foco), (z_foco, z_foco_2)
    curvas = np.load(config / "curvas.npz")
    assert {"zs", "grueso_tamura", "grueso_pearson", "grueso_ssim"} <= set(curvas.files)


def test_main_con_el_motor_de_carlos(config, monkeypatch):
    monkeypatch.setattr(mp, "MOTOR", "carlos")
    z_foco = mp.main()
    carpeta = config.parent / config.name.replace("_exacto_", "_carlos_")
    assert (carpeta / "curvas.npz").exists()
    assert all(abs(z - Z) < PASO for z in z_foco.values()), z_foco


def test_main_en_fase_y_solo_con_tamura(config, monkeypatch):
    """El caso del laboratorio: sin referencia, solo Tamura. Aqui con un objeto
    de fase simulado con fondo en g = 0 (INVERTIR = False) y holograma
    complejo, que es donde la Tamura invertida tiene su pico limpio."""
    for k, v in dict(OBJETO="fase", INVERTIR=False, CAMPO="complejo",
                     METRICAS=("tamura",), EN_CPU=set()).items():
        monkeypatch.setattr(mp, k, v)
    z_foco = mp.main()
    assert abs(z_foco["tamura"] - Z) < PASO / 2, z_foco


# ── el segundo simulador: el modelo de Lopera et al. 2024 ───────────────────

REFERENCIA_CARLOS = RAIZ / "referencia/carlos/DLHM-model-main/DLHM-model-main/dlhm.py"
#: Las que viajan copiadas desde su dlhm.py: su modelo de simulacion y las
#: cinco de la cadena de reconstruccion.
DE_CARLOS = ("ifts", "fts", "resize", "point_src", "angular_spectrum", "dlhm")


@pytest.mark.skipif(not REFERENCIA_CARLOS.exists(),
                    reason="referencia/carlos no va en el repo (.gitignore)")
def test_las_seis_funciones_de_carlos_son_copias_literales():
    """dlhm() -el modelo de simulacion de Lopera et al., Opt. Express 32,
    48509 (2024)- y las cinco de la cadena de reconstruccion, identicas a las
    de su dlhm.py, docstrings incluidos: su valor es que nadie las ha tocado."""
    def defs(ruta):
        arbol = ast.parse(pathlib.Path(ruta).read_text(encoding="utf-8"))
        return {n.name: ast.dump(n) for n in arbol.body
                if isinstance(n, ast.FunctionDef) and n.name in DE_CARLOS}
    assert defs(mp.__file__) == defs(REFERENCIA_CARLOS)


def test_simular_con_lopera_es_su_dlhm_pasado_a_0_1():
    """simular(..., "lopera") es dlhm() tal cual, de 8 bits a [0, 1].

    Con dx_in = 0 se salta su paso de remuestreo (el i del articulo), que con
    un objeto ya en la malla proyectada -paso delta*z/L y la forma del
    sensor- es la identidad: da lo mismo, bit a bit, que pedirle
    dx_in = delta*z/L, y no pasa por la rama del recorte, que esta rota.
    """
    t = 1 - 0.8 * objeto_de_prueba()
    M, N = FORMA
    h = mp.simular(t, LAMB, DELTA, L, Z, "intensidad", device="cpu",
                   simulador="lopera", na_fuente=0.1)
    h8 = mp.dlhm(t, DELTA * Z / L, L, Z, N * DELTA, M * DELTA, DELTA, LAMB, NA_s=0.1)
    assert h8.dtype == np.uint8
    np.testing.assert_array_equal(h, h8 / 255.0)


def test_un_simulador_desconocido_aborta():
    with pytest.raises(SystemExit):
        mp.comprobar_configuracion(("tamura",), "png", "amplitud", "exacto",
                                   "intensidad", None, "objeto.png", 3.0,
                                   simulador="rayleigh")


def test_main_con_el_simulador_de_lopera(config, monkeypatch):
    """Su holograma -paraxial, con la distorsion de Brown-Conrady, la
    envolvente de la fuente y 8 bits- reconstruido con el motor exacto y la
    misma seleccion, sin fondo (se divide por la media, como un holograma de
    laboratorio sin FONDO). Cae en su propia carpeta."""
    monkeypatch.setattr(mp, "SIMULADOR", "lopera")
    z_foco = mp.main()
    carpeta = config.parent / "sim_objeto_z0.24mm_lopera_exacto_amplitud_roi48-32-64x64"
    assert (carpeta / "sim_objeto_z0.24mm_lopera.npy").exists()
    assert all(abs(z - Z) < PASO for z in z_foco.values()), z_foco


# ── la fuente (2026-09-29) ──────────────────────────────────────────────────

@pytest.mark.parametrize("cambio", [
    dict(fuente="pinhole", ruta="holo.npy"),       # un holograma grabado ya trae su fuente
    dict(fuente="pinhole", simulador="lopera"),     # el suyo trae su envolvente
    dict(fuente="pinhole", campo="complejo"),
    dict(fuente="led"), dict(fuente=3.0),
    dict(piso=1.0), dict(piso=-0.1)])
def test_una_fuente_o_un_piso_que_no_casan_abortan(cambio):
    kw = dict(metricas=("tamura",), formato="png", objeto="amplitud", motor="exacto",
              campo="intensidad", ruta=None, referencia="objeto.png", z_real=3.0)
    kw.update(cambio)
    with pytest.raises(SystemExit):
        mp.comprobar_configuracion(**kw)


def test_fuente_pinhole_o_una_funcion_pasan():
    kw = dict(metricas=("tamura",), formato="png", objeto="amplitud", motor="exacto",
              campo="intensidad", ruta=None, referencia="objeto.png", z_real=3.0,
              piso=0.05)
    mp.comprobar_configuracion(fuente="pinhole", **kw)
    mp.comprobar_configuracion(fuente=lambda s: np.ones_like(s), **kw)


def test_fuente_de_y_su_sufijo():
    assert mp.fuente_de(None, 5e-3, LAMB) is None
    a = mp.fuente_de("pinhole", 5e-3, LAMB)
    s = np.array([0.0, 0.05, 0.2])
    np.testing.assert_array_equal(a(s), dlhm.fuente_pinhole(5e-3, LAMB)(s))
    f = lambda s: np.ones_like(s)                   # noqa: E731
    assert mp.fuente_de(f, 5e-3, LAMB) is f
    assert mp.sufijo_fuente(None, 5e-3) == ""
    assert mp.sufijo_fuente("pinhole", 5e-3) == "_pinhole5um"
    assert mp.sufijo_fuente(f, 5e-3) == "_fuente"


def test_la_carpeta_de_un_simulado_con_fuente_lleva_su_sufijo():
    carpeta, nombre = mp.carpeta_de_salida("raiz", None, "dir/BenchmarkTarget.png", 3.0,
                                           "intensidad", "exacto", "amplitud", None,
                                           "_pinhole5um")
    assert nombre == "sim_BenchmarkTarget_z3mm_intensidad_pinhole5um"
    assert carpeta.name == "sim_BenchmarkTarget_z3mm_intensidad_pinhole5um_exacto_amplitud"
    # con RUTA el nombre es el del archivo, lleve lo que lleve
    _, nombre = mp.carpeta_de_salida("raiz", "x/holo.npy", None, None, "intensidad",
                                     "exacto", "amplitud", None, "_pinhole5um")
    assert nombre == "holo"


def test_simular_con_fuente_es_la_ida_con_fuente():
    a = dlhm.fuente_pinhole(3.2e-3, LAMB)
    t = 1 - 0.8 * objeto_de_prueba()
    h = mp.simular(t, LAMB, DELTA, L, Z, "intensidad", device="cpu", fuente=a)
    esperado = dlhm.a_numpy(dlhm.holograma(t, DELTA * Z / L, LAMB, Z, L, FORMA, DELTA,
                                           fuente=a, device="cpu"))
    np.testing.assert_array_equal(h, esperado)


def test_zona_iluminada_mide_la_fraccion_oscura_y_el_mayor_cuadrado_centrado():
    """Disco de radio 50 centrado en (100, 150): el mayor cuadrado centrado que
    cabe tiene medio lado m con sqrt(2) m <= 50, o sea m = 35 y lado 70."""
    yy, xx = np.mgrid[:200, :300]
    fondo = ((yy - 100) ** 2 + (xx - 150) ** 2 <= 50 ** 2).astype(float)
    fr, lado = mp.zona_iluminada(fondo, (80, 130, 40, 40), 0.05)
    assert fr == 0.0 and lado == 70
    fr, _ = mp.zona_iluminada(fondo, (0, 0, 64, 64), 0.05)
    assert fr == 1.0
    fr, _ = mp.zona_iluminada(fondo, (100, 100, 64, 64), 0.05)
    assert 0.0 < fr < 1.0


def test_el_motor_exacto_pasa_el_piso(globales):
    """Con fuente y piso el motor exacto reconstruye lo mismo que el
    Reconstructor con esos argumentos."""
    a = dlhm.fuente_pinhole(3.2e-3, LAMB)
    t = 1 - 0.8 * objeto_de_prueba()
    h = mp.simular(t, LAMB, DELTA, L, Z, "intensidad", device="cpu", fuente=a)
    fondo = dlhm.iluminacion(FORMA, DELTA, L, a)
    m = mp.MotorExacto(h, LAMB, DELTA, L, (Z - 0.01, Z + 0.01), fondo=fondo, piso=0.05,
                       device="cpu")
    R = dlhm.Reconstructor(h, DELTA, LAMB, L, (Z - 0.01, Z + 0.01), fondo=fondo, piso=0.05,
                           device="cpu")
    np.testing.assert_allclose(dlhm.a_numpy(m.campo(Z, (0, 0, *FORMA))),
                               dlhm.a_numpy(R.campo(Z)), rtol=1e-12, atol=1e-12)


#: Pinhole que da, con L = 0.8 mm, un disco (|a|² >= 0.05) de ~69 px de radio:
#: la ventana de config (48, 32, 64, 64), con las esquinas a 45 px, cae dentro.
PINHOLE_PRUEBA = 3.2e-3


def test_main_con_fuente_guarda_el_fondo_y_encuentra_el_foco(config, monkeypatch):
    """Medido con el prototipo (normalizando por amplitud, recorte de 96): el
    exacto a 2.0 / 1.0 / 1.0 um (Tamura / Pearson / SSIM), el de Carlos a
    1.0 / 1.0 / 0.0 um, las mismas cotas que sin fuente."""
    a = dlhm.fuente_pinhole(PINHOLE_PRUEBA, LAMB)
    monkeypatch.setattr(mp, "FUENTE", a)
    z_foco = mp.main()
    carpeta = config.parent / config.name.replace("_intensidad_", "_intensidad_fuente_")
    nombre = "sim_objeto_z0.24mm_intensidad_fuente"
    holo = np.load(carpeta / f"{nombre}.npy")
    fondo = np.load(carpeta / f"{nombre}_fondo.npy")
    np.testing.assert_allclose(fondo, dlhm.iluminacion(FORMA, DELTA, L, a), rtol=1e-12)
    M, N = FORMA
    # el disco brillante: medido 283 veces el nivel de la esquina
    assert holo[M // 2 - 5:M // 2 + 5, N // 2 - 5:N // 2 + 5].mean() > 100 * holo[:10, :10].mean()
    # y dividido por su fondo, dentro del disco, es un holograma de contraste
    luz = fondo >= 0.05 * fondo.max()
    assert 0.5 < (holo[luz] / fondo[luz]).mean() < 1.5
    assert all(abs(z - Z) < PASO / 2 for z in z_foco.values()), z_foco
    # releido con RUTA y FONDO, sin FUENTE: el mismo foco, en la misma carpeta
    monkeypatch.setattr(mp, "FUENTE", None)
    monkeypatch.setattr(mp, "RUTA", carpeta / f"{nombre}.npy")
    monkeypatch.setattr(mp, "FONDO", carpeta / f"{nombre}_fondo.npy")
    z_foco_2 = mp.main()
    assert all(abs(z_foco_2[n] - z_foco[n]) <= 2 * PASO / 10 for n in z_foco), (z_foco, z_foco_2)


def test_main_con_fuente_y_el_motor_de_carlos(config, monkeypatch):
    """Con la fuente simulada Carlos recibe el contraste; releido con RUTA y
    FONDO tambien, asi que da el mismo foco. Antes de dividir tambien con FONDO
    el releido le daba el holograma crudo y la Tamura se iba ~50 um."""
    monkeypatch.setattr(mp, "FUENTE", dlhm.fuente_pinhole(PINHOLE_PRUEBA, LAMB))
    monkeypatch.setattr(mp, "MOTOR", "carlos")
    z_foco = mp.main()
    assert all(abs(z - Z) < PASO for z in z_foco.values()), z_foco
    carpeta = config.parent / config.name.replace("_intensidad_exacto_",
                                                  "_intensidad_fuente_carlos_")
    nombre = "sim_objeto_z0.24mm_intensidad_fuente"
    monkeypatch.setattr(mp, "FUENTE", None)
    monkeypatch.setattr(mp, "RUTA", carpeta / f"{nombre}.npy")
    monkeypatch.setattr(mp, "FONDO", carpeta / f"{nombre}_fondo.npy")
    z_foco_2 = mp.main()
    assert all(abs(z_foco_2[n] - z_foco[n]) <= 2 * PASO / 10 for n in z_foco), (z_foco, z_foco_2)


def test_main_avisa_si_la_region_medida_sale_de_la_luz(config, monkeypatch, capsys):
    """Una ventana casi entera: sus esquinas (a 91 px) salen del disco de 69."""
    for k, v in dict(FUENTE=dlhm.fuente_pinhole(PINHOLE_PRUEBA, LAMB),
                     ROI=(8, 8, 144, 112), METRICAS=("tamura",), EN_CPU=set()).items():
        monkeypatch.setattr(mp, k, v)
    mp.main()
    salida = capsys.readouterr().out
    assert "AVISO" in salida and "PISO" in salida
    lado = mp.zona_iluminada(dlhm.iluminacion(FORMA, DELTA, L, mp.FUENTE),
                             (8, 8, 112, 144), 0.05)[1]
    assert f"{lado}, {lado})" in salida


BENCHMARK = RAIZ / "referencia/carlos/DLHM-model-main/DLHM-model-main/data/BenchmarkTarget.png"


@pytest.mark.skipif(not BENCHMARK.exists(),
                    reason="referencia/carlos no va en el repo (.gitignore)")
def test_el_roi_por_defecto_encierra_el_usaf_central():
    """Spec D8: un cuadrado con el grupo USAF del centro del BenchmarkTarget
    entero, 48 px de margen como minimo, ningun otro grupo dentro y todo el
    cuadrado en la luz del pinhole del banco (PISO 0.05)."""
    from scipy import ndimage as ndi
    from CamposT.montaje import MONTAJE
    img = cv.imread(str(BENCHMARK), cv.IMREAD_GRAYSCALE) > 127
    M, N = img.shape
    x0, y0, ancho, alto = mp.ROI
    assert ancho == alto
    etiquetas, _ = ndi.label(ndi.binary_dilation(img, iterations=12))
    objetos = ndi.find_objects(etiquetas)

    def distancia(sl):
        return np.hypot((sl[0].start + sl[0].stop) / 2 - M / 2,
                        (sl[1].start + sl[1].stop) / 2 - N / 2)
    usaf = 1 + min(range(len(objetos)), key=lambda i: distancia(objetos[i]))
    ys, xs = np.nonzero(img & (etiquetas == usaf))
    assert ys.min() - y0 >= 48 and y0 + alto - 1 - ys.max() >= 48
    assert xs.min() - x0 >= 48 and x0 + ancho - 1 - xs.max() >= 48
    assert set(np.unique(etiquetas[y0:y0 + alto, x0:x0 + ancho])) <= {0, usaf}
    a = dlhm.fuente_pinhole(MONTAJE.pinhole, MONTAJE.lamb)
    fondo = dlhm.iluminacion((M, N), MONTAJE.delta_x, MONTAJE.L, a)
    assert mp.zona_iluminada(fondo, (y0, x0, alto, ancho), 0.05)[0] == 0.0


# ── el recorte propio de Carlos (2026-09-29) ────────────────────────────────

def test_carlos_con_recorte_mide_un_trozo_de_su_recorte():
    """Spec D9: reconstruye el cuadrado centrado de `recorte` y las metricas
    se miden en la ventana, dentro de el."""
    _, c = holograma("amplitud", "intensidad")
    m = mp.MotorCarlos(c, VENTANA, LAMB, DELTA, L, recorte=96)
    assert m.ventana_maxima == (16, 32, 96, 96)
    entera = m.campo(Z, m.ventana_maxima)
    f0, c0, alto, ancho = VENTANA
    np.testing.assert_array_equal(m.campo(Z, VENTANA),
                                  entera[f0 - 16:f0 - 16 + alto, c0 - 32:c0 - 32 + ancho])


@pytest.mark.parametrize("ventana, recorte", [((0, 0, 64, 64), 96), (VENTANA, 200)])
def test_carlos_aborta_si_la_ventana_no_cabe_en_su_recorte(ventana, recorte):
    _, c = holograma("amplitud", "intensidad")
    with pytest.raises(SystemExit):
        mp.MotorCarlos(c, ventana, LAMB, DELTA, L, recorte=recorte)


def test_carlos_con_fondo_divide_su_salida_por_la_raiz():
    """Spec D5bis: con raiz, la entrada 1 + c1 va sin normalizar y la salida
    es 1 + (t - 1)/raiz en cada pixel del recorte, 1 bajo el piso."""
    a = dlhm.fuente_pinhole(PINHOLE_PRUEBA, LAMB)
    t = 1 - 0.8 * objeto_de_prueba()
    h = mp.simular(t, LAMB, DELTA, L, Z, "intensidad", device="cpu", fuente=a)
    fondo = dlhm.iluminacion(FORMA, DELTA, L, a)
    c1, raiz, raiz_min = dlhm.contraste_amplitud(h, fondo, 0.05)
    uno = mp.MotorCarlos(1 + c1, VENTANA, LAMB, DELTA, L, recorte=96,
                         raiz=np.ones_like(raiz))
    m = mp.MotorCarlos(1 + c1, VENTANA, LAMB, DELTA, L, recorte=96, raiz=raiz,
                       raiz_min=raiz_min)
    f0, c0, alto, ancho = VENTANA
    r = raiz[f0:f0 + alto, c0:c0 + ancho]
    luz = r >= raiz_min
    base = uno.campo(Z, VENTANA)
    esperado = 1 + np.where(luz, (base - 1) / np.where(luz, r, 1), 0)
    np.testing.assert_allclose(m.campo(Z, VENTANA), esperado, rtol=1e-10, atol=1e-12)


def test_carlos_con_fondo_pone_exactamente_1_bajo_el_piso():
    """Spec D5bis, la rama de la mascara: bajo raiz_min la salida es EXACTAMENTE
    1, en todo el recorte, y sobre el piso es 1 + (t - 1)/raiz.

    La prueba de arriba no la pisa: con PISO = 0.05 el piso de la raiz es 0.22 y
    el minimo del recorte de 96 es 0.25, asi que ningun pixel cae bajo el y un
    (t - 1)/r sin mascara la dejaria pasar. Con PISO = 0.3 (raiz_min = 0.55) caen
    1961 de 9216, las esquinas del recorte, donde el pinhole ya no ilumina."""
    a = dlhm.fuente_pinhole(PINHOLE_PRUEBA, LAMB)
    t = 1 - 0.8 * objeto_de_prueba()
    h = mp.simular(t, LAMB, DELTA, L, Z, "intensidad", device="cpu", fuente=a)
    fondo = dlhm.iluminacion(FORMA, DELTA, L, a)
    c1, raiz, raiz_min = dlhm.contraste_amplitud(h, fondo, 0.3)
    uno = mp.MotorCarlos(1 + c1, VENTANA, LAMB, DELTA, L, recorte=96,
                         raiz=np.ones_like(raiz))
    m = mp.MotorCarlos(1 + c1, VENTANA, LAMB, DELTA, L, recorte=96, raiz=raiz,
                       raiz_min=raiz_min)
    rf0, rc0, lado, _ = m.ventana_maxima
    r = raiz[rf0:rf0 + lado, rc0:rc0 + lado]
    bajo = r < raiz_min
    # que no pase en vacio: hay pixeles bajo el piso y tambien sobre el
    assert bajo.any() and not bajo.all()
    salida = m.campo(Z, m.ventana_maxima)
    assert np.isfinite(salida).all()
    assert (salida[bajo] == 1.0).all()
    base = uno.campo(Z, uno.ventana_maxima)
    esperado = 1 + (base - 1) / np.where(bajo, 1, r)
    np.testing.assert_allclose(salida[~bajo], esperado[~bajo], rtol=1e-10, atol=1e-12)
