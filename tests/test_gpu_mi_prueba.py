"""Los cuatro scripts de scripts/mi_prueba_*.py corriendo en CPU y en GPU.

QUE SE PROTEGE AQUI. Que acelerar no cambie el resultado. Cada uno de los
cuatro scripts lleva una version de su propagador escrita contra `xp` -NumPy
o CuPy- y por bloques de filas, y cada uno sabe contrastarla contra algo:

    mi_prueba_angular   contra angularSpectrum(), la de pyDHM copiada tal cual
    mi_prueba_dlhm      contra la cadena intacta de dlhm.py (point_src +
                        angular_spectrum), que es la seccion 2 de ese archivo
    mi_prueba_blas      contra si misma en CPU y complex128; en referencia/ no
                        hay ningun BL-ASM contra el que medir
    mi_prueba_mpasm     contra si misma en CPU y complex128; el MatrixDftCPU de
                        Zhao vive en scripts/retro_mpasm.py y se contrasta alli

Los scripts ya corren esa comprobacion en cada invocacion, porque cuesta
milisegundos. Este fichero es el que la ejecuta sin abrir ninguna figura y el
unico sitio donde los cuatro se miran a la vez.

LAS TOLERANCIAS NO ESTAN AJUSTADAS HASTA QUE LA SUITE PASE. Salen de la
mantisa del dtype, igual que en tests/test_propagadores.py, y hay tres casos
que conviene no mezclar:

    CPU, complex128     EXACTO. Es la misma aritmetica en la misma maquina y
                        la misma biblioteca de FFT: no hay de donde salga un
                        bit de diferencia. Si esto deja de dar 0, alguien
                        cambio el ALGORITMO y no solo su ejecucion.

    GPU, complex128     1e-10. Aqui cambia la biblioteca -cuFFT contra
                        pocketfft-, que redondea distinto en la ultima cifra,
                        y la cadena encadena dos transformadas y varios
                        productos. Medido el 08/09: 6e-16 (blas), 2e-15
                        (mpasm), 2e-13 (angular) y 1e-12 (dlhm). El tope deja
                        dos ordenes de margen sobre el peor y sigue estando
                        seis por debajo de lo que produce tocar la formula.

    GPU, complex64      1e-5, que es la tolerancia que el repo usa para simple
                        precision. Medido: 2.8e-7 (angular), 2.5e-7 (blas),
                        9.0e-7 (mpasm) y 5.7e-7 (dlhm).

Que el caso de en medio exista es lo que separa las dos cosas que "correr en
GPU" mezcla: el DISPOSITIVO y la PRECISION. Lo que se pierde en complex64 es
la mantisa, no la tarjeta.
"""

import matplotlib
matplotlib.use("Agg")           # antes de importar los scripts, que traen pyplot

import importlib
import inspect

import numpy as np
import pytest

#: Los cuatro barridos puntuados. La prueba corre entera sobre cada uno.
MODULOS = ("scripts.mi_prueba_angular", "scripts.mi_prueba_blas",
           "scripts.mi_prueba_mpasm", "scripts.mi_prueba_dlhm")

#: De donde salieron las copias literales del bloque backend.
ORIGEN = "scripts.retro_fft_angular"

#: Las que viajan copiadas desde ORIGEN y no deben divergir.
#:
#: sincronizar() NO esta en esta lista porque no sale de ahi: es
#: CamposT.backend.sincronizar() sin el argumento opcional, y los retro_* no la
#: llevan. Lo que se le exige es lo otro -que sea la misma en los cuatro
#: mi_prueba_*-, y de eso se ocupa su propia prueba.
#:
#: comprobar_memoria tampoco esta, y ahi divergir es lo CORRECTO: cada script
#: cuenta lo suyo -la matriz espectral en mpasm, la malla remuestreada en dlhm-.
COPIADAS = ("elegir_dispositivo", "a_cpu", "liberar")


@pytest.fixture(params=MODULOS)
def mod(request):
    return importlib.import_module(request.param)


def hay_gpu():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


sin_gpu = pytest.mark.skipif(not hay_gpu(), reason="no hay CUDA en esta maquina")


def equivalencia(mod, xp, dtype):
    """El numero que compara la version rapida contra su referencia.

    Los cuatro scripts la exponen como comprobar_equivalencia(), pero no con
    la misma firma: angular necesita saber si los ejes van cruzados, y blas y
    mpasm devuelven ademas una segunda cantidad que no depende del dispositivo
    (la fraccion de banda y el Kf). Aqui se normaliza a un solo float.
    """
    if mod.__name__.endswith("angular"):
        return mod.comprobar_equivalencia(xp, dtype, mod.EJES_CRUZADOS)
    r = mod.comprobar_equivalencia(xp, dtype)
    return r[0] if isinstance(r, tuple) else r


# ── seleccion de dispositivo ────────────────────────────────────────────────

def test_cpu_siempre_disponible(mod):
    xp, dev = mod.elegir_dispositivo("cpu")
    assert xp is np and dev == "cpu"


def test_auto_no_falla_nunca(mod):
    """'auto' cae a NumPy sin ruido: es lo que hace que el script arranque en
    cualquier maquina sin tocar una constante."""
    xp, dev = mod.elegir_dispositivo("auto")
    assert dev in ("cpu", "gpu")
    assert dev == ("gpu" if hay_gpu() else "cpu")


@pytest.mark.skipif(hay_gpu(), reason="esta maquina si tiene CUDA")
def test_gpu_explicita_aborta_sin_cuda(mod):
    """DISPOSITIVO = 'gpu' sin CUDA ABORTA, no cae a CPU en silencio.

    Un barrido que dice 'gpu' y corre en CPU da un tiempo que parece una
    medida y no lo es. Es el error que solo se descubre al comparar tablas.
    """
    with pytest.raises(SystemExit):
        mod.elegir_dispositivo("gpu")


def test_a_cpu_devuelve_numpy(mod):
    a = mod.a_cpu(np.arange(4))
    assert isinstance(a, np.ndarray)


# ── que acelerar no cambio el resultado ─────────────────────────────────────

def test_equivalencia_exacta_en_cpu_doble(mod):
    """En CPU y complex128 la version rapida es la referencia, bit a bit.

    No es una tolerancia generosa: es la misma aritmetica en la misma maquina.
    Si esto deja de dar 0, alguien cambio el ALGORITMO y no solo su ejecucion.
    """
    assert equivalencia(mod, np, np.complex128) == 0.0


@sin_gpu
def test_equivalencia_en_gpu_simple(mod):
    """En GPU y complex64, dentro de lo que da la mantisa de la simple."""
    import cupy as cp
    assert equivalencia(mod, cp, np.complex64) < 1e-5


@sin_gpu
def test_equivalencia_en_gpu_doble(mod):
    """La GPU en complex128 tampoco cambia el resultado.

    Separa las dos cosas que 'correr en GPU' mezcla: el DISPOSITIVO y la
    PRECISION. Lo que se pierde en complex64 es la mantisa, no la tarjeta.

    No se le exige 0 exacto como en CPU: aqui la FFT la hace cuFFT y no
    pocketfft, y dos bibliotecas correctas redondean distinto en la ultima
    cifra. Ver las tolerancias en la cabecera.
    """
    import cupy as cp
    assert equivalencia(mod, cp, np.complex128) < 1e-10


# ── lo que no depende del dispositivo, no depende del dispositivo ───────────

@sin_gpu
def test_la_banda_de_blas_no_depende_del_dispositivo():
    """La mascara de BL-ASM decide con searchsorted sobre ejes en float64.

    O sea que la FRACCION de banda que sobrevive es la misma en las dos
    maquinas y en los dos dtypes, y por eso se puede comparar entre corridas.
    """
    import cupy as cp
    m = importlib.import_module("scripts.mi_prueba_blas")
    assert m.comprobar_equivalencia(cp, np.complex64)[1] == 0.0


@sin_gpu
def test_el_kf_de_mpasm_no_depende_del_dispositivo():
    """kf_paper() es aritmetica de escalares en float64 y no toca la tarjeta."""
    import cupy as cp
    m = importlib.import_module("scripts.mi_prueba_mpasm")
    assert m.comprobar_equivalencia(cp, np.complex64)[1] == 0.0


# ── que la comprobacion no dependa de lo que estas probando ─────────────────

def test_la_equivalencia_de_dlhm_no_depende_de_la_constante_editable():
    """SOBREMUESTREO se edita a diario: comprobar_equivalencia no puede leerla.

    REGRESION, y de las feas. reconstruir() leia el global SOBREMUESTREO y
    comprobar_equivalencia lo llamaba sin fijarlo, asi que con
    SOBREMUESTREO = None la cadena rapida remuestreaba por geometria -221x221
    sobre una entrada de 256- y la de referencia no. Segun el valor que
    estuviera puesto en ese momento, la suite reventaba con un ValueError de
    formas o pasaba. Fallar segun cuando se corre es peor que fallar siempre:
    parece flakiness del backend y no lo es.

    Se comprueba en CPU/complex128 porque ahi el resultado exacto es 0.0 y no
    hace falta tolerancia ninguna para ver si las dos llamadas coinciden.
    """
    m = importlib.import_module("scripts.mi_prueba_dlhm")
    antes = m.SOBREMUESTREO
    try:
        m.SOBREMUESTREO = None
        con_none = m.comprobar_equivalencia(np, np.complex128)
        m.SOBREMUESTREO = 1
        con_uno = m.comprobar_equivalencia(np, np.complex128)
    finally:
        m.SOBREMUESTREO = antes
    assert con_none == con_uno == 0.0


@pytest.mark.parametrize("sobremuestreo", [1, None])
def test_dlhm_equivale_en_las_dos_ramas(sobremuestreo):
    """Las dos ramas de reconstruir(): malla del sensor, y remuestreada.

    La segunda es la que pide la geometria y la que no cabe en la tarjeta a
    tamano real; a 256x256 sale una malla de 221 y se puede comprobar.
    """
    m = importlib.import_module("scripts.mi_prueba_dlhm")
    assert m.comprobar_equivalencia(np, np.complex128, sobremuestreo) == 0.0


@sin_gpu
@pytest.mark.parametrize("sobremuestreo", [1, None])
def test_dlhm_equivale_en_las_dos_ramas_en_gpu(sobremuestreo):
    import cupy as cp
    m = importlib.import_module("scripts.mi_prueba_dlhm")
    assert m.comprobar_equivalencia(cp, np.complex64, sobremuestreo) < 1e-5


# ── la correlacion, que es lo que puntua el barrido ─────────────────────────

def test_correlacion_reconoce_su_propio_mapa(mod):
    rng = np.random.default_rng(0)
    a = rng.random((64, 64))
    assert mod.correlacion(a, a) == pytest.approx(1.0)
    assert mod.correlacion(a, -a) == pytest.approx(-1.0)


def test_correlacion_de_un_mapa_constante_es_cero_y_no_nan(mod):
    """Un NaN aqui viajaria hasta el argmax y elegiria un z cualquiera."""
    a = np.ones((16, 16))
    assert mod.correlacion(a, np.arange(256.0).reshape(16, 16)) == 0.0


@sin_gpu
def test_correlacion_igual_en_las_dos_maquinas(mod):
    """La reduccion va en float64 aunque el campo venga en complex64.

    Sobre una malla de 3000x4000 son 1.2e7 sumandos: en float32 el error
    acumulado es del orden de lo que separa dos pasos vecinos del barrido, y
    el argmax se iria a un z equivocado.
    """
    import cupy as cp
    rng = np.random.default_rng(1)
    a = rng.random((256, 256))
    b = rng.random((256, 256))
    en_cpu = mod.correlacion(a, b)
    en_gpu = mod.correlacion(cp.asarray(a, dtype=cp.float32),
                             cp.asarray(b, dtype=cp.float32))
    assert en_gpu == pytest.approx(en_cpu, abs=1e-6)


# ── las copias literales, que no deben divergir ─────────────────────────────

@pytest.mark.parametrize("nombre", COPIADAS)
def test_el_bloque_backend_es_el_mismo_en_toda_la_familia(mod, nombre):
    """Las cuatro funciones de seleccion viajan copiadas entre los scripts.

    Es la misma decision que tests/test_nitidez_foco.py toma con nitidez():
    se permite la copia -cada script tiene que poder leerse solo- pero se fija
    que no diverja. Si hay que cambiar una, se cambian todas.
    """
    origen = importlib.import_module(ORIGEN)
    assert (inspect.getsource(getattr(mod, nombre))
            == inspect.getsource(getattr(origen, nombre)))


def test_sincronizar_es_la_misma_en_toda_la_familia(mod):
    """sincronizar() no viene de los retro_*, asi que se compara consigo misma.

    Sale de CamposT.backend.sincronizar(), sin el argumento opcional porque
    aqui el xp siempre se sabe. Lo que hay que fijar es que los cuatro scripts
    lleven la misma: si uno se queda sin sincronizar, su barrido mide el
    tiempo de LANZAMIENTO de los kernels y sale absurdamente rapido, que es
    justo el error que no se ve mirando la consola.
    """
    primero = importlib.import_module(MODULOS[0])
    assert (inspect.getsource(mod.sincronizar)
            == inspect.getsource(primero.sincronizar))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
