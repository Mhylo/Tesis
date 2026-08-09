"""Selección de backend de cómputo: CuPy (GPU, CUDA) o NumPy (CPU).

El resto del paquete se escribe contra el módulo `xp` que devuelve get_xp(),
de modo que el mismo algoritmo corre en CPU y en GPU. Los tiempos que se
comparan en la tesis salen así del mismo código, no de dos implementaciones
distintas.

Dos cuidados que condicionan todo lo demás:

Precisión. En GPUs de consumo (GeForce) la doble precisión va a 1/32 del
ritmo de la simple, así que en GPU el dtype por defecto es complex64. Pero
las fases NO se pueden calcular en simple: el argumento de la función de
transferencia es k·z, que con lamb = 632.8 nm y z = 200 m vale ~2e12 rad, y
en float32 el error absoluto ahí es de ~1e5 rad, es decir, fase aleatoria.
Por eso todo argumento de fase se evalúa en float64 y sólo se convierte a
complex64 el resultado ya acotado en [-1, 1] (ver fase_a_complejo).

Memoria. La matriz espectral de MPASM es (s·M, s·N); con s=10 sobre una
entrada de 1024 son 10240² elementos, 839 MB en complex64 y 1.7 GB en
complex128. En una tarjeta de 4 GB eso obliga a construir las matrices por
bloques de filas en vez de materializarlas en doble precisión.
"""

import numpy as np

try:
    import cupy as cp
    _GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:                                    # cupy ausente o sin driver
    cp = None
    _GPU = False


# ------------------------------------------------------------------ selección
def gpu_disponible():
    return _GPU


def get_xp(device="auto"):
    """Devuelve (módulo_array, nombre_dispositivo).

    device: 'gpu' | 'cpu' | 'auto'. Con 'auto' usa la GPU si hay una.
    """
    if device == "auto":
        device = "gpu" if _GPU else "cpu"
    if device == "gpu":
        if not _GPU:
            raise RuntimeError(
                "Se pidió device='gpu' pero CuPy no encuentra ninguna GPU CUDA. "
                "Instala cupy-cuda12x y comprueba con nvidia-smi.")
        return cp, "gpu"
    if device == "cpu":
        return np, "cpu"
    raise ValueError(f"device desconocido: {device!r}")


def es_gpu(xp):
    return cp is not None and xp is cp


# --------------------------------------------------------------------- dtypes
def dtype_por_defecto(device):
    """complex64 en GPU (velocidad y memoria), complex128 en CPU (referencia)."""
    return np.complex64 if device == "gpu" else np.complex128


def real_de(cdtype):
    """Tipo real asociado a un complejo: complex64 -> float32."""
    return np.zeros(0, dtype=cdtype).real.dtype


# ------------------------------------------------------- transferencia de datos
def a_dispositivo(a, xp, dtype=None):
    """Sube (o baja) un array al backend pedido, opcionalmente convirtiendo."""
    if es_gpu(xp):
        a = cp.asarray(a)
    elif cp is not None and isinstance(a, cp.ndarray):
        a = cp.asnumpy(a)
    else:
        a = np.asarray(a)
    return a.astype(dtype, copy=False) if dtype is not None else a


def a_numpy(a):
    """Baja a NumPy sea cual sea el origen. Para guardar, graficar o medir."""
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


# ------------------------------------------------------------------ sincronía
def sincronizar(xp=None):
    """Espera a que la GPU termine.

    CuPy encola los kernels y devuelve el control de inmediato: sin esta
    llamada, un perf_counter() alrededor de una operación en GPU mide el
    tiempo de lanzamiento, no el de cómputo. Es obligatoria en cualquier
    medida de tiempo.
    """
    if _GPU and (xp is None or es_gpu(xp)):
        cp.cuda.Stream.null.synchronize()


def liberar_memoria():
    """Devuelve al sistema la memoria retenida por los pools de CuPy."""
    if _GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.fft.config.get_plan_cache().clear()


def memoria_gpu():
    """(libre, total) en bytes, o None si no hay GPU."""
    if not _GPU:
        return None
    return cp.cuda.Device().mem_info


def info_gpu():
    if not _GPU:
        return None
    libre, total = cp.cuda.Device().mem_info
    props = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "nombre": props["name"].decode(),
        "capacidad": f"{props['major']}.{props['minor']}",
        "VRAM total [GB]": total / 2**30,
        "VRAM libre [GB]": libre / 2**30,
        "cupy": cp.__version__,
    }


def comprobar_memoria(bytes_necesarios, etiqueta="", margen=0.85, dtype=None):
    """Aborta con un mensaje útil antes de un OOM opaco de CUDA."""
    mem = memoria_gpu()
    if mem is None:
        return
    libre, total = mem
    if bytes_necesarios > margen * libre:
        salidas = ["baja s", "reduce el padding", "pasa device='cpu'"]
        if dtype is not None and np.dtype(dtype).itemsize > 8:
            salidas.insert(1, "usa dtype=complex64 (mitad de memoria)")
        raise MemoryError(
            f"{etiqueta}: harían falta ~{bytes_necesarios / 2**30:.2f} GB y hay "
            f"{libre / 2**30:.2f} GB libres de {total / 2**30:.2f} GB. "
            f"Opciones: {', '.join(salidas)}.")


# --------------------------------------------------------- construcción por bloques
#: presupuesto por bloque intermedio en float64/complex128 (bytes)
PRESUPUESTO_BLOQUE = 64 << 20


def bloques(n_filas, n_cols, itemsize=16, presupuesto=PRESUPUESTO_BLOQUE):
    """Trocea n_filas para que cada bloque intermedio quepa en el presupuesto.

    Genera pares (i0, i1) de filas. Sirve para construir matrices grandes
    evaluando las fases en doble precisión sin materializar nunca la matriz
    completa en complex128.
    """
    por_bloque = max(1, presupuesto // max(1, n_cols * itemsize))
    for i0 in range(0, n_filas, por_bloque):
        yield i0, min(i0 + por_bloque, n_filas)


def fase_a_complejo(fase, dtype, xp):
    """exp(i·fase) con fase en float64, resultado en el dtype de trabajo.

    La exponencial se evalúa en doble precisión (reducción de argumento
    correcta para fases enormes) y sólo después se recorta la mantisa.
    """
    return xp.exp(1j * fase).astype(dtype, copy=False)


def phasor(a, b, signo, xp, dtype):
    """Matriz exp(signo·2πi·outer(a, b)), construida por bloques de filas.

    Es el núcleo de la DFT matricial de MPASM. El producto externo se calcula
    en float64 aunque la salida sea complex64.
    """
    a = xp.asarray(a, dtype=np.float64)
    b = xp.asarray(b, dtype=np.float64)
    out = xp.empty((a.size, b.size), dtype=dtype)
    for i0, i1 in bloques(a.size, b.size):
        out[i0:i1] = fase_a_complejo(signo * 2 * np.pi * xp.outer(a[i0:i1], b),
                                     dtype, xp)
    return out


# ------------------------------------------------------------------ cronómetro
def cronometrar(fn, *args, repeticiones=1, calentar=True, xp=None, **kw):
    """Ejecuta fn y devuelve (resultado, tiempo_medio_en_segundos).

    Sincroniza antes y después, y hace una pasada de calentamiento: la primera
    llamada a CuPy incluye la inicialización de cuBLAS y la compilación de
    kernels, que no forma parte del coste del algoritmo.
    """
    import time

    if calentar:
        fn(*args, **kw)
        sincronizar(xp)
    sincronizar(xp)
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        res = fn(*args, **kw)
    sincronizar(xp)
    return res, (time.perf_counter() - t0) / repeticiones
