"""Ida y vuelta con el espectro angular de banda limitada: objeto -> +z -> holograma -> -z -> objeto.

Propaga la imagen de entrada una distancia Z y retropropaga el resultado -Z
para ver si vuelve a enfocar donde debe. Un solo metodo, escrito a mano, sin
importar CamposT: sirve de contraste independiente del paquete.

Es el hermano de retro_fft_angular.py, con BL-ASM en vez de FFT-ASM. Merece
script propio porque el limite de banda cambia justo lo que aquel script deja
medido como su punto debil.

CPU o GPU con el mismo cuerpo de algoritmo (ver DISPOSITIVO). Lo unico que
cambia es si el modulo de arrays es NumPy o CuPy.

EL PROPAGADOR
-------------
BL-ASM es el espectro angular de siempre con una mascara: Matsushima y
Shimobaba, Opt. Express 17, 19662 (2009). La funcion de transferencia
H = exp(i k z sqrt(1 - (lamb fx)^2 - (lamb fy)^2)) oscila cada vez mas rapido
al alejarse del origen del plano de frecuencias, y a partir de cierta
frecuencia da mas de media vuelta entre dos muestras de la malla: eso ya no es
propagacion, es aliasing. El limite es la frecuencia a la que ocurre,

    flim_x = 1 / (lamb * sqrt((2 z dfx)^2 + 1)),   dfx = 1/(N dx)

y una por eje. Por encima de el, H se anula en vez de dejarse muestrear mal.

AQUI NO HAY PROPAGADOR DE REFERENCIA, y conviene saberlo antes de leer los
numeros. retro_fft_angular.py contrasta contra el angularSpectrum de pyDHM
copiado tal cual; para BL-ASM no hay equivalente en referencia/: pyDHM trae
fresnel, bluestein y angularSpectrum, y ninguno limita la banda. Escribir yo
una "referencia" y compararla con mi implementacion no comprobaria nada, asi
que el script no finge tenerla. Donde vive la verificacion de BL-ASM es en
tests/test_propagadores.py, contra el gaussiano analitico y contra la
Rayleigh-Sommerfeld de referencias.py. Lo que este script aporta es otra cosa:
ver el metodo trabajar sobre una imagen real, en una pieza que se lee entera.

LO QUE EL LIMITE DE BANDA ARREGLA AL RETROPROPAGAR
--------------------------------------------------
El docstring de retro_fft_angular.py deja medido su punto debil: las ondas
evanescentes (lamb^2 f^2 > 1) se dejan decaer, y propagando hacia adelante eso
esta bien, pero RETROPROPAGANDO el mismo factor crece, hasta 2.87e+110 con
delta = lamb/4.

BL-ASM no tiene ese problema, y no porque lo trate mejor: porque el limite de
banda cae SIEMPRE por debajo de 1/lamb, que es donde empiezan las evanescentes.
Se ve en la formula: flim = 1/(lamb sqrt((2 z dfx)^2 + 1)) y la raiz vale al
menos 1. La mascara las descarta antes de que nadie las evalue. El script lo
comprueba en cada corrida (ver comprobar_evanescentes).

EL BARRIDO DE FOCO SALE SESGADO, Y NO ES UN FALLO
--------------------------------------------------
La nitidez (energia del gradiente) baja al desenfocar, pero la banda que la
mascara deja pasar TAMBIEN baja al alejarse: a z corto sobrevive mas espectro
y la imagen parece mas nitida se enfoque o no. Las dos caidas se suman y el
pico del barrido sale corrido hacia distancias cortas. Medido con este script
sobre BenchmarkTarget reducido a 512 px con Z = 2000 mm, donde la mascara deja
pasar el 26 %: la vuelta A pica en 1500 mm en vez de 2000.

Por eso foco_blas.png lleva un segundo panel con la banda que sobrevive a cada
z. No corrige el sesgo: lo enseña, que es lo que se puede hacer sin inventarse
una normalizacion. Con la mascara inactiva el sesgo no existe y el pico cae
donde debe.

Y EL SIGNO NO LE AFECTA
-----------------------
flim depende de z^2, asi que el limite de banda es el mismo a +z y a -z. Es lo
contrario del Kf de MPASM, que estaba escrito para z > 0 y al aplicarlo tal
cual a z < 0 apagaba la compresion sin avisar (ver retro_mpasm.py). Aqui no hay
nada que corregir, pero el script lo mide en vez de suponerlo, porque suponerlo
es exactamente lo que fallo en el otro sitio.

PRECISION EN GPU
----------------
El campo va en complex64 y la fase en float64 SIEMPRE, se propague donde se
propague. La fase 2*pi*z*sqrt(1/lamb^2 - f^2) vale ~2e5 rad a z = 20 mm, y en
float32 eso son 0.02 rad de error solo por la mantisa. Calcularla en doble y
bajar al dtype de trabajo unicamente el fasor —que ya esta acotado a modulo
1— cuesta cero y quita el problema.

UNIDADES: milimetros para todo. Da igual cual sea mientras sea la MISMA en
lambda, delta y z: el espectro angular solo ve lambda*z/delta^2.

    633 nm -> 633e-6 mm     3.45 um -> 3.45e-3 mm     20 mm -> 20.0

LAS DOS VUELTAS, Y POR QUE SON DISTINTAS
----------------------------------------
  A) desde el CAMPO COMPLEJO que sale de la ida. Es propagacion invertida y
     punto: tiene que devolver el objeto, salvo el espectro que la mascara
     descarto por el camino. Ahi esta la diferencia con FFT-ASM, que es
     unitario y devuelve el objeto exacto: BL-ASM tira banda a proposito, asi
     que su vuelta A no puede ser perfecta y la correlacion lo enseña.

  B) desde sqrt(|U|^2), que es lo unico que da un sensor real. La fase se
     perdio en la medida y vuelve el objeto con su imagen gemela encima.

Cuanto estropea la gemela depende del objeto. Si es mayormente OPACO (barras
claras sobre fondo negro) no queda haz sin tocar que haga de onda de
referencia, y sin referencia no hay holograma de Gabor que reconstruir. Con el
target invertido (barras oscuras sobre fondo claro, que es como se ve un DLHM
real) si reconstruye: INVERTIR lo cambia.

EL SIGNO DE Z
-------------
Z es la separacion objeto-sensor, POSITIVA. La ida usa +Z y la vuelta -Z.

Ese signo no se puede comprobar mirando la intensidad: el campo de la vuelta B
es real (sqrt de una intensidad), y para entrada real U(-z) = conj(U(+z)),
luego |U(-z)|^2 = |U(+z)|^2 exactamente. Con el signo cambiado la figura sale
identica. Solo se nota en la fase, y en cuanto se encadene algo no real
(correccion de fuente puntual, filtro complejo, recuperacion de fase).
"""

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import cupy as cp
except Exception:                      # sin CuPy, sin CUDA, o CuPy roto
    cp = None

# ════════════════════════════════════════════════════════════════════════════
#  EDITA ESTOS CUATRO
# ════════════════════════════════════════════════════════════════════════════

#: Imagen del OBJETO (transmitancia), no un holograma. Barras normales o
#: r"..." para que \U no se lea como escape.
RUTA = r"C:\Users\User\Desktop\Tesis\referencia\carlos\DLHM-model-main\DLHM-model-main\data\BenchmarkTarget.png"

#: Longitud de onda [mm].
LAMB = 633e-6

#: Paso de pixel [mm].
DELTA = 3.45e-3

#: Distancia objeto <-> sensor [mm], POSITIVA. La ida va a +Z y la vuelta a -Z.
Z = 100.0

# ════════════════════════════════════════════════════════════════════════════
#  Y esto solo si hace falta
# ════════════════════════════════════════════════════════════════════════════

#: "auto" usa la GPU si hay CuPy con CUDA; "cpu" y "gpu" fuerzan.
DISPOSITIVO = "auto"

#: dtype del campo. complex64 en GPU (ver PRECISION EN GPU); la fase va en
#: float64 en los dos casos pase lo que pase.
DTYPE = None                 # None = complex64 en GPU, complex128 en CPU

#: Reduce la imagen a este lado mayor antes de propagar, o None para dejarla.
#: OJO: si lo cambias, DELTA deja de ser el de tu sensor y hay que escalarlo
#: por (lado_original / lado_nuevo). El script lo hace y lo dice.
REDUCIR_A = None

#: Relleno de ceros. La FFT convoluciona de forma circular: lo que sale por un
#: borde reentra por el opuesto. Con 2 ese doblez queda fuera del recorte.
PAD = 2

#: True invierte la imagen (barras oscuras sobre fondo claro). Es lo que le da
#: onda de referencia al holograma y hace que la vuelta B reconstruya.
INVERTIR = False

#: Distancias del barrido de foco, como fraccion de Z. None lo desactiva.
BARRIDO = np.linspace(0.4, 1.6, 25)

#: Carpeta donde escribir las figuras, o None para solo mostrarlas.
SALIDA = None

#: Filas por bloque al evaluar el kernel. Baja si la GPU se queda sin memoria.
FILAS_POR_BLOQUE = 512


# ------------------------------------------------------ propagador de trabajo
def espectro_angular_bl(field, z, wavelength, dx, dy, xp=np,
                        dtype=np.complex128, filas=FILAS_POR_BLOQUE):
    """BL-ASM: espectro angular con el limite de banda de Matsushima (2009).

    Devuelve (campo, fraccion_de_banda_que_sobrevive). La fraccion es el area
    del rectangulo que la mascara deja pasar dividida por la de la malla, y es
    el numero que dice cuanto esta trabajando el limite: 1.00 quiere decir que
    a esta z el metodo es FFT-ASM, y 0.05 que esta tirando el 95 % del plano
    de frecuencias.

    Tres cosas de la implementacion, todas de ejecucion y ninguna de algoritmo:

    1. La funcion de transferencia NO se materializa entera. Sobre la malla
       6000x8000 de BenchmarkTarget con PAD = 2 una H completa son 0.72 GB en
       complex64, y el producto por el espectro otros 0.72. Aqui el fasor se
       evalua por bloques de filas y se multiplica in situ, asi que el pico
       extra es un bloque.

    2. La mascara tampoco. El limite de banda es un RECTANGULO CENTRADO y fx,
       fy van en orden creciente, asi que las frecuencias que sobreviven son un
       tramo contiguo: basta anular las cuatro bandas de fuera por rodajas. La
       alternativa habitual -construir dos meshgrid y una mascara booleana del
       tamano del plano- son tres arrays completos mas.

    3. La fase se calcula en float64 y solo el fasor -acotado a modulo 1- baja
       a dtype.

    searchsorted con side='right' en el extremo negativo y 'left' en el
    positivo reproduce la desigualdad ESTRICTA |f| < flim: a la izquierda se
    anula todo lo que cumple f <= -flim, y a la derecha todo lo que cumple
    f >= +flim.
    """
    U = xp.asarray(field, dtype=dtype)
    M, N = U.shape

    # cada eje con SU longitud: dfx sale del numero de columnas y dfy del de
    # filas. (angularSpectrum de pyDHM los cruza; ver retro_fft_angular.py.)
    dfx, dfy = 1 / (dx * N), 1 / (dy * M)
    fx = ((np.arange(N) - N / 2) * dfx).astype(np.float64)
    fy = ((np.arange(M) - M / 2) * dfy).astype(np.float64)

    # Ec. (21) de Matsushima & Shimobaba. Depende de z^2: mismo limite a +z
    # que a -z, que es lo que hace que este metodo no necesite el arreglo de
    # signo que si necesito el Kf de MPASM.
    flim_x = 1 / (wavelength * np.sqrt((2 * z * dfx) ** 2 + 1))
    flim_y = 1 / (wavelength * np.sqrt((2 * z * dfy) ** 2 + 1))

    F = xp.fft.fftshift(xp.fft.fft2(U))

    fx_d = xp.asarray(fx)[None, :]
    fy_d = xp.asarray(fy)[:, None]
    k = 2 * np.pi / wavelength
    for i0 in range(0, M, filas):
        i1 = min(i0 + filas, M)
        arg = 1.0 - (wavelength * fx_d) ** 2 - (wavelength * fy_d[i0:i1]) ** 2
        # las evanescentes se anulan, no se dejan decaer: retropropagando
        # crecerian. La mascara de banda las descarta igualmente (lo comprueba
        # comprobar_evanescentes), asi que esto es un cinturon sobre tirantes.
        propagante = arg > 0
        fase = k * z * xp.sqrt(xp.where(propagante, arg, 0.0))
        F[i0:i1] *= xp.where(propagante, xp.exp(1j * fase).astype(dtype), 0)
        del arg, propagante, fase

    x0 = int(np.searchsorted(fx, -flim_x, side="right"))
    x1 = int(np.searchsorted(fx, flim_x, side="left"))
    y0 = int(np.searchsorted(fy, -flim_y, side="right"))
    y1 = int(np.searchsorted(fy, flim_y, side="left"))
    F[:, :x0] = 0
    F[:, x1:] = 0
    F[:y0] = 0
    F[y1:] = 0
    fraccion = ((x1 - x0) * (y1 - y0)) / (M * N)

    return xp.fft.ifft2(xp.fft.ifftshift(F)), fraccion


# ------------------------------------------------------------------ backend
def elegir_dispositivo(preferencia="auto"):
    """(modulo de arrays, nombre). Cae a NumPy sin ruido si no hay CUDA."""
    hay_gpu = False
    if cp is not None:
        try:
            hay_gpu = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            hay_gpu = False
    if preferencia == "gpu":
        if not hay_gpu:
            raise SystemExit("DISPOSITIVO = 'gpu' pero no hay CUDA disponible.")
        return cp, "gpu"
    if preferencia == "cpu" or not hay_gpu:
        return np, "cpu"
    return cp, "gpu"


def a_cpu(a):
    return a.get() if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


def liberar(xp):
    if xp is cp:
        cp.get_default_memory_pool().free_all_blocks()


def comprobar_memoria(xp, M, N, dtype):
    """Aborta con un mensaje util en vez de con un OOM de CUDA.

    Se cuentan cuatro arrays del tamano de la malla: el campo, el fftshift, la
    salida de fft2 y el espacio de trabajo de cuFFT. El kernel no cuenta, que
    para eso va por bloques, y la mascara tampoco, que para eso va por rodajas.
    """
    if xp is not cp:
        return
    libre, _ = cp.cuda.runtime.memGetInfo()
    hacen_falta = 4 * M * N * np.dtype(dtype).itemsize
    if hacen_falta > 0.85 * libre:
        raise SystemExit(
            f"No cabe en la GPU: la malla {M}x{N} en "
            f"{np.dtype(dtype).name} pide ~{hacen_falta / 2**30:.2f} GB y hay "
            f"{libre / 2**30:.2f} GB libres.\nBaja PAD, pon REDUCIR_A (por "
            f"ejemplo 2048) o usa DISPOSITIVO = 'cpu'.")


def comprobar_simetria_del_limite(delta, z):
    """El limite de banda tiene que ser el mismo a +z y a -z.

    flim depende de z^2, asi que la igualdad es exacta y esta comprobacion
    parece de sobra. Existe porque en MPASM la suposicion equivalente -que Kf
    valia lo mismo en los dos sentidos- resulto FALSA en el codigo, no en la
    fisica: la Ec. (14) esta escrita para z > 0 y aplicada a z < 0 devolvia
    fmax negativo y apagaba la compresion en silencio. La leccion no fue sobre
    Kf, fue sobre suponer.
    """
    df = 1 / (delta * 1024)
    mas = 1 / (LAMB * np.sqrt((2 * abs(z) * df) ** 2 + 1))
    menos = 1 / (LAMB * np.sqrt((2 * -abs(z) * df) ** 2 + 1))
    return abs(mas - menos) / mas


def comprobar_evanescentes(delta, z, n=1024):
    """Cuantas frecuencias evanescentes deja pasar la mascara. Tiene que ser 0.

    Las evanescentes son lamb*f > 1. El limite de banda vale
    flim = 1/(lamb*sqrt((2 z df)^2 + 1)) y la raiz es >= 1 siempre, luego
    flim <= 1/lamb y la mascara las corta todas. Es la razon de que BL-ASM no
    reviente al retropropagar donde FFT-ASM llega a 2.87e+110.
    """
    df = 1 / (delta * n)
    f = (np.arange(n) - n / 2) * df
    flim = 1 / (LAMB * np.sqrt((2 * z * df) ** 2 + 1))
    pasan = np.abs(f) < flim
    return int(np.count_nonzero(pasan & (LAMB * np.abs(f) > 1.0)))


# ------------------------------------------------------------------ utilidades
def cargar_objeto(ruta, invertir=False, reducir_a=None, delta=DELTA):
    """Imagen -> (transmitancia en [0,1] como campo complejo, delta efectivo).

    La entrada es el OBJETO, no un holograma: la imagen ES la transmitancia y
    el campo es t, no sqrt(t). (Un holograma es intensidad medida y ahi si va
    la raiz; ver el docstring del modulo.)

    Reducir la imagen agranda el pixel: delta se escala por el mismo factor, o
    la escala fisica de la propagacion cambiaria sin que nadie lo note.
    """
    img = Image.open(ruta).convert("L")
    if reducir_a is not None and max(img.size) > reducir_a:
        factor = reducir_a / max(img.size)
        nuevo = (max(1, round(img.size[0] * factor)),
                 max(1, round(img.size[1] * factor)))
        delta = delta * (img.size[0] / nuevo[0])
        img = img.resize(nuevo, Image.LANCZOS)
    t = np.asarray(img, dtype=float) / 255.0
    if invertir:
        t = 1.0 - t
    return t, delta


def propagar(U, z, delta, lamb, pad, xp, dtype):
    """espectro_angular_bl con relleno de ceros y recorte al tamano original.

    Devuelve (campo recortado, fraccion de banda que sobrevivio).
    """
    M, N = U.shape
    if pad > 1:
        rel = xp.zeros((M * pad, N * pad), dtype=dtype)
        i, j = (rel.shape[0] - M) // 2, (rel.shape[1] - N) // 2
        rel[i:i + M, j:j + N] = xp.asarray(U, dtype=dtype)
    else:
        rel, i, j = xp.asarray(U, dtype=dtype), 0, 0
    fuera, fraccion = espectro_angular_bl(rel, z, lamb, delta, delta, xp=xp,
                                          dtype=dtype)
    del rel
    # copia, no vista: una vista mantendria viva la malla rellena entera, que
    # con PAD = 2 sobre 4000x3000 son 0.36 GB por reconstruccion del barrido
    recorte = fuera[i:i + M, j:j + N].copy()
    del fuera
    return recorte, fraccion


def nitidez(I):
    """Energia del gradiente, normalizada. Maxima en el plano de foco.

    Un objeto de amplitud enfocado tiene bordes duros; desenfocado los tiene
    difuminados. La suma de |grad I|^2 lo mide sin necesitar saber cual era el
    objeto, que es lo que hace falta en un barrido de una medida real.
    """
    I = np.asarray(a_cpu(I), dtype=float)
    I = I / I.max()
    gy, gx = np.gradient(I)
    return float(np.mean(gx**2 + gy**2))


def parecido(a, b):
    """Correlacion entre dos mapas, normalizados por su maximo."""
    a = np.asarray(a_cpu(a), float).ravel()
    b = np.asarray(a_cpu(b), float).ravel()
    return float(np.corrcoef(a / a.max(), b / b.max())[0, 1])


# ---------------------------------------------------------------------- main
def main():
    if not pathlib.Path(RUTA).is_file():
        raise SystemExit(f"No encuentro la imagen en:\n    {RUTA}\n\n"
                         "Edita la constante RUTA al principio del archivo.")
    if Z <= 0:
        raise SystemExit(
            f"Z = {Z} pero es la distancia objeto-sensor y va POSITIVA: los "
            f"signos los ponen la ida (+Z) y la vuelta (-Z).\nPon Z = {abs(Z)}.")

    xp, dev = elegir_dispositivo(DISPOSITIVO)
    dtype = DTYPE or (np.complex64 if dev == "gpu" else np.complex128)

    t_obj, delta = cargar_objeto(RUTA, INVERTIR, REDUCIR_A, DELTA)
    M, N = t_obj.shape
    comprobar_memoria(xp, M * PAD, N * PAD, dtype)

    print(f"objeto {RUTA}")
    print(f"  malla {M}x{N}, ventana {N * delta:.3f} x {M * delta:.3f} mm "
          f"(con relleno x{PAD}: {N * PAD * delta:.3f} mm)")
    if delta != DELTA:
        print(f"  imagen reducida a {max(M, N)} px: delta escalado de "
              f"{DELTA * 1e3:.3f} a {delta * 1e3:.3f} um")
    print(f"  lambda {LAMB * 1e6:.1f} nm | delta {delta * 1e3:.3f} um | "
          f"ida +{Z:.3f} mm, vuelta {-Z:+.3f} mm")
    print(f"  dispositivo {dev.upper()} | dtype {np.dtype(dtype).name} | "
          f"fase en float64")
    if dev == "gpu":
        libre, total = cp.cuda.runtime.memGetInfo()
        print(f"  {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, "
              f"{libre / 2**30:.2f} de {total / 2**30:.2f} GB libres")

    print("\n  BL-ASM no tiene propagador de referencia de terceros en el repo:")
    print("  pyDHM trae fresnel, bluestein y angularSpectrum, y ninguno limita")
    print("  la banda. La verificacion del metodo vive en la suite de CamposT.")

    sim = comprobar_simetria_del_limite(delta, Z)
    ev = comprobar_evanescentes(delta, Z)
    print(f"\n  limite de banda a +z y a -z: difieren en {sim:.1e}  "
          f"<- tiene que ser 0")
    print(f"  evanescentes que pasan la mascara: {ev}  <- tiene que ser 0")
    if ev:
        print("  AVISO: la mascara deja pasar evanescentes. Retropropagando "
              "esas crecen en vez de decaer.")

    f_nyq = 1 / (2 * delta)
    Np = N * PAD
    dfx = 1 / (delta * Np)
    flim = 1 / (LAMB * np.sqrt((2 * Z * dfx) ** 2 + 1))
    print(f"\n  f_Nyquist {f_nyq:.1f} 1/mm | limite de banda {flim:.1f} 1/mm "
          f"({min(flim / f_nyq, 1.0) * 100:.1f} % de la malla por eje)")
    if flim >= f_nyq:
        print("  A esta z la mascara no recorta nada: BL-ASM es aqui FFT-ASM. "
              "Sube Z para verlo trabajar.")
    fondo = t_obj.mean() / t_obj.max()
    print(f"  fondo del objeto: {fondo:.2f} del maximo"
          + ("" if fondo >= 0.25 else
             "  <- oscuro: sin onda de referencia, la vuelta B no reconstruye"))

    def cronometrar(f):
        if dev == "gpu":
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        r = f()
        if dev == "gpu":
            cp.cuda.Stream.null.synchronize()
        return r, time.perf_counter() - t0

    kw = dict(xp=xp, dtype=dtype)
    U0 = xp.asarray(t_obj, dtype=dtype)

    (U_h, fr_ida), t_ida = cronometrar(
        lambda: propagar(U0, +Z, delta, LAMB, PAD, **kw))
    I_h = xp.abs(U_h) ** 2
    (U_a, fr_a), t_a = cronometrar(
        lambda: propagar(U_h, -Z, delta, LAMB, PAD, **kw))
    (U_b, _), t_b = cronometrar(
        lambda: propagar(xp.sqrt(I_h).astype(dtype), -Z, delta, LAMB, PAD, **kw))

    print(f"\n  ida {t_ida * 1e3:.0f} ms | vuelta A {t_a * 1e3:.0f} ms | "
          f"vuelta B {t_b * 1e3:.0f} ms")
    print(f"  banda que sobrevive a la mascara: {fr_ida * 100:.2f} % en la ida, "
          f"{fr_a * 100:.2f} % en la vuelta  <- iguales, el limite no ve el signo")
    print(f"  vuelta A (campo complejo)  correlacion con el objeto = "
          f"{parecido(xp.abs(U_a), xp.abs(U0)):+.4f}")
    print(f"    OJO: aqui NO tiene que dar 1 como en FFT-ASM. BL-ASM descarta")
    print(f"    espectro a proposito, y lo descartado no vuelve: la ida tiro "
          f"{(1 - fr_ida) * 100:.2f} %")
    print(f"  vuelta B (solo intensidad) correlacion con el objeto = "
          f"{parecido(xp.abs(U_b), xp.abs(U0)):+.4f}")

    # --- barrido de foco -----------------------------------------------------
    zs = curva_a = curva_b = fracciones = None
    if BARRIDO is not None:
        zs = np.asarray(BARRIDO, float) * Z
        raiz_I = xp.sqrt(I_h).astype(dtype)
        curva_a, curva_b, fracciones = [], [], []
        t0 = time.perf_counter()
        for k, z in enumerate(zs):
            Ua, fr = propagar(U_h, -z, delta, LAMB, PAD, **kw)
            curva_a.append(nitidez(xp.abs(Ua) ** 2)); fracciones.append(fr)
            del Ua
            Ub, _ = propagar(raiz_I, -z, delta, LAMB, PAD, **kw)
            curva_b.append(nitidez(xp.abs(Ub) ** 2)); del Ub
            liberar(xp)
            print(f"    barrido {k + 1}/{len(zs)}   z = {z:7.2f} mm", end="\r")
        if dev == "gpu":
            cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        del raiz_I
        liberar(xp)
        curva_a, curva_b = np.array(curva_a), np.array(curva_b)
        fracciones = np.array(fracciones)
        print(" " * 48, end="\r")
        print(f"\n  barrido de foco: {len(zs)} distancias de {zs[0]:.1f} a "
              f"{zs[-1]:.1f} mm en {dt:.1f} s "
              f"({dt / (2 * len(zs)) * 1e3:.0f} ms por propagacion)")
        print(f"    vuelta A enfoca en z = {zs[curva_a.argmax()]:.2f} mm")
        print(f"    vuelta B enfoca en z = {zs[curva_b.argmax()]:.2f} mm")
        print(f"    esperado:            z = {Z:.2f} mm")
        if fracciones.max() - fracciones.min() > 1e-6:
            print(f"    banda que sobrevive: de {fracciones.max() * 100:.2f} % a "
                  f"{fracciones.min() * 100:.2f} %, estrechandose al alejarse")
            print(f"    OJO: la nitidez y la banda caen las dos con z, asi que "
                  f"el pico de foco\n    sale sesgado hacia distancias cortas. "
                  f"El panel de abajo de foco_blas.png\n    es el que deja ver "
                  f"ese sesgo; no lo corrige.")
        else:
            print(f"    la mascara no recorta en todo el barrido "
                  f"({fracciones.max() * 100:.2f} %): aqui BL-ASM es FFT-ASM")

    # --- figuras -------------------------------------------------------------
    obj = a_cpu(xp.abs(U0))
    paneles = [("objeto (entrada)", obj),
               (f"holograma  |U|^2 a +{Z:g} mm", a_cpu(I_h)),
               (f"vuelta A: del campo complejo\ncorr = {parecido(xp.abs(U_a), xp.abs(U0)):+.3f}",
                a_cpu(xp.abs(U_a)) ** 2),
               (f"vuelta B: de la intensidad\ncorr = {parecido(xp.abs(U_b), xp.abs(U0)):+.3f}",
                a_cpu(xp.abs(U_b)) ** 2)]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
    for a, (titulo, im) in zip(ax, paneles):
        im = np.asarray(im, float)
        a.imshow((im / im.max()) ** 0.5, cmap="gray")
        a.set_title(titulo, fontsize=10)
        a.axis("off")
    fig.suptitle(f"BL-ASM, banda util {fr_ida * 100:.2f} % a z = {Z:g} mm",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    figuras = [("ida_y_vuelta_blas.png", fig)]

    if zs is not None:
        # Dos paneles y no dos escalas en el mismo eje: la nitidez va
        # normalizada a 1 y la banda en tanto por ciento, y superponerlas con
        # un twinx deja al lector decidiendo que curva mira que eje. Comparten
        # la x, que es lo unico que de verdad tienen en comun.
        fig2, (a2, a3) = plt.subplots(
            2, 1, figsize=(7.2, 5.4), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
        a2.plot(zs, curva_a / curva_a.max(), "o-", ms=3, label="A: campo complejo")
        a2.plot(zs, curva_b / curva_b.max(), "s-", ms=3, label="B: solo intensidad")
        a2.axvline(Z, color="k", ls="--", lw=1, label=f"z real = {Z:g} mm")
        a2.set_ylabel("nitidez (energia del gradiente,\nnormalizada)")
        a2.set_title(f"Donde enfoca la retropropagacion con BL-ASM  ({dev.upper()})",
                     fontsize=11)
        a2.legend(fontsize=9)

        # lo que distingue a este metodo de FFT-ASM: la mascara se estrecha al
        # alejarse, justo donde el otro empieza a aliasar
        a3.plot(zs, fracciones * 100, color="0.45", lw=1.4)
        a3.axvline(Z, color="k", ls="--", lw=1)
        a3.set_xlabel("distancia de reconstruccion [mm]")
        a3.set_ylabel("banda que sobrevive\na la mascara [%]", fontsize=9)
        a3.set_ylim(bottom=0)
        fig2.tight_layout()
        figuras.append(("foco_blas.png", fig2))

    if SALIDA is not None:
        destino = pathlib.Path(SALIDA)
        destino.mkdir(parents=True, exist_ok=True)
        for nombre, f in figuras:
            f.savefig(destino / nombre, dpi=150, bbox_inches="tight")
            print(f"  -> {destino / nombre}")
    plt.show()


if __name__ == "__main__":
    main()
