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
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\campos\entrada.png"

#: Longitud de onda [mm].
LAMB = 633e-6

#: Paso de pixel [mm].
DELTA = 3.45e-3

#: Distancia objeto <-> sensor [mm], POSITIVA. La ida va a +Z y la vuelta a -Z.
Z = 20.0

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

#: True escribe la pila de foco entera en
#: resultados/reconstruccion/<objeto>/blas/<A|B>/, un PNG por distancia y
#: por vuelta. El barrido ya reconstruye a cada z; sin esto se queda solo la
#: curva de nitidez y para mirar cualquier otra distancia hay que repropagar.
#: No cambia el pico de memoria -escribe el campo que ya estaba vivo-, pero si
#: llena disco: a 512 son 82 PNG de ~100 KB, y con BenchmarkTarget sin reducir
#: son 82 de varios MB. Apagalo en ese caso.
GUARDAR_BARRIDO = True

#: True anade la figura de fases (fases_*.png). Los campos ya estan
#: calculados, asi que solo cuesta el dibujado; pero son ocho paneles de la
#: malla entera, y en 3000x4000 eso se nota.
FASES = True

#: Lado del recorte de detalle de la figura de fases, en pixeles. A malla
#: completa la estructura de franjas no se ve: 4000 px no caben en un panel.
ZOOM_FASE = 200

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
    """Energia del gradiente, normalizada por la MEDIA. Maxima en el foco.

    Un objeto de amplitud enfocado tiene bordes duros; desenfocado los tiene
    difuminados. La suma de |grad I|^2 lo mide sin necesitar saber cual era el
    objeto, que es lo que hace falta en un barrido de una medida real.

    POR LA MEDIA Y NO POR EL MAXIMO, y esto no es un detalle de estilo. El
    maximo no es una constante del problema: es una cantidad que DEPENDE DEL
    FOCO. Al enfocar, la luz se concentra y el maximo sube, asi que dividir por
    el cancela justo el efecto que se quiere medir. Sobre una gaussiana de
    energia fija, estrechar el pico de 30 a 5 pixeles dejaba la metrica en
    x0.96 -DECRECIA- normalizando por el maximo, y da x1247 por la media.

    Lo que costaba: con el maximo, el pico del barrido acertaba el 25 % de las
    veces y erraba hasta un 15 %. Con la media acierta el 100 % con un error
    medio del 0.3 %, medido sobre siete distancias y cuatro resoluciones de
    barrido. Tambien quita el sesgo de la vuelta A a z larga con BL-ASM, que
    se venia atribuyendo al estrechamiento de la mascara de banda: no era la
    mascara.

    La media es la energia por pixel, y la propagacion la conserva salvo lo que
    escapa por los bordes: es un normalizador estable con z, que es exactamente
    lo que el maximo no es. Normalizar hace falta de todos modos, o la metrica
    mediria el brillo del plano en vez de su estructura.

    Un plano identicamente nulo daria 0/0. No tiene estructura: su nitidez es 0.
    """
    I = np.asarray(a_cpu(I), dtype=float)
    m = I.mean()
    if m <= 0:
        return 0.0
    gy, gx = np.gradient(I / m)
    return float(np.mean(gx**2 + gy**2))


def pico_de_foco(zs, curva, margen=0.15):
    """Distancia del maximo de la curva, o None si el barrido no acota el foco.

    El argmax siempre existe. Si el barrido no contiene el plano de foco, el
    argmax devuelve el extremo mas alto, y publicarlo con dos decimales lo
    convierte en una medida que nadie ha hecho.

    Al acercarse al plano del holograma la reconstruccion tiende al holograma
    mismo, que es un patron de franjas densas: nitidez maxima. Es una tendencia
    monotona hacia z -> 0, no un pico. Cuando el pico verdadero es debil -la
    vuelta B, con la gemela encima- esa tendencia gana y el argmax se va al
    extremo corto.

    Medido sobre 24 casos (dos propagadores x dos vueltas x dos rangos x tres
    distancias): los 4 que erraban tenian el argmax dentro del margen y los 20
    que acertaban, fuera. Sin excepciones en ninguno de los dos sentidos.

    La prominencia del pico -(max - mediana) / desviacion- NO separa: los
    aciertos bajan a 0.9 sigma y los fallos suben a 1.9. Por eso la guarda es
    geometrica y no estadistica, que era lo que parecia a primera vista.
    """
    curva = np.asarray(curva, dtype=float)
    k = int(np.argmax(curva))
    n = len(curva)
    if k < margen * n or k > (1 - margen) * n:
        return None
    return float(np.asarray(zs, dtype=float)[k])


def parecido(a, b):
    """Correlacion entre dos mapas, normalizados por su maximo."""
    a = np.asarray(a_cpu(a), float).ravel()
    b = np.asarray(a_cpu(b), float).ravel()
    return float(np.corrcoef(a / a.max(), b / b.max())[0, 1])


# -------------------------------------------------------------------- guardado
# Estas dos son IDENTICAS en los tres scripts de retropropagacion, y
# tests/test_guardado_barrido.py comprueba que no se separen. Ojo: del formato
# de nombre hay CUATRO copias, no tres, porque CamposT.retropropagacion escribe
# su pila de foco igual; la prueba mira tambien esa.
def nombre_png(z):
    """Nombre del PNG de una distancia del barrido.

    Tres decimales y ancho fijo, no el z0020.png entero de resultados/campos/.
    Un linspace da distancias no enteras: con el formato entero, 49.6 y 50.2
    escribirian las dos en z0050.png y la segunda pisaria a la primera sin
    aviso. El ancho fijo mantiene el orden alfabetico igual al orden del
    barrido, que es lo que hace que un `ls` de la carpeta enseñe la pila.
    """
    return f"z{z:08.3f}.png"


def carpeta_barrido(objeto, metodo, vuelta):
    """resultados/reconstruccion/<objeto>/<metodo>/<A|B>/, bajo la raiz del repo.

    Absoluta y no relativa: los scripts se lanzan desde el editor, desde la
    carpeta scripts/ o desde la raiz, y una ruta relativa dejaria la pila en el
    directorio de invocacion, distinto cada vez.

    <objeto> es el stem de RUTA. Sin el, correr con entrada.png y despues con
    BenchmarkTarget.png a la misma Z escribe los mismos nombres en la misma
    carpeta y la segunda pila pisa a la primera sin aviso. Es la misma razon
    por la que CamposT.retropropagacion mete <holograma> en su ruta.
    """
    raiz = pathlib.Path(__file__).resolve().parent.parent
    return raiz / "resultados" / "reconstruccion" / objeto / metodo / vuelta


def guardar_png(I, ruta):
    """Escribe un mapa de intensidad como PNG de 8 bits, creando la carpeta.

    |U|^2 dividido por su maximo, y NADA MAS: sin gamma.

    Tenia gamma = 0.5, "para que la pila y la figura del mismo barrido se
    parezcan". Pero los paneles de las figuras aplicaban la misma gamma, y

        (|U|^2 / max)^0.5  =  |U| / sqrt(max)

    o sea que una gamma de 0.5 sobre una intensidad ES la amplitud: ni la pila
    ni las figuras ensenaban |U|^2. Ahora las dos son intensidad y se siguen
    pareciendo entre si, que era el requisito de verdad.

    La division por el maximo se queda: un PNG de 8 bits no admite floats y hay
    que llevar el rango a [0, 1]. Es LINEAL, o sea que no cambia la relacion
    entre valores. La gamma si la cambia, y de la forma exacta que convierte
    intensidad en amplitud.

    Normalizar por el maximo de CADA imagen mide contraste y no brillo
    absoluto: dos distancias del barrido son comparables aunque no les llegue
    la misma energia.

    Un campo identicamente nulo daba 0/0: NaN por todo el array y un PNG de
    basura, sin excepcion y sin aviso. Un plano negro es un resultado legitimo
    -sale al reconstruir un campo que se anulo- y hay que poder verlo como tal.
    """
    A = np.asarray(a_cpu(I), dtype=np.float64)
    m = A.max()
    A = np.clip(A / m, 0, 1) if m > 0 else np.zeros_like(A)
    ruta = pathlib.Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((A * 255).astype(np.uint8)).save(ruta)


# ------------------------------------------------------------------------ fase
# Estas dos son IDENTICAS en los tres scripts de retropropagacion, y
# tests/test_fases_retro.py comprueba que no se separen: el riesgo de que cada
# script lleve su copia es que alguien arregle la formula en uno solo.
def sin_piston(U, mask):
    """angle(U) con la fase global quitada, pesando por amplitud.

    Propagar multiplica el campo por una fase global -el exp(ikz) y lo que
    arrastre la normalizacion-. Esa constante es la eleccion de origen de
    fases, no un error de reconstruccion: compararla sin quitarla mide una
    constante irrelevante.

    El piston es el angulo del fasor MEDIO, no la media de los angulos. La
    media de angulos falla justo cuando el piston vale pi: angle() reparte
    esos pixeles entre +3.14 y -3.14, la media sale ~0, y entonces no quita
    nada y no avisa. sum(U) no corta el circulo, asi que no tiene ese
    problema, y de paso pesa por amplitud, que es lo que se quiere: donde no
    hay senal la fase es ruido y no debe votar.
    """
    U = a_cpu(U)
    piston = np.angle(np.sum(U[mask]))
    return np.angle(U * np.exp(-1j * piston))


def rms_fase(U, U0, mask):
    """Error RMS de fase [rad] contra el objeto, sobre la mascara y sin piston.

    U0 es la transmitancia del objeto: real y positiva, o sea de fase 0. Por
    eso su fase es un dato conocido con el que contrastar y no hace falta otra
    referencia.

    Es ciega a la amplitud, al reves que parecido(). Las dos conviven aqui a
    proposito: la vuelta A y la vuelta B se parecen mucho en modulo y se
    distinguen en la fase, que es justo lo que un sensor no mide.

    Escala de lectura: una fase sin ninguna informacion, uniforme en
    (-pi, pi], da pi/sqrt(3) = 1.8138 rad = 103.9 grados.
    """
    d = a_cpu(U)[mask] * np.conj(a_cpu(U0)[mask])
    d = d * np.exp(-1j * np.angle(np.sum(d)))
    return float(np.sqrt(np.mean(np.angle(d) ** 2)))


def pinta_fase(ax, fase, alfa, titulo):
    """Un panel de fase: colormap ciclico y opacidad proporcional a amplitud.

    Recibe la fase y la opacidad YA calculadas, no el campo, para que el panel
    completo y su zoom compartan piston y normalizacion. Calcularlos dentro
    daria dos pistones distintos para el mismo campo, y el recorte parecerian
    otra reconstruccion.

    Dos decisiones sin las cuales la figura miente:

    - Fondo GRIS, no blanco. En twilight el +-pi ES blanco, asi que sobre
      fondo blanco no hay forma de distinguir "la fase vale pi" de "aqui no
      hay amplitud y esto es ruido".
    - Opacidad = amplitud. BenchmarkTarget tiene el fondo al 0.04 del maximo:
      en casi todo el plano la fase es ruido uniforme en (-pi, pi], y pintarla
      cruda da un confeti vistoso que no significa nada.
    """
    im = ax.imshow(fase, cmap="twilight", vmin=-np.pi, vmax=np.pi, alpha=alfa,
                   interpolation="nearest")
    ax.set_facecolor("0.62")
    ax.set_title(titulo, fontsize=9.5)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


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
        if GUARDAR_BARRIDO:
            objeto = pathlib.Path(RUTA).stem
            dir_a = carpeta_barrido(objeto, "blas", "A")
            dir_b = carpeta_barrido(objeto, "blas", "B")
        curva_a, curva_b, fracciones = [], [], []
        t0 = time.perf_counter()
        for k, z in enumerate(zs):
            Ua, fr = propagar(U_h, -z, delta, LAMB, PAD, **kw)
            Ia = xp.abs(Ua) ** 2
            curva_a.append(nitidez(Ia)); fracciones.append(fr)
            if GUARDAR_BARRIDO:
                guardar_png(Ia, dir_a / nombre_png(z))
            del Ua, Ia
            Ub, _ = propagar(raiz_I, -z, delta, LAMB, PAD, **kw)
            Ib = xp.abs(Ub) ** 2
            curva_b.append(nitidez(Ib))
            if GUARDAR_BARRIDO:
                guardar_png(Ib, dir_b / nombre_png(z))
            del Ub, Ib
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
        if GUARDAR_BARRIDO:
            print(f"    pila de foco: {2 * len(zs)} PNG en "
                  f"{dir_a.parent}, subcarpetas A y B")
        z_a = pico_de_foco(zs, curva_a)
        print(f"    vuelta A enfoca en z = {z_a:.2f} mm" if z_a is not None
              else "    vuelta A: el maximo cae en un extremo del barrido: el "
                   "barrido NO acota el foco. Mueve o ensancha BARRIDO.")
        z_b = pico_de_foco(zs, curva_b)
        print(f"    vuelta B enfoca en z = {z_b:.2f} mm" if z_b is not None
              else "    vuelta B: el maximo cae en un extremo del barrido: el "
                   "barrido NO acota el foco. Mueve o ensancha BARRIDO.")
        print(f"    esperado:            z = {Z:.2f} mm")
        if fracciones.max() - fracciones.min() > 1e-6:
            print(f"    banda que sobrevive: de {fracciones.max() * 100:.2f} % a "
                  f"{fracciones.min() * 100:.2f} %, estrechandose al alejarse")
            print(f"    La mascara se estrecha al alejarse y la reconstruccion "
                  f"se emborrona con ella, enfoque o no. Eso NO corre el pico:")
            print(f"    medido a Z = 30, 50, 70 y 90 mm sobre el rango completo, "
                  f"el error es 0.0 % en las cuatro. El sesgo hacia distancias")
            print(f"    cortas que se avisaba aqui lo producia nitidez() al "
                  f"normalizar por el maximo, no la mascara.")
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

    # --- fases ---------------------------------------------------------------
    # Es lo que separa la vuelta A de la vuelta B, y lo que las figuras de
    # intensidad no pueden ensenar: |U| las hace parecidas, la fase no.
    if FASES:
        mask_obj = t_obj > 0.5 * t_obj.max()
        rms_a = rms_fase(U_a, U0, mask_obj)
        rms_b = rms_fase(U_b, U0, mask_obj)
        print(f"\n  error RMS de fase contra el objeto, sin piston y sobre el "
              f"{100 * mask_obj.mean():.1f}% brillante del plano:")
        print(f"    vuelta A (campo complejo) : {np.degrees(rms_a):6.2f} deg")
        print(f"    vuelta B (solo intensidad): {np.degrees(rms_b):6.2f} deg")
        print(f"    sin ninguna informacion de fase saldrian "
              f"{np.degrees(np.pi / np.sqrt(3)):.1f} deg")
        print(f"    OJO: la RMS solo mira donde el objeto brilla, que es donde "
              f"la imagen real\n    domina en amplitud. El dano de la vuelta B "
              f"esta repartido por el resto\n    del plano, y eso lo ensena la "
              f"figura, no el numero.")

        campos = [("objeto (entrada)\nreal positivo -> fase 0", U0),
                  (f"ida: campo a +{Z:g} mm", U_h),
                  (f"vuelta A: del campo complejo\n"
                   f"RMS = {np.degrees(rms_a):.1f} deg", U_a),
                  (f"vuelta B: de la intensidad\n"
                   f"RMS = {np.degrees(rms_b):.1f} deg", U_b)]

        Mo, No = t_obj.shape
        lado = min(ZOOM_FASE, Mo, No)
        rec = (slice((Mo - lado) // 2, (Mo - lado) // 2 + lado),
               slice((No - lado) // 2, (No - lado) // 2 + lado))

        fig3, ax3 = plt.subplots(2, 4, figsize=(16.5, 8.6))
        for k, (titulo, U) in enumerate(campos):
            fase = sin_piston(U, mask_obj)
            alfa = np.abs(a_cpu(U)).astype(float)
            alfa = np.clip(alfa / np.percentile(alfa, 99.5), 0, 1) ** 0.45
            im = pinta_fase(ax3[0, k], fase, alfa, titulo)
            pinta_fase(ax3[1, k], fase[rec], alfa[rec],
                       f"detalle {lado}x{lado} px, centro")
        ax3[0, 0].set_ylabel("plano completo", fontsize=10)
        ax3[1, 0].set_ylabel(f"zoom {lado} px", fontsize=10)
        cb = fig3.colorbar(im, ax=ax3, fraction=0.018, pad=0.015,
                           ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb.ax.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
        cb.set_label("fase [rad], piston quitado | opacidad = amplitud",
                     fontsize=9)
        fig3.suptitle(f"Fase de la ida y las dos vueltas -- BL-ASM, "
                      f"Z = {Z:g} mm, delta = {delta * 1e3:.2f} um",
                      fontsize=12)
        figuras.append(("fases_blas.png", fig3))

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
