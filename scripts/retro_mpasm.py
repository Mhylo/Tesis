"""Ida y vuelta con el espectro angular matricial: objeto -> +z -> holograma -> -z -> objeto.

Propaga la imagen de entrada una distancia Z y retropropaga el resultado -Z
para ver si vuelve a enfocar donde debe. Un solo metodo, escrito a mano, sin
importar CamposT: sirve de contraste independiente del paquete.

Es el tercero de la serie, con MPASM en vez de FFT-ASM (retro_fft_angular.py) o
BL-ASM (retro_blas.py). Y es el unico de los tres que retropropaga con el
metodo del paper que sostiene la tesis, asi que es donde importa que la vuelta
salga bien.

CPU o GPU con el mismo cuerpo de algoritmo (ver DISPOSITIVO). Lo unico que
cambia es si el modulo de arrays es NumPy o CuPy.

EL PROPAGADOR
-------------
MPASM sustituye las dos FFT por productos matriciales, Zhao et al., Opt. Lett.
45, 5937 (2020). La DFT directa se escribe como My @ U0 @ Mx con Mx y My
matrices de fasores, y eso permite elegir el muestreo del plano de frecuencias
en vez de heredarlo de la malla de entrada. Cuatro parametros lo controlan:

    s    sobremuestreo del espectro. La matriz espectral es (s*M, s*N).
    Kf   compresion del intervalo espectral, Ec. (14). Kf > 1 acerca las
         muestras al origen, que es donde vive el espectro cuando el haz se
         ensancha al alejarse.
    r    numero de puntos del plano de salida, en unidades de la malla.
    mag  razon entre el paso de salida y el de entrada: agranda la ventana de
         observacion sin tocar el plano de entrada.

Kf es lo que distingue a MPASM de un FFT-ASM caro. Si Kf = 1, MPASM ES un
FFT-ASM rellenado de ceros y no aporta nada, solo cuesta mas.

EL PROPAGADOR DE REFERENCIA
---------------------------
MatrixDftCPU va TAL CUAL, copiada de referencia/zhao2020/MatrixDft.py sin
tocar una linea: es la maquinaria de DFT matricial del propio Wanli Zhao.

propagacion_original() y kf_original() son transcripciones de
LightPropagate.propagate_cpu y .propagate_pro_CPU de referencia/zhao2020/
Light.py: mismo cuerpo linea por linea, con self.X sustituido por el parametro
X. Se transcriben en vez de importarse porque Light.py hace `import cupy` en la
primera linea y `from MatrixDft import ...` con ruta relativa, asi que no se
puede importar tal cual desde aqui (es lo que arregla scripts/
parchar_referencias.py, y aqui se prefiere tener el codigo a la vista).

mpasm_bloques() calcula lo mismo pero cabe en la tarjeta. La equivalencia entre
las dos se comprueba en cada corrida y se imprime (comprobar_equivalencia).

DOS COSAS DEL ORIGINAL, MEDIDAS, NO SUPUESTAS
---------------------------------------------
  - La Ec. (14) esta escrita `A**2 * B` en el codigo y `A**2 + B` en el paper.
    Es un `*+` donde iba un `+`: en Python `a*+b` es `a*(+b)`, o sea el
    producto. La version con producto no es dimensionalmente consistente.
    kf_original() lo reproduce; kf_paper() usa la suma. El script imprime los
    dos valores en cada corrida.

  - k2 se calcula a partir de z SIN VALOR ABSOLUTO. Con z < 0 el denominador
    4*sqrt(2)*z*lamb sale negativo, fmax sale negativo, y el `if k2<=1: k2=1`
    devuelve 1: la compresion se apaga entera y en silencio. Como retropropagar
    es precisamente propagar a -z, el original corre TODA reconstruccion como
    un FFT-ASM rellenado de ceros. Kf depende de |z| y no de z, porque el ancho
    de banda de la funcion de transferencia es el mismo en los dos sentidos:
    H(-z) = conj(H(z)). El script lo enseña llamando al original a +Z y a -Z.

PRECISION EN GPU
----------------
El campo va en complex64 y la fase en float64 SIEMPRE, se propague donde se
propague. Los fasores de Mx y My son exp(-2i*pi*x*fx) con productos x*fx que
llegan a ~1e5: en float32 la reduccion de argumento pierde la fase entera.
Calcular el producto externo en doble y bajar al dtype de trabajo unicamente el
fasor -que ya esta acotado a modulo 1- cuesta cero y quita el problema.

MEMORIA: ES EL CUELLO DE BOTELLA DE ESTE METODO
------------------------------------------------
La matriz espectral es (s*M, s*N) y las cuatro de fasores son (N, s*N),
(s*M, M), (s*N, r*N) y (r*M, s*M). Sobre la malla 6000x8000 que sale de
BenchmarkTarget con PAD = 2, y con s = 1, eso ya son 2.4 GB en complex64: no
cabe en una tarjeta de 4 con nada mas dentro. Con s = 2 son cuatro veces mas.

Por eso REDUCIR_A viene puesto a 1024 por defecto en este script y a None en
sus dos hermanos. No es una preferencia: es que MPASM no admite las mallas que
FFT-ASM se traga sin pestanear, y ese es justo el precio que compra el control
del muestreo.

UNIDADES: milimetros para todo. Da igual cual sea mientras sea la MISMA en
lambda, delta y z.

    633 nm -> 633e-6 mm     3.45 um -> 3.45e-3 mm     20 mm -> 20.0

COMPRIMIR TAMBIEN ES DESCARTAR
------------------------------
Kf > 1 acerca las muestras del espectro al origen, y eso estrecha el rango de
frecuencias que se muestrea a +-1/(2*delta*Kf). Lo que queda fuera no vuelve.
O sea que MPASM con compresion NO es reversible, igual que BL-ASM no lo es: la
diferencia es que BL-ASM tira banda con una mascara y MPASM la deja fuera
cambiando donde muestrea. Medido con este script sobre BenchmarkTarget
reducido a 512 px: a Z = 100 mm, donde Kf = 1, la vuelta A correlaciona +0.940
con el objeto -lo mismo que FFT-ASM y BL-ASM, porque los tres son ahi el mismo
metodo-; a Z = 2000 mm, con Kf = 1.305, baja a +0.662, que es practicamente lo
que da BL-ASM ahi mismo (+0.662).

Lo que compra ese precio es no aliasar: FFT-ASM a esa distancia no pierde
banda, la dobla sobre si misma, que es peor y no se ve venir.

LAS DOS VUELTAS, Y POR QUE SON DISTINTAS
----------------------------------------
  A) desde el CAMPO COMPLEJO que sale de la ida. Con Kf = 1 es propagacion
     invertida y punto: tiene que devolver el objeto. Con Kf > 1, ver arriba.

  B) desde sqrt(|U|^2), que es lo unico que da un sensor real. La fase se
     perdio en la medida y vuelve el objeto con su imagen gemela encima.

EL SIGNO DE Z
-------------
Z es la separacion objeto-sensor, POSITIVA. La ida usa +Z y la vuelta -Z.

Ese signo no se puede comprobar mirando la intensidad: el campo de la vuelta B
es real (sqrt de una intensidad), y para entrada real U(-z) = conj(U(+z)),
luego |U(-z)|^2 = |U(+z)|^2 exactamente. Con el signo cambiado la figura sale
identica. Solo se nota en la fase, y en cuanto se encadene algo no real.
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

#: Sobremuestreo del espectro. La matriz espectral es (s*M, s*N) POR
#: DISTANCIA: en un barrido, subirlo multiplica la memoria de cada paso.
S = 1

#: Puntos del plano de salida (r) y razon entre el paso de salida y el de
#: entrada (mag). Con mag > 1 la ventana de observacion se agranda sin tocar
#: el plano de entrada, que es lo que ningun otro metodo del repo sabe hacer.
#: OJO: r*mag <= s*Kf o la salida trae copias periodicas superpuestas. El
#: script lo comprueba y aborta.
R = 1
MAG = 1.0

#: None calcula Kf con la Ec. (14) del paper a cada z. Un numero lo fija.
KF = None

#: "auto" usa la GPU si hay CuPy con CUDA; "cpu" y "gpu" fuerzan.
DISPOSITIVO = "auto"

#: dtype del campo. complex64 en GPU (ver PRECISION EN GPU); las fases van en
#: float64 en los dos casos pase lo que pase.
DTYPE = None                 # None = complex64 en GPU, complex128 en CPU

#: Reduce la imagen a este lado mayor antes de propagar. En este script viene
#: PUESTO, no en None: ver MEMORIA en el docstring. OJO: al cambiarlo, DELTA
#: deja de ser el de tu sensor y hay que escalarlo; el script lo hace y lo dice.
REDUCIR_A = 1024

#: Relleno de ceros. MPASM no lo necesita -no convoluciona de forma circular-,
#: pero se deja para comparar en igualdad de condiciones con los otros dos
#: scripts. Con 1 se desactiva y la malla espectral baja a la cuarta parte.
PAD = 2

#: True invierte la imagen (barras oscuras sobre fondo claro). Es lo que le da
#: onda de referencia al holograma y hace que la vuelta B reconstruya.
INVERTIR = False

#: Distancias del barrido de foco, como fraccion de Z. None lo desactiva.
#: Menos puntos que en los otros dos scripts: cada paso de MPASM cuesta dos
#: productos matriciales densos, no dos FFT.
BARRIDO = np.linspace(0.4, 1.6, 13)

#: Carpeta donde escribir las figuras, o None para solo mostrarlas.
SALIDA = None

#: Filas por bloque al construir los fasores. Baja si la GPU se queda corta.
FILAS_POR_BLOQUE = 512


# ------------------------------------------------- propagador de referencia
class MatrixDftCPU():
    """TAL CUAL de referencia/zhao2020/MatrixDft.py, autor Wanli Zhao.

    Copiada sin tocar una linea, incluido el estilo. k1 es s, k2 es Kf, k3 es
    r y k4 es mag.
    """

    def __init__(self,In,delta,k1,k2):
        self.In = In
        self.delta = delta
        self.k1 = k1
        self.k2 = k2
        self.N = np.size(In,1)
        self.M = np.size(In,0)
        self.x = np.reshape(np.linspace(-self.N/2,self.N/2-1,self.N)*self.delta,(self.N,1))
        self.y = np.reshape(np.linspace(-self.M/2,self.M/2-1,self.M)*self.delta,(1,self.M))
        self.Lx = k1*delta*self.N
        self.Ly = k1*delta*self.M
        self.fx = np.reshape(np.linspace(-k1*self.N/2,k1*self.N/2-1,k1*self.N)/self.Lx/k2,(1,k1*self.N))
        self.fy = np.reshape(np.linspace(-k1*self.M/2,k1*self.M/2-1,k1*self.M)/self.Ly/k2,(k1*self.M,1))

    def mdft(self):
        Mx = np.exp(-2*np.pi*1j*np.dot(self.x,self.fx))
        My = np.exp(-2*np.pi*1j*np.dot(self.fy,self.y))
        Out = np.dot(np.dot(My,self.In),Mx)/(np.power(self.k1,2)*self.M*self.N)/np.power(self.k2,2)
        return Out

    def midft(self,Fin,k3,k4):
        x1 = np.reshape(np.linspace(-k3*self.N/2,k3*self.N/2-1,k3*self.N)*self.delta*k4,(1,k3*self.N))
        y1 = np.reshape(np.linspace(-k3*self.M/2,k3*self.M/2-1,k3*self.M)*self.delta*k4,(k3*self.M,1))
        Mx1 = np.exp(2*np.pi*1j*np.dot(self.fx.T,x1))
        My1 = np.exp(2*np.pi*1j*np.dot(y1,self.fy.T))
        Out = np.dot(np.dot(My1,Fin),Mx1)
        return Out


def propagacion_original(U0, z, lamb, delta, k1, k2, k3, k4):
    """Transcripcion de LightPropagate.propagate_cpu (Light.py, Wanli Zhao).

    Mismo cuerpo linea por linea; self.lamb, self.delta, self.M, self.N y
    self.k pasan a ser parametros o variables locales. No se importa el
    original porque Light.py hace `import cupy` en la primera linea.
    """
    M, N = U0.shape
    k = 2 * np.pi / lamb
    M1 = k1 * M
    N1 = k1 * N
    LX = delta * N1
    LY = delta * M1
    u = lamb / LX * np.linspace(-N1 / 2, N1 / 2 - 1, N1) / k2
    v = lamb / LY * np.linspace(-M1 / 2, M1 / 2 - 1, M1) / k2
    uu, vv = np.meshgrid(u, v)
    H = np.exp(1j * k * z * np.sqrt(1 - np.power(uu, 2) - np.power(vv, 2)))
    mad = MatrixDftCPU(U0, delta, k1, k2)
    Fu = mad.mdft()
    Uz = mad.midft(Fu * H, k3, k4)
    return Uz


def kf_original(N, delta, lamb, z, k1=1):
    """Transcripcion del k2 de LightPropagate.propagate_pro_CPU.

    Se conservan las dos cosas del original: el `*+` de la Ec. (14) -que en
    Python es un producto- y la ausencia de valor absoluto sobre z, que es lo
    que apaga la compresion al retropropagar.
    """
    N = N * k1
    if z != 0:
        with np.errstate(invalid="ignore"):
            fmax = np.sqrt(np.sqrt(np.power(N * lamb, 4) * +np.power(8 * N * lamb * z, 2))
                           - np.power(N * lamb, 2)) / (4 * np.sqrt(2) * z * lamb)
            k2 = 1 / (2 * delta) / fmax
        if not (k2 > 1):        # `if k2<=1: k2=1` del original, y ademas NaN -> 1
            k2 = 1
    else:
        k2 = 1
    return float(k2)


def kf_paper(N, delta, lamb, z, s=1):
    """Ec. (14) tal como esta impresa: A**2 + B, y sobre |z|.

    Las dos diferencias con kf_original() estan documentadas en el docstring
    del modulo. Es la que usa el propagador de trabajo.
    """
    if z == 0:
        return 1.0
    z = abs(z)
    Ns = s * N
    A = (Ns * lamb) ** 2
    B = (8 * Ns * lamb * z) ** 2
    fmax = np.sqrt(np.sqrt(A**2 + B) - A) / (4 * np.sqrt(2) * z * lamb)
    return float(max(1.0, (1 / (2 * delta)) / fmax))


# ------------------------------------------------------ propagador de trabajo
def fasores(a, b, signo, xp, dtype, filas=FILAS_POR_BLOQUE):
    """exp(signo*2i*pi*outer(a,b)) por bloques de filas, producto en float64.

    Es el nucleo de la DFT matricial. Materializar el producto externo en
    complex128 y convertirlo despues costaria el doble de memoria en el pico y
    no ganaria nada: lo que hay que calcular en doble es el ARGUMENTO, no el
    fasor, que sale acotado a modulo 1.
    """
    a = xp.asarray(a, dtype=np.float64).ravel()
    b = xp.asarray(b, dtype=np.float64).ravel()
    out = xp.empty((a.size, b.size), dtype=dtype)
    for i0 in range(0, a.size, filas):
        i1 = min(i0 + filas, a.size)
        out[i0:i1] = xp.exp(signo * 2j * np.pi
                            * xp.outer(a[i0:i1], b)).astype(dtype)
    return out


def comprobar_ventana(r, mag, s, Kf):
    """r*mag <= s*Kf, o la salida trae copias periodicas del campo superpuestas.

    El espectro se muestrea cada 1/(s*N*delta*Kf), asi que el campo que
    describe se repite en el espacio cada s*N*delta*Kf. La malla de salida
    abarca r*N*delta*mag. Si lo segundo supera a lo primero, la salida no
    falla ni avisa: devuelve otra cosa. N se cancela, de modo que la condicion
    no depende del tamano de la imagen.
    """
    if r * mag > s * Kf * (1 + 1e-12):
        raise SystemExit(
            f"La ventana de salida no cabe en el periodo del espectro: "
            f"r*mag = {r * mag:g} > s*Kf = {s * Kf:g}.\nEl campo saldria con "
            f"copias periodicas superpuestas. Sube S a >= "
            f"{int(np.ceil(r * mag / Kf))}, o baja R o MAG.")


def mpasm_bloques(field, z, lamb, delta, s=1, Kf=None, r=1, mag=1.0,
                  xp=np, dtype=np.complex128, filas=FILAS_POR_BLOQUE):
    """Lo mismo que propagacion_original(), pero cabe en la tarjeta.

    Devuelve (campo, Kf_usado). Tres diferencias, todas de ejecucion y ninguna
    de algoritmo:

    1. Los fasores se construyen por bloques de filas y la transferencia se
       aplica in situ sobre el espectro, sin materializar H entera ni el
       producto Fu*H.

    2. Las fases van en float64 y solo los fasores bajan a dtype.

    3. xp es NumPy o CuPy. El cuerpo es el mismo, de modo que comparar tiempos
       compara dispositivos y no dos implementaciones.

    Y una diferencia que SI es de algoritmo, a proposito: Kf sale de kf_paper,
    o sea con la Ec. (14) como esta impresa y sobre |z|. Con kf_original la
    compresion se apagaria en toda retropropagacion.
    """
    U = xp.asarray(field, dtype=dtype)
    M, N = U.shape
    Ms, Ns = s * M, s * N
    if Kf is None:
        Kf = min(kf_paper(M, delta, lamb, z, s), kf_paper(N, delta, lamb, z, s))
    Kf = float(Kf)
    comprobar_ventana(r, mag, s, Kf)

    # coordenadas en float64: son la malla, no datos de campo
    x = (np.arange(N) - N / 2) * delta
    y = (np.arange(M) - M / 2) * delta
    fx = (np.arange(Ns) - Ns / 2) / (s * delta * N) / Kf
    fy = (np.arange(Ms) - Ms / 2) / (s * delta * M) / Kf

    Mx = fasores(x, fx, -1, xp, dtype, filas)                 # (N, Ns)
    My = fasores(fy, y, -1, xp, dtype, filas)                 # (Ms, M)
    F = ((My @ U @ Mx) / float(s**2 * M * N * Kf**2)).astype(dtype, copy=False)
    del Mx, My

    # H por bloques de filas, in situ. Las evanescentes se anulan: dejarlas
    # decaer haria que al retropropagar crecieran.
    k = 2 * np.pi / lamb
    uu = (lamb * xp.asarray(fx, dtype=np.float64))[None, :]
    for i0 in range(0, Ms, filas):
        i1 = min(i0 + filas, Ms)
        vv = (lamb * xp.asarray(fy[i0:i1], dtype=np.float64))[:, None]
        arg = 1.0 - uu**2 - vv**2
        propagante = arg > 0
        fase = k * z * xp.sqrt(xp.where(propagante, arg, 0.0))
        F[i0:i1] *= xp.where(propagante, xp.exp(1j * fase).astype(dtype), 0)
        del vv, arg, propagante, fase

    x1 = (np.arange(r * N) - r * N / 2) * delta * mag
    y1 = (np.arange(r * M) - r * M / 2) * delta * mag
    Mx1 = fasores(fx, x1, +1, xp, dtype, filas)               # (Ns, rN)
    My1 = fasores(y1, fy, +1, xp, dtype, filas)               # (rM, Ms)
    out = My1 @ F @ Mx1
    del Mx1, My1, F
    return out, Kf


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


def memoria_mpasm(M, N, s, r, dtype):
    """Bytes de pico: el espectro, los cuatro fasores y la salida."""
    it = np.dtype(dtype).itemsize
    Ms, Ns = s * M, s * N
    elementos = (Ms * Ns          # espectro
                 + N * Ns         # Mx
                 + Ms * M         # My
                 + Ns * (r * N)   # Mx1
                 + (r * M) * Ms   # My1
                 + (r * M) * (r * N))   # salida
    return elementos * it


def comprobar_memoria(xp, M, N, s, r, dtype):
    """Aborta con un mensaje util en vez de con un OOM de CUDA."""
    hacen_falta = memoria_mpasm(M, N, s, r, dtype)
    if xp is not cp:
        if hacen_falta > 8 << 30:
            raise SystemExit(
                f"La malla {M}x{N} con s = {s} pide ~"
                f"{hacen_falta / 2**30:.2f} GB en CPU.\nBaja REDUCIR_A, PAD o S.")
        return
    libre, _ = cp.cuda.runtime.memGetInfo()
    if hacen_falta > 0.85 * libre:
        raise SystemExit(
            f"No cabe en la GPU: la malla {M}x{N} con s = {s} en "
            f"{np.dtype(dtype).name} pide ~{hacen_falta / 2**30:.2f} GB y hay "
            f"{libre / 2**30:.2f} GB libres.\nMPASM no admite las mallas que "
            f"FFT-ASM se traga: baja REDUCIR_A (por ejemplo 512), PAD o S, o "
            f"usa DISPOSITIVO = 'cpu'.")


def comprobar_equivalencia(xp, dtype):
    """mpasm_bloques() contra propagacion_original() en una malla pequena.

    Con Kf FIJADO A MANO: es la prueba de que la maquinaria matricial acelerada
    da lo mismo que la de Zhao, no de que los dos elijan el mismo Kf, que es
    otra cosa y se mide aparte (ver el informe de Kf en main).
    """
    rng = np.random.default_rng(0)
    U = rng.random((128, 128)) + 1j * rng.random((128, 128))
    kf = 1.7
    ref = propagacion_original(U, 7.0, LAMB, DELTA, 1, kf, 1, 1.0)
    rap, _ = mpasm_bloques(U, 7.0, LAMB, DELTA, s=1, Kf=kf, r=1, mag=1.0,
                           xp=xp, dtype=dtype)
    rap = a_cpu(rap)
    return float(np.max(np.abs(rap - ref)) / np.max(np.abs(ref)))


# ------------------------------------------------------------------ utilidades
def cargar_objeto(ruta, invertir=False, reducir_a=None, delta=DELTA):
    """Imagen -> (transmitancia en [0,1], delta efectivo).

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
    """mpasm_bloques con relleno de ceros y recorte al tamano original.

    Devuelve (campo recortado, Kf usado). Con R o MAG distintos de sus valores
    por defecto la malla de salida NO es la de entrada y no se recorta: ahi
    recortar tiraria justo lo que se pidio calcular.
    """
    M, N = U.shape
    if pad > 1:
        rel = xp.zeros((M * pad, N * pad), dtype=dtype)
        i, j = (rel.shape[0] - M) // 2, (rel.shape[1] - N) // 2
        rel[i:i + M, j:j + N] = xp.asarray(U, dtype=dtype)
    else:
        rel, i, j = xp.asarray(U, dtype=dtype), 0, 0
    fuera, Kf = mpasm_bloques(rel, z, lamb, delta, s=S, Kf=KF, r=R, mag=MAG,
                              xp=xp, dtype=dtype)
    del rel
    if R != 1 or MAG != 1.0:
        return fuera, Kf
    # copia, no vista: una vista mantendria viva la malla rellena entera
    recorte = fuera[i:i + M, j:j + N].copy()
    del fuera
    return recorte, Kf


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
    comprobar_memoria(xp, M * PAD, N * PAD, S, R, dtype)

    print(f"objeto {RUTA}")
    print(f"  malla {M}x{N}, ventana {N * delta:.3f} x {M * delta:.3f} mm "
          f"(con relleno x{PAD}: {N * PAD * delta:.3f} mm)")
    if delta != DELTA:
        print(f"  imagen reducida a {max(M, N)} px: delta escalado de "
              f"{DELTA * 1e3:.3f} a {delta * 1e3:.3f} um")
    print(f"  lambda {LAMB * 1e6:.1f} nm | delta {delta * 1e3:.3f} um | "
          f"ida +{Z:.3f} mm, vuelta {-Z:+.3f} mm")
    print(f"  s = {S} | r = {R} | mag = {MAG:g} | "
          f"Kf = {'automatico' if KF is None else KF}")
    print(f"  dispositivo {dev.upper()} | dtype {np.dtype(dtype).name} | "
          f"fases en float64")
    print(f"  memoria de pico estimada: "
          f"{memoria_mpasm(M * PAD, N * PAD, S, R, dtype) / 2**30:.2f} GB")
    if dev == "gpu":
        libre, total = cp.cuda.runtime.memGetInfo()
        print(f"  {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, "
              f"{libre / 2**30:.2f} de {total / 2**30:.2f} GB libres")

    eq = comprobar_equivalencia(xp, dtype)
    print(f"\n  mpasm_bloques vs propagacion_original (Zhao, Kf fijo): {eq:.2e}")

    # --- el Kf que elige cada formula ---------------------------------------
    Np = N * PAD
    kf_p_mas = kf_paper(Np, delta, LAMB, +Z, S)
    kf_p_menos = kf_paper(Np, delta, LAMB, -Z, S)
    kf_o_mas = kf_original(Np, delta, LAMB, +Z, S)
    kf_o_menos = kf_original(Np, delta, LAMB, -Z, S)
    print(f"\n  Kf a z = {+Z:+.1f} mm:  paper (A^2+B, |z|) {kf_p_mas:7.3f}   "
          f"original (A^2*B, z) {kf_o_mas:7.3f}")
    print(f"  Kf a z = {-Z:+.1f} mm:  paper (A^2+B, |z|) {kf_p_menos:7.3f}   "
          f"original (A^2*B, z) {kf_o_menos:7.3f}")
    if kf_o_menos == 1.0 and kf_p_menos > 1.0:
        print(f"  <- ahi esta: el original apaga la compresion al "
              f"retropropagar (Kf = 1),\n     o sea que corre la vuelta como "
              f"un FFT-ASM rellenado de ceros. Sin avisar.")
    if kf_p_mas == 1.0:
        print(f"  A esta z el paper tampoco comprime (Kf = 1): MPASM es aqui "
              f"un FFT-ASM caro.\n     Sube Z para verlo trabajar.")

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

    (U_h, kf_ida), t_ida = cronometrar(
        lambda: propagar(U0, +Z, delta, LAMB, PAD, **kw))
    I_h = xp.abs(U_h) ** 2
    (U_a, kf_a), t_a = cronometrar(
        lambda: propagar(U_h, -Z, delta, LAMB, PAD, **kw))
    (U_b, _), t_b = cronometrar(
        lambda: propagar(xp.sqrt(I_h).astype(dtype), -Z, delta, LAMB, PAD, **kw))

    print(f"\n  ida {t_ida * 1e3:.0f} ms | vuelta A {t_a * 1e3:.0f} ms | "
          f"vuelta B {t_b * 1e3:.0f} ms")
    print(f"  Kf usado: {kf_ida:.3f} en la ida, {kf_a:.3f} en la vuelta  "
          f"<- iguales, que es el arreglo")
    corr_a = parecido(xp.abs(U_a), xp.abs(U0))
    if kf_ida <= 1.0:
        print(f"  vuelta A (campo complejo)  correlacion con el objeto = "
              f"{corr_a:+.4f}   <- tiene que ser ~1")
        print(f"    (con Kf = 1 esto ES un FFT-ASM rellenado de ceros, y esa "
              f"propagacion si es reversible)")
    else:
        print(f"  vuelta A (campo complejo)  correlacion con el objeto = "
              f"{corr_a:+.4f}")
        print(f"    OJO: con Kf = {kf_ida:.3f} > 1 esto NO tiene que dar 1. "
              f"Comprimir el intervalo\n    espectral estrecha el rango de "
              f"frecuencias muestreado a "
              f"+-1/(2*delta*Kf), y\n    lo que queda fuera no vuelve. Es la "
              f"misma perdida que la mascara de BL-ASM,\n    por otro camino: "
              f"MPASM cambia DONDE muestrea, no que descarta.")
    print(f"  vuelta B (solo intensidad) correlacion con el objeto = "
          f"{parecido(xp.abs(U_b), xp.abs(U0)):+.4f}")

    # --- barrido de foco -----------------------------------------------------
    zs = curva_a = curva_b = kfs = None
    if BARRIDO is not None:
        zs = np.asarray(BARRIDO, float) * Z
        raiz_I = xp.sqrt(I_h).astype(dtype)
        curva_a, curva_b, kfs = [], [], []
        t0 = time.perf_counter()
        for k, z in enumerate(zs):
            Ua, kf = propagar(U_h, -z, delta, LAMB, PAD, **kw)
            curva_a.append(nitidez(xp.abs(Ua) ** 2)); kfs.append(kf); del Ua
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
        kfs = np.array(kfs)
        print(" " * 48, end="\r")
        print(f"\n  barrido de foco: {len(zs)} distancias de {zs[0]:.1f} a "
              f"{zs[-1]:.1f} mm en {dt:.1f} s "
              f"({dt / (2 * len(zs)) * 1e3:.0f} ms por propagacion)")
        print(f"    vuelta A enfoca en z = {zs[curva_a.argmax()]:.2f} mm")
        print(f"    vuelta B enfoca en z = {zs[curva_b.argmax()]:.2f} mm")
        print(f"    esperado:            z = {Z:.2f} mm")
        print(f"    Kf recalculado a cada z: de {kfs.min():.3f} a {kfs.max():.3f}")
        if kfs.max() - kfs.min() > 1e-9:
            print(f"    OJO: Kf sube con z, y a mas Kf mas estrecho el rango "
                  f"muestreado, o sea\n    mas borrosa la reconstruccion se "
                  f"enfoque o no. Esa caida se suma a la del\n    desenfoque y "
                  f"corre el pico hacia distancias cortas. El panel de abajo de"
                  f"\n    foco_mpasm.png deja ver el sesgo; no lo corrige.")

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
    fig.suptitle(f"MPASM, s = {S}, Kf = {kf_ida:.3f} a z = {Z:g} mm",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    figuras = [("ida_y_vuelta_mpasm.png", fig)]

    if zs is not None:
        # Dos paneles y no dos escalas en el mismo eje: la nitidez va
        # normalizada a 1 y Kf es un numero suyo, y superponerlas con un twinx
        # deja al lector decidiendo que curva mira que eje.
        fig2, (a2, a3) = plt.subplots(
            2, 1, figsize=(7.2, 5.4), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
        a2.plot(zs, curva_a / curva_a.max(), "o-", ms=3, label="A: campo complejo")
        a2.plot(zs, curva_b / curva_b.max(), "s-", ms=3, label="B: solo intensidad")
        a2.axvline(Z, color="k", ls="--", lw=1, label=f"z real = {Z:g} mm")
        a2.set_ylabel("nitidez (energia del gradiente,\nnormalizada)")
        a2.set_title(f"Donde enfoca la retropropagacion con MPASM  ({dev.upper()})",
                     fontsize=11)
        a2.legend(fontsize=9)

        # Kf a cada z: es lo que MPASM hace y los otros dos metodos no
        a3.plot(zs, kfs, color="0.45", lw=1.4)
        a3.axvline(Z, color="k", ls="--", lw=1)
        a3.axhline(1.0, color="0.75", lw=0.8)
        a3.set_xlabel("distancia de reconstruccion [mm]")
        a3.set_ylabel("Kf elegido\na cada z", fontsize=9)
        fig2.tight_layout()
        figuras.append(("foco_mpasm.png", fig2))

    if SALIDA is not None:
        destino = pathlib.Path(SALIDA)
        destino.mkdir(parents=True, exist_ok=True)
        for nombre, f in figuras:
            f.savefig(destino / nombre, dpi=150, bbox_inches="tight")
            print(f"  -> {destino / nombre}")
    plt.show()


if __name__ == "__main__":
    main()
