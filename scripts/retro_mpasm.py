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

Por eso PAD va a 1 en este script y a 2 en sus dos hermanos. No es una
preferencia: es que MPASM no admite las mallas que FFT-ASM se traga sin
pestanear, y ese es justo el precio que compra el control del muestreo.

Con REDUCIR_A = None y PAD = 1 la malla es 3000x4000 y el pico son 1.10 GB en
complex128, que si cabe en RAM. Medido en CPU: 10.8 s por propagacion, o sea
unos 5 min el barrido de 13 distancias por dos vueltas.

UNIDADES: milimetros para todo. Da igual cual sea mientras sea la MISMA en
lambda, delta y z.

    633 nm -> 633e-6 mm     3.45 um -> 3.45e-3 mm     20 mm -> 20.0

EL UMBRAL DE Kf: CUANDO EMPIEZA A COMPRIMIR
-------------------------------------------
Kf = max(1, (1/(2*delta)) / fmax), con fmax el de la Ec. (14). O sea que solo
comprime cuando al muestreo le SOBRA Nyquist. A z corto pasa lo contrario: el
campo pide mas banda de la que la malla puede dar, el cociente sale menor que
1 y el max() lo sube a 1 sin avisar. Ahi MPASM es un FFT-ASM caro.

Del limite de z grande de la Ec. (14) sale la forma cerrada

    Kf ~= sqrt(z*lamb)/(delta*sqrt(s*N))   =>   Kf > 1  <=>  z > delta^2*s*N/lamb

y ese umbral es lo que hay que mirar ANTES de creerse un barrido de Kf:

    REDUCIR_A=1024, PAD=2, S=1 -> delta 13.48 um, N 2048 -> z* = 588 mm
    REDUCIR_A=None, PAD=1, S=1 -> delta  3.45 um, N 4000 -> z* =  75 mm

Con la primera config el barrido 40-160 mm cae ENTERO por debajo del umbral y
el panel de Kf de foco_mpasm.png sale plano en 1: no es un fallo del codigo,
es que ahi no hay nada que comprimir. Con la segunda, Kf recorre 1.000 a 1.460
y ademas se ve soltarse el max() sobre los 75 mm.

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
Z = 150.0

# ════════════════════════════════════════════════════════════════════════════
#  Y esto solo si hace falta
# ════════════════════════════════════════════════════════════════════════════

#: Sobremuestreo del espectro. La matriz espectral es (s*M, s*N) POR
#: DISTANCIA: en un barrido, subirlo multiplica la memoria de cada paso.
#: OJO, va al reves de lo que parece: s entra bajo raiz en el DENOMINADOR de
#: Kf, asi que subirlo ALEJA el umbral de compresion (ver EL UMBRAL DE Kf).
S = 2

#: Puntos del plano de salida (r) y razon entre el paso de salida y el de
#: entrada (mag). Con mag > 1 la ventana de observacion se agranda sin tocar
#: el plano de entrada, que es lo que ningun otro metodo del repo sabe hacer.
#: OJO: r*mag <= s*Kf o la salida trae copias periodicas superpuestas. El
#: script lo comprueba y aborta.
R = 1
MAG = 1.0

#: None calcula Kf con la Ec. (14) del paper a cada z. Un numero lo fija.
#: Fijarlo por encima del que pide la Ec. (14) no mejora nada: estrecha el
#: rango muestreado a +-1/(2*delta*Kf) sin que hubiera nada que comprimir.
KF = None

#: "auto" usa la GPU si hay CuPy con CUDA; "cpu" y "gpu" fuerzan.
DISPOSITIVO = "auto"

#: dtype del campo. complex64 en GPU (ver PRECISION EN GPU); las fases van en
#: float64 en los dos casos pase lo que pase.
DTYPE = None                 # None = complex64 en GPU, complex128 en CPU

#: Reduce la imagen a este lado mayor antes de propagar. OJO: al cambiarlo,
#: DELTA deja de ser el de tu sensor y hay que escalarlo; el script lo hace y
#: lo dice. Va en None -resolucion nativa- porque reducir SUBE el umbral de
#: compresion: delta crece como delta^2 y N solo baja como N (ver EL UMBRAL DE
#: Kf). A 1024 px el umbral se iba a 588 mm y Kf salia 1 en TODO el barrido.
REDUCIR_A = None

#: Relleno de ceros. MPASM no lo necesita -no convoluciona de forma circular-,
#: y aqui va desactivado por eso y porque duplicar N duplica el umbral de
#: compresion. Con 2 se compara en igualdad de condiciones con los otros dos
#: scripts, a costa de multiplicar por cuatro la malla espectral.
PAD = 1

#: True invierte la imagen (barras oscuras sobre fondo claro). Es lo que le da
#: onda de referencia al holograma y hace que la vuelta B reconstruya.
INVERTIR = False

#: Distancias del barrido de foco, como fraccion de Z. None lo desactiva.
#: Menos puntos que en los otros dos scripts: cada paso de MPASM cuesta dos
#: productos matriciales densos, no dos FFT.
BARRIDO = np.linspace(0.4, 1.6, 13)

#: True escribe la pila de foco entera en
#: resultados/reconstruccion/<objeto>/mpasm/<A|B>/, un PNG por distancia y
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
    if kf_o_menos == 1.0 and kf_o_mas > 1.0 and kf_p_menos > 1.0:
        print(f"  <- ahi esta: el original apaga la compresion al "
              f"retropropagar (Kf = 1),\n     o sea que corre la vuelta como "
              f"un FFT-ASM rellenado de ceros. Sin avisar.")
    elif kf_o_menos == 1.0 and kf_p_menos > 1.0:
        # Los dos sentidos a 1: aqui NO manda el signo, manda la errata `*`.
        # Atribuirlo al signo con original(+z) tambien a 1 seria falso, y es
        # justo la clase de frase que en una defensa se comprueba evaluando
        # la formula dos veces. El efecto del signo aparece mas lejos.
        print(f"  <- el original no comprime en NINGUN sentido a esta z: aqui "
              f"no manda la\n     falta de abs(z), manda la errata `*` de la "
              f"Ec. (14), que agranda fmax.\n     El efecto del signo se ve a "
              f"z mas larga, donde original(+z) > 1.")
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
        if GUARDAR_BARRIDO:
            objeto = pathlib.Path(RUTA).stem
            dir_a = carpeta_barrido(objeto, "mpasm", "A")
            dir_b = carpeta_barrido(objeto, "mpasm", "B")
        curva_a, curva_b, kfs = [], [], []
        t0 = time.perf_counter()
        for k, z in enumerate(zs):
            Ua, kf = propagar(U_h, -z, delta, LAMB, PAD, **kw)
            Ia = xp.abs(Ua) ** 2
            curva_a.append(nitidez(Ia)); kfs.append(kf)
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
        kfs = np.array(kfs)
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
        print(f"    Kf recalculado a cada z: de {kfs.min():.3f} a {kfs.max():.3f}")
        if kfs.max() - kfs.min() > 1e-9:
            print(f"    Kf sube con z, y a mas Kf mas estrecho el rango "
                  f"muestreado: la reconstruccion se emborrona al alejarse,")
            print(f"    enfoque o no. Eso NO corre el pico: medido a Z = 30, 50, "
                  f"70 y 90 mm sobre el rango completo, el error es 0.0 % en las")
            print(f"    cuatro. El sesgo hacia distancias cortas que se avisaba "
                  f"aqui lo producia nitidez() al normalizar por el maximo, no Kf.")

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
        fig3.suptitle(f"Fase de la ida y las dos vueltas -- MPASM, "
                      f"Z = {Z:g} mm, Kf = {kf_ida:.3f}, delta = {delta * 1e3:.2f} um",
                      fontsize=12)
        figuras.append(("fases_mpasm.png", fig3))

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
