"""Ida y vuelta con MPASM: objeto -> +Z -> sensor -> -Z -> reconstruccion.

Matrix-based Angular Spectrum Method. Es el hermano de retro_fft_angular.py y
retro_blas.py: mismo esqueleto, otro propagador.

QUE HACE MPASM QUE LOS OTROS DOS NO

El espectro angular por FFT muestrea el plano de frecuencias con el paso que
le impone la malla, y ahi se acaba lo que puede hacer. MPASM lo muestrea con
una DFT matricial, o sea eligiendo el paso: puede COMPRIMIR el espectro por un
factor Kf para meter en la malla frecuencias que la FFT aliaria, y puede sacar
el resultado en una malla de salida distinta de la de entrada (R puntos con
paso MAG veces el de entrada). Agrandar la ventana de observacion sin tocar el
plano de entrada es lo que ningun otro metodo del repo sabe hacer.

El precio es la memoria: la matriz espectral es (S*M, S*N) POR DISTANCIA. Con
BenchmarkTarget (4000x3000) y S = 2 son 2.38 GB por paso del barrido, asi que
la RUTA por defecto es el USAF de 512, igual que en los otros dos scripts.

DOS COSAS QUE SE COMPRUEBAN AL ARRANCAR

  - r*mag <= s*Kf, o la salida trae copias periodicas del campo superpuestas.
    No falla ni avisa por su cuenta: devuelve otra cosa. comprobar_ventana()
    aborta con el S minimo que haria falta.

  - mpasm_bloques() contra propagacion_original(), que es la implementacion de
    Zhao copiada tal cual y no acepta CuPy. Es la prueba de que la maquinaria
    matricial acelerada da lo mismo que la del paper, con Kf fijado a mano: que
    los dos ELIJAN el mismo Kf es otra cosa distinta.

LAS FIGURAS, en tres filas de tres:

    |U|^2           sin gamma. Una gamma de 0.5 sobre una intensidad ES la
                    amplitud, porque (|U|^2/max)^0.5 = |U|/sqrt(max).
    fase            con PESAR_FASE, la opacidad va por amplitud.
    campo complejo  coloreado de dominio: tono = fase, brillo = amplitud. Las
                    dos filas de arriba parten el campo en mitades; esta las
                    junta.

BARRIDO con dos metricas independientes: nitidez (donde enfoca) y RMS de fase.

LO QUE ESTE SCRIPT NO ES: una reconstruccion holografica. La vuelta parte del
CAMPO COMPLEJO en el plano del sensor (campo_sensor), no de lo que un sensor
mide. Un sensor entrega |U|^2 y tira la fase; retropropagar sqrt(|U|^2) da
correlacion 0.53 con el objeto en vez de 1.00, y ahi aparece la imagen gemela.
Por eso la vuelta sale exacta: no reconstruye un holograma, deshace una
propagacion. Para lo primero esta scripts/retro_holograma.py.

UNIDADES: milimetros para todo.  633 nm -> 633e-6    3.45 um -> 3.45e-3
"""
import pathlib

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import cupy as cp
except Exception:                      # sin CuPy, sin CUDA, o CuPy roto
    cp = None


# ════════════════════════════════════════════════════════════════════════════
#  PARAMETROS
# ════════════════════════════════════════════════════════════════════════════

#: Imagen del OBJETO (transmitancia), no un holograma.
#: OJO con el tamano: la matriz espectral es (S*M, S*N) POR DISTANCIA. Con
#: BenchmarkTarget (4000x3000) y S = 2 son 2.38 GB por paso del barrido.
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\campos\entrada.png"

#: Longitud de onda [mm].
LAMB = 633e-6

#: Paso de pixel [mm].
DELTA = 3.45e-3

#: Distancia objeto <-> sensor [mm], POSITIVA: la ida usa +Z y la vuelta -Z.
Z = 360.0

#: Sobremuestreo del espectro. La matriz espectral es (S*M, S*N) POR DISTANCIA:
#: en un barrido, subirlo multiplica la memoria de cada paso. OJO, va al reves
#: de lo que parece: s entra bajo raiz en el DENOMINADOR de Kf, asi que subirlo
#: ALEJA el umbral de compresion.
S = 12

#: Puntos del plano de salida (R) y razon entre el paso de salida y el de
#: entrada (MAG). Con MAG > 1 la ventana de observacion se agranda sin tocar el
#: plano de entrada, que es lo que ningun otro metodo del repo sabe hacer.
#: OJO: R*MAG <= S*Kf o la salida trae copias periodicas superpuestas. El
#: script lo comprueba con comprobar_ventana() y aborta.
R = 1
MAG = 1.0

#: None calcula Kf con la Ec. (14) del paper a cada z. Un numero lo fija.
#: Fijarlo por encima del que pide la Ec. (14) no mejora nada: estrecha el
#: rango muestreado a +-1/(2*delta*Kf) sin que hubiera nada que comprimir.
KF = None

#: QUE REPRESENTA LA IMAGEN.
#:
#:   "intensidad"  transmitancia en INTENSIDAD. El campo es su raiz: A = sqrt(img).
#:                 Es lo que hay que poner si la imagen es una foto de lo que
#:                 pasa por la muestra, porque un sensor registra |U|^2.
#:   "amplitud"    la imagen ES la amplitud del campo: A = img.
#:
#: Sobre un target BINARIO da igual -sqrt(0)=0 y sqrt(1)=1-, pero en escala de
#: grises los dos campos difieren hasta 0.43 con PHI_MAX = pi.
ENTRADA = "intensidad"

#: PROFUNDIDAD DE FASE del objeto, en radianes:  U0 = A * exp(i * PHI_MAX * A)
#:
#: 0 es un objeto de amplitud puro (fase 0 en todo el plano), que es lo que
#: habia antes. pi es el maximo que cabe SIN ENVOLVER: np.angle() devuelve
#: (-pi, pi], asi que por encima la fase da la vuelta y el panel enseña saltos
#: de 2*pi que son del display, no del objeto. Medido: con PHI_MAX = 1.2*pi la
#: fase pedida llega a 3.770 rad y angle() devuelve 3.138.
#:
#: OJO: SOBRE UN OBJETO BINARIO ESTO NO HACE NADA. Con A en {0, 1} la fase
#: queda en {0, PHI_MAX}, o sea el campo entero por una constante de modulo 1:
#: una fase GLOBAL, que se cancela en |U|^2 y conmuta con la propagacion. El
#: holograma sale identico para PHI_MAX = 0 que para pi. usaf_like es binario;
#: el modelo necesita escala de grises. El script lo comprueba y avisa.
PHI_MAX = 0.0

#: "auto" usa la GPU si hay CuPy con CUDA; "cpu" y "gpu" fuerzan.
DISPOSITIVO = "auto"

#: dtype del campo. complex64 en GPU: en una GeForce la doble precision va a
#: 1/32 del ritmo de la simple. Las fases se calculan en float64 siempre.
DTYPE = None                 # None = complex64 en GPU, complex128 en CPU

#: Filas por bloque al construir los fasores. Baja si la GPU se queda corta.
FILAS_POR_BLOQUE = 512

#: True pinta la fase con la OPACIDAD proporcional a la amplitud; False la
#: pinta cruda, como np.angle() a secas.
#:
#: Con True, donde no hay campo no se pinta nada. Con False ves lo que hay de
#: verdad en el array, incluido el fondo. Cual quieres depende del caso: en una
#: ida y vuelta desde el campo complejo el fondo es cero NUMERICO -amplitudes
#: de 1e-16, o sea 1e-16 veces el maximo- y np.angle() de esos numeros da ruido
#: uniforme en (-pi, pi] que se come la figura. Con un holograma real o un
#: objeto de fase, ese fondo tiene amplitud de verdad y su fase SIGNIFICA algo.
#:
#: Es una decision de dibujo, no un filtro sobre los datos: el array es el
#: mismo. En la duda, mira las dos.
PESAR_FASE = True

#: Distancias del barrido de foco, como fraccion de Z. None lo desactiva.
#: A cada z se retropropaga y se miden las dos metricas: nitidez (donde
#: enfoca) y RMS de fase (cuanto se parece la fase a la del objeto).
BARRIDO = np.linspace(0.4, 1.6, 25)

#: Carpeta donde guardar las figuras, o None para solo mostrarlas.
SALIDA = None

#: Nombre corto del propagador, para la ruta de salida del holograma. Es el
#: mismo vocabulario que usa METODOS en CamposT: fft, blas, mpasm.
METODO = "mpasm"


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

# ════════════════════════════════════════════════════════════════════════════
#  METRICAS  --  copias literales de las de retro_blas.py y retro_mpasm.py,
#  que tests/test_nitidez_foco.py, test_pico_de_foco.py y test_fases_retro.py
#  comprueban que no diverjan. No las edites solo aqui.
# ════════════════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════════════════
#  FIGURA
# ════════════════════════════════════════════════════════════════════════════

def pinta_intensidad(ax, U, titulo):
    """|U|^2 dividido por su maximo. Sin gamma."""
    I = np.abs(U) ** 2
    m = I.max()
    ax.imshow(I / m if m > 0 else I, cmap="gray", vmin=0, vmax=1)
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def pinta_fase(ax, U):
    """angle(U), con la opacidad pesada por la amplitud si PESAR_FASE.

    El fondo del panel es GRIS y no blanco: en twilight el +-pi ES blanco, asi
    que sobre blanco no se distingue "la fase vale pi" de "aqui no hay campo".
    """
    if not PESAR_FASE:
        alfa = None
    else:
        alfa = np.abs(U).astype(float)
        p = np.percentile(alfa, 99.5)
        alfa = np.clip(alfa / p, 0, 1) ** 0.45 if p > 0 else np.zeros_like(alfa)
    im = ax.imshow(np.angle(U), cmap="twilight", vmin=-np.pi, vmax=np.pi,
                   alpha=alfa, interpolation="nearest")
    ax.set_facecolor("0.62")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def pinta_complejo(ax, U):
    """El campo complejo ENTERO en un panel: tono = fase, brillo = amplitud.

    Es coloreado de dominio. Los paneles de arriba parten el campo en dos
    mitades -|U|^2 pierde la fase, angle(U) pierde la amplitud- y ninguna de
    las dos sola dice donde esta el campo Y cuanto vale su fase a la vez. Aqui
    van juntas: el color dice la fase, la luminosidad dice cuanta amplitud hay
    detras de esa fase.

    Negro = no hay campo. Por eso este panel no necesita PESAR_FASE: el peso no
    es una decision de dibujo aqui, es la mitad de la informacion.

    El brillo va a la 0.45 y no lineal, por la misma razon que en pinta_fase:
    con un objeto que es 91% fondo, lineal deja el panel casi todo negro.
    """
    fase, mag = np.angle(U), np.abs(U)
    p = np.percentile(mag, 99.5)
    v = np.clip(mag / p, 0, 1) ** 0.45 if p > 0 else np.zeros_like(mag)
    hsv = np.stack([(fase + np.pi) / (2 * np.pi), np.ones_like(v), v], axis=-1)
    ax.imshow(mcolors.hsv_to_rgb(hsv), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

def objeto(ruta, entrada, phi_max):
    """Imagen -> campo complejo de entrada. -> (U0, imagen cruda).

        A  = sqrt(img)  si entrada == "intensidad",  img si "amplitud"
        U0 = A * exp(i * phi_max * A)

    Un objeto que absorbe Y retarda, con la fase acoplada a la amplitud. Los
    dos extremos son phi_max = 0 (amplitud pura) y A constante (fase pura).
    """
    img = np.asarray(Image.open(ruta).convert("L"), dtype=np.float64) / 255.0
    if entrada not in ("intensidad", "amplitud"):
        raise SystemExit(f'ENTRADA = "{entrada}" no existe. '
                         'Pon "intensidad" o "amplitud".')
    A = np.sqrt(img) if entrada == "intensidad" else img
    return A * np.exp(1j * phi_max * A), img


def guardar_holograma(campo, z, parametros):
    """campo_sensor -> tres archivos, y una linea por archivo en consola.

        .png   |campo|^2 normalizado por su maximo, SIN GAMMA. Es lo que mide
               un sensor: al retropropagarlo sale el objeto CON su imagen
               gemela encima.
        .npy   el campo complejo tal cual, en el dtype en que se calculo. Es lo
               que habia ANTES de medirlo, con su fase. Retropropagarlo deshace
               la ida SOLO si el propagador que la hizo es la identidad: exacto
               con FFT-ASM, filtrado con BL-ASM -que descarta banda a proposito-,
               y truncado con MPASM, que tira la energia que sale de su ventana.
        .txt   con que se hizo.

    Los dos primeros no son redundantes: el PNG es lo que un sensor te habria
    dado, el .npy es lo que habia antes de que lo midiera.
    scripts/retro_holograma.py distingue los dos por la extension.

    SIN GAMMA A PROPOSITO. Un PNG de intensidad guardado como I^0.6 -lo que
    hace CamposT/pipeline.py por defecto- se lee luego como I^0.3 al tomarle la
    raiz, y eso no es solo contraste feo: MUEVE LA DISTANCIA a la que enfoca la
    reconstruccion, que es justo lo que un barrido intenta medir.

    EL .txt NO ES ADORNO. El nombre del archivo lleva la z y nada mas, asi que
    dos corridas con la misma z y distinta lambda escriben el mismo nombre y la
    segunda pisa a la primera. El .txt es lo que impide que eso sea un error
    mudo: dice con que parametros se hizo el archivo que hay ahi.

    ESTA FUNCION ESTA DUPLICADA en los tres scripts retro_* A PROPOSITO. Podria
    importarse de CamposT.pipeline.guardar(), que hace casi esto mismo, pero
    estos tres no importan el paquete: son el contraste INDEPENDIENTE contra el.
    Si dependieran de el, coincidir con el dejaria de significar algo. Es la
    misma duplicacion deliberada que ya tienen nitidez(), pico_de_foco(),
    a_cpu(), objeto() y los tres pinta_*().
    """
    A = a_cpu(campo)

    # Bajo la raiz del repo, se lance el script desde donde se lance: una ruta
    # relativa lo dejaria en el directorio de invocacion.
    destino = (pathlib.Path(__file__).resolve().parent.parent / "resultados"
               / "hologramas" / pathlib.Path(RUTA).stem / METODO)
    destino.mkdir(parents=True, exist_ok=True)

    # z{:08.3f}: el mismo formato que nombre_png() en
    # CamposT/retropropagacion.py, escrito a mano porque aqui no se importa el
    # paquete. Tres decimales para que dos z distintas no escriban el mismo
    # archivo, y ancho fijo para que el orden alfabetico siga al de la z.
    #
    # OJO: las extensiones se PEGAN con f-string, NO con Path.with_suffix().
    # Para pathlib "z0010.000" ya tiene sufijo -".000"- y with_suffix(".png")
    # lo SUSTITUIRIA en vez de anadirlo, dejando z0010.png. Eso se cargaria los
    # tres decimales enteros: z = 10.0 y z = 10.5 escribirian en el MISMO
    # archivo, en silencio, que es exactamente lo que este formato existe para
    # impedir.
    base = destino / f"z{z:08.3f}"

    I = np.abs(A) ** 2
    m = I.max()
    # Un campo identicamente nulo daria 0/0: NaN por todo el array y un PNG de
    # basura, sin error y sin aviso. Un negro es un resultado legitimo.
    I = I / m if m > 0 else np.zeros_like(I)
    Image.fromarray((I * 255).astype(np.uint8)).save(f"{base}.png")

    np.save(f"{base}.npy", A)

    with open(f"{base}.txt", "w", encoding="utf-8") as f:
        for clave, valor in parametros.items():
            f.write(f"{clave} = {valor}\n")

    print("holograma guardado:")
    for ext in (".png", ".npy", ".txt"):
        print(f"  -> {base}{ext}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

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

    U0_obj, img = objeto(RUTA, ENTRADA, PHI_MAX)
    M, N = img.shape

    if PHI_MAX != 0 and len(np.unique(img)) <= 2:
        print(f"\n  AVISO: el objeto es BINARIO ({len(np.unique(img))} niveles) "
              f"y PHI_MAX no hace nada.\n  Con A en {{0, 1}} la fase queda en "
              f"{{0, PHI_MAX}}: el campo entero por una constante de\n  modulo "
              f"1, o sea una fase GLOBAL. Se cancela en |U|^2 y conmuta con la "
              f"propagacion.\n  Usa una imagen en escala de grises.")
    comprobar_memoria(xp, M, N, S, R, dtype)

    kw = dict(s=S, Kf=KF, r=R, mag=MAG, xp=xp, dtype=dtype)

    # mpasm_bloques devuelve (campo, Kf usado). Kf es el factor de compresion
    # del espectro que la Ec. (14) del paper pide a esa z: con Kf = 1 no hay
    # compresion y MPASM es un espectro angular matricial; por encima, esta
    # comprimiendo. Va impreso porque es lo que distingue a este metodo.
    # OJO CON EL NOMBRE. campo_sensor es el CAMPO COMPLEJO en el plano del
    # sensor: |h| y su fase. NO es un holograma. Un holograma es |h|^2, un mapa
    # real y no negativo, y en este script no existe como variable: solo se
    # calcula al dibujar, dentro de pinta_intensidad().
    #
    # La distincion no es de vocabulario. Retropropagar campo_sensor deshace la
    # propagacion -exp(+i*phi)*exp(-i*phi) = 1- y devuelve el objeto exacto.
    # Retropropagar sqrt(|h|^2), que es lo unico que entrega un sensor, da
    # correlacion 0.53 con el objeto en vez de 1.00: ahi aparece la imagen
    # gemela. Los dos campos de partida difieren en 1.707 sobre un modulo
    # maximo de 1.300, y la diferencia es entera de fase.
    #
    # O sea: este script NO reconstruye un holograma, deshace una propagacion.
    # Para lo primero esta scripts/retro_holograma.py.
    campo_sensor, kf_ida = mpasm_bloques(U0_obj, +Z, LAMB, DELTA, **kw)
    retropropagado, kf_vuelta = mpasm_bloques(campo_sensor, -Z, LAMB, DELTA, **kw)

    print(f"objeto {RUTA}")
    print(f"  malla {M}x{N} | lambda {LAMB * 1e6:.1f} nm | "
          f"delta {DELTA * 1e3:.3f} um | ida +{Z:g} mm, vuelta {-Z:+g} mm")
    print(f"  dispositivo {dev.upper()} | dtype {np.dtype(dtype).name} | "
          f"fases en float64")
    if dev == "gpu":
        libre, total = cp.cuda.runtime.memGetInfo()
        print(f"  {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, "
              f"{libre / 2**30:.2f} de {total / 2**30:.2f} GB libres")
    print(f"  S = {S} | R = {R} | MAG = {MAG} | matriz espectral "
          f"{S * M}x{S * N} por distancia "
          f"({memoria_mpasm(M, N, S, R, dtype) / 2**30:.2f} GB)")
    print(f"  Kf: {kf_ida:.4f} en la ida, {kf_vuelta:.4f} en la vuelta"
          + ("   <- 1.0 = sin compresion" if max(kf_ida, kf_vuelta) <= 1.0
             else "   <- comprimiendo el espectro"))

    # r*mag <= s*Kf, o la salida trae copias periodicas superpuestas y no avisa
    comprobar_ventana(R, MAG, S, kf_ida)
    comprobar_ventana(R, MAG, S, kf_vuelta)

    # propagacion_original() es la referencia de Zhao y no acepta CuPy: corre
    # siempre en CPU. Contrastarla con mpasm_bloques() es la prueba de que la
    # maquinaria matricial acelerada da lo mismo que la del paper.
    eq = comprobar_equivalencia(xp, dtype)
    print(f"  mpasm_bloques vs propagacion_original (referencia): {eq:.2e}")

    err = (np.abs(a_cpu(retropropagado) - U0_obj).max()
           / np.abs(U0_obj).max())
    print(f"  ida y vuelta del campo complejo: error max relativo = {err:.2e}")

    # Se guarda aqui y no justo tras la ida para que la consola se lea en
    # orden: primero con que se hizo, despues que se escribio. Ademas deja el
    # Kf impreso justo encima del .txt que lo anota.
    #
    # kf_ida no es un ajuste, es un RESULTADO de la ida: el coeficiente de
    # compresion frecuencial que MPASM calculo. KF es lo que se le PIDIO -None
    # significa "calculalo tu"-. Van los dos porque no son lo mismo, y sin
    # kf_ida dos hologramas con los mismos parametros de entrada pueden diferir
    # sin que se pueda saber por que.
    guardar_holograma(campo_sensor, Z, {
        "objeto": RUTA,
        "propagador": "MPASM (espectro angular por producto matricial)",
        "lambda [mm]": LAMB,
        "delta [mm]": DELTA,
        "Z [mm]": Z,
        "malla": f"{M}x{N}",
        "ENTRADA": ENTRADA,
        "PHI_MAX": PHI_MAX,
        "dispositivo": dev,
        "dtype": np.dtype(dtype).name,
        "S": S,
        "R": R,
        "MAG": MAG,
        "KF": KF,
        "kf_ida": f"{kf_ida:.6f}",
    })

    # ---- barrido de foco ----------------------------------------------------
    zs = nit = rms = None
    if BARRIDO is not None:
        U0 = xp.asarray(U0_obj, dtype=dtype)
        mascara = np.abs(U0_obj) > 0.5 * np.abs(U0_obj).max()
        zs = np.asarray(BARRIDO, float) * Z
        nit, rms = [], []
        for k, z in enumerate(zs):
            U, _ = mpasm_bloques(campo_sensor, -z, LAMB, DELTA, **kw)
            nit.append(nitidez(xp.abs(U) ** 2))
            rms.append(rms_fase(U, U0, mascara))
            del U
            liberar(xp)
            print(f"    barrido {k + 1}/{len(zs)}   z = {z:7.2f} mm", end="\r")
        nit, rms = np.array(nit), np.array(rms)
        print(" " * 48, end="\r")

        z_nit = pico_de_foco(zs, nit)
        z_rms = pico_de_foco(zs, -rms)        # el RMS MINIMIZA en el foco
        print(f"\nbarrido de foco: {len(zs)} distancias de {zs[0]:.1f} a "
              f"{zs[-1]:.1f} mm")
        print(f"    nitidez maxima     en z = {z_nit:.2f} mm" if z_nit is not None
              else "    nitidez: el maximo cae en un extremo del barrido, que "
                   "NO acota el foco. Ensancha BARRIDO.")
        print(f"    RMS de fase minimo en z = {z_rms:.2f} mm" if z_rms is not None
              else "    RMS: el minimo cae en un extremo del barrido.")
        print(f"    esperado:            z = {Z:.2f} mm")
        print(f"    RMS en el foco: "
              f"{np.degrees(rms[np.argmin(np.abs(zs - Z))]):.2f} deg"
              f"   (sin informacion de fase saldrian "
              f"{np.degrees(np.pi / np.sqrt(3)):.1f} deg)")

    # ---- figuras ------------------------------------------------------------
    # a_cpu antes de pintar: matplotlib no dibuja arrays de CuPy
    campos = [("objeto (entrada)", U0_obj),
              (f"holograma $|U|^2$ a +{Z:g} mm", a_cpu(campo_sensor)),
              (f"retropropagado a {-Z:+g} mm", a_cpu(retropropagado))]

    fig, ax = plt.subplots(3, 3, figsize=(13.5, 12.6))
    for k, (titulo, U) in enumerate(campos):
        pinta_intensidad(ax[0, k], U, titulo)
        im = pinta_fase(ax[1, k], U)
        pinta_complejo(ax[2, k], U)
    for fila, etiqueta in ((0, "$|U|^2$"), (1, "fase"), (2, "campo complejo")):
        ax[fila, 0].text(-0.04, 0.5, etiqueta, transform=ax[fila, 0].transAxes,
                         rotation=90, va="center", ha="right", fontsize=11)

    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    etiq = ["-pi", "-pi/2", "0", "pi/2", "pi"]
    cb = fig.colorbar(im, ax=ax[1, :], fraction=0.030, pad=0.015, ticks=ticks)
    cb.ax.set_yticklabels(etiq)
    cb.set_label("fase [rad]" + (" | opacidad = amplitud" if PESAR_FASE else ""),
                 fontsize=9)

    # la barra de la fila compleja es HSV porque es el tono que se pinta ahi.
    # No se puede reutilizar la de arriba: twilight y hsv no son el mismo mapa.
    sm = plt.cm.ScalarMappable(cmap="hsv",
                               norm=mcolors.Normalize(-np.pi, np.pi))
    cb2 = fig.colorbar(sm, ax=ax[2, :], fraction=0.030, pad=0.015, ticks=ticks)
    cb2.ax.set_yticklabels(etiq)
    cb2.set_label("fase = tono | amplitud = brillo", fontsize=9)
    fig.suptitle(f"MPASM, ida y vuelta -- Z = {Z:g} mm, S = {S}, "
                 f"delta = {DELTA * 1e3:.2f} um, lambda = {LAMB * 1e6:.0f} nm",
                 fontsize=12)

    figuras = [("ida_y_vuelta.png", fig)]

    if zs is not None:
        fig2, a2 = plt.subplots(figsize=(7.6, 4.4))
        l_nit, = a2.plot(zs, nit / nit.max(), "o-", ms=3, color="C0",
                         label="nitidez (energia del gradiente)")
        a2.set_xlabel("distancia de reconstruccion [mm]")
        a2.set_ylabel("nitidez normalizada", color="C0")
        a2.tick_params(axis="y", labelcolor="C0")
        a3 = a2.twinx()
        l_rms, = a3.plot(zs, np.degrees(rms), "s-", ms=3, color="C3",
                         label="RMS de fase")
        a3.set_ylabel("RMS de fase [deg]", color="C3")
        a3.tick_params(axis="y", labelcolor="C3")
        l_z = a2.axvline(Z, color="k", ls="--", lw=1, label=f"z real = {Z:g} mm")
        # a mano: a2.get_lines() arrastraria la axvline con su nombre interno
        # y la leyenda saldria con un "_child1" dentro
        lineas = [l_nit, l_rms, l_z]
        a2.legend(lineas, [l.get_label() for l in lineas], fontsize=9,
                  loc="center right")
        a2.set_title("Donde enfoca la retropropagacion (MPASM)", fontsize=11)
        fig2.tight_layout()
        figuras.append(("foco.png", fig2))

    if SALIDA is not None:
        destino = pathlib.Path(SALIDA)
        destino.mkdir(parents=True, exist_ok=True)
        for nombre, f in figuras:
            f.savefig(destino / nombre, dpi=150, bbox_inches="tight")
            print(f"  -> {destino / nombre}")
    plt.show()


if __name__ == "__main__":
    main()
