"""Ida y vuelta con el espectro angular: objeto -> +Z -> sensor -> -Z -> objeto.

Un solo propagador, angularSpectrum(), copiado tal cual de la implementacion de
referencia y sin tocar una linea. Ida, vuelta, y una figura de seis paneles:
|U|^2 arriba, fase abajo.

Tres cosas que conviene saber al leer la figura:

  - LA INTENSIDAD ES |U|^2, sin gamma. Una gamma de 0.5 sobre una intensidad es
    la amplitud, porque (|U|^2/max)^0.5 = |U|/sqrt(max).

  - LA FASE VA PESADA POR LA AMPLITUD. Donde el campo vale ~1e-16, np.angle()
    de un negativo de 1e-17 devuelve pi: sin pesar, el fondo sale de confeti y
    tapa la estructura. El fondo del panel es gris y no blanco porque en
    twilight el +-pi ES blanco.

  - NO HAY RELLENO DE CEROS. La FFT convoluciona en circulo y lo que sale por
    un borde reentra por el opuesto. La vuelta no lo nota -el doblez es
    reversible-, pero el |U|^2 que se pinta lo lleva dentro.

La vuelta se hace desde el CAMPO COMPLEJO, asi que devuelve el objeto exacto.
Desde sqrt(|U|^2) -lo unico que da un sensor- devolveria el objeto con su
imagen gemela encima, y solo el 46% de la energia caeria sobre el.

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
RUTA = r"C:\Users\User\Desktop\Tesis\referencia\carlos\DLHM-model-main\DLHM-model-main\data\BenchmarkTarget.png"

#: Longitud de onda [mm].
LAMB = 633e-6

#: Paso de pixel [mm].
DELTA = 3.45e-3

#: Distancia objeto <-> sensor [mm], POSITIVA: la ida usa +Z y la vuelta -Z.
Z = 10.0

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
#: 1/32 del ritmo de la simple. La fase se calcula en float64 siempre.
DTYPE = None                 # None = complex64 en GPU, complex128 en CPU

#: True reproduce los ejes cruzados de angularSpectrum(). Hace falta en True
#: para que comprobar_equivalencia() cierre en malla rectangular.
EJES_CRUZADOS = True

#: Filas por bloque al evaluar el kernel. Baja si la GPU se queda sin memoria.
FILAS_POR_BLOQUE = 512

#: True pinta la fase con la OPACIDAD proporcional a la amplitud; False la
#: pinta cruda, como np.angle() a secas.
#:
#: Con True, donde no hay campo no se pinta nada. Con False ves lo que hay de
#: verdad en el array, incluido el fondo. Cual quieres depende del caso:
#:
#:   - IDA Y VUELTA DESDE EL CAMPO COMPLEJO (lo que hace este script por
#:     defecto): el fondo es cero NUMERICO. Medido sobre el USAF a Z = 100
# , el
#:     91.3% del plano tiene |U| ~ 1.3e-16 -o sea 1e-16 veces el maximo- y
#:     contiene el 2.9e-29% de la energia. np.angle() de esos numeros da ruido
#:     uniforme en (-pi, pi], y con False se come la figura entera. Ahi True es
#:     lo correcto.
#:
#:   - HOLOGRAMA REAL, OBJETO DE FASE, O RECONSTRUCCION DESDE LA INTENSIDAD:
#:     el fondo tiene amplitud de verdad y su fase SIGNIFICA algo. Ahi False te
#:     ensena cosas que True esconde.
#:
#: En la duda, mira las dos. La opacidad es una decision de dibujo, no un
#: filtro sobre los datos: el array es el mismo.
PESAR_FASE = False

#: Distancias del barrido de foco, como fraccion de Z. None lo desactiva.
#: A cada z se retropropaga y se miden las dos metricas: nitidez (donde
#: enfoca) y RMS de fase (cuanto se parece la fase a la del objeto).
BARRIDO = np.linspace(0.4, 1.6, 25)

#: Carpeta donde guardar las figuras, o None para solo mostrarlas.
SALIDA = None

#: Nombre corto del propagador, para la ruta de salida del holograma. Es el
#: mismo vocabulario que usa METODOS en CamposT: fft, blas, mpasm.
METODO = "fft"


# ════════════════════════════════════════════════════════════════════════════
#  PROPAGADOR
# ════════════════════════════════════════════════════════════════════════════

def angularSpectrum(field, z, wavelength, dx, dy, scale_factor=1):
    """
    Propagación angular del frente de onda usando el espectro angular
    field: campo complejo
    z: distancia de propagación
    wavelength: longitud de onda
    dx, dy: pasos espaciales
    """
    # NO EDITAR: copiada tal cual de la implementacion de referencia. Ojo a
    # dfx = 1/(dx*M) y dfy = 1/(dy*N): estan CRUZADOS. En malla cuadrada da
    # igual; en rectangular no.
    field = np.array(field)
    M, N = field.shape
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)

    kernel = np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2)) + 0j
    phase = np.exp(1j * z * scale_factor * 2 * np.pi * np.sqrt(kernel))

    tmp = field_spec * phase
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)

    return out


# ------------------------------------------------------ propagador de trabajo
def indices_centrados(n):
    """Indices que le corresponden a fftshift(fft(...)):  ..., -1, 0, 1, ...

    Es arange(n) - n/2 cuando n es PAR, y NO lo es cuando n es impar. fftshift
    deja la componente continua en el indice n//2 en los dos casos, y
    arange(n) - n/2 vale 0 ahi solo si n es par: con n impar la rejilla queda
    medio paso corrida, el kernel se evalua fuera de sitio y el campo sale
    desplazado

        lamb * z * (0.5 / (delta * n)) / delta   pixeles

    Medido contra el gaussiano analitico en malla 65x65: RMS 4.2e-2 con
    arange(n) - n/2 y 5.5e-4 con esta. En malla par las dos son la misma
    rejilla hasta el ultimo bit, asi que esto no mueve ningun resultado de
    lado par.

    El fallo es MUDO: el medio paso se aplica en la ida y en la vuelta y se
    cancela, asi que la ida y vuelta sigue saliendo a 1e-16 y ninguna prueba
    de reversibilidad puede verlo. Lo fija tests/test_propagadores.py.

    Copia literal de la de CamposT.propagadores.frecuencias_fft(), que la
    devuelve ya multiplicada por el paso. Aqui hace falta suelta porque dfx y
    dfy no siempre salen de n.
    """
    return np.fft.fftshift(np.fft.fftfreq(n)) * n


def espectro_angular(field, z, wavelength, dx, dy, scale_factor=1,
                     xp=np, dtype=np.complex128, cruzados=True,
                     filas=FILAS_POR_BLOQUE):
    """Lo mismo que angularSpectrum(), pero cabe en la tarjeta.

    Misma matematica, mismo orden de shifts, mismo tratamiento de las
    evanescentes (se dejan decaer, no se anulan). Tres diferencias, todas de
    ejecucion y ninguna de algoritmo:

    1. El kernel NO se materializa entero. angularSpectrum() construye X e Y
       con meshgrid y luego kernel y phase completos: sobre la malla 6000x8000
       de BenchmarkTarget con PAD = 2 eso son 1.44 GB en coordenadas mas 1.44
       en kernel mas 1.44 en phase, imposible en 4 GB. Aqui el fasor se evalua
       por bloques de filas y se multiplica in situ sobre el espectro, asi que
       el pico extra es un bloque.

    2. La fase se calcula en float64 y solo el fasor —acotado a modulo 1— baja
       a dtype. En complex64 puro, 2e5 rad de fase perderian 0.02 rad en la
       mantisa.

    3. xp es NumPy o CuPy. El cuerpo es el mismo, de modo que comparar tiempos
       compara dispositivos y no dos implementaciones.

    cruzados=True reproduce el dfx = 1/(dx*M), dfy = 1/(dy*N) del original.
    Con False cada eje lleva su longitud. Solo difieren en malla rectangular.

    Y una CUARTA diferencia, esta si de algoritmo: la rejilla se centra con
    indices_centrados() y no con arange(n) - n/2, que angularSpectrum() usa y
    que solo es correcta con n PAR. En malla par las dos coinciden bit a bit
    -y por eso comprobar_equivalencia(), que corre en 256x256, sigue cerrando-;
    con lado impar angularSpectrum() sale desplazado medio paso de frecuencia y
    esta no. Ver el docstring de indices_centrados().
    """
    U = xp.asarray(field, dtype=dtype)
    M, N = U.shape

    if cruzados:
        dfx, dfy = 1 / (dx * M), 1 / (dy * N)
    else:
        dfx, dfy = 1 / (dx * N), 1 / (dy * M)

    # (1, N) y (M, 1): la aritmetica difunde igual que con meshgrid y no
    # materializa dos mallas completas
    fx = (indices_centrados(N) * dfx).astype(np.float64)
    fy = (indices_centrados(M) * dfy).astype(np.float64)
    fx = xp.asarray(fx)[None, :]
    fy = xp.asarray(fy)[:, None]

    F = xp.fft.fftshift(U)
    F = xp.fft.fft2(F)
    F = xp.fft.fftshift(F)

    inv_l2 = (1.0 / wavelength) ** 2
    for i0 in range(0, M, filas):
        i1 = min(i0 + filas, M)
        kernel = inv_l2 - (fx**2 + fy[i0:i1]**2) + 0j
        F[i0:i1] *= xp.exp(1j * z * scale_factor * 2 * np.pi
                           * xp.sqrt(kernel)).astype(dtype)
        del kernel

    out = xp.fft.ifftshift(F)
    out = xp.fft.ifft2(out)
    return xp.fft.ifftshift(out)


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
    para eso va por bloques.
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


def comprobar_equivalencia(xp, dtype, cruzados):
    """espectro_angular() contra angularSpectrum() en una malla pequena.

    Es la prueba de que acelerar no cambio el resultado. Se corre en cada
    invocacion porque cuesta milisegundos y porque una version rapida que
    nadie contrasta contra la lenta no vale nada.
    """
    rng = np.random.default_rng(0)
    U = (rng.random((256, 256)) + 1j * rng.random((256, 256)))
    ref = angularSpectrum(U, 7.0, LAMB, DELTA, DELTA)
    rap = a_cpu(espectro_angular(U, 7.0, LAMB, DELTA, DELTA, xp=xp,
                                 dtype=dtype, cruzados=cruzados))
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
               que habia ANTES de medirlo: al retropropagarlo la vuelta deshace
               la ida y devuelve el objeto exacto.
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
    base = destino / f"z{z:08.3f}"

    I = np.abs(A) ** 2
    m = I.max()
    # Un campo identicamente nulo daria 0/0: NaN por todo el array y un PNG de
    # basura, sin error y sin aviso. Un negro es un resultado legitimo.
    I = I / m if m > 0 else np.zeros_like(I)
    Image.fromarray((I * 255).astype(np.uint8)).save(base.with_suffix(".png"))

    np.save(base.with_suffix(".npy"), A)

    with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        for clave, valor in parametros.items():
            f.write(f"{clave} = {valor}\n")

    print("holograma guardado:")
    for ext in (".png", ".npy", ".txt"):
        print(f"  -> {base.with_suffix(ext)}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    if not pathlib.Path(RUTA).is_file():
        raise SystemExit(f"No encuentro la imagen en:\n    {RUTA}\n\n"
                         "Edita la constante RUTA al principio del archivo.")

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
    comprobar_memoria(xp, M, N, dtype)

    # kw viaja a todas las llamadas: el cuerpo del algoritmo es el mismo en las
    # dos maquinas, asi que comparar tiempos compara dispositivos y no dos
    # implementaciones distintas.
    kw = dict(xp=xp, dtype=dtype, cruzados=EJES_CRUZADOS)

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
    campo_sensor = espectro_angular(U0_obj, +Z, LAMB, DELTA, DELTA, **kw)
    retropropagado = espectro_angular(campo_sensor, -Z, LAMB, DELTA, DELTA, **kw)

    print(f"objeto {RUTA}")
    print(f"  malla {M}x{N} | lambda {LAMB * 1e6:.1f} nm | "
          f"delta {DELTA * 1e3:.3f} um | ida +{Z:g} mm, vuelta {-Z:+g} mm")
    print(f"  dispositivo {dev.upper()} | dtype {np.dtype(dtype).name} | "
          f"fase en float64")
    if dev == "gpu":
        libre, total = cp.cuda.runtime.memGetInfo()
        print(f"  {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, "
              f"{libre / 2**30:.2f} de {total / 2**30:.2f} GB libres")

    # angularSpectrum() es la referencia y NO acepta CuPy: su primera linea es
    # np.array(field). Corre siempre en CPU y complex128, y sirve para
    # comprobar que espectro_angular -que si va en la tarjeta- calcula lo mismo.
    eq = comprobar_equivalencia(xp, dtype, EJES_CRUZADOS)
    print(f"  espectro_angular vs angularSpectrum (referencia): {eq:.2e}")

    err = (np.abs(a_cpu(retropropagado) - U0_obj).max()
           / np.abs(U0_obj).max())
    print(f"  ida y vuelta del campo complejo: error max relativo = {err:.2e}"
          f"   <- es la identidad, tiene que ser ~1e-15 (1e-7 en complex64)")

    # Se guarda aqui y no justo tras la ida para que la consola se lea en
    # orden: primero con que se hizo, despues que se escribio. campo_sensor ya
    # existe y nadie lo toca en medio.
    guardar_holograma(campo_sensor, Z, {
        "objeto": RUTA,
        "propagador": "FFT-ASM (espectro angular por FFT)",
        "lambda [mm]": LAMB,
        "delta [mm]": DELTA,
        "Z [mm]": Z,
        "malla": f"{M}x{N}",
        "ENTRADA": ENTRADA,
        "PHI_MAX": PHI_MAX,
        "dispositivo": dev,
        "dtype": np.dtype(dtype).name,
        "EJES_CRUZADOS": EJES_CRUZADOS,
    })

    # ---- barrido de foco ----------------------------------------------------
    zs = nit = rms = None
    if BARRIDO is not None:
        U0 = xp.asarray(U0_obj, dtype=dtype)
        mascara = np.abs(U0_obj) > 0.5 * np.abs(U0_obj).max()
        zs = np.asarray(BARRIDO, float) * Z
        nit, rms = [], []
        for k, z in enumerate(zs):
            U = espectro_angular(campo_sensor, -z, LAMB, DELTA, DELTA, **kw)
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
        print(f"    nitidez maxima  en z = {z_nit:.2f} mm" if z_nit is not None
              else "    nitidez: el maximo cae en un extremo del barrido, que "
                   "NO acota el foco. Ensancha BARRIDO.")
        print(f"    RMS de fase minimo en z = {z_rms:.2f} mm" if z_rms is not None
              else "    RMS: el minimo cae en un extremo del barrido.")
        print(f"    esperado:            z = {Z:.2f} mm")
        print(f"    RMS en el foco: {np.degrees(rms[np.argmin(np.abs(zs - Z))]):.2f} deg"
              f"   (sin informacion de fase saldrian "
              f"{np.degrees(np.pi / np.sqrt(3)):.1f} deg)")

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
    fig.suptitle(f"Espectro angular, ida y vuelta -- Z = {Z:g} mm, "
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
        a2.set_title("Donde enfoca la retropropagacion", fontsize=11)
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
