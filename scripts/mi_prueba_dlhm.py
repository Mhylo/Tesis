"""Reconstruccion DLHM de fuente puntual, con barrido puntuado contra el objeto.

El hermano de scripts/mi_prueba_angular.py, que hace lo mismo pero con onda
plana. Aqui la iluminacion es una FUENTE PUNTUAL DIVERGENTE, que es lo que hay
en un banco DLHM de verdad, y eso cambia tres cosas que ninguna cantidad de
reescalado de imagenes arregla:

  1. Hay que multiplicar el holograma por la ONDA ESFERICA de referencia antes
     de propagar, y dividirla despues: Rec = U * conj(U0).
  2. La distancia de propagacion es L - z, no z.
  3. La reconstruccion sale MAGNIFICADA por M = L/z, asi que la referencia hay
     que escalarla por M para poder compararla.

POR QUE UN SCRIPT APARTE. mi_prueba_angular.py propaga onda plana y funciona.
Meter aqui un interruptor de geometria lo llevaria a ~620 lineas y obligaria a
cada constante a decir en que modo aplica. El repo ya tiene el patron de un
script por concepto: retro_fft_angular, retro_blas, retro_mpasm.

DE DONDE SALE LA FISICA. point_src, angular_spectrum, fts, ifts y resize estan
COPIADAS TAL CUAL de referencia/carlos/DLHM-model-main/.../dlhm.py, igual que
mi_prueba_angular copia angularSpectrum: su valor entero es que nadie las ha
tocado. Si hay que cambiar algo, se cambia en una copia de trabajo, no aqui.

OJO: el angular_spectrum de aqui usa dfx = 1/Wx, con los ejes EN SU SITIO. El
angularSpectrum de mi_prueba_angular los lleva CRUZADOS. En malla cuadrada
coinciden; en rectangular no. No los mezcles.

EL ROI NO ES UNA COMODIDAD AQUI, ES LO QUE HACE VIABLE EL BARRIDO. El
remuestreo por geometria exige una malla de Wx/s puntos, y eso depende del
ANCHO FISICO del sensor, no del paso de pixel: submuestrear el holograma no
ayuda nada. Medido con lambda 532 nm, delta 1.85 um y z = 2 mm:

    recorte    malla        memoria    por paso
      3000    13144x13144    15.4 GB    ~24 s
      1024     2802x2802      0.70 GB   ~1.1 s
       750     1624x1624      0.24 GB   ~0.4 s

Sin recortar, un solo paso pide 15 GB. El script calcula esa malla ANTES de
empezar y aborta con la tabla si no cabe, en vez de morir sin explicar nada.

SE COMPARA LA FASE, NO LA INTENSIDAD. El objeto del modelo de Carlos es
exp(-i*2pi*(n-1)*h*I/lambda): fase pura, |objeto|^2 = 1 en todas partes.
Correlacionar amplitud seria comparar contra algo uniforme. Y por ese signo
menos, la fase corre AL REVES que los grises de la referencia: una correlacion
NEGATIVA aqui es el modelo cumpliendose, no un fallo. Por eso el pico se busca
en |corr| y se reporta con su signo.

EL BARRIDO VA SOBRE z, CON L FIJO. L -fuente a sensor- se mide una vez en el
banco. z -fuente a muestra- es lo que no se sabe. Y como M = L/z, mover z mueve
el foco Y la escala a la vez: la referencia se reescala en cada paso, asi que
la correlacion solo sube cuando las dos cosas coinciden. Eso hace el pico mas
agudo, no mas confuso.

ESTADO: LA RECONSTRUCCION FUNCIONA, EL BARRIDO PUNTUADO NO. Leelo antes de
fiarte de un numero de este script.

Lo que SI se comprobo, mirando la imagen: con el sensor ENTERO y SOBREMUESTREO
= 1, angle(Rec) muestra estructura del objeto reconocible -grupos de barras,
numeros-. La fisica DLHM esta bien puesta.

Lo que NO funciona: la correlacion contra la referencia sale ~0.03 a cualquier
escala, o sea ruido, y el barrido no encuentra el z = 2 mm que dice
main_dlhm.py. Se descartaron tres causas y ninguna era:

  1. El recorte del holograma. Con ROI de 750x750 la reconstruccion queda
     DOMINADA por la difraccion del borde de la ventana: la log-amplitud es un
     cuadrado borroso y el objeto desaparece. Aqui el ROI hace el barrido
     viable y a la vez insignificante.
  2. El sobremuestreo por geometria. Con of ~ 4.4 la malla pide 15 GB por paso;
     con of = 1 cabe el sensor entero y ahi SI aparece el objeto. Pero of = 1
     alia, y la log-amplitud sale replicada en una rejilla 3x3.
  3. La escala de la referencia. Se barrio M de 0.8 a 6.0 y el mejor ajuste da
     -0.029: no hay escala que case.

VALIDADO CONTRA reconstruction_dlhm.py, y pasa. Corriendo las dos cadenas
sobre el mismo holograma reducido a 512x512, con sus parametros (L = 11 mm,
z = 4.95 mm):

    razon mio/Carlos    1.000000e-06   (desviacion relativa 6.6e-06)
    corr(|Rec|)         1.000000
    corr(angle(Rec))    0.999997

O sea que reconstruir() es fiel. Eso DESCARTA la reconstruccion como causa: el
fallo esta en la puntuacion.

OJO A ESE 1e-6, que es (1e-3)^2 y no un error: point_src devuelve exp(ikr)/r, y
el 1/r NO es invariante de escala. Trabajar en milimetros en vez de metros lo
cambia por 1000, y como Rec = U*conj(U0) el factor entra al cuadrado. Es un
factor real global: no afecta a ninguna correlacion, pero desconcierta si
alguien compara salidas numero a numero.

EL PORTADOR DE FASE TAMBIEN QUEDA DESCARTADO. Se estimo con un paso bajo del
propio Rec y se dividio en el plano complejo -que es como se quita un portador
sobre fase ENVUELTA; ajustar un polinomio a angle() no vale-. Con sigma de 10,
30, 80 y 200 la correlacion se queda en ~0.03 y la desviacion de la fase ni se
mueve: 1.317 a 1.335. Si el objeto estuviera debajo de una rampa suave, esto lo
habria sacado.

Y mirando de cerca: en el cuadrante donde la referencia a M = 4 tiene una
estrella radial, la fase reconstruida tiene bandas y bloques rectangulares. No
se parecen. Lo que parecia "estructura del objeto" a tamano completo es textura
de difraccion.

CINCO HIPOTESIS DESCARTADAS, Y LA CONCLUSION ES QUE EL ENFOQUE ES EL EQUIVOCADO.
Puntuar contra la referencia exige saber a la vez la escala, el centrado y la
geometria (L, z), y de las tres solo se conoce una relacion: M = L/z ~ 4. Los
dos scripts de Carlos ni siquiera concuerdan entre si -main_dlhm usa L=8, z=2;
reconstruction_dlhm usa L=11, z=4.95-, asi que (L, z) es justo lo que habria que
buscar... con el barrido que no funciona porque no sabe puntuar. Es circular.

LA SALIDA es puntuar SIN referencia: una metrica de NITIDEZ sobre la
reconstruccion, como la nitidez() de scripts/retro_holograma.py. Una
reconstruccion enfocada tiene mas contraste que una desenfocada, sea cual sea el
objeto, y eso no necesita ni escala, ni centrado, ni que la fase este
desenvuelta. Rompe la circularidad.

MIENTRAS TANTO, este script sirve para MIRAR la reconstruccion, no para
puntuarla. El numero que imprime no significa nada todavia.

EL MODELO DE REFERENCIA SOLO GENERA EN UNA CONFIGURACION, Y ESO EXPLICA MUCHO.

dlhm() decide entre RECORTAR el objeto al campo que el sensor abarca, o
estirarlo entero a la malla del sensor:

    if W_provided > W_s:  <recorta>
    else:                 <estira>

Con los parametros de main_dlhm.py los dos valen 1.3875e-3 EXACTOS, asi que la
comparacion sale False por un empate de flotantes y corre el else: el objeto
entero, bajado x4 y estirado a 3000x3000, con las columnas aplastadas x0.75.

Y la otra rama ESTA ROTA. Recorta con los N, M ORIGINALES en vez de los del
sample ya remuestreado:

    sample = resize(sample, int(M/res_d), int(N/res_d))   # ahora es 750x1000
    N_s, M_s = sample.shape                                # 750, 1000 <- se calcula
    ...
    start_x = int(N/2 - Q/2 + x0)                          # ...pero usa N = 3000
    sample = sample[1000:2000, 1500:2500]                  # sobre 750x1000: VACIO

De ahi sale un ZeroDivisionError en cuanto se pide cualquier otro sensor. O sea
que dlhm() solo produce hologramas en la configuracion exacta de main_dlhm.py.

Y no se arregla cambiando N, M por N_s, M_s: con eso los indices salen
NEGATIVOS -int(375-500) = -125- y el recorte pide Q x P = 1000x1000 pixeles
cuando el campo que el sensor abarca son W_s/dx_in = 250. El remuestreo previo
va ademas en sentido contrario al que la magnificacion pide: deja el paso del
sample en dx_in*Mag = 7.4 um cuando deberia ser dx_out/Mag = 0.4625 um.

CONSECUENCIA PARA data/Simulated_hologram.png: se comporta como un recorte
central magnificado x4 -medido barriendo escalas contra el objeto- y eso es
justo lo que la rama rota haria si funcionase. Asi que ese archivo NO salio del
codigo tal como esta hoy. Reconstruirlo a ciegas es perseguir una geometria que
nadie puede reproducir.

UNIDADES: milimetros para todo.  532 nm -> 532e-6    1.85 um -> 1.85e-3
"""

import pathlib
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
except Exception:                      # sin CuPy, sin CUDA, o CuPy roto
    cp = None

from CamposT.roi import Roi, elegir, informe

# ════════════════════════════════════════════════════════════════════════════
#  1. PARAMETROS
# ════════════════════════════════════════════════════════════════════════════

#: EL HOLOGRAMA. Se usa su INTENSIDAD tal cual, sin tomar la raiz: la cadena
#: DLHM multiplica la intensidad registrada por la onda de referencia, no su
#: amplitud. Es lo que hace reconstruction_dlhm.py.
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\dlhm_propio\holo.png"

#: EL OBJETO, contra el que se puntua cada z del barrido.
REFERENCIA = r"C:\Users\User\Desktop\Tesis\referencia\carlos\DLHM-model-main\DLHM-model-main\data\BenchmarkTarget.png"

#: Longitud de onda [mm]. El modelo de Carlos usa 532 nm, no 633.
LAMB = 532e-6

#: Paso de pixel del sensor [mm]. El modelo usa 1.85 um, no 3.45.
DELTA = 1.85e-3

#: Distancia FUENTE -> SENSOR [mm]. Se mide una vez en el banco y NO se barre.
L = 8.0

#: Barrido de la distancia FUENTE -> MUESTRA [mm], POSITIVA y menor que L.
#: Es lo que no se sabe. Con L = 8 el pico deberia caer en z = 2 (M = L/z = 4).
Z = (1.0, 4.0)
PASOS = 25

#: QUE SE CORRELACIONA CONTRA LA REFERENCIA.
#:
#:   "fase"        np.angle(Rec). Lo correcto para el objeto de este modelo,
#:                 que es de FASE PURA: su |U|^2 es uniforme y no dice nada.
#:   "intensidad"  np.abs(Rec)**2, por si el objeto absorbe.
COMPARAR = "intensidad"

#: EL RECORTE, una ventana por imagen. Mismos cuatro valores que en
#: mi_prueba_angular: None, True (raton sobre ESA imagen), (X0, Y0, ANCHO,
#: ALTO), o "misma" (el rectangulo que resolvio la otra).
#:
#: AQUI CASI NUNCA QUIERES None EN EL HOLOGRAMA: sin recortar, el remuestreo
#: pide 15 GB por paso. Lee "EL ROI NO ES UNA COMODIDAD" arriba.
ROI_HOLOGRAMA = "misma"
ROI_REFERENCIA = True

#: Presupuesto de memoria [GB] para la malla remuestreada. Si el barrido pide
#: mas, aborta ANTES de empezar con la tabla de recortes, en vez de morir en el
#: primer paso con un MemoryError que no dice que hacer.
MEMORIA_MAX_GB = 4.0

#: SOBREMUESTREO de la malla en el plano de la muestra.
#:
#:   None   el factor que exige la geometria, of = delta/s. Es lo que hace
#:          reconstruction_dlhm.py, y con el sensor entero pide 15 GB por paso.
#:   1      la malla se queda como el sensor. Es la linea que Carlos dejo
#:          COMENTADA en su reconstruction_dlhm.py, y es la unica configuracion
#:          en la que se ha visto salir el objeto: sensor entero y of = 1.
#:
#: El precio de 1 es el aliasing: la reconstruccion sale replicada en una
#: rejilla 3x3. El precio de None es que no cabe salvo recortando, y recortando
#: la difraccion del borde de la ventana tapa el objeto. Ese es el callejon en
#: el que esta este script; lee ESTADO en la cabecera.
SOBREMUESTREO = None

#: Centinela para "usa la constante de arriba". No vale None, porque None YA
#: significa algo distinto -el sobremuestreo que exige la geometria- y no vale
#: 1 porque entonces no habria forma de pedir el otro. Existe para que
#: reconstruir() siga leyendo la constante cuando la llama main(), y a la vez
#: pueda fijarse desde fuera: comprobar_equivalencia() tiene que dar el mismo
#: numero se edite lo que se edite aqui arriba.
DE_LA_CONSTANTE = object()

#: DISPOSITIVO de calculo: "auto" (la GPU si la hay), "cpu" o "gpu".
#:
#: LO QUE LA TARJETA ARREGLA Y LO QUE NO, que aqui no es obvio. El barrido son
#: PASOS reconstrucciones, y cada una son dos pares de FFT sobre la malla
#: entera: eso la GPU lo hace mucho mas rapido. Lo que NO arregla es el
#: callejon de la cabecera: con SOBREMUESTREO = None la malla que exige la
#: geometria es de 13312x13312, o sea 1.42 GB POR ARRAY en complex64, y la
#: cadena necesita varios a la vez. En una tarjeta de 4 GB sigue sin caber.
#: Lo que complex64 compra es aproximadamente el DOBLE de recorte que en CPU
#: con complex128, no el sensor entero.
#:
#: Con "gpu" y sin CUDA ABORTA en vez de caer a CPU en silencio.
DISPOSITIVO = "auto"

#: dtype de trabajo. None = complex64 en GPU, complex128 en CPU.
#:
#: Es la politica de CamposT.backend. Las FASES se evaluan en float64 pase lo
#: que pase: el argumento de la onda esferica es k*r, que con lambda = 532 nm
#: y r ~ 8 mm vale ~9.4e4 rad, y en float32 eso pierde 0.006 rad de mantisa.
#: Lo que baja a simple es el fasor ya acotado a modulo 1.
DTYPE = None

#: Filas por bloque al construir la onda esferica y el kernel. NO cambia el
#: resultado, solo la memoria de pico.
FILAS_POR_BLOQUE = 512


# ════════════════════════════════════════════════════════════════════════════
#  2. LA FISICA DLHM  --  COPIADA TAL CUAL DE dlhm.py, NO EDITAR
# ════════════════════════════════════════════════════════════════════════════
#
# Estas cinco funciones son de
# referencia/carlos/DLHM-model-main/DLHM-model-main/dlhm.py, sin tocar una
# linea. Su valor entero es que nadie las ha tocado: son contra lo que se
# contrasta cualquier version de trabajo. Si hay que cambiar algo, se hace en
# una copia, no aqui.

def ifts(A):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(A)))


def fts(A):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(A)))


def resize(B, dx, dy):
    r = np.real(B)
    img = np.imag(B)
    r_r = cv.resize(r, (int(dx), int(dy)), interpolation=cv.INTER_LINEAR)
    img_r = cv.resize(img, (int(dx), int(dy)), interpolation=cv.INTER_LINEAR)
    B_r = r_r + 1j * img_r
    return B_r


def point_src(N, M, z, x0, y0, lambda_, dx):
    """
    Generates a point source illumination centered at (x0, y0)
    and observed in a plane at distance z.
    """
    dy = dx  # Set y-pitch same as x-pitch
    m, n = np.meshgrid(np.arange(-M / 2, M / 2), np.arange(-N / 2, N / 2))  # Coordinate mesh grid
    k = 2 * np.pi / lambda_  # Wavenumber
    r = np.sqrt(z ** 2 + (m * dx - x0) ** 2 + (n * dy - y0) ** 2)  # Radial distance from source
    return np.exp(1j * k * r) / r  # Complex field with spherical phase


def angular_spectrum(A, Wx, Wy, k, z):
    """
    Propagates field A using angular spectrum method.
    """
    Q, P = A.shape  # Get size of input field
    dfx = 1 / Wx  # Frequency sampling interval in x
    dfy = 1 / Wy  # Frequency sampling interval in y

    # Generate frequency grid using linspace
    fx = np.linspace(-P/2 * dfx, (P/2 - 1) * dfx, P)
    fy = np.linspace(-Q/2 * dfy, (Q/2 - 1) * dfy, Q)
    fx, fy = np.meshgrid(fx, fy)


    # Complex exponential term for propagation
    E = np.exp(-1j * z * np.sqrt(k ** 2 - (2 * np.pi * fx) ** 2 - (2 * np.pi * fy) ** 2))

    # Perform Fourier transform, apply propagation, and inverse Fourier transform
    return ifts(fts(A) * E)


# ════════════════════════════════════════════════════════════════════════════
#  2b. COPIA DE TRABAJO  --  la misma fisica, en la tarjeta
# ════════════════════════════════════════════════════════════════════════════
#
# La seccion 2 se queda intacta porque es la REFERENCIA. Estas son las mismas
# cuentas escritas contra xp -NumPy o CuPy- y por bloques de filas, que es lo
# que el resto del repo hace en CamposT.backend y en los retro_*.
#
# comprobar_equivalencia() corre las dos cadenas sobre la misma entrada en
# cada invocacion: si alguna vez dejan de coincidir, se ve en la consola antes
# de mirar ningun resultado.

def _fts(A, xp):
    return xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(A)))


def _ifts(A, xp):
    return xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(A)))


def fuente_puntual(N, M, z, x0, y0, lamb, dx, xp=np, dtype=np.complex128,
                   filas=FILAS_POR_BLOQUE):
    """Lo mismo que point_src(), pero cabe en la tarjeta. -> (N, M).

    Dos diferencias, las dos de ejecucion:

    1. No se materializan las dos mallas de meshgrid. Los ejes van sueltos y
       la aritmetica difunde igual, asi que el pico extra es un bloque de
       filas en vez de dos arrays completos.

    2. La fase k*r se evalua en float64 y solo el resultado baja a dtype. Con
       lambda = 532 nm y r ~ 8 mm, k*r vale ~9.4e4 rad.

    La rejilla es la misma hasta el ultimo bit: arange(n) - n/2 es exactamente
    arange(-n/2, n/2), tambien con n impar.

    OJO AL 1/r, que no es invariante de escala: en milimetros la salida vale
    1000 veces la de metros, y como Rec = U*conj(U0) el factor entra al
    cuadrado. Es global y real, no afecta a ninguna correlacion. Esta anotado
    en la cabecera del modulo.
    """
    dy = dx
    k = 2 * np.pi / lamb
    m = (xp.arange(M, dtype=np.float64) - M / 2) * dx - x0     # a lo ancho
    n = (xp.arange(N, dtype=np.float64) - N / 2) * dy - y0     # a lo alto
    out = xp.empty((N, M), dtype=dtype)
    for i0 in range(0, N, filas):
        i1 = min(i0 + filas, N)
        r = xp.sqrt(z ** 2 + m[None, :] ** 2 + n[i0:i1, None] ** 2)
        out[i0:i1] = (xp.exp(1j * k * r) / r).astype(dtype)
        del r
    return out


def espectro_angular(A, Wx, Wy, k, z, xp=np, dtype=np.complex128,
                     filas=FILAS_POR_BLOQUE):
    """Lo mismo que angular_spectrum(), pero cabe en la tarjeta.

    Mismo signo en el exponente -negativo, que es el de esta cadena-, mismos
    ejes EN SU SITIO (dfx = 1/Wx), mismo orden de shifts. Dos diferencias, las
    dos de ejecucion:

    1. E no se materializa entera: se evalua por bloques de filas y se
       multiplica in situ sobre el espectro.

    2. La fase va en float64 y solo el fasor baja a dtype.

    LAS EVANESCENTES NO SE TOCAN, igual que en el original: si el argumento de
    la raiz se hiciera negativo saldria nan y lo veria todo el mundo. En esta
    geometria no pasa -k = 11809 rad/mm y la frecuencia mas alta que la malla
    describe son 2*pi*1200 = 7540- pero es una propiedad de estos parametros,
    no de la funcion. Se deja como esta porque el trato de la copia de trabajo
    es no cambiar el algoritmo.
    """
    U = xp.asarray(A, dtype=dtype)
    Q, P = U.shape
    dfx, dfy = 1 / Wx, 1 / Wy

    # linspace y no arange: es la rejilla del original bit a bit
    fx = np.linspace(-P / 2 * dfx, (P / 2 - 1) * dfx, P)
    fy = np.linspace(-Q / 2 * dfy, (Q / 2 - 1) * dfy, Q)
    fx = xp.asarray(fx, dtype=np.float64)[None, :]
    fy = xp.asarray(fy, dtype=np.float64)[:, None]

    F = _fts(U, xp)
    dospi = 2 * np.pi
    for i0 in range(0, Q, filas):
        i1 = min(i0 + filas, Q)
        raiz = xp.sqrt(k ** 2 - (dospi * fx) ** 2 - (dospi * fy[i0:i1]) ** 2)
        F[i0:i1] *= xp.exp(-1j * z * raiz).astype(dtype)
        del raiz
    return _ifts(F, xp)


# ------------------------------------------------------------------ backend
#
# elegir_dispositivo, a_cpu y liberar son copias literales de las de
# scripts/retro_fft_angular.py. sincronizar() no: esa es
# CamposT.backend.sincronizar() sin el argumento opcional, porque aqui el xp
# siempre se sabe. tests/test_gpu_mi_prueba.py comprueba que no diverjan.

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


def sincronizar(xp):
    """Espera a que la tarjeta termine.

    CuPy encola los kernels y devuelve el control de inmediato: sin esto, un
    perf_counter() alrededor del barrido mide el tiempo de LANZAMIENTO y no el
    de computo, y la GPU sale ridiculamente rapida. Es obligatoria en toda
    medida de tiempo.
    """
    if xp is cp:
        cp.cuda.Stream.null.synchronize()


def _reconstruir_referencia(holo, z, lamb, delta, L_fuente, sobremuestreo=1):
    """La cadena con las funciones INTACTAS de la seccion 2. Solo CPU.

    point_src, angular_spectrum y resize tal como vienen de dlhm.py, sin una
    linea cambiada -incluido el `referencia * np.ones((N, M))` que la copia de
    trabajo se ahorra-. Existe para que comprobar_equivalencia() tenga contra
    que medir.

    Lleva las DOS ramas de sobremuestreo, no solo la de 1: si solo tuviera una,
    la comparacion valdria en una configuracion y en la otra compararia mallas
    de distinto tamano. Que es exactamente el fallo que motivo este argumento.
    """
    Q, P = holo.shape
    Wcx, Wcy = P * delta, Q * delta
    k = 2 * np.pi / lamb
    if sobremuestreo is None:
        N, M = malla_remuestreada(P, Q, z, lamb, delta)
        h = resize(holo.astype(complex), M, N)
    else:
        N, M = Q, P
        h = holo.astype(complex)
    referencia = point_src(N, M, L_fuente, 0, 0, lamb, (delta * P) / N)
    U = angular_spectrum(referencia * h, Wcx, Wcy, k, L_fuente - z)
    U0 = angular_spectrum(referencia * np.ones((N, M)), Wcx, Wcy, k,
                          L_fuente - z)
    return U * np.conj(U0)


def comprobar_equivalencia(xp, dtype, sobremuestreo=1):
    """reconstruir() contra la cadena intacta de la seccion 2.

    Es la prueba de que acelerar no cambio el resultado. Se corre en cada
    invocacion porque cuesta milisegundos y porque una version rapida que
    nadie contrasta contra la lenta no vale nada.

    Se mide sobre |Rec| y no sobre Rec entero por lo que dice la cabecera del
    modulo: el 1/r de point_src arrastra un factor global -(1e-3)^2 al pasar
    de metros a milimetros, y al cuadrado por Rec = U*conj(U0)- que es real y
    no afecta a nada, pero que un error relativo sobre el complejo si acusaria
    si algun dia alguien cambia de unidades. El error relativo sobre el modulo
    es la cantidad que de verdad importa aqui.

    Y el sobremuestreo se FIJA en 1 por defecto, en vez de leer la constante:
    lo que esta funcion tiene que responder es "el dispositivo cambia el
    resultado?", y esa respuesta no puede depender de que estes probando la
    malla remuestreada o no. La otra rama se comprueba igual pasando None, y
    tests/test_gpu_mi_prueba.py pasa por las dos.
    """
    rng = np.random.default_rng(0)
    holo = rng.random((256, 256))
    ref = _reconstruir_referencia(holo, 1.875, LAMB, DELTA, L,
                                  sobremuestreo=sobremuestreo)
    rap = a_cpu(reconstruir(holo, 1.875, LAMB, DELTA, L, xp=xp, dtype=dtype,
                            sobremuestreo=sobremuestreo))
    return float(np.max(np.abs(np.abs(rap) - np.abs(ref)))
                 / np.max(np.abs(ref)))


# ════════════════════════════════════════════════════════════════════════════
#  3. RECONSTRUCCION Y BARRIDO
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

def malla_remuestreada(lado_x, lado_y, z, lamb, delta):
    """Cuantos puntos por lado exige el remuestreo por geometria. -> (N, M).

    El factor sale de la frecuencia de muestreo en el plano de la muestra:

        s  = lambda * sqrt((Wx/2)^2 + (Wy/2)^2 + z^2) / Wx
        of = delta / s

    y la malla es lado*of = Wx/s. OJO A ESO: el resultado depende del ANCHO
    FISICO del sensor, no del paso de pixel. Submuestrear el holograma no
    reduce la malla ni un punto; recortarlo si, y linealmente. Por eso aqui el
    ROI es lo que hace viable el barrido y no una comodidad.
    """
    Wx, Wy = lado_x * delta, lado_y * delta
    s = lamb * np.sqrt((Wx / 2) ** 2 + (Wy / 2) ** 2 + z ** 2) / Wx
    of = delta / s
    return int(lado_y * of), int(lado_x * of)


def comprobar_memoria(holo, zs, lamb, delta, tope_gb, xp=np,
                      dtype=np.complex128):
    """Aborta ANTES del barrido si la malla remuestreada no cabe.

    Sin esto, el primer paso pide la memoria que pida y muere con un
    MemoryError que no dice que hacer. La tabla dice exactamente cuanto hay que
    recortar, que es la unica palanca que mueve este numero.

    EL TOPE NO ES EL MISMO EN LOS DOS DISPOSITIVOS. En CPU manda
    MEMORIA_MAX_GB, que es una cifra que pones tu. En GPU manda la VRAM libre
    de verdad, que es un hecho: se toma la menor de las dos, porque de nada
    sirve autorizar 4 GB en una tarjeta que tiene 3.2 libres.

    Y el tamano depende del dtype: en complex64 la misma malla ocupa la mitad,
    que es lo que hace que en la tarjeta quepa aproximadamente el doble de
    recorte que en CPU.
    """
    Q, P = holo.shape
    itemsize = np.dtype(dtype).itemsize
    if xp is cp:
        libre, _ = cp.cuda.runtime.memGetInfo()
        tope_gb = min(tope_gb, 0.85 * libre / 2 ** 30)
    # el peor caso es la z mas chica: s crece con z, y la malla es Wx/s
    if SOBREMUESTREO is not None:
        gb = 6 * Q * P * itemsize / 2 ** 30
        print(f"  SOBREMUESTREO = {SOBREMUESTREO}: la malla se queda en "
              f"{Q}x{P} ({Q * P / 1e6:.1f} Mpx, ~{gb:.2f} GB de pico). "
              f"Alia, ver ESTADO.")
        if gb > tope_gb:
            raise SystemExit(
                f"Ni siquiera sin remuestrear cabe: {gb:.2f} GB de pico "
                f"contra un tope de {tope_gb:.2f} GB.\nRecorta con "
                f"ROI_HOLOGRAMA, o usa DISPOSITIVO = 'cpu'.")
        return
    N, M = malla_remuestreada(P, Q, min(zs), lamb, delta)
    # seis mallas complejas de pico: h, ref, dos productos y dos espectros
    gb = 6 * N * M * itemsize / 2 ** 30
    if gb <= tope_gb:
        print(f"  malla remuestreada: hasta {N}x{M} ({N * M / 1e6:.1f} Mpx, "
              f"~{gb:.2f} GB de pico)")
        return
    lineas = ["  recorte    malla          memoria"]
    for lado in (3000, 2048, 1500, 1024, 750, 512):
        if lado > min(P, Q):
            continue
        n, m = malla_remuestreada(lado, lado, min(zs), lamb, delta)
        lineas.append(f"  {lado:6d}   {n:5d}x{m:<5d}   "
                      f"{6 * n * m * itemsize / 2**30:5.2f} GB")
    raise SystemExit(
        f"El remuestreo por geometria pide una malla de {N}x{M} "
        f"({gb:.1f} GB de pico) y el tope es {tope_gb} GB.\n\n"
        f"Eso depende del ANCHO FISICO del sensor, no del paso de pixel: "
        f"submuestrear no ayuda,\nhay que RECORTAR. Con ROI_HOLOGRAMA:\n\n"
        + "\n".join(lineas)
        + f"\n\nO sube MEMORIA_MAX_GB si sabes que tu maquina lo aguanta "
          f"(en GPU manda la VRAM libre,\nque MEMORIA_MAX_GB no puede subir).")


def reconstruir(holo, z, lamb, delta, L_fuente, xp=np,
                dtype=np.complex128, sobremuestreo=DE_LA_CONSTANTE):
    """Un holograma DLHM -> el campo reconstruido en el plano de la muestra.

    Es la cadena de reconstruction_dlhm.py: remuestrear por geometria,
    multiplicar por la onda esferica, propagar L-z, y dividir la referencia.

    Rec = U * conj(U0) es lo que quita la onda esferica del resultado. Sin ese
    paso, la fase del objeto queda montada sobre la del frente divergente y no
    se parece a nada.

    Corre en NumPy o en CuPy segun xp. Dos avisos sobre eso:

    EL REMUESTREO SE QUEDA EN CPU. resize() es cv.resize, que no acepta arrays
    de CuPy, asi que con SOBREMUESTREO = None la interpolacion se hace en la
    CPU y solo despues sube la malla ya remuestreada. Es una sola vez por
    paso, y es la parte barata: lo caro son las cuatro FFT que vienen detras.

    NO SE MULTIPLICA POR ones((N, M)). El original escribe
    `referencia * np.ones((N, M))` para el campo de referencia, que es la
    referencia sin mas: multiplicar por unos no cambia un numero y si
    materializa una malla entera de mas. Es la unica linea que esta copia no
    reproduce literalmente, y comprobar_equivalencia() fija que el resultado
    no se mueve.

    sobremuestreo vale por defecto la constante SOBREMUESTREO, que es lo que
    quiere main(). Se puede fijar por argumento, y comprobar_equivalencia() lo
    hace: si leyera la constante, su resultado cambiaria cada vez que alguien
    la edita, y una comprobacion que depende de lo que estas probando no
    comprueba nada.
    """
    if sobremuestreo is DE_LA_CONSTANTE:
        sobremuestreo = SOBREMUESTREO
    Q, P = holo.shape
    Wcx, Wcy = P * delta, Q * delta
    k = 2 * np.pi / lamb

    if sobremuestreo is None:
        N, M = malla_remuestreada(P, Q, z, lamb, delta)
        h = resize(holo.astype(complex), M, N)      # cv.resize: siempre CPU
    else:
        N, M = Q, P
        h = holo

    kw = dict(xp=xp, dtype=dtype)
    h = xp.asarray(h, dtype=dtype)
    referencia = fuente_puntual(N, M, L_fuente, 0, 0, lamb, (delta * P) / N,
                                **kw)
    U = espectro_angular(referencia * h, Wcx, Wcy, k, L_fuente - z, **kw)
    del h
    U0 = espectro_angular(referencia, Wcx, Wcy, k, L_fuente - z, **kw)
    del referencia
    return U * xp.conj(U0)


def referencia_a_escala(ref, mag, forma):
    """El objeto tal como lo ve el sensor: su parte central, magnificada.

    Con magnificacion mag, el sensor solo abarca 1/mag del objeto, y lo enseña
    llenando toda la malla. Asi que se recorta el central 1/mag y se lleva al
    tamano de la reconstruccion.

    Se asume que el objeto y el sensor comparten paso de pixel y estan
    centrados en el eje optico. Lo primero lo cumple el modelo de Carlos
    (dx_in = dx_out). Lo segundo, casi: midiendo la escala contra el holograma
    aparecio un desplazamiento de unos 50 px en filas, que aqui se ignora
    porque es el 1.7% del lado y no mueve el pico del barrido.
    """
    H, W = ref.shape
    ly, lx = max(2, int(round(H / mag))), max(2, int(round(W / mag)))
    y0, x0 = (H - ly) // 2, (W - lx) // 2
    recorte = ref[y0:y0 + ly, x0:x0 + lx]
    return cv.resize(recorte, (forma[1], forma[0]), interpolation=cv.INTER_AREA)


def correlacion(a, b):
    """Correlacion de Pearson entre dos mapas reales.

    Invariante a escala y a desplazamiento, que es lo que hace falta cuando ni
    el brillo absoluto ni el cero de la fase significan nada. Un mapa constante
    devuelve 0 en vez de un NaN que viajaria hasta el pico del barrido.
    """
    # La REDUCCION va en float64 aunque el campo venga en complex64. Sobre
    # una malla de 3000x4000 son 1.2e7 sumandos: en float32 el error relativo
    # acumulado es ~1e-4, del orden de lo que separa dos pasos vecinos del
    # barrido, y el argmax se iria a un z equivocado. 96 MB de copia es barato
    # comparado con eso. En CPU no cambia nada: ya venia en doble.
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = float(a @ a) * float(b @ b)
    return float(a @ b) / den ** 0.5 if den > 0 else 0.0


def observable(Rec, comparar):
    """Que se compara contra la referencia. -> mapa real.

    El objeto del modelo es de FASE PURA -exp(-i*phi*I)-, asi que su |U|^2 es
    uniforme y correlacionarlo no diria nada. La estructura esta en angle(Rec).
    """
    if comparar == "fase":
        return np.angle(Rec)
    if comparar == "intensidad":
        return np.abs(Rec) ** 2
    raise SystemExit(f'COMPARAR = "{comparar}" no existe. '
                     'Pon "fase" o "intensidad".')


# ════════════════════════════════════════════════════════════════════════════
#  4. MAIN
# ════════════════════════════════════════════════════════════════════════════

def _roi_de(valor, imagen, titulo, nombre):
    """Un valor de ROI_* -> Roi o None. "misma" lo resuelve el llamante."""
    if valor is None or valor is False:
        return None
    if valor is True:
        return elegir(imagen, titulo)
    try:
        x0, y0, ancho, alto = valor
    except (TypeError, ValueError):
        raise SystemExit(
            f"{nombre} = {valor!r} no es ninguno de los cuatro valores "
            f"validos:\n\n"
            f"    None                    sin recorte\n"
            f"    True                    la arrastras con el raton\n"
            f"    (X0, Y0, ANCHO, ALTO)   ventana fija\n"
            f'    "misma"                 el rectangulo de la otra\n')
    return Roi(x0, y0, ancho, alto)


def main():
    holo = cv.imread(str(pathlib.Path(RUTA)), cv.IMREAD_GRAYSCALE)
    obj = cv.imread(str(pathlib.Path(REFERENCIA)), cv.IMREAD_GRAYSCALE)
    if holo is None:
        raise SystemExit(f"No pude leer el holograma:\n    {RUTA}")
    if obj is None:
        raise SystemExit(f"No pude leer la referencia:\n    {REFERENCIA}")
    holo = holo.astype(np.float64) / 255.0
    ref = obj.astype(np.float64) / 255.0

    xp, dev = elegir_dispositivo(DISPOSITIVO)
    dtype = DTYPE or (np.complex64 if dev == "gpu" else np.complex128)

    z0, z1 = float(Z[0]), float(Z[1])
    if min(z0, z1) <= 0:
        raise SystemExit(f"Z = {Z}: la distancia fuente-muestra es POSITIVA.")
    if max(z0, z1) >= L:
        raise SystemExit(
            f"Z = {Z} llega a {max(z0, z1)} mm y L = {L} mm. La muestra esta "
            f"ENTRE la fuente y el sensor:\nz < L siempre, o L - z se hace "
            f"negativa y la propagacion va al otro lado.")
    zs = np.linspace(z0, z1, PASOS)

    # --- recortes ---------------------------------------------------------
    h, r = ROI_HOLOGRAMA, ROI_REFERENCIA
    if h == "misma" and r == "misma":
        raise SystemExit(
            'ROI_HOLOGRAMA y ROI_REFERENCIA valen las dos "misma" y no hay de '
            'donde copiar.')
    T_H = "holograma: arrastra la ventana"
    T_R = "referencia: arrastra la zona del objeto"
    if h == "misma":
        roi_h = roi_r = _roi_de(r, ref, T_R, "ROI_REFERENCIA")
    elif r == "misma":
        roi_h = roi_r = _roi_de(h, holo, T_H, "ROI_HOLOGRAMA")
    else:
        roi_h = _roi_de(h, holo, T_H, "ROI_HOLOGRAMA")
        roi_r = _roi_de(r, ref, T_R, "ROI_REFERENCIA")

    forma_original = holo.shape
    if roi_h is not None:
        holo = roi_h.recortar(holo)
    if roi_r is not None:
        ref = roi_r.recortar(ref)

    print(f"holograma  {RUTA}")
    print(f"referencia {REFERENCIA}")
    print(f"  lambda {LAMB * 1e6:.1f} nm | delta {DELTA * 1e3:.3f} um | "
          f"L = {L} mm")
    print(f"  holograma {holo.shape}, referencia {ref.shape}")
    print(f"  {len(zs)} distancias de {zs[0]:.3f} a {zs[-1]:.3f} mm  "
          f"(M = L/z de {L / zs[-1]:.2f} a {L / zs[0]:.2f})")
    if roi_h is not None:
        print(informe(roi_h, forma_original, zs, LAMB, DELTA))
    print(f"  dispositivo {dev.upper()} | dtype {np.dtype(dtype).name} | "
          f"fase en float64")
    if dev == "gpu":
        libre, total = cp.cuda.runtime.memGetInfo()
        print(f"  {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, "
              f"{libre / 2**30:.2f} de {total / 2**30:.2f} GB libres")
    comprobar_memoria(holo, zs, LAMB, DELTA, MEMORIA_MAX_GB, xp, dtype)

    eq = comprobar_equivalencia(xp, dtype)
    print(f"  reconstruir() vs la cadena intacta de dlhm.py: {eq:.2e}")
    print()

    # --- barrido ----------------------------------------------------------
    curva = np.empty(len(zs))
    mejor_Rec, mejor_ref, mejor = None, None, -1

    sincronizar(xp)
    t0 = time.perf_counter()
    for i, z in enumerate(zs):
        Rec = reconstruir(holo, z, LAMB, DELTA, L, xp=xp, dtype=dtype)
        obs = observable(Rec, COMPARAR)
        del Rec
        # La referencia se reescala EN CADA PASO: mover z mueve M = L/z, o sea
        # el foco y la escala a la vez. Asi la correlacion solo sube cuando
        # coinciden las dos.
        #
        # El reescalado es cv.resize, o sea CPU, y la escala cambia en cada
        # paso: no hay nada que subir una sola vez. Lo que sube es el
        # resultado, que es del tamano de la reconstruccion.
        r_esc = referencia_a_escala(ref, L / z, obs.shape)
        curva[i] = correlacion(obs, xp.asarray(r_esc))
        # No se acumulan las reconstrucciones: son mallas de varios miles de
        # lado. Se guarda solo la mejor hasta ahora.
        if mejor < 0 or abs(curva[i]) > abs(curva[mejor]):
            mejor_Rec, mejor_ref, mejor = obs, r_esc, i
        elif obs is not mejor_Rec:
            del obs
        print(f"  z = {z:6.3f} mm   M = {L / z:5.2f}   corr = {curva[i]:+.4f}")
    sincronizar(xp)
    t = time.perf_counter() - t0

    # La mejor reconstruccion baja UNA vez, al final: lo que queda es
    # dibujarla. mejor_ref nunca subio, es de cv.resize.
    mejor_Rec = a_cpu(mejor_Rec)
    liberar(xp)

    print(f"\n{len(zs)} pasos en {t:.2f} s "
          f"({t / len(zs) * 1e3:.0f} ms/paso) en {dev.upper()}")
    print(f"enfoca en z = {zs[mejor]:.3f} mm   M = {L / zs[mejor]:.2f}   "
          f"corr = {curva[mejor]:+.4f}")
    if curva[mejor] < 0:
        print("  (la correlacion es NEGATIVA y eso es lo esperado: el objeto "
              "del modelo es\n  exp(-i*phi*I), asi que su fase corre al reves "
              "que los grises de la referencia.)")
    if mejor in (0, len(zs) - 1):
        print("  AVISO: el maximo cae en un EXTREMO del barrido, que por tanto "
              "NO acota el foco.\n  Ensancha Z.")

    # --- figura -----------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(18.5, 4.8))
    ax[0].imshow(mejor_ref, cmap="gray")
    ax[0].set_title(f"referencia a escala M = {L / zs[mejor]:.2f}", fontsize=10)
    ax[1].imshow(holo, cmap="gray")
    ax[1].set_title("holograma (entrada)", fontsize=10)
    ax[2].imshow(mejor_Rec, cmap="gray")
    ax[2].set_title(f"reconstruccion ({COMPARAR}) a z = {zs[mejor]:.3f} mm\n"
                    f"corr = {curva[mejor]:+.4f}", fontsize=10)
    for a in ax[:3]:
        a.set_xticks([])
        a.set_yticks([])
    ax[3].plot(zs, curva, "o-", ms=3)
    ax[3].axvline(zs[mejor], color="r", ls="--", lw=1,
                  label=f"z = {zs[mejor]:.3f} mm")
    ax[3].axhline(0, color="0.7", lw=0.8)
    ax[3].set_xlabel("distancia fuente-muestra z [mm]")
    ax[3].set_ylabel(f"correlacion ({COMPARAR}) con la referencia")
    ax[3].set_title("donde enfoca", fontsize=10)
    ax[3].legend(fontsize=9)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
