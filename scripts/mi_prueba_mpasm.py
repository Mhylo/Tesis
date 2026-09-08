"""Barrido puntuado contra una referencia, con MPASM.

El hermano de scripts/mi_prueba_angular.py y de mi_prueba_blas.py. Misma
geometria -onda plana-, mismas constantes, mismo barrido puntuado, mismo
recorte de dos ventanas. Solo cambia el propagador.

MPASM es el espectro angular por PRODUCTO MATRICIAL (Zhao 2020). No usa FFT
para la transformada inversa: la evalua sobre una malla de salida explicita, y
de ahi salen sus dos rasgos propios.

    Kf, el coeficiente de compresion frecuencial. El barrido lo imprime en cada
    paso, y NO es la constante KF: KF es lo que se le PIDE -None se lo deja
    calcular a el- y Kf es lo que USA. Por encima de 1.0 esta comprimiendo el
    espectro. Subir S aleja ese umbral, aunque parezca lo contrario: s entra
    bajo raiz en el DENOMINADOR de Kf.

    OJO, Y ESTO ESTA MEDIDO: la correlacion que no llega a 1.00 NO es culpa de
    esa compresion. Sobre el holograma de resultados/hologramas/entrada/mpasm/
    se probo S = 1, 6 y 12 -o sea Kf de 2.2799 a 1.0000- y la correlacion se
    queda en 0.8903 en los tres, con la energia en el sensor clavada en 74.9%.
    Lo que se pierde es el 25% que MPASM tira FUERA DE SU VENTANA DE SALIDA al
    evaluar la transformada inversa en una malla explicita; FFT-ASM envuelve esa
    energia en circulo y por eso si es exactamente invertible (1.0000, 100%).
    Es condiciones de contorno, no compresion. Agrandar la ventana con R lo
    recupera: R = 2 sube al 92.9% y R = 4 al 98.2%.

    MAG y R, que agrandan la ventana de observacion sin tocar el plano de
    entrada. Es lo unico que ninguno de los otros dos propagadores sabe hacer.

OJO A LA MEMORIA: la matriz espectral es (S*M, S*N) POR DISTANCIA. En un
barrido de 25 pasos, subir S de 1 a 4 multiplica por 16 la memoria de cada
paso. El defecto de aqui es S = 1 por eso.

NO HAY GUARDA DE PROPORCION, al reves que en mi_prueba_angular.py. Alli
angularSpectrum lleva los ejes CRUZADOS y deshacer ese cruce obliga a que el
recorte conserve la razon del marco. mpasm_bloques los tiene en su sitio y
ademas calcula Kf POR EJE, asi que aqui cualquier forma de ventana vale.

DE DONDE SALE EL PROPAGADOR. kf_paper, fasores, comprobar_ventana y
mpasm_bloques estan copiadas de scripts/retro_mpasm.py, que es la version de
trabajo del repo, contrastada alli contra el MatrixDftCPU de Zhao.

QUE LE ENTREGAS. La extension de RUTA decide: un .npy trae el CAMPO COMPLEJO
-con fase- y cualquier imagen es INTENSIDAD medida -con imagen gemela-.

Las distancias de Z van POSITIVAS: el menos lo pone el barrido.

UNIDADES: milimetros para todo.  633 nm -> 633e-6    3.45 um -> 3.45e-3
"""
import pathlib
import time

import numpy as np
from PIL import Image

try:
    import cupy as cp
except Exception:                      # sin CuPy, sin CUDA, o CuPy roto
    cp = None

import matplotlib.pyplot as plt

# POR QUE ESTE IMPORT SI, Y EL DEL PROPAGADOR NO. Este script no importa el
# propagador de CamposT a proposito: lleva angularSpectrum copiada tal cual de
# la implementacion de referencia, y si usara la del paquete, coincidir con el
# paquete no probaria nada. Esa independencia es sobre PROPAGADORES.
#
# roi.py no propaga: recorta un rectangulo. No valida ninguna cuenta de este
# archivo, asi que importarlo no compromete nada, y ya viene con 34 pruebas
# detras. Duplicar Roi + elegir + informe serian ~90 lineas en un script de
# 250: la copia dominaria el archivo que pretende ilustrar un propagador.
from CamposT.roi import Roi, elegir, informe

# ════════════════════════════════════════════════════════════════════════════
#  1. PARAMETROS  --  es lo unico que hay que editar
# ════════════════════════════════════════════════════════════════════════════

#: El HOLOGRAMA (imagen de intensidad), no un objeto. Usa barras normales o
#: antepon r a las comillas para que \U no se lea como escape.
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\hologramas\BenchmarkTarget\fft\z0010.000.npy"

#: Longitud de onda [mm]. 633 nm se escribe 633e-6.
LAMB = 633e-6

#: Paso de pixel de TU sensor [mm]. 3.45 um se escribe 3.45e-3.
DELTA = 3.45e-3

#: La IMAGEN DE REFERENCIA: el objeto que produjo el holograma, la verdad de
#: terreno contra la que se puntua cada distancia del barrido. Sale de la clave
#: "objeto" del .txt que acompana al holograma.
#:
#: Tiene que tener la MISMA FORMA que el holograma. Si no, el script aborta en
#: vez de reescalar: reescalar daria una correlacion perfectamente calculable y
#: sin sentido, porque compararia dos muestreos distintos del mismo objeto.
REFERENCIA = r"C:\Users\User\Desktop\Tesis\referencia\carlos\DLHM-model-main\DLHM-model-main\data\BenchmarkTarget.png"

#: COMO SE LEE LA IMAGEN DE RUTA. "intensidad" toma sqrt(I), que es lo correcto
#: para un holograma; "amplitud" la toma tal cual, que es lo que hacia antes
#: este script. Corre las dos y mira cuanto se mueve el pico: esa diferencia es
#: lo que cuesta interpretar mal la imagen.
ENTRADA = "intensidad"

#: Barrido de distancias holograma <-> objeto [mm], POSITIVAS: el menos lo pone
#: el barrido. Para una sola distancia, pon los dos extremos iguales y PASOS = 1.
#:
#: EL PASO IMPORTA MAS DE LO QUE PARECE, porque el foco es AGUDO. Medido sobre
#: el holograma que trae RUTA por defecto, desde su .npy:
#:
#:      z [mm]    9.80    9.90   10.00   10.10   10.20
#:      corr     0.8188  0.9648  1.0000  0.9648  0.8188
#:
#: o sea que equivocarse 0.2 mm cuesta 18 puntos de correlacion.
#:
#: Con (5.0, 20.0) y 30 pasos el paso es 0.517 mm, y esa rejilla NI SIQUIERA
#: muestrea 10.000: cae en 9.655 y en 10.172. Por eso el barrido reporta 0.9418
#: y no 1.0000 aunque el archivo lleve la fase intacta. No es que no enfoque:
#: es que pasa de largo por encima del foco.
#:
#: ASI QUE VA EN DOS PASADAS:
#:
#:   1. esta, ancha, para localizar la zona.
#:   2. una estrecha alrededor del pico que salga. Para este holograma,
#:      Z = (9.5, 10.5) con PASOS = 21 da un paso de 0.05 mm y SI contiene
#:      10.000 exacto.
#:
#: Subir PASOS en la ancha no es la salida: cada paso son ~1.6 s sobre esta
#: malla de 3000x4000, asi que la pasada ancha con 300 pasos serian 8 minutos
#: para lo que la estrecha resuelve en 35 segundos.
Z = (5.0, 20.0)
PASOS = 30

#: QUE HACER SI LA REFERENCIA Y EL HOLOGRAMA NO TIENEN LA MISMA FORMA.
#:
#:   False   aborta diciendo las dos formas. Es lo seguro y el defecto.
#:   True    RECORTA la mayor, centrada, hasta la forma comun.
#:
#: Se recorta y NO se reescala: un resize pasa cada pixel por un filtro de
#: interpolacion, y un recorte no toca ninguno de los que sobreviven. Tampoco
#: se rellena con ceros: meterian negro en la correlacion y falsearian el
#: numero que la metrica existe para dar.
#:
#: LO QUE ESTO NO ARREGLA, y es lo importante: reconcilia el CAMPO DE VISION y
#: nada mas. Si las dos imagenes vienen de geometrias distintas -otra lambda,
#: otro delta, una fuente divergente contra onda plana, un objeto de fase
#: contra uno de amplitud- la correlacion seguira sin significar nada aunque
#: las formas casen. El script lo avisa cada vez que recorta.
AJUSTAR_FORMA = True

#: Sobremuestreo del espectro de MPASM. La matriz espectral es (S*M, S*N) POR
#: DISTANCIA: en un barrido, subirlo multiplica la memoria de cada paso. Va al
#: reves de lo que parece: s entra bajo raiz en el DENOMINADOR de Kf, asi que
#: subirlo ALEJA el umbral de compresion.
S = 1

#: Puntos del plano de salida (R) y razon entre el paso de salida y el de
#: entrada (MAG). Con MAG > 1 la ventana de observacion se agranda sin tocar el
#: plano de entrada, que es lo que solo MPASM sabe hacer.
R = 1
MAG = 1.0

#: Kf que se le PIDE. None se lo deja calcular a el, y es lo normal. No
#: confundir con el Kf que USA, que el barrido imprime en cada paso.
KF = None

#: Filas por bloque al evaluar los fasores. NO cambia el resultado, solo la
#: memoria de pico.
FILAS_POR_BLOQUE = 512

#: EL RECORTE, una ventana por imagen. Las dos aceptan los mismos valores:
#:
#:     None                    sin recorte en ESA imagen.
#:     True                    la arrastras con el raton SOBRE ESA imagen.
#:     (X0, Y0, ANCHO, ALTO)   ventana fija.
#:     "misma"                 copia el rectangulo que resolvio LA OTRA.
#:
#: "misma" evita teclear el mismo rectangulo dos veces, o arrastrarlo dos
#: veces. Solo una de las dos puede llevarlo: si las dos dicen "misma", no hay
#: de donde copiar y aborta.
#:
#: LOS TRES CASOS UTILES:
#:
#:   ROI_HOLOGRAMA = "misma";  ROI_REFERENCIA = True
#:       arrastras sobre la REFERENCIA -que es la legible: un holograma a 10 mm
#:       es un patron de franjas- y el mismo rectangulo va a las dos.
#:
#:   ROI_HOLOGRAMA = True;     ROI_REFERENCIA = "misma"
#:       al reves, arrastras sobre el holograma.
#:
#:   ROI_HOLOGRAMA  = (100, 200, 1000, 750)
#:   ROI_REFERENCIA = (340, 180, 1000, 750)
#:       independientes: MISMO TAMANO, distinta posicion. Es lo que sirve
#:       cuando los dos planos estan descentrados entre si.
#:
#: EL TAMANO FINAL TIENE QUE COINCIDIR o aborta: no se reescala ninguna, por lo
#: mismo que no se reescala en AJUSTAR_FORMA.
#:
#: LA PROPORCION solo se le exige a la ventana que toca el HOLOGRAMA, porque
#: QUE GANAS: 1000x750 frente a 3000x4000 es el 6% de los pixeles, y el barrido
#: fino de 21 pasos baja de 74 s a 4.4 s.
#:
#: QUE PIERDES: el recorte es SECO, sin margen de guarda. Medido con esa misma
#: ventana, la correlacion en el foco cae de 1.0000 a 0.9585.
ROI_HOLOGRAMA = "misma"
ROI_REFERENCIA = True

#: DISPOSITIVO de calculo: "auto" (la GPU si la hay), "cpu" o "gpu".
#:
#: MPASM es el de la familia que MAS gana con la tarjeta y el que antes se
#: queda sin memoria, y por la misma razon: su nucleo son dos productos de
#: matrices densas -My @ U @ Mx- en vez de una FFT. El producto es justo lo que
#: una GPU hace bien, y la matriz espectral (S*M, S*N) es justo lo que no cabe.
#: Sube S solo despues de mirar lo que dice comprobar_memoria().
#:
#: Con "gpu" y sin CUDA ABORTA en vez de caer a CPU en silencio: un tiempo
#: medido en el dispositivo equivocado no dice nada.
DISPOSITIVO = "auto"

#: dtype de trabajo. None = complex64 en GPU, complex128 en CPU.
#:
#: Es la politica de CamposT.backend, y en MPASM es la que decide si el barrido
#: cabe: la matriz espectral ocupa la mitad en complex64. Las FASES siguen
#: evaluandose en float64 dentro de fasores() y del kernel, asi que lo que baja
#: a simple es el fasor ya acotado a modulo 1, nunca el argumento -que aqui
#: llega a 2e5 rad y en float32 seria ruido-.
DTYPE = None



# ════════════════════════════════════════════════════════════════════════════
#  2. RETROPROPAGADOR  --  el nucleo. Si borras algo de aqui, no queda script.
# ════════════════════════════════════════════════════════════════════════════

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
#
# elegir_dispositivo, a_cpu y liberar son copias literales de las de
# scripts/retro_mpasm.py. sincronizar() no: esa es CamposT.backend.sincronizar()
# sin el argumento opcional. comprobar_memoria tampoco, y ahi divergir es lo
# correcto: aqui cuenta la matriz espectral de MPASM, que es lo que manda en
# este propagador. tests/test_gpu_mi_prueba.py fija que las tres primeras sigan
# a retro_fft_angular y que sincronizar sea la misma en toda la familia.

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


def memoria_mpasm(M, N, s, r, dtype):
    """Bytes de pico que pide mpasm_bloques() sobre una malla (M, N).

    NO es el tamano de la malla, y esa es toda la diferencia con los otros
    scripts de la familia. Lo que manda es la matriz espectral F, que es
    (s*M, s*N): CUADRATICA en s. Con s = 4 sobre 1024x1024 son 4096x4096, o
    sea 16 veces la entrada.

    Se cuentan, en el momento de mas ocupacion -el primer producto-:

        U   M*N          el campo de entrada
        Mx  N*(s*N)      fasor de columnas
        My  (s*M)*M      fasor de filas
        F   (s*M)*(s*N)  la matriz espectral

    Los fasores de vuelta (Mx1, My1) se construyen despues de que `del Mx, My`
    haya soltado los de ida, asi que no se suman: entran en su sitio. La
    salida (r*M, r*N) si, porque convive con F.
    """
    elementos = M * N + N * (s * N) + (s * M) * M + (s * M) * (s * N) \
        + (r * M) * (r * N)
    return elementos * np.dtype(dtype).itemsize


def comprobar_memoria(xp, M, N, s, r, dtype):
    """Aborta con un mensaje util en vez de con un OOM de CUDA.

    Con S alto el mensaje importa mas que en los otros scripts: la palanca no
    es solo recortar, es BAJAR S, y no es obvio cual toca.
    """
    if xp is not cp:
        return
    libre, _ = cp.cuda.runtime.memGetInfo()
    hacen_falta = memoria_mpasm(M, N, s, r, dtype)
    if hacen_falta > 0.85 * libre:
        cabe = ""
        for s_ok in range(s - 1, 0, -1):
            if memoria_mpasm(M, N, s_ok, r, dtype) <= 0.85 * libre:
                cabe = f" Con S = {s_ok} si cabria."
                break
        raise SystemExit(
            f"No cabe en la GPU: la malla {M}x{N} con S = {s} pide una matriz "
            f"espectral de {s * M}x{s * N},\n~{hacen_falta / 2**30:.2f} GB en "
            f"{np.dtype(dtype).name}, y hay {libre / 2**30:.2f} GB libres."
            f"{cabe}\nBaja S, recorta con ROI_HOLOGRAMA, o usa "
            f"DISPOSITIVO = 'cpu'.")


def comprobar_equivalencia(xp, dtype):
    """mpasm_bloques() en el dispositivo contra ella misma en CPU doble.

    Es la prueba de que acelerar no cambio el resultado, y se corre en cada
    invocacion porque cuesta milisegundos.

    POR QUE CONTRA SI MISMA. La referencia de este propagador -el MatrixDftCPU
    de Zhao- vive en scripts/retro_mpasm.py, que la lleva copiada tal cual y la
    contrasta alli. Este script no la duplica: lo que aqui hace falta fijar es
    que el dispositivo y el dtype no muevan el resultado.

    Devuelve (error relativo del campo, diferencia de Kf). La segunda tiene que
    salir 0.0 exacto: kf_paper() es aritmetica de escalares en float64 y no
    toca el dispositivo.
    """
    rng = np.random.default_rng(0)
    U = rng.random((128, 128)) + 1j * rng.random((128, 128))
    ref, kf_ref = mpasm_bloques(U, -7.0, LAMB, DELTA, s=2,
                                xp=np, dtype=np.complex128)
    rap, kf_rap = mpasm_bloques(U, -7.0, LAMB, DELTA, s=2,
                                xp=xp, dtype=dtype)
    err = float(np.max(np.abs(a_cpu(rap) - ref)) / np.max(np.abs(ref)))
    return err, abs(kf_rap - kf_ref)


# ════════════════════════════════════════════════════════════════════════════
#  3. BARRIDO CON REFERENCIA
# ════════════════════════════════════════════════════════════════════════════

def campo_de_entrada(ruta, entrada):
    """Archivo -> (campo complejo, mapa para pintar, etiqueta de que es).

    LA EXTENSION DECIDE, y no es comodidad: son dos experimentos distintos.

      .npy   el CAMPO COMPLEJO tal cual, CON SU FASE. La vuelta deshace la ida
             y devuelve el objeto exacto. Medido sobre el holograma que trae
             RUTA por defecto: correlacion 1.0000 contra el objeto. NO es lo
             que entrega un sensor.

      resto  una IMAGEN, o sea INTENSIDAD medida. La fase se perdio al medir,
             asi que la reconstruccion sale con la IMAGEN GEMELA desenfocada
             encima. Medido: 0.8618. Y los 8 bits del PNG cuestan encima de eso
             solo 0.0003, asi que toda la perdida es la fase, no el formato.

    Esos dos numeros son la razon de que esta funcion mire la extension. Si la
    reconstruccion "no enfoca" desde un PNG, no es el propagador: es que el
    archivo ya no lleva la fase. Ningun propagador la devuelve.

    ENTRADA solo aplica a la rama de imagen. Con .npy se ignora y se avisa: un
    campo complejo ya es un campo, no hay nada que interpretar.

    El mapa para pintar se devuelve aparte porque no siempre es |campo|^2: con
    ENTRADA = "amplitud" eso seria I^2 y no la imagen que abriste.
    """
    ruta = pathlib.Path(ruta)
    if ruta.suffix.lower() == ".npy":
        U = np.load(ruta)
        if not np.iscomplexobj(U):
            raise SystemExit(
                f"{ruta.name} es un .npy REAL, no un campo complejo. Si es una "
                f"intensidad guardada en .npy, guardala como imagen o toma tu "
                f"la raiz antes: aqui un .npy se interpreta siempre como el "
                f"campo, y tratarlo como intensidad seria una conversion muda.")
        if U.ndim != 2:
            raise SystemExit(f"{ruta.name} tiene forma {U.shape} y hace falta "
                             f"un array 2D (M, N).")
        if entrada != "intensidad":
            print(f'  AVISO: ENTRADA = "{entrada}" se ignora con un .npy. Un '
                  f'campo complejo ya es un campo.')
        U = np.asarray(U)
        return U, np.abs(U) ** 2, "campo complejo con fase (vuelta exacta, sin gemela)"

    img = np.asarray(Image.open(ruta).convert("L"), dtype=np.float64) / 255.0
    if entrada == "intensidad":
        return (np.sqrt(img).astype(complex), img,
                "intensidad medida, campo = sqrt(I) (con gemela)")
    if entrada == "amplitud":
        return img.astype(complex), img, "la imagen ES la amplitud, campo = I"
    raise SystemExit(f'ENTRADA = "{entrada}" no existe. '
                     'Pon "intensidad" o "amplitud".')


def correlacion(a, b):
    """Correlacion de Pearson entre dos mapas reales.

    Se usa esta y no un rms porque el PNG del holograma se guardo normalizado
    por su maximo: la escala absoluta ya no significa nada. La correlacion es
    invariante a escala y a desplazamiento; el rms exigiria recalibrar el brillo
    primero, que seria un parametro mas que ajustar a ojo -justo lo que el
    barrido puntuado existe para quitar-.

    Un mapa constante no tiene con que correlacionar: se devuelve 0 en vez de
    dividir por cero y sacar un NaN que viajaria hasta el pico del barrido.
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


def _sidecar(ruta):
    """El .txt que acompana al holograma, si existe. -> dict, o None.

    OJO AL NOMBRE. Para pathlib, "z0010.000.npy" tiene sufijo ".npy", y al
    quitarlo queda "z0010.000", cuyo sufijo APARENTE es ".000". Por eso la
    extension se corta a mano en vez de con with_suffix(), que se comeria los
    tres decimales. Es el mismo fallo que ya mordio al escribir estos archivos.
    """
    ruta = pathlib.Path(ruta)
    if not ruta.suffix:
        return None
    lado = pathlib.Path(str(ruta)[:-len(ruta.suffix)] + ".txt")
    if not lado.is_file():
        return None
    datos = {}
    for linea in lado.read_text(encoding="utf-8").splitlines():
        if " = " in linea:
            clave, valor = linea.split(" = ", 1)
            datos[clave.strip()] = valor.strip()
    return datos


def _comprobar_con_el_sidecar(ruta, lamb, delta):
    """Si el holograma trae .txt, su lambda y su delta tienen que ser los tuyos.

    Hoy nada impide reconstruir un holograma de 633 nm con LAMB = 532e-6: el
    barrido devolveria una z perfectamente creible y equivocada, porque lambda
    entra en la fase del propagador. Cuando el archivo DICE con que se hizo, no
    hay ninguna razon para adivinarlo.

    Los hologramas que escribe scripts/retro_fft_angular.py traen ese .txt. Los
    de terceros no, y por eso esta guarda no puede ser la unica: ver el aviso
    de _ajustar_forma().
    """
    datos = _sidecar(ruta)
    if datos is None:
        return
    malos = []
    for clave, mio in (("lambda [mm]", lamb), ("delta [mm]", delta)):
        suyo = datos.get(clave)
        if suyo is not None and not np.isclose(float(suyo), mio, rtol=1e-9):
            malos.append(f"    {clave}: el archivo dice {suyo}, tu pusiste {mio}")
    if malos:
        raise SystemExit(
            "El .txt del holograma no cuadra con tus constantes:\n"
            + "\n".join(malos)
            + "\n\nCon otra lambda o otro delta el barrido devuelve una z "
              "creible y equivocada,\nporque las dos entran en la fase del "
              "propagador. Corrige las constantes.")


def _recorte_centrado(a, M, N):
    """Los M x N centrales de a. Sin interpolar: cada pixel que queda es el
    mismo pixel que habia."""
    y0, x0 = (a.shape[0] - M) // 2, (a.shape[1] - N) // 2
    return a[y0:y0 + M, x0:x0 + N]


def _ajustar_forma(campo, img, ref, ajustar):
    """Deja holograma y referencia con la misma forma, o aborta.

    -> (campo, img, ref) ya recortados.

    RECORTA la mayor, centrada. No reescala, porque lo que se pidio fue ponerlas
    a la misma escala SIN que la imagen sufra, y un resize pasa cada pixel por
    un filtro de interpolacion. Un recorte no toca ninguno de los que quedan.

    Si el recortado acaba siendo el HOLOGRAMA, se le exige la proporcion del
    marco: ahi el recorte cambia la fisica, porque angularSpectrum lleva los
    ejes cruzados. Sobre la referencia no se exige nada: es solo el blanco
    contra el que se puntua.

    LO QUE NO ARREGLA: solo el campo de vision. Dos imagenes de geometrias
    distintas seguiran dando una correlacion sin sentido con las formas ya
    casadas, y eso no se puede ver en los pixeles. Por eso avisa siempre.
    """
    if campo.shape == ref.shape:
        return campo, img, ref
    if not ajustar:
        raise SystemExit(
            f"La referencia y el holograma no tienen la misma forma:\n"
            f"    holograma  {campo.shape}\n"
            f"    referencia {ref.shape}\n\n"
            f"Pon AJUSTAR_FORMA = True para recortar la mayor, centrada, hasta "
            f"la forma comun.\nEso reconcilia el campo de vision y NADA MAS: "
            f"si vienen de geometrias\ndistintas, la correlacion seguira sin "
            f"significar nada.")

    M = min(campo.shape[0], ref.shape[0])
    N = min(campo.shape[1], ref.shape[1])
    if (M, N) != campo.shape:
        campo = _recorte_centrado(campo, M, N)
        img = _recorte_centrado(img, M, N)
        print(f"  holograma recortado, centrado, a {N}x{M}.")
    if (M, N) != ref.shape:
        ref = _recorte_centrado(ref, M, N)
        print(f"  referencia recortada, centrada, a {N}x{M}.")
    print("  AVISO: eso reconcilia el CAMPO DE VISION y nada mas. Si las dos "
          "vienen de geometrias\n  distintas -otra lambda, otro delta, fuente "
          "divergente contra onda plana, objeto\n  de fase contra objeto de "
          "amplitud- la correlacion seguira sin significar nada.")
    return campo, img, ref


def _roi_fija(valor, nombre):
    """La constante ROI cuando trae coordenadas -> Roi, o un error que ENSENA.

    Sin esto, ROI = True daba un "TypeError: argument after * must be an
    iterable, not bool" de Python crudo. El nombre ROI se lee como "la
    funcion", asi que activarla escribiendo True es lo primero que uno intenta,
    y el mensaje tiene que decir QUE ESCRIBIR, no solo que algo fallo.
    """
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


def _resolver_ventanas(forma, campo_pinta, ref):
    """Las dos constantes -> (roi del holograma, roi de la referencia).

    "misma" copia el rectangulo que resolvio LA OTRA, para no teclear ni
    arrastrar dos veces lo mismo. Solo una puede llevarlo.

    AQUI NO SE EXIGE NINGUNA PROPORCION. Esa guarda existe en
    mi_prueba_angular.py porque su angularSpectrum lleva los ejes CRUZADOS;
    mpasm_bloques los tiene en su sitio y calcula Kf POR EJE, asi que cualquier
    forma de ventana vale.
    """
    h, r = ROI_HOLOGRAMA, ROI_REFERENCIA
    if h == "misma" and r == "misma":
        raise SystemExit(
            'ROI_HOLOGRAMA y ROI_REFERENCIA valen las dos "misma", y entonces '
            'no hay de donde copiar.\nPon una de las dos a None, a True, o a '
            'un (X0, Y0, ANCHO, ALTO).')

    T_H = "holograma: arrastra la ventana"
    T_R = "referencia: arrastra lo que quieres reconstruir"

    def resolver(valor, imagen, titulo, nombre, toca_holograma):
        if valor is None or valor is False:
            return None
        if valor is True:
            return elegir(imagen, titulo)
        return _roi_fija(valor, nombre)

    if h == "misma":
        roi = resolver(r, ref, T_R, "ROI_REFERENCIA", toca_holograma=True)
        return roi, roi
    if r == "misma":
        roi = resolver(h, campo_pinta, T_H, "ROI_HOLOGRAMA", toca_holograma=True)
        return roi, roi
    return (resolver(h, campo_pinta, T_H, "ROI_HOLOGRAMA", True),
            resolver(r, ref, T_R, "ROI_REFERENCIA", False))


def main():
    # Antes de nada: si el holograma trae .txt, que sus parametros sean los
    # tuyos. Reconstruir con otra lambda da una z creible y equivocada.
    _comprobar_con_el_sidecar(RUTA, LAMB, DELTA)

    xp, dev = elegir_dispositivo(DISPOSITIVO)
    dtype = DTYPE or (np.complex64 if dev == "gpu" else np.complex128)

    ref = np.asarray(Image.open(REFERENCIA).convert("L"), dtype=np.float64) / 255.0
    campo, img, etiqueta = campo_de_entrada(RUTA, ENTRADA)

    campo, img, ref = _ajustar_forma(campo, img, ref, AJUSTAR_FORMA)

    if min(Z) <= 0:
        raise SystemExit(f"Z = {Z} y son distancias holograma-objeto, "
                         f"POSITIVAS: el menos lo pone el barrido.")
    zs = np.linspace(float(Z[0]), float(Z[1]), PASOS)

    # El recorte va DESPUES de comprobar que las formas completas casan: si no
    # casan, el mensaje util es "estas dos imagenes no son del mismo objeto", no
    # un fallo de recorte. Y va ANTES del barrido, porque recortar es lo que lo
    # abarata.
    roi_h, roi_r = _resolver_ventanas(campo.shape, img, ref)
    if roi_h is not None or roi_r is not None:
        forma = campo.shape
        if roi_h is not None:
            campo = roi_h.recortar(campo)
            img = roi_h.recortar(img)
        if roi_r is not None:
            ref = roi_r.recortar(ref)
        # Las dos ventanas pueden estar en posiciones distintas -esa es la
        # gracia, sirve cuando los planos estan descentrados- pero el TAMANO
        # tiene que coincidir o no hay nada que correlacionar.
        if campo.shape != ref.shape:
            raise SystemExit(
                f"Las dos ventanas no dan el mismo tamano:\n"
                f"    holograma  {campo.shape}\n"
                f"    referencia {ref.shape}\n\n"
                f"No se reescala ninguna. Dales el mismo ANCHO y ALTO, o pon "
                f'una de las dos a "misma".')
        if roi_h is not None and roi_r is not None and roi_h != roi_r:
            print(f"  ventanas independientes: holograma en "
                  f"({roi_h.x0}, {roi_h.y0}), referencia en "
                  f"({roi_r.x0}, {roi_r.y0}), las dos {roi_h.ancho}x{roi_h.alto}.")
        if roi_h is not None:
            print(informe(roi_h, forma, zs, LAMB, DELTA))

    M, N = campo.shape
    print(f"holograma  {RUTA}")
    print(f"  {etiqueta}")
    print(f"referencia {REFERENCIA}")
    print(f"  malla {M}x{N} | lambda {LAMB * 1e6:.1f} nm | "
          f"delta {DELTA * 1e3:.3f} um")
    print(f"  {len(zs)} distancias de {zs[0]:.3f} a {zs[-1]:.3f} mm")
    if M != N:
        print("  malla RECTANGULAR, y aqui eso VALE: mpasm_bloques lleva "
              "cada eje con SU\n  longitud y calcula Kf POR EJE, quedandose "
              "con el menor de los dos. La guarda\n  de proporcion que exige "
              "mi_prueba_angular.py no hace falta en este script.")
    print(f"  dispositivo {dev.upper()} | dtype {np.dtype(dtype).name} | "
          f"fase en float64")
    print(f"  S = {S}: matriz espectral {S * M}x{S * N}, "
          f"~{memoria_mpasm(M, N, S, R, dtype) / 2**30:.2f} GB de pico")
    if dev == "gpu":
        libre, total = cp.cuda.runtime.memGetInfo()
        print(f"  {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, "
              f"{libre / 2**30:.2f} de {total / 2**30:.2f} GB libres")

    comprobar_memoria(xp, M, N, S, R, dtype)

    eq, dkf = comprobar_equivalencia(xp, dtype)
    print(f"  mpasm_bloques en {dev.upper()} vs CPU/complex128: "
          f"campo {eq:.2e}, Kf {dkf:.1e}")
    print()

    curva = np.empty(len(zs))
    # MPASM devuelve ademas el Kf que USO. No es lo mismo que KF, que es lo que
    # se le PIDE: con KF = None lo calcula el. Y por encima de 1.0 esta
    # COMPRIMIENDO el espectro, que es con perdida: una correlacion baja puede
    # ser eso y no un desenfoque.
    kfs = np.empty(len(zs))
    mejor_U, mejor = None, -1

    # La referencia sube UNA vez, no una por paso: son 96 MB y el barrido la
    # usa igual en los PASOS pasos. Bajar el campo a la CPU en cada paso solo
    # para correlacionar seria el cuello de botella del barrido en la tarjeta.
    ref_d = xp.asarray(ref)

    sincronizar(xp)
    t0 = time.perf_counter()
    for i, z in enumerate(zs):
        U, kf = mpasm_bloques(campo, -z, LAMB, DELTA, s=S, Kf=KF, r=R,
                              mag=MAG, xp=xp, dtype=dtype,
                              filas=FILAS_POR_BLOQUE)
        kfs[i] = kf if np.isscalar(kf) else max(kf)
        curva[i] = correlacion(xp.abs(U) ** 2, ref_d)
        # No se acumulan los campos: 30 mallas de 3000x4000 en complex128 son
        # 5.7 GB. Se guarda solo el mejor hasta ahora, que son 192 MB.
        if mejor < 0 or curva[i] > curva[mejor]:
            mejor_U, mejor = U, i
        # Se suelta la referencia al campo del paso salvo que sea el mejor.
        # NO se llama a liberar() aqui: el pool de CuPy reutiliza el bloque
        # del paso anterior, que es justo lo que hace baratos los pasos
        # siguientes. Vaciarlo cada vuelta obligaria a un cudaMalloc nuevo por
        # paso y el barrido saldria mas lento que en CPU.
        if mejor != i:
            del U
        print(f"  z = {z:8.3f} mm   corr = {curva[i]:+.4f}   "
              f"Kf = {kfs[i]:6.4f}"
              + ("   <- comprimiendo" if kfs[i] > 1.0 else ""))
    sincronizar(xp)
    t = time.perf_counter() - t0

    # El mejor campo baja UNA vez, al final: lo que queda es dibujarlo.
    mejor_U = a_cpu(mejor_U)
    del ref_d
    liberar(xp)

    print(f"\n{len(zs)} pasos en {t:.2f} s "
          f"({t / len(zs) * 1e3:.0f} ms/paso) en {dev.upper()}")
    print(f"enfoca en z = {zs[mejor]:.3f} mm   corr = {curva[mejor]:+.4f}   "
          f"Kf = {kfs[mejor]:.4f}")
    if mejor in (0, len(zs) - 1):
        print("  AVISO: el maximo cae en un EXTREMO del barrido, que por tanto "
              "NO acota el foco.\n  Ensancha Z.")

    fig, ax = plt.subplots(1, 4, figsize=(18.5, 4.8))
    ax[0].imshow(ref, cmap="gray")
    ax[0].set_title("referencia (el objeto)", fontsize=10)
    ax[1].imshow(img, cmap="gray")
    ax[1].set_title(f"holograma de entrada\n{ENTRADA}", fontsize=10)
    ax[2].imshow(np.abs(mejor_U) ** 2, cmap="gray")
    ax[2].set_title(f"reconstruccion a z = {zs[mejor]:.3f} mm\n"
                    f"corr = {curva[mejor]:+.4f}", fontsize=10)
    for a in ax[:3]:
        a.set_xticks([])
        a.set_yticks([])
    ax[3].plot(zs, curva, "o-", ms=3)
    ax[3].axvline(zs[mejor], color="r", ls="--", lw=1,
                  label=f"z = {zs[mejor]:.3f} mm")
    ax[3].set_xlabel("distancia de reconstruccion [mm]")
    ax[3].set_ylabel("correlacion con la referencia")
    ax[3].set_title("donde enfoca", fontsize=10)
    ax[3].legend(fontsize=9)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
