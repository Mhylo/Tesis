"""Barrido puntuado contra una referencia, con BL-ASM.

El hermano de scripts/mi_prueba_angular.py. Misma geometria -onda plana-,
mismas constantes, mismo barrido puntuado contra el objeto, mismo recorte de
dos ventanas. Lo unico que cambia es el propagador.

BL-ASM es el espectro angular con el LIMITE DE BANDA de Matsushima y Shimobaba
(2009). En vez de aliar las frecuencias que la malla ya no puede muestrear, las
CORTA. Eso tiene una consecuencia que hay que tener delante al leer cualquier
numero de aqui:

    BL-ASM NO ES LA IDENTIDAD. La vuelta no devuelve el objeto, devuelve el
    objeto FILTRADO. Ese es el precio de no aliar, y no es un fallo.

Por eso el barrido imprime, en cada z, QUE FRACCION del espectro dejo pasar la
mascara. No es un ajuste que puedas tocar: es un resultado que dice cuanta
banda se descarto. Una correlacion que no llega a 1.00 con este propagador
puede ser eso y no un desenfoque.

NO HAY GUARDA DE PROPORCION, al reves que en mi_prueba_angular.py. Alli
angularSpectrum lleva los ejes CRUZADOS -dfx con el numero de filas y dfy con
el de columnas-, y deshacer ese cruce obliga a que el recorte conserve la razon
del marco; con un 1.1% de desvio el foco desaparece. espectro_angular_bl tiene
los ejes en su sitio, asi que aqui cualquier forma de ventana vale.

DE DONDE SALE EL PROPAGADOR. espectro_angular_bl e indices_centrados estan
copiadas de scripts/retro_blas.py. No hay implementacion de referencia de
terceros contra la que contrastar BL-ASM -no hay ningun BL-ASM en
referencia/-, asi que la de retro_blas.py es la version de trabajo del repo.

QUE LE ENTREGAS. La extension de RUTA decide: un .npy trae el CAMPO COMPLEJO
-con fase- y cualquier imagen es INTENSIDAD medida -sin fase, sale con la
imagen gemela encima-.

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
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\hologramas\BenchmarkTarget\blas\z0200.000.npy"

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
Z = (0.1, 20.0)
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

#: Filas por bloque al evaluar el fasor. NO cambia el resultado, solo la
#: memoria de pico: el fasor se construye por trozos en vez de materializar la
#: malla entera de una vez.
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
#: El barrido son PASOS retropropagaciones de la misma malla, o sea PASOS
#: pares de FFT sobre el mismo array: es justo la forma de trabajo que la
#: tarjeta acelera. Con "gpu" y sin CUDA ABORTA en vez de caer a CPU en
#: silencio, porque un tiempo medido en el dispositivo equivocado no dice
#: nada, y es el tipo de error que solo se descubre al comparar tablas.
DISPOSITIVO = "auto"

#: dtype de trabajo. None = complex64 en GPU, complex128 en CPU.
#:
#: Es la politica de CamposT.backend. Las FASES se evaluan igualmente en
#: float64 dentro de espectro_angular_bl(), asi que lo que baja a simple es el
#: fasor ya acotado a modulo 1, nunca el argumento.
#:
#: OJO CON LO QUE complex64 SIGNIFICA AQUI. La mascara de banda decide con
#: searchsorted sobre fx y fy, que se construyen en float64 pasen lo que pase:
#: la FRACCION de banda que sobrevive no depende del dtype, y por eso se puede
#: comparar entre dispositivos. Lo unico que cambia es la mantisa del campo.
DTYPE = None



# ════════════════════════════════════════════════════════════════════════════
#  2. RETROPROPAGADOR  --  el nucleo. Si borras algo de aqui, no queda script.
# ════════════════════════════════════════════════════════════════════════════

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

    4. La rejilla se centra con indices_centrados() y no con arange(n) - n/2.
       Esta si es de algoritmo, pero solo se nota con lado IMPAR: en malla par
       las dos son la misma rejilla bit a bit. Ver alli el porque.

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
    fx = (indices_centrados(N) * dfx).astype(np.float64)
    fy = (indices_centrados(M) * dfy).astype(np.float64)

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
#
# elegir_dispositivo, a_cpu y liberar son copias literales de las de
# scripts/retro_blas.py, que es el hermano de este archivo. sincronizar() no:
# esa es CamposT.backend.sincronizar() sin el argumento opcional, porque aqui
# el xp siempre se sabe. tests/test_gpu_mi_prueba.py comprueba que ninguna de
# las cuatro diverja del resto de la familia.

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


def comprobar_memoria(xp, M, N, dtype):
    """Aborta con un mensaje util en vez de con un OOM de CUDA.

    Se cuentan cuatro arrays del tamano de la malla: el campo, el espectro
    desplazado, la salida de fft2 y el espacio de trabajo de cuFFT. El fasor
    no cuenta, que para eso va por bloques de filas.
    """
    if xp is not cp:
        return
    libre, _ = cp.cuda.runtime.memGetInfo()
    hacen_falta = 4 * M * N * np.dtype(dtype).itemsize
    if hacen_falta > 0.85 * libre:
        raise SystemExit(
            f"No cabe en la GPU: la malla {M}x{N} en "
            f"{np.dtype(dtype).name} pide ~{hacen_falta / 2**30:.2f} GB y hay "
            f"{libre / 2**30:.2f} GB libres.\nRecorta con ROI_HOLOGRAMA "
            f"(ver ahi lo que cuesta), o usa DISPOSITIVO = 'cpu'.")


def comprobar_equivalencia(xp, dtype):
    """espectro_angular_bl() en el dispositivo contra ella misma en CPU doble.

    Es la prueba de que acelerar no cambio el resultado, y se corre en cada
    invocacion porque cuesta milisegundos.

    POR QUE CONTRA SI MISMA Y NO CONTRA UNA REFERENCIA. Los hermanos de este
    script contrastan su version rapida contra la implementacion publicada de
    la que salen -angularSpectrum de pyDHM, MatrixDftCPU de Zhao-. Aqui no hay
    contra que: en referencia/ no hay ningun BL-ASM, y por eso este script
    mide la banda descartada en vez de correlacionar contra un tercero. Lo que
    si se puede fijar es que el dispositivo y el dtype no muevan el resultado,
    que es exactamente lo que esta funcion comprueba.

    Devuelve (error relativo del campo, diferencia de la fraccion de banda).
    La segunda tiene que salir 0.0 exacto: la mascara decide en float64 en los
    dos casos.
    """
    rng = np.random.default_rng(0)
    U = rng.random((256, 256)) + 1j * rng.random((256, 256))
    ref, frac_ref = espectro_angular_bl(U, -7.0, LAMB, DELTA, DELTA,
                                        xp=np, dtype=np.complex128)
    rap, frac_rap = espectro_angular_bl(U, -7.0, LAMB, DELTA, DELTA,
                                        xp=xp, dtype=dtype)
    err = float(np.max(np.abs(a_cpu(rap) - ref)) / np.max(np.abs(ref)))
    return err, abs(frac_rap - frac_ref)


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

    AQUI NO SE EXIGE NINGUNA PROPORCION, y esa es la diferencia con
    mi_prueba_angular.py. Alli angularSpectrum lleva los ejes CRUZADOS -dfx con
    el numero de filas y dfy con el de columnas- y deshacer ese cruce obliga a
    que el recorte conserve la razon M/N del marco. espectro_angular_bl los
    tiene en su sitio, asi que cualquier forma de ventana vale.
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
        print("  malla RECTANGULAR, y aqui eso VALE: espectro_angular_bl "
              "lleva cada eje con SU\n  longitud -dfx del numero de columnas, "
              "dfy del de filas- y el limite de banda\n  de Matsushima sale "
              "por eje. La guarda de proporcion que exige\n  "
              "mi_prueba_angular.py no hace falta en este script.")
    print(f"  dispositivo {dev.upper()} | dtype {np.dtype(dtype).name} | "
          f"fase en float64")
    if dev == "gpu":
        libre, total = cp.cuda.runtime.memGetInfo()
        print(f"  {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, "
              f"{libre / 2**30:.2f} de {total / 2**30:.2f} GB libres")

    comprobar_memoria(xp, M, N, dtype)

    eq, dfrac = comprobar_equivalencia(xp, dtype)
    print(f"  espectro_angular_bl en {dev.upper()} vs CPU/complex128: "
          f"campo {eq:.2e}, banda {dfrac:.1e}")
    print()

    curva = np.empty(len(zs))
    # BL-ASM devuelve ademas que FRACCION del espectro dejo pasar su mascara.
    # No es un ajuste, es un resultado: dice cuanta banda descarto para no
    # aliar, y por eso la vuelta no es la identidad sino el objeto FILTRADO.
    fracs = np.empty(len(zs))
    mejor_U, mejor = None, -1

    # La referencia sube UNA vez, no una por paso: son 96 MB y el barrido la
    # usa igual en los PASOS pasos. Bajar el campo a la CPU en cada paso solo
    # para correlacionar seria el cuello de botella del barrido en la tarjeta.
    ref_d = xp.asarray(ref)

    sincronizar(xp)
    t0 = time.perf_counter()
    for i, z in enumerate(zs):
        U, frac = espectro_angular_bl(campo, -z, LAMB, DELTA, DELTA,
                                      xp=xp, dtype=dtype,
                                      filas=FILAS_POR_BLOQUE)
        fracs[i] = frac
        curva[i] = correlacion(xp.abs(U) ** 2, ref_d)
        # No se acumulan los campos: 30 mallas de 3000x4000 en complex128 son
        # 5.7 GB. Se guarda solo el mejor hasta ahora, que son 192 MB.
        if mejor < 0 or curva[i] > curva[mejor]:
            mejor_U, mejor = U, i
        print(f"  z = {z:8.3f} mm   corr = {curva[i]:+.4f}   "
              f"banda que pasa = {100 * fracs[i]:5.1f}%")
    sincronizar(xp)
    t = time.perf_counter() - t0

    # El mejor campo baja UNA vez, al final: lo que queda es dibujarlo.
    mejor_U = a_cpu(mejor_U)
    del ref_d
    liberar(xp)

    print(f"\n{len(zs)} pasos en {t:.2f} s "
          f"({t / len(zs) * 1e3:.0f} ms/paso) en {dev.upper()}")
    print(f"enfoca en z = {zs[mejor]:.3f} mm   corr = {curva[mejor]:+.4f}   "
          f"banda {100 * fracs[mejor]:.1f}%")
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
