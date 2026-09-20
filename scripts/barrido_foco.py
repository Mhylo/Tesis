"""Holograma de una imagen con scripts/mpasm_minimo.py, y barrido de
retropropagacion para encontrar la z de foco.

Propaga la imagen a Z_REAL, retropropaga el holograma a cada z del barrido y
mide CADA reconstruccion con tres metricas: una SIN referencia (Tamura) y dos
CON referencia, que la comparan contra la imagen de partida (Pearson y SSIM).
Muestra las tres curvas, el holograma y la reconstruccion en el foco que elige
cada una.

POR QUE DOS FAMILIAS DE METRICA. Tamura no mira la imagen original: mide el
contraste de la reconstruccion y punto. Es la unica que va a servir en el
laboratorio, donde no hay original con el que comparar. Pearson y SSIM si la
miran, y por eso aqui -en simulacion, donde U0 y Z_REAL se conocen- hacen de
patron: dicen donde esta el foco de verdad, y contra ellas se juzga si Tamura
acierta y si su pico es igual de estrecho. En una medida real no se pueden
calcular, asi que son una VARA DE MEDIR, no un metodo.

Las tres se evaluan sobre la MISMA retropropagacion: una propagacion por z, no
tres. Comparar curvas que salen de campos distintos no compara metricas.

LO QUE SALIO A Z_REAL = 7 mm, en BenchmarkTarget.png, CAMPO "intensidad",
barrido de 1 a 20 mm con paso 0.25 mm:

    metrica    foco       pico sobre el fondo   alt/riz
    tamura     7.000 mm   +2.5%                     4.6
    pearson    7.000 mm   +27.7%                    7.2
    ssim       7.000 mm   +20.1%                   20.7

Las tres clavan la MISMA z, con error 0.000 mm, y ese es el resultado que
importa: Tamura acierta el foco sin mirar la imagen original, o sea que las dos
con referencia no le estan descubriendo ningun sesgo.

Donde si se separan es en el margen. Un pico alto no basta: lo que decide si el
argmax es el foco o es suerte es cuanto sobresale el pico COMPARADO CON lo que
la curva ya oscila entre puntos vecinos (alt/riz, ver margen_del_pico()). Y ahi
el orden no es el que sugiere la altura:

  - SSIM gana con 20.7, y no por tener el pico mas alto -no lo tiene- sino por
    tener el fondo mas liso (rizado 0.97% contra el 3.87% de Pearson).
  - Pearson tiene el pico mas alto de los tres y aun asi se queda en 7.2,
    porque su fondo oscila casi tanto como crece el pico.
  - Tamura es la peor de las tres, 4.6, pero 4.6 no es poco: el pico sigue
    midiendo casi cinco veces el rizado. Por eso funciona, y por eso es
    sensible al paso del barrido de un modo que las otras dos no son.

Conviene remedir esto en cada imagen nueva antes de fiarse del autofoco: son
tres numeros que el script ya imprime.

═══════════════════════════════════════════════════════════════════════════

LO QUE SALIO A Z_REAL = 60 mm, ya con el paso corregido (0.25 mm, PASOS = 277),
BORDE_PX = 1000 y S = 1. Region medida 1000x2000 px de 3000x4000, 298
retropropagaciones en 225.9 s de GPU (propagar 420 ms/z, ssim 320 ms/z, o sea
que SSIM sigue siendo el 42 % del barrido).

    metrica    foco        error       alt/riz
    tamura     60.000 mm   +0.000 mm      26.9
    pearson    60.000 mm   +0.000 mm      54.4
    ssim       60.000 mm   +0.000 mm      54.6

Las tres clavan la z, y con mucho mas margen que a 7 mm: Tamura pasa de 4.6 a
26.9. Ademas el codo de Kf cae DENTRO del barrido por primera vez -Kfy va de
1.0000 a 1.1159 desde z = 56.4 mm mientras Kfx se queda en 1.0000 exacto-, o
sea los dos ejes en regimenes distintos dentro de una misma curva, que es lo
que dice la tarea 61.

Y NO SE NOTA, que tambien es un resultado. Ajustando una recta al fondo de
Tamura a cada lado del codo: -3.59e-05 por mm antes (40-56 mm) y -4.09e-05
despues (56.5-59 mm). Con Kfy = 1.03 en el foco la banda retenida en y baja un
3 %, por debajo del rizado. Para que Kf muerda hay que irse mucho mas lejos, y
ahi vuelve el problema del margen: ver BORDE_PX.

EL PICO ES SIETE VECES MAS ESTRECHO DE LO QUE DICE LA TABLA. El FWHM que
imprime sale del barrido GRUESO y lo limita su propio paso. Medido sobre el
fino (0.025 mm):

    metrica    FWHM grueso   FWHM fino
    tamura       0.505 mm      0.074 mm
    pearson      1.184 mm      0.127 mm
    ssim         0.841 mm      0.095 mm

La regla paso <= FWHM/2 pide entonces 0.037 mm: ~1870 puntos gruesos y ~22 min
de GPU, no 277 y 4 min. Con 0.25 mm acerto porque 60.000 cae EXACTAMENTE sobre
la rejilla: (60 - 1)/0.25 = 236, entero. Eso no es una propiedad del metodo,
es una coincidencia de tres constantes.

QUE PASA CUANDO NO CAE. Dos corridas mas, mismo paso, solo cambia Z_REAL. El
desfase es la distancia del pico verdadero al punto de rejilla mas cercano, y
su maximo posible es medio paso, 0.1266 mm:

    Z_REAL        desfase     tamura       pearson      ssim
    60.0 mm       0.0000     +0.000 mm    +0.000 mm    +0.000 mm
    60.1 mm       0.0145     -0.014 mm    -0.014 mm    -0.014 mm
    60.8734 mm    0.1266     +0.000 mm    +0.203 mm    +0.658 mm

60.1 no prueba gran cosa -cae dentro de la media anchura del pico, 0.037 mm- y
aun asi deja ver algo: las tres devuelven exactamente el desfase, o sea que el
barrido FINO, que abarca +-1 paso grueso con paso 0.025 mm, no mejoro el punto
del grueso en ninguna de las tres.

60.8734 es el caso peor, medio paso, 3.4 veces la media anchura. Ahi se
separan, y el grueso ni siquiera elige el punto de al lado:

  - tamura acierta, +0.000 mm: grueso 60.747 -el vecino- y el fino baja a
    60.873.
  - pearson se va a 61.076 (+0.203 mm): su grueso eligio 61.253, un paso MAS
    ALLA del correcto.
  - ssim se va a 61.532 (+0.658 mm): su grueso eligio 61.759, DOS pasos mas
    alla, y su FWHM sale 13.075 mm, o sea que ya no hay pico que medir.

LA UNICA QUE SOBREVIVE FUERA DE REJILLA ES LA UNICA QUE SIRVE EN EL
LABORATORIO. Pero sobrevive por poco, y el numero exacto importa: muestrear el
pico en su flanco lo baja lo justo para que compita con lo que la curva ya
tiene alto en otro sitio. Maximo del fondo dividido por el maximo de la curva,
con el fondo a mas de 1 mm del pico:

    metrica    Z_REAL = 60.0        Z_REAL = 60.8734
    tamura     0.9881 (z = 5.0)     0.9995 (z = 4.8)
    pearson    0.7560 (z = 58.5)    0.9307 (z = 59.7)
    ssim       0.8528 (z = 63.5)    0.9789 (z = 63.3)

Tamura gana en el caso peor por un 0.05 %. No lo pierde, pero el rival no es el
rizado: es el hombro del arranque del barrido, z -> Z_MIN, donde la
reconstruccion se parece al holograma sin propagar y hereda su contraste. Los
rivales de Pearson y SSIM son otros -lobulos anchos a 59.7 y 63.3 mm, la imagen
gemela- y por eso fallan CERCA del foco en vez de lejos.

Y OJO A DONDE ESTA EL FALLO, porque no es donde parece. En esa misma corrida
de 60.8734 se lanzaron TRES barridos finos, uno por cada pico grueso distinto
(236, 238, 240), y las tres metricas se evaluan en los tres. El maximo de
Pearson sobre los tres finos esta en 60.873 (0.9002, contra 0.8441 en el fino
que eligio), y el de SSIM tambien (0.5762 contra 0.5362). O sea que el dato
correcto YA ESTABA CALCULADO Y GUARDADO: las tres metricas miden bien el foco,
y lo que falla es que cada una solo mira DENTRO del fino centrado en SU propio
argmax grueso. Tomar el argmax sobre todos los finos las arregla a las tres,
con +0.000 mm y sin propagar ni una vez mas. Esta sin hacer: cambia como se
elige el foco, no como se mide.

POR ESO alt/riz NO PREDICE ESTO. En la corrida de 60.8734 da 13.5 para Tamura y
20.2 y 23.0 para las otras dos, y las que fallan son esas dos. alt/riz mide el
rizado punto a punto, y el peligro no es el rizado: es un lobulo ancho, que por
ser suave sobrevive a la resta de vecinos. Si hace falta un numero que avise,
es la tabla de arriba -maximo del fondo sobre maximo de la curva-, no alt/riz.
"""

import pathlib
import sys
import time

# ════════════════════════════════════════════════════════════════════════════
#  QUE SE MUESTRA Y QUE SE GUARDA
# ════════════════════════════════════════════════════════════════════════════

#: True abre las figuras de resultados al terminar (bloquea hasta cerrarlas).
MOSTRAR = True

#: True guarda ademas esas figuras como PNG en SALIDA_DIR.
GUARDAR_FIGURA = True

#: True guarda la reconstruccion de CADA z del barrido (grueso y fino).
#: OJO: medido a 3000x4000, cada PNG cuesta 2.3 s de matplotlib y 18 MB, mas
#: que la propagacion (0.42 s) y las tres metricas juntas. Con los 98 z de este
#: barrido son ~4 min y ~1.8 GB de solo escribir en disco, asi que por defecto
#: va apagado: las figuras de resumen no lo necesitan.
GUARDAR_BARRIDO = False

#: Que se guarda por cada z:
#:   "png"    |U|^2 normalizada a su propio maximo, en gris. ~unos MB por z.
#:   "npy"    el campo complejo tal cual. OJO: 3000x4000 en complex64 son
#:            ~92 MB POR z, y el barrido son ~98 z (~9 GB).
#:   "ambos"
FORMATO_BARRIDO = "png"

#: Que metricas se evaluan en cada z, en orden de dibujo. Se puede quitar
#: cualquiera; "tamura" es la unica que sobrevive fuera de la simulacion.
#: COSTE MEDIDO (3000x4000 en GPU, region de 1930x2930 px, 98 z): propagar
#: 424 ms/z, tamura 18 ms/z, pearson 16 ms/z, ssim 1147 ms/z. O sea que SSIM
#: son 112 s de los 159 s del barrido entero: quitarla lo deja en ~46 s. El
#: script reimprime este reparto al terminar, asi que no hay que creerse estos
#: numeros en otra maquina.
METRICAS = ("tamura", "pearson", "ssim")

#: Cuales de METRICAS viven en la CPU. Son las que NO tocan el dispositivo:
#: reciben A = None y solo pueden usar A_cpu. SSIM es la unica hoy, porque
#: skimage no tiene version GPU.
#:
#: Meter aqui una metrica que use A la rompe, y no en silencio: peta con
#: AttributeError sobre None a la primera z.
EN_CPU = {"ssim"}

#: True encadena el barrido: lanza la propagacion del paso i+1 y, MIENTRAS la
#: GPU la hace, calcula en la CPU las metricas de EN_CPU del paso i.
#:
#: NO CAMBIA NINGUN NUMERO. Las curvas salen identicas bit a bit; lo unico que
#: cambia es el reloj. Se escriben por INDICE y no con append justo por eso:
#: el paso i-1 se cierra una vuelta mas tarde que el i.
#:
#: POR QUE NO HACEN FALTA HILOS. CuPy lanza los kernels de forma asincrona, asi
#: que el solapamiento lo da el driver: basta con no sincronizar entre el
#: lanzamiento y el trabajo de CPU. Un ThreadPoolExecutor aqui solo anadiria la
#: duda del GIL.
#:
#: LO QUE GANA DEPENDE DE S, porque S solo encarece el lado GPU: el techo pasa
#: de ser la SUMA de los dos lados a ser el MAXIMO. MEDIDO sobre 62
#: retropropagaciones, BORDE_PX = 1000, Z_REAL = 60, en una RTX 3050 Laptop:
#:
#:     S    serie    solapado   ahorro    esa etapa en la GPU
#:     1    42.2 s    27.9 s     34 %      391 ->  157 ms/z
#:     2   112.8 s   102.0 s     10 %     1503 -> 1333 ms/z
#:
#: SSIM cuesta ~17 s en las cuatro corridas. Con S = 1 se esconden 14.3 de
#: esos 17, el 86 %. Con S = 2 solo 10.8, el 63 %, Y ESO NO ERA LO ESPERADO:
#: ahi la GPU tarda 1503 ms por z contra 276 de SSIM, o sea que sobraba sitio
#: para esconderlo ENTERO. No se esconde. Lo que queda fuera crece con S, asi
#: que apunta a lo que el encadenado no cubre -la copia a la CPU y el reparto
#: del pool de CuPy van en el mismo stream que la propagacion-. Sin medir.
#:
#: Y LAS CURVAS SALEN IDENTICAS BIT A BIT en los dos S, comprobadas contra la
#: version en serie clave por clave del .npz: esto mueve el reloj, no los
#: numeros. Si alguna vez dejan de serlo, el encadenado esta mal.
SOLAPAR = True

#: Lado en pixeles del recuadro central que se amplia en la figura de campos.
#: Sin el, las barras del target son subpixel en un panel de figura y las tres
#: reconstrucciones se ven identicas aunque no lo sean.
ZOOM_PX = 200

import matplotlib
if not MOSTRAR:
    matplotlib.use("Agg")   # sin ventana: no hace falta backend grafico
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mpasm_minimo import cp, kf, mpasm

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
# solo la formula del cono, z*tan(asin(lambda/(2 delta)))/delta en pixeles: es
# la que dice cuanto borde contamina la metrica, y ya vive ahi
from CamposT.roi import radio_del_cono

# ════════════════════════════════════════════════════════════════════════════
#  PARAMETROS FISICOS
# ════════════════════════════════════════════════════════════════════════════

RUTA = pathlib.Path(r"C:\Users\User\Desktop\Tesis\referencia\carlos\DLHM-model-main\DLHM-model-main\data\BenchmarkTarget.png")   # la imagen es la AMPLITUD, fase 0
LAMB = 633e-6       # mm
DELTA = 3.45e-3     # mm
#: 60.0 mm. Cae EXACTAMENTE sobre la rejilla del grueso -(60-1)/0.25 = 236- y
#: por eso las tres metricas dan +0.000. Las corridas fuera de rejilla (60.1 y
#: 60.8734, el caso peor) estan en el docstring: cambiar este numero es todo lo
#: que hace falta para repetirlas.
Z_REAL = 60.0        # mm, a la que se genera el holograma. El barrido NO la usa.
#: S = 1 A PROPOSITO, y aqui no es por memoria: es lo unico que mete el codo de
#: Kf DENTRO del barrido. El umbral es delta^2*S*n/lambda, o sea que S lo aleja:
#: con S = 2 esta en 112.8 mm (eje y) y el barrido entero corre con Kf = 1.0000.
#: Con S = 1 cae en 56.4 mm, 3.6 mm antes de Z_REAL, y Kfy llega a 1.1159 en
#: Z_MAX mientras Kfx sigue clavado en 1.0000: los dos ejes en regimenes
#: distintos dentro de una misma curva.
S = 2
GPU = cp is not None   # True/False para forzar; con True y sin CuPy aborta
xp = cp if GPU else np

#: True: fondo transparente y barras opacas (holograma en linea de Gabor). El
#: fondo es la onda de referencia; sin el, la intensidad no guarda la fase y el
#: barrido con CAMPO = "intensidad" no enfoca a ninguna z.
INVERTIR = True

#: "intensidad": se retropropaga sqrt(|U|^2), lo unico que mide un sensor
#:               (la fase se pierde y aparece la imagen gemela).
#: "complejo":   se retropropaga el campo con su fase; el foco es exacto.
CAMPO = "intensidad"

#: El sensor solo capta la difraccion hasta NA ~ (N*DELTA/2)/Z_REAL. A 50 mm
#: son 0.018 -resolucion ~36 um, lo que miden las barras- y no enfoca.
#: OJO AL PASO. El barrido fino solo afina alrededor del maximo del grueso, asi
#: que si el grueso salta el pico, el fino afina otra cosa. Medido en esta
#: imagen a Z_REAL = 150 mm: el pico de Tamura mide 0.30 mm de anchura a media
#: altura con CAMPO = "complejo" (1.35 mm con "intensidad"), y con pasos de
#: 0.53 mm el grueso daba 2.58 y 5.22 mm. Con 0.25 mm los dos dan 7.000 mm.
#: Regla: paso <= la mitad de la anchura del pico. Si subes Z_MAX o bajas
#: DELTA, el pico se ensancha y puedes usar menos pasos.
#:
#: Y PASOS NO ES UN NUMERO, ES UN PASO: depende del RANGO, asi que mover Z_REAL
#: lo invalida en silencio. Los 77 de antes daban 0.25 mm sobre el barrido 1-20
#: de Z_REAL = 7 mm; con Z_REAL = 60 ese mismo 77 sobre 1-70 son 0.908 mm, por
#: encima del techo de 0.68 que fija la regla para CAMPO = "intensidad", y el
#: grueso podia saltarse el pico. La forma robusta es fijar el paso:
#:     PASOS = round((Z_MAX - Z_MIN) / paso) + 1
#: 277 son exactamente 0.25 mm sobre 1-70 mm.
Z_MIN, Z_MAX, PASOS = 1.0, Z_REAL + 10.0, 277      # grueso, mm (0.25 si Z_REAL = 60)
PASOS_FINO = 21                            # barrido fino, +-1 paso grueso alrededor del pico

#: Pixeles de cada lado que las METRICAS no miran. None = radio del cono a Z_MAX.
#: No se dibuja en la figura, y las imagenes guardadas salen enteras: esto solo
#: decide sobre que region se mide.
#:
#: El borde de la ventana es un corte seco, y al retropropagar a z ese corte se
#: reparte en anillos sobre un radio radio_del_cono(z). Esos anillos tienen
#: mucho contraste y crecen con z, asi que entran en la cuenta de una metrica
#: que los mire. Con el radio a Z_MAX el margen los cubre a toda z del barrido
#: y la region medida es la MISMA en todos los puntos de la curva, que es lo
#: que los hace comparables.
#:
#: Con referencia el margen ademas alinea las dos imagenes: |U0| se recorta
#: EXACTAMENTE igual que la reconstruccion, porque mpasm() con r = mag = 1
#: devuelve la misma malla, y comparar pixel a pixel dos recortes distintos
#: seria comparar otra cosa.
#:
#: OJO: con esta imagen el margen NO es lo que decide el foco -con el paso de
#: barrido correcto, medido, el pico sale en 7.000 mm con margen y sin el-. Es
#: una guarda, no la razon de que el barrido funcione; la razon es PASOS.
#:
#: POR QUE AQUI VA UN NUMERO Y NO None. Con None el margen es el cono a Z_MAX, y
#: a Z_MAX = 70 mm eso son 1870 px por lado: mas que las 3000 filas de la malla,
#: asi que el script abortaba antes de propagar nada.
#:
#: Y no es mala suerte de esta geometria. Evaluando el cono justo en el umbral
#: de compresion sale una identidad que no depende de nada:
#:     cono(z_umbral) = (delta^2*S*n/lambda) * (lambda/(2*delta)) / delta = S*n/2
#: o sea 2*BORDE = S*n, como minimo la malla entera. Comprobado sobre cinco
#: geometrias distintas (633/3.45/3000, 528/1.83/2048, 405/1.0/512, 632.8/9.77/
#: 512 con S = 4...): 2*cono/n da S salvo el 0.4-2 % del tan no paraxial. O sea
#: que con BORDE_PX = None, Kf > 1 es INALCANZABLE en este script, siempre. Es
#: la misma frase que z_umbral = S*z_lim dicha desde el otro lado: Kf empieza a
#: comprimir justo cuando el cono deja de caber en la ventana.
#:
#: 1000 px es el compromiso: deja 1000x2000 px de region medida -un tercio del
#: lado corto- y cubre el cono hasta z = 37.4 mm, o sea la mitad del barrido.
#: De ahi en adelante los anillos del borde SI entran en la cuenta, y por eso
#: el fondo de las curvas no es comparable con el de una corrida con None.
BORDE_PX = 1000

SALIDA_DIR = RAIZ / "resultados" / "barrido_foco" / f"{RUTA.stem}_z{Z_REAL:g}mm_{CAMPO}_s{S}"


# ════════════════════════════════════════════════════════════════════════════
#  METRICAS
# ════════════════════════════════════════════════════════════════════════════
#
# Todas reciben la amplitud YA recortada -A donde se propago, A_cpu la misma en
# NumPy- y devuelven un float que se MAXIMIZA en el foco. Esa firma comun es lo
# que deja evaluarlas todas sobre la misma propagacion.
#
# Todas trabajan en float64: son sumas sobre millones de pixeles y en float32 el
# error acumulado se come la diferencia entre pasos vecinos del barrido.

def amplitud(U, borde):
    """|U| en float64, sin `borde` px por lado. La entrada de toda metrica."""
    A = xp.abs(U).astype(np.float64)
    if borde:
        A = A[borde:-borde, borde:-borde]
    return A


def tamura(A, A_cpu):
    """Coeficiente de Tamura, sqrt(std(A) / media(A)). SIN REFERENCIA.

    Mide el contraste de la amplitud y no depende de la escala del campo: si
    una z devuelve un campo 10 veces mas brillante, std y media se multiplican
    las dos por 10 y el cociente no cambia. Esa invariancia es lo que hace
    comparables entre si los puntos de la curva.

    Es la unica de las tres que se puede calcular sobre un holograma de camara.
    """
    m = A.mean()
    if m <= 0:
        return 0.0
    return float(xp.sqrt(A.std() / m))


def pearson(A, A_cpu):
    """Correlacion de Pearson entre A y la referencia |U0|. CON REFERENCIA.

    corr = <(A - mediaA)(R - mediaR)> / (sigmaA sigmaR), en [-1, 1], y vale 1
    cuando A = a*R + b con a > 0. O sea: invariante a escala y a offset, igual
    que Tamura, que es justo lo que hace que las dos curvas se puedan poner en
    los mismos ejes sin normalizar nada a mano.

    Esa invariancia no es un adorno. La reconstruccion no sale con la escala de
    U0 -mpasm divide por M*N*Kfx*Kfy*s^2- y con CAMPO = "intensidad" trae
    encima el fondo de la onda de referencia, que es un offset. Una metrica que
    no absorbiera los dos mediria ese desajuste en vez de medir el foco.
    """
    a = A - A.mean()
    r = REF - REF.mean()
    d = xp.sqrt((a * a).sum() * (r * r).sum())
    if d <= 0:
        return 0.0
    return float((a * r).sum() / d)


def ssim(A, A_cpu):
    """SSIM entre A y la referencia |U0|, tras llevar A a su escala. CON REFERENCIA.

    SSIM NO es invariante a escala: sus terminos de luminancia y contraste
    comparan medias y desviaciones absolutas, asi que aplicada en crudo sobre
    una reconstruccion que vive en otra escala que U0 daria un numero bajo a
    TODA z, foco incluido. Por eso antes se ajusta la recta a*A + b que mejor
    lleva A a R por minimos cuadrados, y se compara eso. Lo que queda medido es
    la estructura local, que es lo que se le pide a SSIM y lo que Pearson -un
    solo numero global- no ve.

    Va en CPU: skimage no tiene version GPU, asi que esta metrica es la que
    manda el campo de vuelta por PCIe en cada z. Es la cara de las tres.
    """
    va = A_cpu.var()
    if va <= 0:
        return 0.0
    a = float(((A_cpu - A_cpu.mean()) * (REF_CPU - REF_CPU.mean())).mean() / va)
    b = float(REF_CPU.mean() - a * A_cpu.mean())
    return float(structural_similarity(a * A_cpu + b, REF_CPU, data_range=REF_RANGO))


FUNCION = {"tamura": tamura, "pearson": pearson, "ssim": ssim}
COLOR = {"tamura": "C0", "pearson": "C1", "ssim": "C2"}
ETIQUETA = {"tamura": "Tamura  (SIN referencia)",
            "pearson": "Pearson vs |U0|  (con referencia)",
            "ssim": "SSIM vs |U0|  (con referencia)"}


def anchura_media(zs, c):
    """Anchura a media altura del pico de c(zs), en mm, o nan si no se puede.

    Se interpola linealmente donde la curva -reescalada a [0, 1] entre su
    minimo y su maximo- cruza 0.5 a cada lado del maximo. Devuelve nan si el
    pico llega al extremo del barrido sin bajar de la mitad: ahi la anchura no
    esta medida, esta fuera de la ventana, y un numero seria inventado.

    LIMITE: sale del barrido GRUESO, asi que no resuelve picos mas estrechos
    que un par de pasos. Con paso 0.25 mm, un 0.30 mm medido aqui significa
    "del orden del paso", no 0.30 exacto.
    """
    lo, hi = c.min(), c.max()
    if hi <= lo:
        return float("nan")
    y = (c - lo) / (hi - lo)
    i = int(np.argmax(y))
    cortes = []
    for sentido in (-1, +1):
        j = i
        while 0 <= j + sentido < len(y) and y[j + sentido] > 0.5:
            j += sentido
        k = j + sentido
        if not (0 <= k < len(y)):
            return float("nan")   # el pico no baja de media altura dentro del barrido
        t = (y[j] - 0.5) / (y[j] - y[k])
        cortes.append(zs[j] + t * (zs[k] - zs[j]))
    return abs(cortes[1] - cortes[0])


def margen_del_pico(zs, c, z_pico, ancho=1.0):
    """(altura del pico, rizado del fondo, y su cociente) de la curva c(zs).

    Decir que un pico "sobresale un 3%" no dice nada por si solo: lo que decide
    si el maximo es el foco o es suerte es cuanto sobresale COMPARADO CON lo que
    la curva ya oscila por su cuenta entre puntos vecinos. Dos curvas con el
    mismo pico relativo y distinto rizado no son igual de fiables.

    El fondo es todo lo que queda a mas de `ancho` mm del pico. El rizado se
    mide como std de las diferencias entre vecinos consecutivos partido por
    sqrt(2), y no como la std del fondo a secas: Pearson y SSIM caen despacio
    con z, y la std de una curva con tendencia mide la tendencia, no el rizado.
    Restar vecinos mata cualquier tendencia suave y deja solo el salto punto a
    punto, que es lo que hace que el argmax se vaya a la casilla de al lado.

    La altura se mide contra la MEDIANA del fondo, no contra su minimo: un
    minimo es un punto y puede ser el propio rizado.
    """
    fuera = np.abs(zs - z_pico) > ancho
    if fuera.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    o = c[fuera]
    base = float(np.median(o))
    rizado = float(np.std(np.diff(o)) / np.sqrt(2))
    altura = float(c.max() - base)
    return altura, rizado, (altura / rizado if rizado > 0 else float("inf"))


def a_cpu(a):
    return a.get() if GPU else a


def guardar_instante(U, z, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    U = a_cpu(U)
    nombre = f"z{z:08.4f}mm"
    if FORMATO_BARRIDO in ("npy", "ambos"):
        np.save(carpeta / f"{nombre}.npy", U)
    if FORMATO_BARRIDO in ("png", "ambos"):
        I = np.abs(U) ** 2
        plt.imsave(carpeta / f"{nombre}.png", I / I.max(), cmap="gray", vmin=0, vmax=1)


#: Metricas repartidas por donde corren. El orden de METRICAS se conserva en
#: las dos listas, asi que las curvas no dependen de este reparto.
EN_GPU_LISTA = [n for n in METRICAS if n not in EN_CPU]
EN_CPU_LISTA = [n for n in METRICAS if n in EN_CPU]

#: Encadenar solo tiene sentido si hay GPU que adelantar y CPU que esconder.
SOLAPANDO = SOLAPAR and GPU and bool(EN_CPU_LISTA)

#: Con SOLAPANDO esta etapa ya NO mide la propagacion: mide lo que la CPU no
#: consiguio esconder, o sea cuanto sobresale la GPU por encima del SSIM.
CLAVE_GPU = "espera GPU" if SOLAPANDO else "propagar"

#: segundos acumulados por etapa; el reparto se imprime al final
RELOJ = {CLAVE_GPU: 0.0, **{n: 0.0 for n in METRICAS}}


def barrido(H, zs, carpeta):
    """Un dict nombre -> curva, todas medidas sobre la MISMA propagacion.

    Con SOLAPANDO el bucle va encadenado: se LANZA la propagacion del paso i
    -asincrona, la llamada vuelve enseguida- y, mientras la GPU la calcula, la
    CPU cierra las metricas de EN_CPU del paso i-1. El coste por z deja de ser
    la suma de los dos lados y pasa a ser el mayor de los dos.

    Las curvas se escriben POR INDICE, no con append: el paso i-1 se cierra una
    vuelta mas tarde que el i, y con append saldrian desordenadas.
    """
    curvas = {n: np.empty(len(zs)) for n in METRICAS}
    pendiente = None                       # (indice, A_cpu) del paso anterior

    def cerrar_en_cpu(i, A_cpu):
        """Las de EN_CPU reciben A = None: no pueden tocar el dispositivo."""
        for n in EN_CPU_LISTA:
            t = time.perf_counter()
            curvas[n][i] = FUNCION[n](None, A_cpu)
            RELOJ[n] += time.perf_counter() - t

    for i, z in enumerate(zs):
        U = mpasm(H, DELTA, LAMB, -z, s=S, gpu=GPU)   # LANZA, no bloquea
        A = amplitud(U, BORDE)                        # LANZA, no bloquea

        # La GPU esta ocupada con lo de arriba. Aprovechar para cerrar el paso
        # anterior en la CPU, que es lo unico que no depende de este z.
        if pendiente is not None:
            cerrar_en_cpu(*pendiente)
            pendiente = None

        # AQUI se espera a la GPU: .get() sincroniza. Lo que quede de espera es
        # lo que la CPU no pudo esconder.
        t = time.perf_counter()
        A_cpu = a_cpu(A)
        RELOJ[CLAVE_GPU] += time.perf_counter() - t

        for n in EN_GPU_LISTA:
            t = time.perf_counter()
            curvas[n][i] = FUNCION[n](A, A_cpu)
            RELOJ[n] += time.perf_counter() - t

        if GUARDAR_BARRIDO:
            guardar_instante(U, z, carpeta)
        del U, A

        if SOLAPANDO:
            pendiente = (i, A_cpu)         # se cierra en la vuelta siguiente
        else:
            cerrar_en_cpu(i, A_cpu)
            del A_cpu

    if pendiente is not None:              # el ultimo paso no tiene siguiente
        cerrar_en_cpu(*pendiente)
    return curvas


# ════════════════════════════════════════════════════════════════════════════
#  CORRIDA
# ════════════════════════════════════════════════════════════════════════════

if FORMATO_BARRIDO not in ("png", "npy", "ambos"):
    raise SystemExit(f'FORMATO_BARRIDO = {FORMATO_BARRIDO!r}: usa "png", "npy" o "ambos".')
if not METRICAS:
    raise SystemExit("METRICAS esta vacio: no hay nada con que elegir el foco.")
for n in METRICAS:
    if n not in FUNCION:
        raise SystemExit(f"METRICAS trae {n!r}, y solo existen {tuple(FUNCION)}.")

img = plt.imread(RUTA).astype(float)
if img.ndim == 3:
    img = img[..., :3].mean(axis=2)
img = img / img.max()
U0 = (1 - img if INVERTIR else img).astype(complex)
M, N = U0.shape

BORDE = (int(np.ceil(radio_del_cono(Z_MAX, LAMB, DELTA))) if BORDE_PX is None
         else int(BORDE_PX))
if min(M, N) - 2 * BORDE < 64:
    # medir con el borde dentro no es una version peor del barrido: es otro
    # barrido, que elige la z mas lejana en vez del foco
    raise SystemExit(
        f"El margen de {BORDE} px por lado (cono a Z_MAX = {Z_MAX:g} mm) no "
        f"deja nada de una malla {M}x{N}.\nBaja Z_MAX, o fija BORDE_PX sabiendo "
        f"que los anillos del borde entran en la metrica.")

# la referencia: la MISMA amplitud de partida, recortada con el MISMO margen
REF = amplitud(xp.asarray(U0), BORDE)
REF_CPU = a_cpu(REF)
REF_RANGO = float(REF_CPU.max() - REF_CPU.min())

print(f"metricas: {', '.join(METRICAS)} sobre {M - 2 * BORDE}x{N - 2 * BORDE} px "
      f"de {M}x{N} (margen {BORDE} px por lado, cono a {Z_MAX:g} mm)")

# ---- propagacion: el holograma ---------------------------------------------
U = mpasm(U0, DELTA, LAMB, Z_REAL, s=S, gpu=GPU)
H = xp.abs(U).astype(U.dtype) if CAMPO == "intensidad" else U   # sqrt(|U|^2) = |U|
del U

# ---- retropropagacion: barrido grueso y fino --------------------------------
t0 = time.perf_counter()
zs = np.linspace(Z_MIN, Z_MAX, PASOS)
paso = zs[1] - zs[0]
curvas = barrido(H, zs, SALIDA_DIR / "grueso")

picos = {n: int(np.argmax(c)) for n, c in curvas.items()}
for n, i in picos.items():
    if i in (0, PASOS - 1):
        print(f"AVISO: el maximo de {n} cae en el extremo z = {zs[i]:g} mm; "
              f"el barrido no acota ese foco.")

# Una pasada fina por cada indice grueso DISTINTO, compartida por las metricas
# que lo eligieron: propagar dos veces a la misma z no da dos respuestas, y el
# barrido fino cuesta lo mismo que el grueso por punto.
finos, z_foco = {}, {}
for i in sorted(set(picos.values())):
    zs_f = np.linspace(zs[i] - paso, zs[i] + paso, PASOS_FINO)
    finos[i] = (zs_f, barrido(H, zs_f, SALIDA_DIR / f"fino_z{zs[i]:.3f}"))
for n, i in picos.items():
    zs_f, cf = finos[i]
    z_foco[n] = float(zs_f[int(np.argmax(cf[n]))])
t = time.perf_counter() - t0

# ---- reconstruccion en cada foco elegido ------------------------------------
# agrupada por z: dos metricas que eligen la misma z no piden dos paneles
z_unicos = sorted({round(z, 4) for z in z_foco.values()})
quien = {z: [n for n in METRICAS if round(z_foco[n], 4) == z] for z in z_unicos}
recon = {z: a_cpu(mpasm(H, DELTA, LAMB, -z, s=S, gpu=GPU)) for z in z_unicos}
H = a_cpu(H)

# ---- Kf a lo largo del barrido ---------------------------------------------
# mpasm() lo calcula por eje: Kfy con las M filas, Kfx con las N columnas
z_kf = np.linspace(Z_MIN, Z_MAX, 200)
kfy = np.array([kf(M, DELTA, LAMB, z, S) for z in z_kf])
kfx = np.array([kf(N, DELTA, LAMB, z, S) for z in z_kf])
# primera z con compresion, buscada hasta 20 veces Z_MAX para poder decirlo
z_lejos = np.linspace(Z_MIN, 20 * Z_MAX, 4000)
con_compresion = [z for z in z_lejos if max(kf(M, DELTA, LAMB, z, S), kf(N, DELTA, LAMB, z, S)) > 1]
z_umbral = con_compresion[0] if con_compresion else None

# ---- lo que se sabe ---------------------------------------------------------
print(f"{RUTA.name} {M}x{N} | holograma a {Z_REAL:g} mm | campo {CAMPO} | s = {S}"
      f" | {'GPU' if GPU else 'CPU'}")
print(f"grueso: {PASOS} pasos de {Z_MIN:g} a {Z_MAX:g} mm (paso {paso:g} mm)")
print(f"fino:   {PASOS_FINO} pasos (paso {2 * paso / (PASOS_FINO - 1):.3g} mm) "
      f"x {len(finos)} pico(s) distinto(s)")
print()
print(f"{'metrica':>9} {'grueso':>9} {'foco':>9} {'error':>9} {'FWHM':>9} "
      f"{'altura':>9} {'rizado':>9} {'alt/riz':>8}")
for n in METRICAS:
    c = curvas[n]
    w = anchura_media(zs, c)
    alt, riz, razon = margen_del_pico(zs, c, zs[picos[n]])
    print(f"{n:>9} {zs[picos[n]]:8.3f}  {z_foco[n]:8.3f}  {z_foco[n] - Z_REAL:+8.3f}  "
          f"{w:8.3f}  {alt:9.4f} {riz:9.4f} {razon:8.1f}")
print(f"(error = foco - Z_REAL = {Z_REAL:g} mm; FWHM del barrido grueso, no "
      f"resuelve por debajo de ~{2 * paso:g} mm; altura sobre la mediana del "
      f"fondo y rizado punto a punto de ese fondo, ambos en unidades de la "
      f"propia metrica -solo alt/riz es comparable entre metricas-)")
print()
n_prop = PASOS + PASOS_FINO * len(finos)
print(f"{n_prop} retropropagaciones en {t:.1f} s"
      + (" (incluye guardar cada z)" if GUARDAR_BARRIDO else ""))
print("  reparto: " + ", ".join(
    f"{k} {v:.1f} s ({v / n_prop * 1e3:.0f} ms/z)" for k, v in RELOJ.items()))
print(f"Kf en el barrido: Kfy {kfy.min():.4f}-{kfy.max():.4f}, Kfx {kfx.min():.4f}-{kfx.max():.4f}"
      + (f" | Kf > 1 a partir de z = {z_umbral:.1f} mm" if z_umbral is not None
         else f" | Kf = 1 hasta {20 * Z_MAX:g} mm"))
if GUARDAR_BARRIDO:
    print(f"barrido guardado ({FORMATO_BARRIDO}) en {SALIDA_DIR}")

# Las curvas en crudo, siempre. Son 98 floats por metrica y cuestan 159 s de
# GPU: volver a dibujarlas, medirles la anchura o compararlas con otra corrida
# no deberia exigir repetir el barrido.
SALIDA_DIR.mkdir(parents=True, exist_ok=True)
volcado = {"zs": zs, **{f"grueso_{n}": curvas[n] for n in METRICAS}}
for i, (zs_f, cf) in finos.items():
    volcado[f"fino_{i}_zs"] = zs_f
    volcado.update({f"fino_{i}_{n}": cf[n] for n in METRICAS})
np.savez(SALIDA_DIR / "curvas.npz", **volcado)
print("->", SALIDA_DIR / "curvas.npz")

# ════════════════════════════════════════════════════════════════════════════
#  FIGURA 1: el barrido
# ════════════════════════════════════════════════════════════════════════════

# cada curva a su propio maximo: lo que se compara es DONDE esta el pico y como
# de estrecho es, no cuanto vale. Tamura y SSIM no viven en la misma escala y
# forzarlas a una comun no significaria nada.
ESCALA = {n: max(curvas[n].max(), max(f[1][n].max() for f in finos.values()))
          for n in METRICAS}

fig1, (b1, b2) = plt.subplots(1, 2, figsize=(15, 5.5),
                              gridspec_kw={"width_ratios": [2, 1]})
for n in METRICAS:
    b1.plot(zs, curvas[n] / ESCALA[n], "o-", ms=3, lw=1.2, color=COLOR[n],
            label=ETIQUETA[n])
    for zs_f, cf in finos.values():
        b1.plot(zs_f, cf[n] / ESCALA[n], ".-", ms=4, lw=1, color=COLOR[n], alpha=0.55)
    b1.axvline(z_foco[n], color=COLOR[n], lw=1, ls=":")
b1.axvline(Z_REAL, color="k", ls="--", lw=1.2, label=f"Z_REAL = {Z_REAL:g} mm")
b1.set_xlabel("z de retropropagacion [mm]")
b1.set_ylabel("metrica / su propio maximo")
b1.set_title("barrido: sin referencia vs con referencia\n"
             "(punteado claro: barrido fino; vertical de color: foco elegido)",
             fontsize=10)
b1.legend(fontsize=8, loc="upper right")
b1.grid(alpha=0.25)

for n in METRICAS:
    m = (zs > Z_REAL - 6 * paso) & (zs < Z_REAL + 6 * paso)
    b2.plot(zs[m], curvas[n][m] / ESCALA[n], "o-", ms=4, lw=1.2, color=COLOR[n], label=n)
    for zs_f, cf in finos.values():
        b2.plot(zs_f, cf[n] / ESCALA[n], ".-", ms=4, lw=1, color=COLOR[n], alpha=0.55)
b2.axvline(Z_REAL, color="k", ls="--", lw=1.2)
b2.set_xlim(Z_REAL - 6 * paso, Z_REAL + 6 * paso)
b2.set_xlabel("z [mm]")
b2.set_title(f"zoom +-{6 * paso:g} mm alrededor de Z_REAL", fontsize=10)
b2.legend(fontsize=8)
b2.grid(alpha=0.25)

fig1.suptitle(f"Barrido de foco con MPASM minimo  ·  {CAMPO}  ·  lambda {LAMB * 1e6:.0f} nm"
              f"  ·  delta {DELTA * 1e3:.2f} um  ·  s = {S}  ·  {M}x{N}")
fig1.tight_layout()

# ════════════════════════════════════════════════════════════════════════════
#  FIGURA 2: holograma y reconstrucciones
# ════════════════════════════════════════════════════════════════════════════

ext = [-N / 2 * DELTA, N / 2 * DELTA, -M / 2 * DELTA, M / 2 * DELTA]
fila0, col0 = (M - ZOOM_PX) // 2, (N - ZOOM_PX) // 2
rec = (slice(fila0, fila0 + ZOOM_PX), slice(col0, col0 + ZOOM_PX))
ext_z = [(col0 - N / 2) * DELTA, (col0 + ZOOM_PX - N / 2) * DELTA,
         (fila0 + ZOOM_PX - M / 2) * DELTA, (fila0 - M / 2) * DELTA]

paneles = [(np.abs(U0) ** 2, "referencia |U0|^2  (objeto a z = 0)"),
           (np.abs(H) ** 2, f"holograma |H|^2 a {Z_REAL:g} mm  ({CAMPO})")]
for z in z_unicos:
    paneles.append((np.abs(recon[z]) ** 2,
                    f"retropropagado a {z:.3f} mm\n" + " + ".join(quien[z])))

fig2, ejes = plt.subplots(2, len(paneles), figsize=(4.1 * len(paneles), 8.6),
                          squeeze=False)
for k, (I, titulo) in enumerate(paneles):
    I = I / I.max()
    ejes[0][k].imshow(I, extent=ext, cmap="gray", vmin=0, vmax=1)
    ejes[0][k].set_title(titulo, fontsize=10)
    ejes[0][k].set_xlabel("x [mm]")
    ejes[1][k].imshow(I[rec], extent=ext_z, cmap="gray", vmin=0, vmax=1)
    ejes[1][k].set_title(f"zoom central {ZOOM_PX}x{ZOOM_PX} px", fontsize=9)
    ejes[1][k].set_xlabel("x [mm]")
ejes[0][0].set_ylabel("y [mm]")
ejes[1][0].set_ylabel("y [mm]")

fig2.suptitle(f"Campos  ·  {RUTA.name}  ·  {CAMPO}  ·  cada panel normalizado a su propio maximo")
fig2.tight_layout()

# ════════════════════════════════════════════════════════════════════════════
#  FIGURA 3: Kf
# ════════════════════════════════════════════════════════════════════════════

fig3, c1 = plt.subplots(figsize=(7.5, 4.8))
c1.plot(z_kf, kfx, label=f"Kfx (N = {N})")
c1.plot(z_kf, kfy, "--", label=f"Kfy (M = {M})")
for n in METRICAS:
    c1.axvline(z_foco[n], color=COLOR[n], lw=1, ls=":",
               label=f"foco {n} = {z_foco[n]:.3f} mm")
c1.set_xlabel("z [mm]")
c1.set_ylabel("Kf")
c1.set_title("evolucion de Kf" + (f"  (Kf > 1 desde z = {z_umbral:.1f} mm)"
                                 if z_umbral is not None else ""))
c1.set_ylim(bottom=min(0.95, kfx.min(), kfy.min()))
c1.legend(fontsize=8)
fig3.tight_layout()

if GUARDAR_FIGURA:
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    for nombre, f in (("barrido_metricas", fig1), ("campos", fig2), ("kf", fig3)):
        f.savefig(SALIDA_DIR / f"{nombre}.png", dpi=110)
        print("->", SALIDA_DIR / f"{nombre}.png")
if MOSTRAR:
    plt.show()
