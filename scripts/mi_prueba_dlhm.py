"""Reconstruccion DLHM de fuente puntual, y seleccion del foco como barrido_foco.

QUE HACE. Toma un holograma DLHM -grabado, o simulado con la ida exacta de
CamposT.dlhm-, lo reconstruye en un barrido de z (fuente -> muestra, con L
fijo) y elige el foco como scripts/barrido_foco.py: tres metricas sobre la
MISMA reconstruccion -Tamura, sin referencia, la unica que sirve en el
laboratorio, y Pearson y SSIM contra el objeto, que en simulacion hacen de vara
de medir-, un barrido grueso y un fino, una region medida (un ROI o el cuarto
central), la tabla de foco, FWHM y alt/riz, curvas.npz y dos figuras. Los
parametros de seleccion son los de barrido_foco, con el mismo nombre y el mismo
significado; las metricas son copias literales de las suyas, y
tests/test_mi_prueba_dlhm_seleccion.py fija que no diverjan.

Es el hermano DLHM de scripts/mi_prueba_angular.py, que propaga onda plana. Va
aparte porque la fuente puntual lo cambia todo: la onda de referencia es
esferica, la distancia de propagacion es L - z y la reconstruccion sale
magnificada por M = L/z. Un interruptor de geometria en un solo script obligaria
a cada constante a decir en que modo aplica.

DOS MOTORES DETRAS DE LA MISMA SELECCION (MOTOR):

    "exacto"  CamposT.dlhm.Reconstructor: la esferica exacta dentro del MPASM,
              sin aproximacion paraxial. El holograma entra ENTERO: el espectro
              del sensor se calcula una vez y cada z cuesta solo la ventana
              medida. Va por defecto porque es el unico en el que cabe el
              sensor entero y en el que el ROI tiene el sentido de
              barrido_foco: evaluar menos, no recortar.

    "carlos"  reconstruir(), la cadena de reconstruction_dlhm.py: remuestrear
              por geometria, multiplicar por la onda esferica, propagar L - z
              y quitar la referencia. Necesita el holograma RECORTADO al ROI
              -con el sensor entero pide 15 GB por paso-, y el recorte tiene
              que ser cuadrado y conviene centrado. Se conserva porque es la
              cadena validada contra el modelo de Carlos, y compararlas sobre
              el mismo holograma es de lo que va la tesis. Ver MotorCarlos.

LA MALLA ES LA DEL SENSOR, PROYECTADA DESDE LA FUENTE: paso DELTA*z/L, un pixel
por pixel del sensor. En ella un detalle cae en el MISMO pixel a toda z -el
rayo principal que llega al pixel X del sensor cruza el plano z en X*z/L-, asi
que el ROI es una ventana fija de pixeles, valida en todo el barrido y alineada
con el holograma y con la referencia. Por eso la referencia tiene que ser el
objeto en esa malla, con la forma del holograma. Al simular lo es por
construccion.

OBJETO = "fase". Un objeto de fase tiene en el foco el MINIMO de contraste de
amplitud (|t| = 1), asi que la Tamura va con el signo cambiado; Pearson y SSIM
comparan angle(t) con la fase de la referencia, Pearson en valor absoluto
porque el signo depende del convenio: el objeto del modelo de Carlos es
exp(-i*2pi*(n-1)*h*I/lambda), y por ese signo menos su fase corre AL REVES que
los grises de la referencia. Todas se maximizan en el foco, como en
barrido_foco.

LO QUE SALIO el 2026-09-28, con los valores por defecto: el BenchmarkTarget
simulado entero (3000x4000 px, z = 3 mm, holograma de INTENSIDAD, con gemela)
con el montaje (528 nm, 1.83 um, L = 10 mm), grueso de 2.5 a 3.5 mm con paso
0.01 mm y fino de 0.001 mm, en la GPU de desarrollo (RTX 3050 de 4 GB):

    variante                            metrica     foco     FWHM   alt/riz
    exacto, cuarto central              tamura      3.0000   0.019    33.6
      (750x1750 px: el defecto)         pearson     3.0000   0.047    57.7
                                        ssim        3.0000   0.016    61.0
    exacto, Z_REAL a medio paso de      tamura      3.0000   0.037    10.8
      la rejilla gruesa (caso peor)     pearson     3.0000   0.066    45.2
                                        ssim        3.0000   0.088    34.1
    exacto, ROI de 1024 centrado        tamura      3.0000   0.023    34.9
                                        pearson     3.0000   0.050    52.3
                                        ssim        3.0000   0.017    65.0
    Carlos, el mismo ROI                tamura      3.0020   0.078    41.1
                                        pearson     2.9990   0.075    54.3
                                        ssim        3.0000   0.055    47.5
    exacto, malla entera (BORDE_PX=0)   tamura      3.0000   0.041    32.7
    fase pi/2, exacto, cuarto central   -tamura     3.0010   0.038    26.4
                                        |pearson|   3.0000   0.035    51.8
                                        ssim        3.0000   0.013   190.8
    fase, exacto, ROI de 1024           -tamura     3.0010   0.034    25.9
                                        |pearson|   3.0000   0.045    50.4
                                        ssim        3.0000   0.014   154.7
    fase, Carlos, ROI de 1024           -tamura     3.0050   0.092    30.7
                                        |pearson|   3.0010   0.078    37.9
                                        ssim        2.9990   0.060    66.1

(foco y FWHM en mm; la FWHM, del barrido grueso). Lo que dice:

  - Con el motor exacto las tres metricas clavan el foco, a 1 um como mucho
    -el paso fino-, tambien en el caso peor: el grueso elige un vecino de
    3.000 y el fino lo corrige. La Tamura, la unica que sirve en el
    laboratorio, acierta en todas las variantes.
  - El de Carlos acierta a 2 um en amplitud y a 5 um -medio paso grueso- con
    la Tamura invertida en fase, con picos unas tres veces mas anchos: solo ve
    la NA de su recorte. En las pruebas, con una malla de 128 px y un recorte
    de 64, su Tamura invertida NO enfoca (se va 19 a 55 um): ahi el borde del
    recorte domina el contraste de amplitud. Con 1024 ya no.
  - Con Tamura la malla entera tambien acierta: el cuarto central no es lo que
    decide el foco, como si lo era con la nitidez de verif_dlhm_esferico.py.
    Queda por defecto porque da el pico la mitad de ancho y cuesta 3.5 veces
    menos por z (ver BORDE_PX).

TIEMPOS en esa GPU: simular el holograma entero, ~45 s, una vez (despues se
lee con RUTA). Construir el motor exacto, ~11 s: 5560x8357 frecuencias, 354 MB.
Cada plano del cuarto central cuesta ~220 ms en la GPU y SSIM otros ~215 ms en
la CPU; con SOLAPAR van a la vez, y los 122 planos del barrido salen en ~30 s.
Sin "ssim" en METRICAS, 27 s: SOLAPAR ya lo esconde casi entero, asi que
quitarlo no compra tiempo. OJO al leer el reparto: con SOLAPAR, "espera GPU"
(~25 ms/z) es lo que SSIM no alcanzo a esconder, no lo que cuesta la GPU. La
malla entera, 775 ms por plano. El de Carlos, con el recorte de 1024, 46 s
por barrido.

EL SEGUNDO SIMULADOR: EL MODELO DE LOPERA (SIMULADOR = "lopera"). Es dlhm() de
Lopera et al., Opt. Express 32, 48509 (2024), copiada tal cual de su
repositorio (seccion 2). No propaga la onda esferica: la sustituye por una
magnificacion -remuestrear la muestra por Mag = L/z-, una onda plana a
Mag*(L - z), la distorsion radial de Brown-Conrady, la envolvente de la fuente
y 8 bits. Su holograma lleva la envolvente, asi que el motor exacto lo divide
por su media, como uno de laboratorio sin FONDO. Y su convenio es e^{-ikz}:
con un objeto de fase su holograma es el de la fase conjugada, que ninguna
metrica nota (Pearson va en valor absoluto, SSIM ajusta el signo y la Tamura
mide amplitud). Verificado el 2026-09-28 que es su modelo y que se le llama
bien:

  - con su main_dlhm.py, nuestra copia y su modulo dan el mismo holograma bit a
    bit; y a escala real (3000x4000) nuestra llamada, con dx_in = 0, da el
    mismo que su camino entero con dx_in = delta*z/L, en amplitud y en fase;
  - su propagacion contra la ida exacta, en un sensor de 1024 (cuadrado
    central de 512): correlacion 1.00000 con un objeto liso -manchas de
    ~12 um, el regimen paraxial, donde las dos son la misma fisica- y 0.895
    con el detalle de ~1 um del BenchmarkTarget, el mismo 0.89 que el
    escalado de la tarea 24. Su modelo entero, 0.954 y 0.850.

Reconstruido con la seleccion de este script, mismo objeto y z = 3 mm, error
del foco en mm:

    simulador       motor                     tamura    pearson     ssim
    exacto          exacto, cuarto central    +0.000    +0.000    +0.000
    lopera          exacto, cuarto central    +0.020    +0.014    +0.170
    lopera          Carlos, ROI de 1024       +0.011    +0.006    +0.007
    lopera, fase    exacto, cuarto central    +0.012    +0.013    +0.020
    lopera, fase    Carlos, ROI de 1024       +0.006    +0.002    +0.007

Su holograma no enfoca donde su modelo puso la muestra: 12-20 um con la
fisica exacta y 2-11 um con la cadena de Carlos, que ve menos NA. En amplitud
SSIM se va a un lobulo lejano, porque el detalle fino no llega a enfocar en
ningun plano. Quitando uno a uno sus pasos el corrimiento no desaparece:

    su holograma                              tamura      pearson
    entero                                    +20 um      +14 um
    sin Brown-Conrady                         +27 um      +15 um
    sin Brown-Conrady, envolvente ni 8 bits   +27 um      +16 um

Es su nucleo paraxial -la onda plana a Mag*(L - z)-, no el posproceso.

EL ~0.03 DE ANTES, EXPLICADO. Hasta el 2026-09-28 este script puntuaba cada z
con una sola correlacion contra la referencia, y salia ~0.03 a cualquier
escala: "la reconstruccion funciona, el barrido puntuado no". Se habian
descartado cinco causas -el recorte, el sobremuestreo, la escala de la
referencia, un portador de fase y la propia reconstruccion, validada abajo- y
la conclusion fue que hacia falta puntuar sin referencia. La causa era la
GEOMETRIA de la puntuacion: referencia_a_escala() estiraba el 1/M central de la
referencia sobre TODA la salida de reconstruir(), cuando el objeto que el
recorte ve ocupa solo el 1/M central de esa salida, su huella. Medido con la
version de entonces sobre el BenchmarkTarget simulado entero (3000x4000,
z = 3 mm, holograma de intensidad) y su recorte central de 1024, en z = 3:

    geometria                                     correlacion
    la vieja, referencia_a_escala()                  +0.051
    la vieja, dividiendo por |U0|^2                  -0.010
    la vieja con la mejor escala, M de 0.8 a 6.0     +0.071   (M = 2.20)
    la huella, sin dividir                           -0.459
    la huella dividiendo por |U0|^2: MotorCarlos     -0.731
    el motor exacto sobre la misma ventana           -0.836

Negativas porque aquella referencia no estaba invertida. Ninguna escala
arreglaba la vieja. La otra mitad del arreglo es dividir por |U0|^2: ver
MotorCarlos.

DE DONDE SALE LA FISICA DE CARLOS. point_src, angular_spectrum, fts, ifts y
resize estan COPIADAS TAL CUAL de referencia/carlos/DLHM-model-main/.../dlhm.py,
igual que mi_prueba_angular copia angularSpectrum: su valor entero es que nadie
las ha tocado. Si hay que cambiar algo, se cambia en una copia de trabajo, no
ahi.

OJO: el angular_spectrum de aqui usa dfx = 1/Wx, con los ejes EN SU SITIO. El
angularSpectrum de mi_prueba_angular los lleva CRUZADOS. En malla cuadrada
coinciden; en rectangular no. No los mezcles.

VALIDADO CONTRA reconstruction_dlhm.py, y pasa. Corriendo las dos cadenas
sobre el mismo holograma reducido a 512x512, con sus parametros (L = 11 mm,
z = 4.95 mm):

    razon mio/Carlos    1.000000e-06   (desviacion relativa 6.6e-06)
    corr(|Rec|)         1.000000
    corr(angle(Rec))    0.999997

O sea que reconstruir() es fiel.

OJO, esa comparacion es de ANTES de que malla_remuestreada() redondeara al
par: con esos parametros la malla salia 337 en las dos cadenas. Hoy
reconstruir() usa 338 y reconstruction_dlhm.py sigue en 337, que desplaza su
reconstruccion. Se aparta a proposito; ver malla_remuestreada().

OJO A ESE 1e-6, que es (1e-3)^2 y no un error: point_src devuelve exp(ikr)/r, y
el 1/r NO es invariante de escala. Trabajar en milimetros en vez de metros lo
cambia por 1000, y como Rec = U*conj(U0) el factor entra al cuadrado. Es un
factor real global: no afecta a ninguna metrica, pero desconcierta si alguien
compara salidas numero a numero.

EL RECORTE DEL MOTOR DE CARLOS NO ES UNA COMODIDAD. El remuestreo por
geometria exige una malla de Wx/s puntos, y eso depende del ANCHO FISICO del
recorte, no del paso de pixel: submuestrear el holograma no ayuda nada. Medido
con lambda 532 nm, delta 1.85 um y z = 2 mm:

    recorte    malla        memoria    por paso
      3000    13144x13144    15.4 GB    ~24 s
      1024     2802x2802      0.70 GB   ~1.1 s
       750     1624x1624      0.24 GB   ~0.4 s

Sin recortar, un solo paso pide 15 GB. El script calcula esa malla ANTES de
empezar y aborta con la tabla si no cabe, en vez de morir sin explicar nada.

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
nadie puede reproducir. Para eso esta RUTA = None: simular con la ida exacta un
holograma cuya geometria si se conoce.

UNIDADES: milimetros para todo.  528 nm -> 528e-6    1.83 um -> 1.83e-3
"""

import pathlib
import sys
import time

# ════════════════════════════════════════════════════════════════════════════
#  1. PARAMETROS
# ════════════════════════════════════════════════════════════════════════════

# ── 1a. QUE SE MUESTRA Y QUE SE GUARDA  (los mismos de barrido_foco) ───────

#: True abre las figuras al terminar (bloquea hasta cerrarlas).
MOSTRAR = True

#: True guarda ademas esas figuras como PNG en la carpeta de salida.
GUARDAR_FIGURA = True

#: True guarda la reconstruccion de CADA z del barrido (grueso y fino). A
#: diferencia de barrido_foco guarda la VENTANA MEDIDA, no la malla entera: el
#: motor de Carlos no tiene otra.
GUARDAR_BARRIDO = False

#: Que se guarda por cada z: "png" (|t|^2, o angle(t) con OBJETO = "fase", a
#: [0, 1]), "npy" (t complejo, tal cual) o "ambos".
FORMATO_BARRIDO = "png"

#: Que metricas se evaluan en cada z, en orden de dibujo. "tamura" es la unica
#: sin referencia: la unica que sirve con un holograma de laboratorio.
METRICAS = ("tamura", "pearson", "ssim")

#: Cuales corren en la CPU: reciben A = None y solo usan A_cpu. SSIM, porque
#: skimage no tiene version GPU. Meter una que use A la rompe a la primera z.
EN_CPU = {"ssim"}

#: True encadena: la GPU reconstruye el paso i+1 mientras la CPU mide el i. No
#: cambia ningun numero, solo el reloj. Sin GPU no hace nada.
SOLAPAR = True

#: Lado en pixeles del recuadro que se amplia en la figura de campos.
ZOOM_PX = 200

import cv2 as cv                                                     # noqa: E402
import matplotlib                                                    # noqa: E402
if not MOSTRAR:
    matplotlib.use("Agg")   # sin ventana: no hace falta backend grafico
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
from skimage.metrics import structural_similarity                    # noqa: E402

try:
    import cupy as cp
except Exception:                      # sin CuPy, sin CUDA, o CuPy roto
    cp = None

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from CamposT import dlhm as dlhm_exacto    # dlhm() es el modelo de Lopera  # noqa: E402
from CamposT.montaje import MONTAJE                                  # noqa: E402
from CamposT.roi import Roi, elegir                                  # noqa: E402

# ── 1b. EL HOLOGRAMA Y EL OBJETO ──────────────────────────────────────────

#: El holograma grabado (.png de 8 bits o .npy), o None para SIMULARLO con la
#: ida exacta de CamposT.dlhm a partir de REFERENCIA. El simulado se guarda en
#: la carpeta de salida como <nombre>.npy: pon aqui esa ruta y la corrida
#: siguiente no vuelve a simular (en 3000x4000 la ida tarda ~40 s en la GPU) y
#: escribe en la misma carpeta.
RUTA = None

#: Solo con RUTA y MOTOR = "exacto": la imagen SIN muestra, por la que se
#: divide el holograma. None divide por su propia media, que deja dentro la
#: envolvente del cono del pinhole. El motor de Carlos no la usa: recibe la
#: intensidad normalizada a [0, 1].
FONDO = None

#: EL OBJETO, en la malla que el sensor proyecta desde la fuente: paso
#: DELTA*z/L y la MISMA forma que el holograma, un pixel por pixel del sensor.
#: Con RUTA = None es lo que se simula; si no, contra lo que se comparan
#: Pearson y SSIM. None solo con METRICAS = ("tamura",).
REFERENCIA = RAIZ / "referencia/carlos/DLHM-model-main/DLHM-model-main/data/BenchmarkTarget.png"

#: "amplitud": t = g, y se mide |t|, como en barrido_foco.
#: "fase":     t = exp(i*FASE_MAX*g). La Tamura va con el signo cambiado -un
#:             objeto de fase enfoca en el MINIMO de contraste de amplitud- y
#:             Pearson (en valor absoluto) y SSIM comparan angle(t) con la
#:             fase de la referencia.
OBJETO = "amplitud"

#: g = 1 - img (fondo transparente, barras opacas: la holografia en linea de
#: Gabor necesita el fondo) o g = img. OJO en "fase": el fondo tiene que quedar
#: en g = 0 o el borde de la ventana es un escalon de fase; el BenchmarkTarget
#: tiene el fondo negro, asi que en fase va con INVERTIR = False.
INVERTIR = True

#: rad: la fase de g = 1. Solo con OBJETO = "fase".
FASE_MAX = np.pi / 2

#: mm. Con RUTA = None, la z a la que se simula. Con RUTA, la verdad si se sabe
#: -la tabla da el error del foco contra ella- o None.
Z_REAL = 3.0

#: Con que se simula, si RUTA = None:
#:   "exacto"  la ida de CamposT.dlhm: la onda esferica e^{ikr}/r sobre la
#:             muestra y propagada sin aproximar, la Ec. (1) de Lopera et al.
#:             2024. Contra esa ecuacion calculada directa: 3.7e-6.
#:   "lopera"  su MODELO de simulacion, dlhm() de la seccion 2: magnificacion
#:             por remuestreo + onda plana + Brown-Conrady + envolvente de la
#:             fuente + 8 bits. Paraxial. Su holograma lleva la envolvente, asi
#:             que el motor exacto lo divide por su media, como uno de
#:             laboratorio sin FONDO.
SIMULADOR = "exacto"

#: Solo al simular con SIMULADOR = "exacto": "intensidad" (|1 + V|^2 = I/I_ref,
#: lo que mide el sensor, con imagen gemela) o "complejo" (1 + V, sin gemela).
#: El modelo de Lopera da siempre intensidad.
CAMPO = "intensidad"

#: Solo con SIMULADOR = "lopera": la NA de la fuente, que fija su envolvente
#: exp(-2r/(L tan(asin(NA)))). 0.1 es la de su main_dlhm.py. Con 0 su codigo
#: usa exp(-r), que depende de las unidades.
NA_FUENTE = 0.1

#: El montaje del laboratorio, CamposT/montaje.py: 528 nm, 1.83 um, L = 10 mm.
#: L -fuente a sensor- se mide una vez en el banco y NO se barre.
LAMB, DELTA, L = MONTAJE.lamb, MONTAJE.delta_x, MONTAJE.L

# ── 1c. EL BARRIDO  (z = fuente -> muestra, con L fijo) ────────────────────

#: Grueso, en mm: paso 0.01 mm. Mover z mueve el foco Y la escala a la vez
#: (M = L/z), pero en la malla proyectada un detalle no cambia de pixel.
Z_MIN, Z_MAX, PASOS = 2.5, 3.5, 101
#: Barrido fino: +-1 paso grueso alrededor de cada pico distinto.
PASOS_FINO = 21

# ── 1d. LA REGION MEDIDA ──────────────────────────────────────────────────

#: Donde miden las metricas, en pixeles de la malla del holograma:
#:     None                    la malla menos BORDE_PX por lado.
#:     True                    la arrastras con el raton sobre la referencia
#:                             (o sobre el holograma si no hay).
#:     (X0, Y0, ANCHO, ALTO)   ventana fija: X0 la columna e Y0 la fila de la
#:                             esquina superior izquierda.
#: Con el motor exacto el holograma entra ENTERO y solo se evalua la ventana.
#: Con el de Carlos la ventana es ademas el RECORTE del holograma, y tiene que
#: ser cuadrada y conviene centrada (ver MotorCarlos).
ROI = None

#: Pixeles por lado que no se miden con ROI = None. None = 3*min(M, N)//8, el
#: cuarto central del lado corto: la periferia trae la imagen gemela y la
#: perdida de NA del borde del sensor. Con la nitidez de
#: scripts/verif_dlhm_esferico.py eso decidia el foco: la ventana entera
#: elegia el extremo del barrido (3.30) y el cuarto central, 3.00. Con Tamura
#: no lo decide -medido el 2026-09-28, la malla entera tambien da 3.0000-, pero
#: el pico sale el doble de ancho (FWHM 0.041 frente a 0.019 mm) y cada z
#: cuesta 3.5 veces mas (775 frente a 220 ms en la GPU). 0 mide la malla entera.
BORDE_PX = None

# ── 1e. EL MOTOR ──────────────────────────────────────────────────────────

#: "exacto" (CamposT.dlhm, el holograma entero) o "carlos" (reconstruir(),
#: sobre el recorte del ROI). Ver la cabecera.
MOTOR = "exacto"

#: DISPOSITIVO de calculo, para los dos motores: "auto" (la GPU si la hay),
#: "cpu" o "gpu".
#:
#: LO QUE LA TARJETA ARREGLA Y LO QUE NO, que con el motor de Carlos no es
#: obvio. Cada paso son dos pares de FFT sobre la malla remuestreada entera:
#: eso la GPU lo hace mucho mas rapido. Lo que NO arregla es el tamano: con
#: SOBREMUESTREO = None y el sensor entero la malla es de 13312x13312, o sea
#: 1.42 GB POR ARRAY en complex64, y la cadena necesita varios a la vez. En
#: una tarjeta de 4 GB sigue sin caber. Lo que complex64 compra es
#: aproximadamente el DOBLE de recorte que en CPU con complex128, no el sensor
#: entero.
#:
#: Con "gpu" y sin CUDA ABORTA en vez de caer a CPU en silencio.
DISPOSITIVO = "auto"

#: dtype de trabajo. None = complex64 en GPU, complex128 en CPU.
#:
#: Es la politica de CamposT.backend. Las FASES se evaluan en float64 pase lo
#: que pase: el argumento de la onda esferica es k*r, que con lambda = 528 nm
#: y r ~ 10 mm vale ~1.2e5 rad, y en float32 eso pierde 0.008 rad de mantisa.
#: Lo que baja a simple es el fasor ya acotado a modulo 1.
DTYPE = None

#: Solo MOTOR = "carlos". SOBREMUESTREO de la malla en el plano de la muestra.
#:
#:   None   el factor que exige la geometria, of = delta/s. Es lo que hace
#:          reconstruction_dlhm.py, y con el sensor entero pide 15 GB por paso.
#:   1      la malla se queda como el sensor. Es la linea que Carlos dejo
#:          COMENTADA en su reconstruction_dlhm.py.
#:
#: El precio de 1 es el aliasing: la reconstruccion sale replicada en una
#: rejilla 3x3. El precio de None es que no cabe salvo recortando; y
#: recortando, la difraccion del borde del recorte tapaba el objeto hasta que
#: MotorCarlos empezo a dividir por |U0|^2.
SOBREMUESTREO = None

#: Solo MOTOR = "carlos". Presupuesto de memoria [GB] para la malla
#: remuestreada. Si el barrido pide mas, aborta ANTES de empezar con la tabla
#: de recortes, en vez de morir en el primer paso con un MemoryError que no dice
#: que hacer. En GPU manda la VRAM libre si es menor.
MEMORIA_MAX_GB = 4.0

#: Centinela para "usa la constante de arriba". No vale None, porque None YA
#: significa algo distinto -el sobremuestreo que exige la geometria- y no vale
#: 1 porque entonces no habria forma de pedir el otro. Existe para que
#: reconstruir() siga leyendo la constante cuando se la llama sin argumento, y
#: a la vez pueda fijarse desde fuera: comprobar_equivalencia() tiene que dar el
#: mismo numero se edite lo que se edite aqui arriba.
DE_LA_CONSTANTE = object()

#: Filas por bloque al construir la onda esferica y el kernel de la cadena de
#: Carlos. NO cambia el resultado, solo la memoria de pico.
FILAS_POR_BLOQUE = 512

#: Carpeta raiz de la salida: <RESULTADOS>/<nombre>_<motor>_<objeto>[_roi...]
RESULTADOS = RAIZ / "resultados" / "mi_prueba_dlhm"


# ════════════════════════════════════════════════════════════════════════════
#  2. LA FISICA DLHM  --  COPIADA TAL CUAL DE dlhm.py, NO EDITAR
# ════════════════════════════════════════════════════════════════════════════
#
# Estas seis funciones son de
# referencia/carlos/DLHM-model-main/DLHM-model-main/dlhm.py, sin tocar una
# linea -los [parche TG] de dlhm() ya estaban ahi: son los arreglos que hubo
# que hacerle para que corriera-. Su valor entero es que nadie las ha tocado:
# son contra lo que se contrasta cualquier version de trabajo. Si hay que
# cambiar algo, se hace en una copia, no aqui. tests/test_mi_prueba_dlhm_seleccion.py
# comprueba que sigan siendo identicas donde referencia/ existe.
#
# Las cinco primeras son la cadena de reconstruccion. dlhm() es el MODELO DE
# SIMULACION de Lopera et al., Opt. Express 32, 48509 (2024), el de
# SIMULADOR = "lopera": remuestrea la muestra por Mag = L/z, la propaga como
# onda plana Mag*(L - z) -su Ec. (14) impresa dice L - z, que no casa con su
# Ec. (1)-, le aplica la distorsion radial de Brown-Conrady, la multiplica por
# la envolvente de la fuente y la pasa a 8 bits. No propaga la onda esferica:
# la sustituye por la magnificacion, que es la aproximacion paraxial.

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


def dlhm(sample, dx_in, L, z, W_cx, W_cy, dx_out, wavelength, x0=0, y0=0, NA_s=0):
    """
    Digital Lensless Holography Model (DLHM) Simulation.

    Args:
    sample: np.ndarray
        Input complex field (wavefront) to be simulated.
    dx_in: float
        Resampling resolution parameter for input.
    L: float
        Distance from the point source to the hologram plane.
    z: float
        Distance from the point source to the sample plane.
    W_c: float
        Width of the sensor.
    dx_out: float
        Pixel size at the sensor.
    wavelength: float
        Wavelength of the light.
    x0, y0: float, optional
        Offsets to adjust the center of the cropped sample.
    NA_s: float, optional
        Numerical Aperture of the source.

    Returns:
    holo: np.ndarray
        Simulated hologram as intensity.
    """

    # Determine the discretization of the input sample
    N, M = sample.shape

    #determine the discretization of the sensor
    # [parche TG] P y Q son conteos de píxeles y se usaban sin redondear:
    # np.linspace(..., num=P) rechaza un flotante desde NumPy 1.18.
    Q = round(W_cy / dx_out)
    P = round(W_cx / dx_out)
    # P = M
    # Q = N

    # Magnification factor
    Mag = L / z
    W_s = W_cy / Mag  # the size of the sample's field of view

    # Re-sample if desired
    if dx_in != 0:
        res_d = dx_in * Mag / dx_out  # ratio of resizing considering the input sample's pixel and the desired
        sample = resize(sample, int(M / res_d), int(N / res_d))
        N_s, M_s = sample.shape

        W_provided = dx_in * N_s  # the size of the provided field of view is computed

        if W_provided > W_s:  # if the field of view provided is larger than the field of view that the system captures,
            # then only the illuminated portion is taken. Else, it is assumed that the provided field
            # of view is the illuminated portion.
            # Crop the sample based on the magnification and offsets
            start_x = int(N / 2 - Q / 2 + x0)
            end_x = int(N / 2 + Q / 2 + x0)
            start_y = int(M / 2 - P / 2 + y0)
            end_y = int(M / 2 + P / 2 + y0)
            sample = sample[start_x:end_x, start_y:end_y]

        else:
            sample = resize(sample, P, Q)


    N, M = sample.shape
    # Wave number
    k = 2 * np.pi / wavelength

    # Spatial coordinates in the camera's plane
    # [parche TG] la firma se actualizó a W_cx/W_cy pero el cuerpo quedó
    # usando W_c, que ya no existe: toda llamada moría con NameError.
    # x recorre P = W_cx/dx_out píxeles y y recorre Q = W_cy/dx_out.
    x = np.linspace(-W_cx / 2, W_cx / 2, P)
    y = np.linspace(-W_cy / 2, W_cy / 2, Q)
    u, v = np.meshgrid(x, y)

    # Radial distances for source wavefront
    r = np.sqrt(u ** 2 + v ** 2 + L ** 2)

    # Determine illumination pattern and normalization
    if NA_s != 0:
        PS = np.exp(-2 * r / (L * np.tan(np.arcsin(NA_s))))
    else:
        PS = np.exp(-r)
    PS = PS - np.min(PS)
    PS = PS / np.max(PS)

    # Spatial frequency coordinates
    dfx = 1 / (dx_out * M)
    dfy = 1 / (dx_out * N)
    fx, fy = np.meshgrid(np.arange(-M / 2 * dfx, M / 2 * dfx, dfx),
                         np.arange(-N / 2 * dfy, N / 2 * dfy, dfy))

    # Propagation kernel for the Angular Spectrum Method (ASM)
    E = np.exp(-1j * Mag * (L - z) * np.sqrt(k ** 2 - 4 * np.pi ** 2 * (fx ** 2 + fy ** 2)))

    # Compute hologram using inverse Fourier transform
    Uz = ifts(fts(sample) * E)
    holo = np.abs(Uz) ** 2

    # # Distortion of the coordinates
    # # Calculate maximum distortion due to distance differences
    # [parche TG] mismo W_c huérfano. sqrt(W_c**2/2 + L**2) es la distancia
    # del origen a la esquina de un sensor cuadrado; la forma rectangular
    # equivalente es sqrt((W_cx**2 + W_cy**2)/4 + L**2), que coincide con
    # la original cuando W_cx == W_cy.
    Mag_max = np.sqrt((W_cx ** 2 + W_cy ** 2) / 4 + L ** 2) / z
    Dist_max = np.abs(Mag_max - Mag)
    # Apply distortion to the hologram
    camMat = np.array([[P, 0, P / 2], [0, Q, Q / 2], [0, 0, 1]])
    distCoeffs = np.array([-Dist_max / (2 * Mag), 0, 0, 0, 0])  # Radial distortion parameters
    holo = cv.undistort(holo.astype(np.float32), camMat, distCoeffs)

    # Normalize and post-process the hologram
    holo = holo - np.min(holo)
    holo = holo / np.max(holo)
    holo = holo * PS
    holo = holo * 2 ** 8
    holo = np.round(holo).astype(np.uint8)

    return holo


# ════════════════════════════════════════════════════════════════════════════
#  2b. COPIA DE TRABAJO  --  la misma fisica, en la tarjeta
# ════════════════════════════════════════════════════════════════════════════
#
# La seccion 2 se queda intacta porque es la REFERENCIA. Estas son las mismas
# cuentas escritas contra xp -NumPy o CuPy- y por bloques de filas, que es lo
# que el resto del repo hace en CamposT.backend y en los retro_*.
#
# comprobar_equivalencia() corre las dos cadenas sobre la misma entrada en
# cada invocacion con MOTOR = "carlos": si alguna vez dejan de coincidir, se ve
# en la consola antes de mirar ningun resultado.

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

    Es la prueba de que acelerar no cambio el resultado. main() la corre en
    cada invocacion con MOTOR = "carlos", porque cuesta milisegundos y porque
    una version rapida que nadie contrasta contra la lenta no vale nada.

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
#  3. LA CADENA DE CARLOS: MALLA, MEMORIA Y RECONSTRUCCION
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

    Y SIEMPRE PAR: int(lado*of) y, si sale impar, el par siguiente. Con P
    impar, fts() deja la continua en el indice (P+1)/2 y la rejilla de
    angular_spectrum(),

        linspace(-P/2*dfx, (P/2 - 1)*dfx, P)

    le asigna ahi +dfx/2: no pasa por f = 0. H se evalua medio paso corrida y
    la reconstruccion sale desplazada lambda*(L - z)/(2W), con W el ancho del
    sensor. Con los numeros del montaje -528 nm, 1.83 um, z = 3, L = 10- un
    sensor de 1024 daba 2027 y un corrimiento de 0.99 um en el plano de la
    muestra, del orden del detalle del objeto. Contra la reconstruccion exacta
    de CamposT.dlhm, con el holograma complejo, la correlacion era 0.875; con
    2028 es 0.990, y 0.993 si ademas se remuestrea en banda limitada en vez de
    con cv.resize (scripts/verif_dlhm_esferico.py, parte carlos). Es el mismo
    fallo que documenta frecuencias_fft() en CamposT/propagadores.py.

    El arreglo va aqui porque angular_spectrum() es copia literal de dlhm.py y
    no se toca, y espectro_angular() tiene que darle lo mismo bit a bit. Hacia
    ARRIBA porque subir no le quita nunca un punto a la malla: el paso W/N
    solo puede afinarse. Y solo los impares: donde int() ya daba par la malla
    es la de siempre, y la tabla de la cabecera -13144, 2802, 1624- sigue
    valiendo con su memoria y sus tiempos.

    El precio: cuando int(lado*of) sale impar, reconstruir() deja de ser bit a
    bit el reconstruction_dlhm.py de Carlos, que se queda con la impar. Es a
    proposito. Lo fija tests/test_malla_remuestreada.py.

    QUEDA ABIERTO: esto solo cubre la rama que remuestrea. Con SOBREMUESTREO =
    1 la malla es la del holograma, y un recorte impar se corre igual: medido
    con 511 de lado, 2 um.
    """
    Wx, Wy = lado_x * delta, lado_y * delta
    s = lamb * np.sqrt((Wx / 2) ** 2 + (Wy / 2) ** 2 + z ** 2) / Wx
    of = delta / s
    N, M = int(lado_y * of), int(lado_x * of)
    return N + N % 2, M + M % 2


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
    # el peor caso es la z mas chica: s crece con z, y la malla es Wx/s. El
    # redondeo al par no lo cambia: n + n % 2 no baja nunca al subir n
    if SOBREMUESTREO is not None:
        gb = 6 * Q * P * itemsize / 2 ** 30
        print(f"  SOBREMUESTREO = {SOBREMUESTREO}: la malla se queda en "
              f"{Q}x{P} ({Q * P / 1e6:.1f} Mpx, ~{gb:.2f} GB de pico). "
              f"Alia, ver SOBREMUESTREO.")
        if gb > tope_gb:
            raise SystemExit(
                f"Ni siquiera sin remuestrear cabe: {gb:.2f} GB de pico "
                f"contra un tope de {tope_gb:.2f} GB.\nRecorta con un "
                f"ROI mas chico, o usa DISPOSITIVO = 'cpu'.")
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
        f"submuestrear no ayuda,\nhay que RECORTAR. Con ROI (cuadrado y centrado):\n\n"
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


# ════════════════════════════════════════════════════════════════════════════
#  4. LOS DOS MOTORES
# ════════════════════════════════════════════════════════════════════════════
#
# La seleccion (metricas, barrido, region) no sabe que motor la alimenta: los
# dos tienen campo(z, ventana) -> t complejo en el dispositivo, sobre la
# ventana (fila, columna, alto, ancho) de la malla que el sensor PROYECTA
# desde la fuente: paso DELTA*z/L, un pixel por pixel del sensor.
#
# En esa malla un detalle cae en el MISMO pixel a toda z: el rayo principal
# que llega al pixel X del sensor cruza el plano z en xi = X*z/L, y
# xi/(DELTA*z/L) = X/DELTA. Asi que el ROI es una ventana fija de pixeles,
# valida en todo el barrido y alineada con el holograma y con la referencia,
# como en barrido_foco, donde la reconstruccion y |U0| comparten malla.


class MotorExacto:
    """CamposT.dlhm.Reconstructor: el holograma ENTERO, y en cada z la ventana.

    El espectro del sensor con la esferica dentro se calcula una vez, al
    construirlo; cada campo() cuesta la funcion de transferencia y los dos
    productos de salida, en proporcion a la ventana. Por eso aqui el ROI tiene
    el sentido de barrido_foco: no recorta el holograma, solo evalua menos.

    Se construye para el rango de z de TODO el barrido, fino incluido (ver
    rango_del_barrido): la rejilla de frecuencias se dimensiona para el peor z
    del rango, y pedir una z fuera aborta.
    """

    def __init__(self, holo, lamb, delta, L_fuente, z_rango, fondo=None,
                 device="auto", dtype=None):
        self.R = dlhm_exacto.Reconstructor(holo, delta, lamb, L_fuente, z_rango,
                                    fondo=fondo, device=device, dtype=dtype)
        self.xp = self.R.xp
        self.forma = self.R.forma
        self.delta, self.L = float(delta), float(L_fuente)
        #: la ventana mas grande que sabe dar: la malla entera
        self.ventana_maxima = (0, 0, *self.forma)

    def campo(self, z, ventana):
        f0, c0, alto, ancho = ventana
        M, N = self.forma
        p = self.delta * z / self.L
        centro = ((f0 + alto / 2 - M / 2) * p, (c0 + ancho / 2 - N / 2) * p)
        return self.R.campo(z, forma=(alto, ancho), centro=centro)

    def describir(self):
        (nfy, nfx), (Py, Px) = self.R.n_frecuencias, self.R.P
        mb = nfy * nfx * np.dtype(self.R.dtype).itemsize / 2 ** 20
        return (f"rejilla de frecuencias {nfy}x{nfx}, periodo {Py:.3f} x "
                f"{Px:.3f} mm, espectro de {mb:.0f} MB")


def interpolar_centrado(A, paso, forma, paso_nuevo, xp=np):
    """A, en una malla centrada de paso `paso`, en la centrada forma x paso_nuevo.

    Lineal por eje, real e imaginaria juntas. "Centrada" es el convenio de todo
    el archivo: el indice i esta en (i - n/2)*paso. Se interpola en las
    coordenadas EXACTAS y no con cv.resize, que alinea los centros de pixel de
    las dos mallas y no sus coordenadas: corre la imagen hasta medio pixel (es
    el residuo de (delta - p)/(2M) que cita malla_remuestreada).

    Aborta si la malla nueva se sale de la vieja: extrapolar seria inventar.
    """
    def eje(A, n_nuevo, axis):
        n = A.shape[axis]
        u = (np.arange(n_nuevo) - n_nuevo / 2) * paso_nuevo / paso + n / 2
        if u[0] < -1e-9 or u[-1] > n - 1 + 1e-9:
            raise ValueError(
                f"la malla nueva ({n_nuevo} puntos de {paso_nuevo:.4g} mm) se sale "
                f"de la vieja ({n} de {paso:.4g} mm)")
        i0 = np.clip(np.floor(u).astype(np.int64), 0, n - 2)
        forma_w = [1, 1]
        forma_w[axis] = n_nuevo
        w = xp.asarray((u - i0).reshape(forma_w), dtype=A.real.dtype)
        return (xp.take(A, xp.asarray(i0), axis=axis) * (1 - w)
                + xp.take(A, xp.asarray(i0 + 1), axis=axis) * w)
    return eje(eje(A, forma[0], 0), forma[1], 1)


def onda_de_referencia(forma, z, lamb, delta, L_fuente, xp=np,
                       dtype=np.complex128, sobremuestreo=None):
    """U0: la onda de la fuente retropropagada a la muestra, la de reconstruir().

    reconstruir() la calcula por dentro y no la devuelve: la esferica en el
    sensor, recortada al holograma, propagada L - z. Esta es la misma cuenta
    -misma malla, remuestreada o no, mismo paso, mismas funciones-, asi que
    reconstruir(unos) es |U0|^2. Lo fija una prueba.
    """
    Q, P = forma
    if sobremuestreo is None:
        N, M = malla_remuestreada(P, Q, z, lamb, delta)
    else:
        N, M = Q, P
    kw = dict(xp=xp, dtype=dtype)
    referencia = fuente_puntual(N, M, L_fuente, 0, 0, lamb, (delta * P) / N, **kw)
    return espectro_angular(referencia, P * delta, Q * delta, 2 * np.pi / lamb,
                            L_fuente - z, **kw)


class MotorCarlos:
    """reconstruir() -la cadena de Carlos- sobre el recorte del ROI.

    TRES COSAS LA HACEN COMPARABLE CON LA EXACTA:

    1. EL RECORTE ES OBLIGADO, CENTRADO Y CUADRADO. Obligado por memoria: con
       el sensor entero la malla remuestreada pide 15 GB por paso (ver
       comprobar_memoria). Centrado porque reconstruir() pone la fuente en el
       centro del recorte: con un ROI descentrado la fisica de este motor ya
       no es la del banco, y main() lo avisa. Cuadrado porque reconstruir()
       usa el mismo paso, delta*P/N, en los dos ejes: con un recorte
       rectangular la esferica sale con otra curvatura en x que en y.

    2. SE DIVIDE POR |U0|^2. reconstruir() devuelve Rec = U*conj(U0), que quita
       la FASE de la referencia pero deja su AMPLITUD, |U0|^2, con la
       difraccion del borde del recorte encima: es la causa 1 de la cabecera,
       "la reconstruccion queda DOMINADA por la difraccion del borde".
       Rec/|U0|^2 = U/U0 la quita. Medido en una malla de 128 px con el objeto
       de las pruebas: correlacion con el objeto 0.84 dividiendo, 0.21 sin
       dividir. Cuesta una propagacion mas por z (tres en vez de dos).

    3. SE DEVUELVE LA HUELLA, EN LA MALLA DEL ROI. La salida de reconstruir()
       abarca en la muestra el mismo ancho que el recorte en el sensor, pero el
       objeto que el recorte ve ocupa solo su 1/M central: los rayos
       principales que llegan al recorte cruzan el plano z en su ancho por z/L.
       Esa huella se interpola en la malla del ROI, paso delta*z/L, que es la
       del motor exacto: asi las metricas no dependen del motor.

    El holograma entra normalizado a [0, 1], como antes: una escala global no
    cambia ninguna metrica.
    """

    def __init__(self, holo, ventana, lamb, delta, L_fuente, xp=np,
                 dtype=np.complex128, sobremuestreo=None):
        holo = np.asarray(holo)
        f0, c0, alto, ancho = (int(v) for v in ventana)
        if alto != ancho:
            M, N = holo.shape
            lado = min(alto, ancho)
            raise SystemExit(
                f"El motor de Carlos necesita un ROI CUADRADO y la region medida "
                f"es de {alto}x{ancho} px (alto x ancho): reconstruir() usa el "
                f"mismo paso en los dos ejes, delta*P/N, y con un recorte "
                f"rectangular la esferica sale con otra curvatura en x que en y."
                f"\nUsa un ROI cuadrado y centrado, por ejemplo\n\n"
                f"    ROI = ({(N - lado) // 2}, {(M - lado) // 2}, {lado}, {lado})\n")
        h = holo[f0:f0 + alto, c0:c0 + ancho]
        self.h = h / np.abs(h).max()
        self.ventana = self.ventana_maxima = (f0, c0, alto, ancho)
        self.lamb, self.delta, self.L = float(lamb), float(delta), float(L_fuente)
        self.xp, self.dtype, self.sobremuestreo = xp, dtype, sobremuestreo

    def campo(self, z, ventana):
        if tuple(int(v) for v in ventana) != self.ventana:
            raise ValueError(f"el motor de Carlos solo reconstruye su recorte, "
                             f"{self.ventana}; se pidio {tuple(ventana)}")
        kw = dict(xp=self.xp, dtype=self.dtype, sobremuestreo=self.sobremuestreo)
        Rec = reconstruir(self.h, z, self.lamb, self.delta, self.L, **kw)
        U0 = onda_de_referencia(self.h.shape, z, self.lamb, self.delta, self.L, **kw)
        Rec /= self.xp.abs(U0) ** 2
        del U0
        n = self.h.shape[0]
        return interpolar_centrado(Rec, self.delta * n / Rec.shape[0], (n, n),
                                   self.delta * z / self.L, self.xp)

    def describir(self, z_rango):
        n = self.h.shape[0]
        if self.sobremuestreo is not None:
            return (f"recorte {n}x{n} sin remuestrear (SOBREMUESTREO = "
                    f"{self.sobremuestreo}): alia, ver la cabecera")
        z_lo, z_hi = min(z_rango), max(z_rango)
        lo = malla_remuestreada(n, n, z_lo, self.lamb, self.delta)[0]
        hi = malla_remuestreada(n, n, z_hi, self.lamb, self.delta)[0]
        return (f"recorte {n}x{n}, malla remuestreada {lo}x{lo} a z = {z_lo:.3f} "
                f"mm y {hi}x{hi} a z = {z_hi:.3f} mm")


# ════════════════════════════════════════════════════════════════════════════
#  5. LAS METRICAS  --  COPIADAS DE scripts/barrido_foco.py
# ════════════════════════════════════════════════════════════════════════════
#
# tamura, pearson, ssim, anchura_media y margen_del_pico son las de
# barrido_foco con el MISMO cuerpo -solo cambian los docstrings- y leen las
# mismas globales de modulo: xp, REF, REF_CPU y REF_RANGO, que
# fijar_referencia() pone antes del barrido.
# tests/test_mi_prueba_dlhm_seleccion.py saca las de barrido_foco de su fuente
# con ast (importarlo correria su barrido entero) y comprueba que den los
# mismos numeros y tengan el mismo cuerpo. Si hay que cambiar una, se cambian
# las dos.
#
# Todas reciben el mapa YA en la ventana medida -A en el dispositivo, A_cpu el
# mismo en NumPy- y se MAXIMIZAN en el foco. El sentido de OBJETO = "fase" va
# fuera, en puntuar(), para que las copias sigan siendo literales.

#: Las globales de las metricas. De arranque, la CPU y sin referencia.
xp = np
REF = REF_CPU = REF_RANGO = None


def tamura(A, A_cpu):
    """sqrt(std(A) / media(A)). SIN REFERENCIA: la unica que sirve en el banco.

    No depende de la escala del campo, que en DLHM no significa nada: la fija
    el fondo por el que se dividio el holograma.
    """
    m = A.mean()
    if m <= 0:
        return 0.0
    return float(xp.sqrt(A.std() / m))


def pearson(A, A_cpu):
    """Correlacion de Pearson entre A y REF. CON REFERENCIA.

    Invariante a escala y a offset: la reconstruccion no sale con la escala del
    objeto y, con un holograma de intensidad, trae la gemela encima.
    """
    a = A - A.mean()
    r = REF - REF.mean()
    d = xp.sqrt((a * a).sum() * (r * r).sum())
    if d <= 0:
        return 0.0
    return float((a * r).sum() / d)


def ssim(A, A_cpu):
    """SSIM entre A -llevada a la escala de REF por minimos cuadrados- y REF.

    SSIM no es invariante a escala; por eso el ajuste a*A + b. Va en la CPU:
    skimage no tiene version GPU.
    """
    va = A_cpu.var()
    if va <= 0:
        return 0.0
    a = float(((A_cpu - A_cpu.mean()) * (REF_CPU - REF_CPU.mean())).mean() / va)
    b = float(REF_CPU.mean() - a * A_cpu.mean())
    return float(structural_similarity(a * A_cpu + b, REF_CPU, data_range=REF_RANGO))


FUNCION = {"tamura": tamura, "pearson": pearson, "ssim": ssim}
COLOR = {"tamura": "C0", "pearson": "C1", "ssim": "C2"}


def anchura_media(zs, c):
    """Anchura a media altura del pico de c(zs), en mm, o nan si no se puede.

    La curva se reescala a [0, 1] y se interpola donde cruza 0.5 a cada lado
    del maximo. nan si el pico llega al extremo sin bajar de la mitad. Sale del
    barrido GRUESO: no resuelve picos de menos de un par de pasos.
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

    El fondo es lo que queda a mas de `ancho` mm del pico. El rizado es la std
    de las diferencias entre vecinos / sqrt(2): mata las tendencias suaves y
    deja el salto punto a punto. La altura, contra la MEDIANA del fondo. En
    DLHM el pico es estrecho: main() pasa ancho = una decima del barrido.
    """
    fuera = np.abs(zs - z_pico) > ancho
    if fuera.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    o = c[fuera]
    base = float(np.median(o))
    rizado = float(np.std(np.diff(o)) / np.sqrt(2))
    altura = float(c.max() - base)
    return altura, rizado, (altura / rizado if rizado > 0 else float("inf"))


def clase_de(nombre, objeto):
    """Que mapa mide una metrica: "amplitud" (|t|) o "fase" (angle(t)).

    Tamura mira siempre la amplitud, porque es contraste. Las de referencia
    miran lo que el objeto modula: con OBJETO = "fase", la fase.
    """
    return "fase" if objeto == "fase" and nombre != "tamura" else "amplitud"


def mapa(t, clase, xp):
    """El mapa real que mide una metrica, en float64.

    En doble aunque t venga en complex64: las metricas suman millones de
    pixeles, y en simple el error acumulado es del orden de lo que separa dos
    pasos vecinos del barrido.
    """
    if clase == "fase":
        return xp.angle(t).astype(np.float64)
    return xp.abs(t).astype(np.float64)


def puntuar(nombre, A, A_cpu, objeto):
    """FUNCION[nombre] con el sentido de OBJETO: todas se maximizan en el foco.

    "fase": la Tamura cambia de signo, porque un objeto de fase enfoca en el
    MINIMO de contraste de amplitud (en el foco |t| = 1), y Pearson va en valor
    absoluto, porque el signo de la fase depende del convenio del modelo
    (Carlos usa e^{-i phi}). SSIM no cambia: ya ajusta a*A + b, y el signo de
    a le da igual.
    """
    v = FUNCION[nombre](A, A_cpu)
    if objeto == "fase":
        if nombre == "tamura":
            return -v
        if nombre == "pearson":
            return abs(v)
    return v


def fijar_referencia(ref, xp_):
    """Las globales que leen las metricas copiadas.

    ref es el mapa de la referencia en la ventana medida -el de la clase de las
    metricas con referencia-, o None sin referencia: entonces solo vale Tamura.
    xp_ es el modulo del motor: REF vive donde vive A.
    """
    global xp, REF, REF_CPU, REF_RANGO
    xp = xp_
    if ref is None:
        REF = REF_CPU = REF_RANGO = None
        return
    REF_CPU = np.asarray(ref, dtype=np.float64)
    REF = xp_.asarray(REF_CPU)
    REF_RANGO = float(REF_CPU.max() - REF_CPU.min())


def etiquetas(objeto):
    """La leyenda de cada curva, segun lo que mide con ese OBJETO."""
    if objeto == "fase":
        return {"tamura": "-Tamura de |t|  (SIN referencia)",
                "pearson": "|Pearson| de angle(t) con la fase  (con referencia)",
                "ssim": "SSIM de angle(t) con la fase  (con referencia)"}
    return {"tamura": "Tamura de |t|  (SIN referencia)",
            "pearson": "Pearson de |t| con la referencia  (con referencia)",
            "ssim": "SSIM de |t| con la referencia  (con referencia)"}


# ════════════════════════════════════════════════════════════════════════════
#  6. EL BARRIDO  --  el de barrido_foco, con el motor en vez de mpasm()
# ════════════════════════════════════════════════════════════════════════════

def barrido(motor, ventana, zs, metricas, objeto, en_cpu=(), solapar=False,
            reloj=None, guardar=None):
    """{nombre: curva}: todas las metricas medidas sobre la MISMA reconstruccion.

    Es el barrido() de barrido_foco con el motor en lugar de mpasm() y hasta
    dos mapas por z en lugar de uno: con OBJETO = "fase" la Tamura mide |t| y
    las de referencia angle(t) (clase_de). Cada mapa baja a la CPU una vez.

    en_cpu   las metricas que corren en la CPU: reciben A = None y solo pueden
             usar A_cpu. Meter una que use A la rompe, y no en silencio.
    solapar  con GPU y alguna metrica en_cpu, encadena: lanza la
             reconstruccion del paso i y, mientras la GPU la calcula, cierra en
             la CPU las metricas del paso i-1. No cambia ningun numero -las
             curvas se escriben por INDICE, porque el paso i-1 se cierra una
             vuelta mas tarde-, solo el reloj.
    reloj    dict de segundos por etapa que se ACUMULA: el grueso y los finos
             comparten uno. "reconstruir" es lanzar el motor y esperarlo; con
             solapar, "espera GPU": lo que la CPU no consiguio esconder.
    guardar  f(t, z), llamada en cada z, o None.
    """
    xp_ = motor.xp
    en_gpu_lista = [n for n in metricas if n not in en_cpu]
    en_cpu_lista = [n for n in metricas if n in en_cpu]
    solapando = solapar and xp_ is not np and bool(en_cpu_lista)
    clave = "espera GPU" if solapando else "reconstruir"
    reloj = {} if reloj is None else reloj
    for k in (clave, *metricas):
        reloj.setdefault(k, 0.0)
    clases = {clase_de(n, objeto) for n in metricas}
    curvas = {n: np.empty(len(zs)) for n in metricas}
    pendiente = None                       # (indice, mapas en la CPU) del paso anterior

    def cerrar_en_cpu(i, mapas_cpu):
        for n in en_cpu_lista:
            t0 = time.perf_counter()
            curvas[n][i] = puntuar(n, None, mapas_cpu[clase_de(n, objeto)], objeto)
            reloj[n] += time.perf_counter() - t0

    for i, z in enumerate(zs):
        t0 = time.perf_counter()
        t = motor.campo(z, ventana)        # en la GPU lanza y vuelve enseguida
        mapas = {c: mapa(t, c, xp_) for c in clases}
        lanzar = time.perf_counter() - t0

        # la GPU esta ocupada con lo de arriba: cerrar el paso anterior
        if pendiente is not None:
            cerrar_en_cpu(*pendiente)
            pendiente = None

        # AQUI se espera al motor: bajar a la CPU sincroniza
        t0 = time.perf_counter()
        mapas_cpu = {c: a_cpu(A) for c, A in mapas.items()}
        reloj[clave] += lanzar + time.perf_counter() - t0

        for n in en_gpu_lista:
            t0 = time.perf_counter()
            c = clase_de(n, objeto)
            curvas[n][i] = puntuar(n, mapas[c], mapas_cpu[c], objeto)
            reloj[n] += time.perf_counter() - t0

        if guardar is not None:
            guardar(t, z)
        del t, mapas

        if solapando:
            pendiente = (i, mapas_cpu)     # se cierra en la vuelta siguiente
        else:
            cerrar_en_cpu(i, mapas_cpu)

    if pendiente is not None:              # el ultimo paso no tiene siguiente
        cerrar_en_cpu(*pendiente)
    return curvas


def enfocar(motor, ventana, zs, pasos_fino, metricas, objeto, en_cpu=(),
            solapar=False, reloj=None, guardar_en=None):
    """El barrido grueso y un fino por cada pico distinto, como barrido_foco.

    -> (curvas, picos, finos, z_foco): las curvas gruesas, el indice grueso del
    maximo de cada metrica, {indice: (zs_fino, curvas_finas)} y la z elegida.

    El fino va de zs[i] - paso a zs[i] + paso, un paso grueso por cada lado: por
    eso rango_del_barrido() da ese rango al motor exacto. Se lanza UNO por
    indice grueso distinto, compartido por las metricas que lo eligieron:
    reconstruir dos veces a la misma z no da dos respuestas.

    guardar_en(subcarpeta) -> f(t, z) o None: "grueso" y "fino_z<zs[i]>".
    """
    paso = zs[1] - zs[0]
    kw = dict(en_cpu=en_cpu, solapar=solapar, reloj=reloj)
    guardar = guardar_en or (lambda _: None)
    curvas = barrido(motor, ventana, zs, metricas, objeto,
                     guardar=guardar("grueso"), **kw)
    picos = {n: int(np.argmax(c)) for n, c in curvas.items()}
    finos, z_foco = {}, {}
    for i in sorted(set(picos.values())):
        zs_f = np.linspace(zs[i] - paso, zs[i] + paso, pasos_fino)
        finos[i] = (zs_f, barrido(motor, ventana, zs_f, metricas, objeto,
                                  guardar=guardar(f"fino_z{zs[i]:.4f}"), **kw))
    for n, i in picos.items():
        zs_f, cf = finos[i]
        z_foco[n] = float(zs_f[int(np.argmax(cf[n]))])
    return curvas, picos, finos, z_foco


# ════════════════════════════════════════════════════════════════════════════
#  7. LA CONFIGURACION Y LA REGION MEDIDA
# ════════════════════════════════════════════════════════════════════════════

def comprobar_configuracion(metricas, formato, objeto, motor, campo, ruta,
                            referencia, z_real, simulador="exacto"):
    """Aborta ANTES de leer ni reconstruir nada si la combinacion no tiene sentido."""
    if formato not in ("png", "npy", "ambos"):
        raise SystemExit(f'FORMATO_BARRIDO = {formato!r}: usa "png", "npy" o "ambos".')
    if not metricas:
        raise SystemExit("METRICAS esta vacio: no hay nada con que elegir el foco.")
    for n in metricas:
        if n not in FUNCION:
            raise SystemExit(f"METRICAS trae {n!r}, y solo existen {tuple(FUNCION)}.")
    if objeto not in ("amplitud", "fase"):
        raise SystemExit(f'OBJETO = {objeto!r}: usa "amplitud" o "fase".')
    if motor not in ("exacto", "carlos"):
        raise SystemExit(f'MOTOR = {motor!r}: usa "exacto" o "carlos".')
    if ruta is None:
        if referencia is None:
            raise SystemExit("RUTA = None simula el holograma de REFERENCIA, y "
                             "REFERENCIA tambien es None: no hay objeto que simular.")
        if z_real is None:
            raise SystemExit("RUTA = None simula el holograma a Z_REAL, y Z_REAL es "
                             "None: di a que z esta la muestra.")
        if simulador not in ("exacto", "lopera"):
            raise SystemExit(f'SIMULADOR = {simulador!r}: usa "exacto" o "lopera".')
        if simulador == "exacto" and campo not in ("intensidad", "complejo"):
            raise SystemExit(f'CAMPO = {campo!r}: usa "intensidad" o "complejo".')
    con_referencia = [n for n in metricas if n != "tamura"]
    if referencia is None and con_referencia:
        raise SystemExit(
            f"METRICAS pide {', '.join(con_referencia)} y REFERENCIA = None: sin el "
            f"objeto no hay contra que compararlas.\nCon un holograma de laboratorio "
            f'pon METRICAS = ("tamura",), la unica que no necesita referencia.')


def rango_del_barrido(z_min, z_max, pasos, L_fuente):
    """-> (zs, paso, (z_lo, z_hi)): el grueso, y el rango que cubre tambien el fino.

    El fino se sale del grueso un paso por cada lado, asi que el motor exacto
    se construye para [z_min - paso, z_max + paso] y ese rango tiene que caber
    en 0 < z < L: la muestra esta ENTRE la fuente y el sensor.
    """
    if pasos < 2:
        raise SystemExit(f"PASOS = {pasos}: hacen falta dos o mas para tener paso.")
    zs = np.linspace(z_min, z_max, pasos)
    paso = zs[1] - zs[0]
    z_lo, z_hi = zs[0] - paso, zs[-1] + paso
    if not 0 < z_lo < z_hi < L_fuente:
        raise SystemExit(
            f"El barrido de {z_min:g} a {z_max:g} mm con paso {paso:g}, y el fino que "
            f"se sale un paso por cada lado, va de {z_lo:g} a {z_hi:g} mm: tiene que "
            f"quedar en 0 < z < L = {L_fuente:g} mm, con z_min < z_max. La muestra "
            f"esta ENTRE la fuente y el sensor.")
    return zs, paso, (z_lo, z_hi)


def region_medida(forma, roi, borde_px, imagen=None, titulo=""):
    """-> ((fila, columna, alto, ancho), Roi o None): donde miden las metricas.

    roi = None                   la malla menos borde_px por lado; con
                                 borde_px = None, 3*min(M, N)//8, que deja el
                                 cuarto central del lado corto (D9: la
                                 periferia trae la gemela y la perdida de NA
                                 del borde del sensor).
    roi = True                   se arrastra con el raton sobre `imagen`.
    roi = (X0, Y0, ANCHO, ALTO)  ventana fija, X0 la columna e Y0 la fila.

    Con un ROI borde_px no se usa, como en barrido_foco.
    """
    M, N = forma
    if roi is None:
        borde = 3 * min(M, N) // 8 if borde_px is None else int(borde_px)
        if borde < 0 or min(M, N) - 2 * borde < 64:
            raise SystemExit(
                f"BORDE_PX = {borde} por lado no deja una region de 64 px de lado en "
                f"una malla {M}x{N}. Bajalo, o fija un ROI.")
        return (borde, borde, M - 2 * borde, N - 2 * borde), None
    if roi is True:
        if imagen is None:
            raise ValueError("ROI = True necesita la imagen sobre la que arrastrar")
        r = elegir(imagen, titulo)
    else:
        if not (isinstance(roi, (tuple, list)) and len(roi) == 4):
            raise SystemExit(
                f"ROI = {roi!r} no es ninguno de los valores validos:\n\n"
                f"    None                    el centro menos BORDE_PX\n"
                f"    True                    la arrastras con el raton\n"
                f"    (X0, Y0, ANCHO, ALTO)   ventana fija\n")
        r = Roi(*roi)
    if r.x0 + r.ancho > N or r.y0 + r.alto > M:
        raise SystemExit(f"ROI {r} se sale de la malla {M}x{N}.")
    if min(r.ancho, r.alto) < 64:
        raise SystemExit(f"ROI {r}: menos de 64 px de lado no da para medir contraste.")
    return (r.y0, r.x0, r.alto, r.ancho), r


# ════════════════════════════════════════════════════════════════════════════
#  8. EL OBJETO, EL HOLOGRAMA Y LA SALIDA
# ════════════════════════════════════════════════════════════════════════════

def leer_imagen(ruta):
    """Un .npy tal cual (real o complejo), o una imagen en gris a [0, 1] (/255)."""
    ruta = pathlib.Path(ruta)
    if not ruta.exists():
        raise SystemExit(f"No existe:\n    {ruta}")
    if ruta.suffix.lower() == ".npy":
        a = np.load(ruta)
    else:
        a = cv.imread(str(ruta), cv.IMREAD_GRAYSCALE)
        if a is None:
            raise SystemExit(f"No pude leer la imagen:\n    {ruta}")
        a = a.astype(np.float64) / 255.0
    if a.ndim != 2:
        raise SystemExit(f"{ruta} no es una imagen 2-D: forma {a.shape}.")
    return a


def objeto_de(img, objeto, invertir, fase_max):
    """La transmitancia t del objeto: g = 1 - img o img, y t = g o exp(i*fase_max*g)."""
    g = 1 - img if invertir else img
    if objeto == "fase":
        return np.exp(1j * fase_max * g)
    return g.astype(np.complex128)


def fondo_de(img):
    """La mediana del marco de la imagen: lo que hay alrededor del objeto."""
    return float(np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])))


def simular(t, lamb, delta, L_fuente, z, campo, device="auto", dtype=None,
            simulador="exacto", na_fuente=0.1):
    """El holograma de t en un sensor de la forma de t, paso delta. En NumPy.

    t va en la malla que el sensor proyecta, paso delta*z/L: un pixel del objeto
    por pixel del sensor.

    simulador = "exacto": la ida de CamposT.dlhm. campo "intensidad" da el de
    contraste |1 + V|^2 = I/I_ref; "complejo", 1 + V, sin imagen gemela.

    simulador = "lopera": dlhm(), su modelo, tal cual, y su salida de 8 bits
    pasada a [0, 1]; campo no aplica. Se le pasa dx_in = 0, que salta su
    remuestreo: con t ya en la malla proyectada y con la forma del sensor ese
    paso es la identidad -una prueba lo fija bit a bit contra dx_in =
    delta*z/L- y asi no pasa por la rama del recorte, que esta rota (ver la
    cabecera).
    """
    if simulador == "lopera":
        M, N = t.shape
        return dlhm(t, 0, L_fuente, z, N * delta, M * delta, delta, lamb,
                    NA_s=na_fuente) / 255.0
    h = a_cpu(dlhm_exacto.holograma(t, delta * z / L_fuente, lamb, z, L_fuente, t.shape,
                                    delta, complejo=campo == "complejo", device=device,
                                    dtype=dtype))
    return 1 + h if campo == "complejo" else h


def vista(a, objeto="amplitud"):
    """Lo que se dibuja o se guarda como PNG, a [0, 1].

    Un campo complejo: |a|^2, o angle(a) con OBJETO = "fase". Un holograma
    real, tal cual. Cada uno a su propio maximo (o rango).
    """
    a = np.asarray(a)
    if objeto == "fase" and np.iscomplexobj(a):
        v = np.angle(a)
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    v = np.abs(a) ** 2 if np.iscomplexobj(a) else a.astype(np.float64)
    m = v.max()
    return v / m if m > 0 else v


def carpeta_de_salida(raiz, ruta, referencia, z_real, campo, motor, objeto, roi):
    """-> (carpeta, nombre). carpeta = raiz/<nombre>_<motor>_<objeto>[_roi...].

    nombre es el del holograma, o sim_<referencia>_z<Z_REAL>mm_<CAMPO o lopera> si se
    simula: el simulado se guarda como <nombre>.npy, asi que leerlo despues con
    RUTA da el mismo nombre y escribe en la misma carpeta.
    """
    if ruta is None:
        nombre = f"sim_{pathlib.Path(referencia).stem}_z{z_real:g}mm_{campo}"
    else:
        nombre = pathlib.Path(ruta).stem
    carpeta = f"{nombre}_{motor}_{objeto}"
    if roi is not None:
        carpeta += f"_roi{roi.x0}-{roi.y0}-{roi.ancho}x{roi.alto}"
    return pathlib.Path(raiz) / carpeta, nombre


def guardar_instante(t, z, carpeta, formato, objeto):
    """La ventana reconstruida a z, en png, npy o los dos."""
    carpeta.mkdir(parents=True, exist_ok=True)
    t = a_cpu(t)
    nombre = f"z{z:08.4f}mm"
    if formato in ("npy", "ambos"):
        np.save(carpeta / f"{nombre}.npy", t)
    if formato in ("png", "ambos"):
        plt.imsave(carpeta / f"{nombre}.png", vista(t, objeto), cmap="gray",
                   vmin=0, vmax=1)


def informe_de_focos(zs, curvas, picos, z_foco, metricas, z_real=None, ancho=None):
    """La tabla de barrido_foco, como texto: grueso, foco, error, FWHM y margen.

    ancho es el que va a margen_del_pico(): el fondo es lo que queda a mas de
    ancho mm del pico. None = una decima del barrido, porque en DLHM el pico
    mide centesimas de mm y el 1 mm de barrido_foco se comeria el barrido.
    """
    paso = zs[1] - zs[0]
    ancho = (zs[-1] - zs[0]) / 10 if ancho is None else ancho
    lineas = [f"{'metrica':>9} {'grueso':>9} {'foco':>9} {'error':>9} {'FWHM':>9} "
              f"{'altura':>9} {'rizado':>9} {'alt/riz':>8}"]
    for n in metricas:
        c = curvas[n]
        w = anchura_media(zs, c)
        alt, riz, razon = margen_del_pico(zs, c, zs[picos[n]], ancho)
        error = f"{z_foco[n] - z_real:+9.4f}" if z_real is not None else f"{'-':>9}"
        lineas.append(f"{n:>9} {zs[picos[n]]:9.4f} {z_foco[n]:9.4f} {error} {w:9.4f} "
                      f"{alt:9.4f} {riz:9.4f} {razon:8.1f}")
    lineas.append(
        f"(error = foco - Z_REAL; FWHM del barrido grueso, no resuelve por debajo de "
        f"~{2 * paso:g} mm; altura sobre la mediana del fondo -a mas de {ancho:g} mm "
        f"del pico- y rizado punto a punto de ese fondo, en unidades de la propia "
        f"metrica: solo alt/riz se compara entre metricas)")
    return "\n".join(lineas)


def guardar_curvas(ruta, zs, curvas, finos, metricas):
    """Las curvas en crudo, siempre: volver a dibujarlas no debe exigir barrer."""
    volcado = {"zs": zs, **{f"grueso_{n}": curvas[n] for n in metricas}}
    for i, (zs_f, cf) in finos.items():
        volcado[f"fino_{i}_zs"] = zs_f
        volcado.update({f"fino_{i}_{n}": cf[n] for n in metricas})
    np.savez(ruta, **volcado)


def figura_barrido(zs, curvas, finos, z_foco, metricas, objeto, z_real=None, titulo=""):
    """Las curvas, cada una reescalada a [0, 1] entre su minimo y su maximo.

    Y no dividida por su maximo como en barrido_foco: con OBJETO = "fase" la
    Tamura cambiada de signo es negativa, y dividir por un maximo negativo le
    da la vuelta. Reescalar conserva donde esta el pico y su anchura, que es lo
    que se compara.
    """
    paso = zs[1] - zs[0]
    etiqueta = etiquetas(objeto)
    escala = {}
    for n in metricas:
        todas = np.concatenate([curvas[n]] + [cf[n] for _, cf in finos.values()])
        lo, hi = float(todas.min()), float(todas.max())
        escala[n] = (lo, hi - lo if hi > lo else 1.0)
    centro = z_real if z_real is not None else z_foco[metricas[0]]

    fig, (b1, b2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                 gridspec_kw={"width_ratios": [2, 1]})
    for n in metricas:
        lo, r = escala[n]
        b1.plot(zs, (curvas[n] - lo) / r, "o-", ms=3, lw=1.2, color=COLOR[n],
                label=etiqueta[n])
        for zs_f, cf in finos.values():
            b1.plot(zs_f, (cf[n] - lo) / r, ".-", ms=4, lw=1, color=COLOR[n], alpha=0.55)
        b1.axvline(z_foco[n], color=COLOR[n], lw=1, ls=":")
    if z_real is not None:
        b1.axvline(z_real, color="k", ls="--", lw=1.2, label=f"Z_REAL = {z_real:g} mm")
    b1.set_xlabel("z = fuente -> muestra [mm]")
    b1.set_ylabel("metrica reescalada a [0, 1]")
    b1.set_title("barrido: sin referencia vs con referencia\n"
                 "(punteado claro: barrido fino; vertical de color: foco elegido)",
                 fontsize=10)
    b1.legend(fontsize=8, loc="upper right")
    b1.grid(alpha=0.25)

    m = (zs > centro - 6 * paso) & (zs < centro + 6 * paso)
    for n in metricas:
        lo, r = escala[n]
        b2.plot(zs[m], (curvas[n][m] - lo) / r, "o-", ms=4, lw=1.2, color=COLOR[n],
                label=n)
        for zs_f, cf in finos.values():
            b2.plot(zs_f, (cf[n] - lo) / r, ".-", ms=4, lw=1, color=COLOR[n], alpha=0.55)
    if z_real is not None:
        b2.axvline(z_real, color="k", ls="--", lw=1.2)
    b2.set_xlim(centro - 6 * paso, centro + 6 * paso)
    b2.set_xlabel("z [mm]")
    donde = "Z_REAL" if z_real is not None else f"el foco de {metricas[0]}"
    b2.set_title(f"zoom +-{6 * paso:g} mm alrededor de {donde}", fontsize=10)
    b2.legend(fontsize=8)
    b2.grid(alpha=0.25)
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def figura_campos(paneles, forma, ventana, zoom_px, con_roi, titulo=""):
    """Cada panel entero y, debajo, su zoom sobre la ventana medida. En pixeles.

    paneles: [(imagen a [0, 1], (fila, columna) de su esquina en la malla,
    titulo)]. Un panel mas chico que la malla -la reconstruccion del motor de
    Carlos, que solo da su recorte- se dibuja en su sitio. Todos comparten la
    malla del holograma: en pixeles un detalle no cambia de sitio con z.
    """
    M, N = forma
    f0, c0, alto, ancho = ventana
    zoom = min(zoom_px, M, N)
    zf = min(max(f0 + alto // 2 - zoom // 2, 0), M - zoom)
    zc = min(max(c0 + ancho // 2 - zoom // 2, 0), N - zoom)
    fig, ejes = plt.subplots(2, len(paneles), figsize=(4.1 * len(paneles), 8.6),
                             squeeze=False)
    for k, (img, (pf, pc), tit) in enumerate(paneles):
        h, w = img.shape
        ext = [pc - 0.5, pc + w - 0.5, pf + h - 0.5, pf - 0.5]
        arriba, abajo = ejes[0][k], ejes[1][k]
        for ax in (arriba, abajo):
            ax.imshow(img, extent=ext, cmap="gray", vmin=0, vmax=1)
            ax.set_xlabel("x [px]")
        arriba.set_xlim(-0.5, N - 0.5)
        arriba.set_ylim(M - 0.5, -0.5)
        arriba.set_title(tit, fontsize=10)
        if con_roi:
            arriba.add_patch(plt.Rectangle((c0 - 0.5, f0 - 0.5), ancho, alto,
                                           fill=False, ec="tab:red", lw=1))
        abajo.set_xlim(zc - 0.5, zc + zoom - 0.5)
        abajo.set_ylim(zf + zoom - 0.5, zf - 0.5)
        abajo.set_title(f"zoom {'en el ROI' if con_roi else 'central'} {zoom}x{zoom} px",
                        fontsize=9)
    ejes[0][0].set_ylabel("y [px]")
    ejes[1][0].set_ylabel("y [px]")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  9. MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Conecta las constantes con las funciones. Devuelve {metrica: z de foco}."""
    comprobar_configuracion(METRICAS, FORMATO_BARRIDO, OBJETO, MOTOR, CAMPO, RUTA,
                            REFERENCIA, Z_REAL, SIMULADOR)
    xp_, dev = elegir_dispositivo(DISPOSITIVO)
    dtype = DTYPE or (np.complex64 if dev == "gpu" else np.complex128)
    zs, paso, z_rango = rango_del_barrido(Z_MIN, Z_MAX, PASOS, L)

    # --- el objeto y el holograma -------------------------------------------
    t_ref = None
    if REFERENCIA is not None:
        img = leer_imagen(REFERENCIA)
        t_ref = objeto_de(img, OBJETO, INVERTIR, FASE_MAX)
        g_fondo = 1 - fondo_de(img) if INVERTIR else fondo_de(img)
        if RUTA is None and OBJETO == "fase" and abs(g_fondo) > 0.1:
            print(f"AVISO: el fondo del objeto vale g = {g_fondo:.2f}, o sea fase "
                  f"{FASE_MAX * g_fondo:.2f} rad, y fuera de la ventana el objeto es "
                  f"transparente:\n  el borde de la ventana es un escalon de fase. "
                  f"Cambia INVERTIR para dejar el fondo en g = 0.")
    # lo que dice el nombre de un holograma simulado: su CAMPO, o "lopera"
    tipo = CAMPO if SIMULADOR == "exacto" else "lopera"
    t0 = time.perf_counter()
    if RUTA is None:
        holo = simular(t_ref, LAMB, DELTA, L, Z_REAL, CAMPO, dev, dtype, SIMULADOR,
                       NA_FUENTE)
        # la ida deja en el pool de CuPy lo que uso -en 3000x4000, ~1.6 GB- y el
        # motor no lo encontraria libre
        liberar(xp_)
        # el de Lopera lleva la envolvente de la fuente y 8 bits: se divide por
        # su media, como uno de laboratorio sin FONDO
        fondo = None if SIMULADOR == "exacto" else "media"
        print(f"holograma simulado ({tipo}) de {pathlib.Path(REFERENCIA).name} a "
              f"z = {Z_REAL:g} mm: {holo.shape[0]}x{holo.shape[1]} en "
              f"{time.perf_counter() - t0:.1f} s")
    else:
        holo = leer_imagen(RUTA)
        fondo = "media" if FONDO is None else leer_imagen(FONDO)
        if FONDO is not None and fondo.shape != holo.shape:
            raise SystemExit(f"FONDO es de {fondo.shape} y el holograma de {holo.shape}.")
        print(f"holograma {RUTA}: {holo.shape[0]}x{holo.shape[1]}")
    if t_ref is not None and t_ref.shape != holo.shape:
        raise SystemExit(
            f"La referencia es de {t_ref.shape[0]}x{t_ref.shape[1]} y el holograma de "
            f"{holo.shape[0]}x{holo.shape[1]}. Tiene que ser el objeto en la malla que "
            f"el sensor proyecta,\nun pixel por pixel del sensor: si no lo es, pon "
            f'REFERENCIA = None y METRICAS = ("tamura",).')
    M, N = holo.shape

    # --- la region medida y la salida ---------------------------------------
    imagen = vista(t_ref if t_ref is not None else holo, OBJETO) if ROI is True else None
    ventana, roi = region_medida((M, N), ROI, BORDE_PX, imagen,
                                 "arrastra la region donde enfocar")
    f0, c0, alto, ancho = ventana
    if roi is not None:
        print(f"ROI = ({roi.x0}, {roi.y0}, {roi.ancho}, {roi.alto})   # repite esta ventana")
    salida, nombre = carpeta_de_salida(RESULTADOS, RUTA, REFERENCIA, Z_REAL, tipo,
                                       MOTOR, OBJETO, roi)
    salida.mkdir(parents=True, exist_ok=True)
    if RUTA is None:
        np.save(salida / f"{nombre}.npy", holo)
        print(f"  -> {salida / f'{nombre}.npy'}  (ponlo en RUTA para no volver a simular)")

    print(f"  lambda {LAMB * 1e6:.0f} nm | delta {DELTA * 1e3:.2f} um | L = {L:g} mm | "
          f"objeto {OBJETO} | motor {MOTOR}")
    print(f"  grueso: {PASOS} pasos de {Z_MIN:g} a {Z_MAX:g} mm (paso {paso:g} mm, "
          f"M = L/z de {L / Z_MAX:.3f} a {L / Z_MIN:.3f}); fino: {PASOS_FINO} pasos "
          f"(paso {2 * paso / (PASOS_FINO - 1):.3g} mm)")
    print(f"  dispositivo {dev.upper()} | dtype {np.dtype(dtype).name} | fase en float64")

    # --- el motor -----------------------------------------------------------
    t0 = time.perf_counter()
    if MOTOR == "exacto":
        motor = MotorExacto(holo, LAMB, DELTA, L, z_rango, fondo=fondo, device=dev,
                            dtype=dtype)
        sincronizar(motor.xp)
        print(f"  motor exacto: {motor.describir()}; construido en "
              f"{time.perf_counter() - t0:.1f} s")
    else:
        if abs(f0 + alto / 2 - M / 2) > 1 or abs(c0 + ancho / 2 - N / 2) > 1:
            print(f"AVISO: el ROI no esta centrado en la malla {M}x{N}. El motor de "
                  f"Carlos pone la fuente en el centro\n  del recorte: su fisica ya no "
                  f"es la del banco.")
        motor = MotorCarlos(holo, ventana, LAMB, DELTA, L, xp_, dtype, SOBREMUESTREO)
        comprobar_memoria(motor.h, z_rango, LAMB, DELTA, MEMORIA_MAX_GB, xp_, dtype)
        print(f"  motor de Carlos: {motor.describir(z_rango)}")
        print(f"  reconstruir() vs la cadena intacta de dlhm.py: "
              f"{comprobar_equivalencia(xp_, dtype):.2e}")

    # --- la referencia de las metricas --------------------------------------
    clase_ref = clase_de("pearson", OBJETO)
    ref = (None if t_ref is None
           else mapa(t_ref[f0:f0 + alto, c0:c0 + ancho], clase_ref, np))
    fijar_referencia(ref, motor.xp)
    print(f"metricas: {', '.join(METRICAS)} sobre {alto}x{ancho} px de {M}x{N} "
          f"(x {c0}-{c0 + ancho}, y {f0}-{f0 + alto} px)")

    # --- el barrido ---------------------------------------------------------
    guardar_en = None
    if GUARDAR_BARRIDO:
        def guardar_en(sub):
            return lambda t, z: guardar_instante(t, z, salida / sub, FORMATO_BARRIDO,
                                                 OBJETO)
    reloj = {}
    sincronizar(motor.xp)
    t0 = time.perf_counter()
    curvas, picos, finos, z_foco = enfocar(motor, ventana, zs, PASOS_FINO, METRICAS,
                                           OBJETO, en_cpu=EN_CPU, solapar=SOLAPAR,
                                           reloj=reloj, guardar_en=guardar_en)
    sincronizar(motor.xp)
    t_barrido = time.perf_counter() - t0
    for n, i in picos.items():
        if i in (0, len(zs) - 1):
            print(f"AVISO: el maximo de {n} cae en el extremo z = {zs[i]:g} mm; el "
                  f"barrido no acota ese foco.")

    # --- la reconstruccion en cada foco elegido -----------------------------
    z_unicos = sorted({round(z, 6) for z in z_foco.values()})
    quien = {z: [n for n in METRICAS if round(z_foco[n], 6) == z] for z in z_unicos}
    recon = {z: a_cpu(motor.campo(z, motor.ventana_maxima)) for z in z_unicos}
    liberar(motor.xp)

    # --- lo que se sabe -----------------------------------------------------
    print()
    print(informe_de_focos(zs, curvas, picos, z_foco, METRICAS, Z_REAL))
    print()
    n_rec = len(zs) + PASOS_FINO * len(finos)
    print(f"{n_rec} reconstrucciones en {t_barrido:.1f} s"
          + (" (incluye guardar cada z)" if GUARDAR_BARRIDO else ""))
    print("  reparto: " + ", ".join(f"{k} {v:.1f} s ({v / n_rec * 1e3:.0f} ms/z)"
                                     for k, v in reloj.items()))
    guardar_curvas(salida / "curvas.npz", zs, curvas, finos, METRICAS)
    print("->", salida / "curvas.npz")

    # --- figuras ------------------------------------------------------------
    cabeza = (f"{nombre}  ·  motor {MOTOR}  ·  objeto {OBJETO}  ·  lambda "
              f"{LAMB * 1e6:.0f} nm  ·  delta {DELTA * 1e3:.2f} um  ·  L = {L:g} mm  ·  "
              f"{M}x{N}")
    fig1 = figura_barrido(zs, curvas, finos, z_foco, METRICAS, OBJETO, Z_REAL, cabeza)
    paneles = []
    if t_ref is not None:
        paneles.append((vista(t_ref, OBJETO), (0, 0),
                        f"referencia ({'fase' if OBJETO == 'fase' else '|t|^2'})"))
    paneles.append((vista(holo), (0, 0), "holograma (entrada)"))
    fila_v, col_v, _, _ = motor.ventana_maxima
    for z in z_unicos:
        paneles.append((vista(recon[z], OBJETO), (fila_v, col_v),
                        f"reconstruido a {z:.4f} mm\n" + " + ".join(quien[z])))
    # en dos lineas: con dos o tres paneles la de fig1 no cabe y se corta
    fig2 = figura_campos(paneles, (M, N), ventana, ZOOM_PX, roi is not None,
                         f"Campos  ·  {nombre}  ·  motor {MOTOR}  ·  objeto {OBJETO}\n"
                         f"lambda {LAMB * 1e6:.0f} nm  ·  delta {DELTA * 1e3:.2f} um  ·  "
                         f"L = {L:g} mm  ·  {M}x{N}  ·  cada panel a su propio rango")
    if GUARDAR_FIGURA:
        for nombre_fig, f in (("barrido_metricas", fig1), ("campos", fig2)):
            f.savefig(salida / f"{nombre_fig}.png", dpi=110)
            print("->", salida / f"{nombre_fig}.png")
    if MOSTRAR:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)
    return z_foco


if __name__ == "__main__":
    main()
