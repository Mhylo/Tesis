"""Retropropaga un holograma YA GRABADO con los propagadores que elijas.

Los otros tres scripts/retro_*.py son demostraciones de un propagador cada uno:
llevan su copia, la contrastan contra la referencia y hacen la ida y la vuelta
sobre un objeto sintetico. Este no demuestra ninguno. Recibe un holograma que
ya existe -no lo genera- y lo lleva de vuelta al plano del objeto, asi que
llama a CamposT.pipeline.propagar(), que ya tiene FFT-ASM, BL-ASM y MPASM bajo
una firma comun. No hay una cuarta copia de los propagadores en este archivo.

DOS INTERRUPTORES, los dos de TRUE/FALSE:

  USAR_ANGULAR / USAR_BLAS / USAR_MPASM   que propagador (o cuales) corren.
                Varios a la vez se comparan en la misma figura, sobre el mismo
                holograma y el mismo recorte, que es la unica forma honesta de
                compararlos: mismo dato de entrada, misma z.

  RECORTAR      True abre el holograma en una ventana, arrastras un rectangulo
                con el raton, cierras, y solo ese trozo se retropropaga.

QUE LE ENTREGAS. La extension decide, y no es un detalle de comodidad:

    .png (o cualquier imagen)   la imagen es INTENSIDAD medida. El campo de
                partida es sqrt(I). Es lo que entrega un sensor, que registra
                |U|^2 y tira la fase. Al retropropagarlo sale el objeto CON SU
                IMAGEN GEMELA desenfocada encima: correlacion ~0.53 con el
                objeto en vez de 1.00. No es un fallo del script ni del
                propagador, es el problema de la imagen gemela, y este script
                no lo suprime -eso es Gerchberg-Saxton, phase-shifting o resta
                de fondo, y es otro problema-.

    .npy        el archivo trae el CAMPO COMPLEJO. Ahi la vuelta deshace la ida
                -exp(+i*phi)*exp(-i*phi) = 1- y devuelve el objeto exacto, sin
                gemela. No es lo que mide un sensor: es lo que hacen los otros
                tres scripts cuando retropropagan campo_sensor.

    La consola dice cual de los dos caminos tomo, porque cambia como se lee la
    figura entera.

OJO A LA GAMMA. Un PNG de intensidad casi nunca guarda I: guarda I^gamma, para
que se vea algo en pantalla. Los de resultados/campos/ los escribe
CamposT/pipeline.py con gamma = 0.6, asi que el archivo es I^0.6 y tomarle la
raiz da I^0.3 en vez de I^0.5. Eso no es solo contraste feo: MUEVE LA DISTANCIA
A LA QUE ENFOCA LA RECONSTRUCCION, que es justo lo que el barrido intenta medir.
GAMMA_GUARDADO la deshace. Vale 1.0 por defecto -no toca nada- porque la gamma
de un archivo que te entregan no se puede adivinar mirandolo: hay que saber con
cual se guardo.

QUE CUESTA RECORTAR. El muestreo a delta acota el angulo de difraccion que la
malla puede representar: sin(theta) = lambda / (2*delta). La luz que sale de UN
punto del objeto llega al sensor repartida sobre un disco de radio

    r = z * tan(theta) / delta   pixeles

que con 633 nm, 3.45 um y z = 20 mm son ~532 px. Si la ventana que recortas es
mas chica que ese disco estas tirando parte del cono de cada punto: la
reconstruccion sale con menos resolucion y con anillos en los bordes. El script
imprime r siempre y avisa cuando la ROI se queda corta, pero RECORTA IGUAL. Es
tu decision, no una guarda.

SIN RELLENO DE CEROS. Se propaga la malla tal cual, sin rodearla de ceros. La
FFT convoluciona EN CIRCULO: lo que sale por un borde reentra por el opuesto, y
eso viaja dentro del |U|^2 que se pinta. Con un recorte pesa mas, porque la
ventana crea dos bordes duros que en el holograma entero no estaban. La ida y
vuelta no lo nota -el doblez es reversible-, pero una reconstruccion sola si.

EL SIGNO. Z es POSITIVA, la separacion sensor->objeto tal como se mide en el
montaje; el menos lo pone la retropropagacion. Y sobre una entrada de
intensidad ese signo NO SE PUEDE COMPROBAR mirando la figura: sqrt(I) es real,
y para entrada real U(-z) = conj(U(+z)), luego |U(-z)|^2 = |U(+z)|^2 exacto. La
pila de foco con el signo invertido sale identica imagen por imagen. Es la
misma simetria que produce la gemela. La convencion se fija por contrato en
tests/test_retropropagacion.py.

EL BARRIDO mide nitidez() a cada z y una curva por metodo. NO lleva rms_fase():
esa metrica compara contra la fase del objeto, y aqui no hay objeto -te dan un
holograma y punto-. Meterla seria inventarse una referencia.

QUE NO HACE: la correccion de fuente puntual divergente. Se asume iluminacion
colimada (in-line clasico, onda plana). En DLHM con pinhole hay magnificacion
M = L/z y hay que reescalar el paso de pixel y usar una distancia efectiva
ANTES de llegar aqui.

UNIDADES: milimetros para todo.  633 nm -> 633e-6    3.45 um -> 3.45e-3
"""

import pathlib

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

from CamposT.backend import a_numpy, info_gpu, liberar_memoria
from CamposT.campos import load_field
from CamposT.pipeline import propagar
from CamposT.propagadores import memoria_mpasm


# ════════════════════════════════════════════════════════════════════════════
#  PARAMETROS
# ════════════════════════════════════════════════════════════════════════════

#: EL HOLOGRAMA, ya grabado. Imagen -> intensidad medida (campo = sqrt(I), con
#: imagen gemela). .npy -> campo complejo (la vuelta es exacta, sin gemela).
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\campos\fft\z0020.png"

#: Longitud de onda [mm].
LAMB = 633e-6

#: Paso de pixel del sensor [mm].
DELTA = 3.45e-3

#: Distancia sensor <-> objeto [mm], POSITIVA. El menos lo pone la
#: retropropagacion: internamente se propaga a -Z.
Z = 20.0

#: QUE PROPAGADORES CORREN. Varios en True se comparan en la misma figura.
#:
#:   USAR_ANGULAR  FFT-ASM, el espectro angular por FFT. La referencia, sin
#:                 control de muestreo: a z larga o delta chico alia.
#:   USAR_BLAS     BL-ASM, con el limite de banda de Matsushima (2009). Corta
#:                 las frecuencias que ya no se pueden muestrear en vez de
#:                 aliarlas, asi que no es la identidad: devuelve el objeto
#:                 FILTRADO. Ese es el precio de no aliar.
#:   USAR_MPASM    espectro angular por producto matricial. El que mas gana con
#:                 CUDA, y el unico que puede ampliar la ventana de salida.
#:                 OJO A LA MEMORIA: la matriz espectral es (S_MPASM * M) por
#:                 (S_MPASM * N), y eso POR DISTANCIA del barrido.
USAR_ANGULAR = True
USAR_BLAS = True
USAR_MPASM = True

#: True abre el holograma, arrastras el rectangulo con el raton, cierras, y
#: solo esa ventana se retropropaga. Al cerrar imprime las coordenadas
#: elegidas para que puedas anotarlas y repetir el recorte.
#:
#: Lee "QUE CUESTA RECORTAR" arriba antes de interpretar el resultado.
RECORTAR = True

#: Sobremuestreo del espectro de MPASM. El defecto de mpasm() es 10, pero la
#: matriz espectral es (s*N)^2 POR DISTANCIA: en un barrido eso no cabe.
#: Subelo para una sola z.
S_MPASM = 1

#: "auto" usa la GPU si hay CuPy con CUDA; "cpu" y "gpu" fuerzan.
DISPOSITIVO = "auto"

#: dtype del campo. None = complex64 en GPU, complex128 en CPU. Las fases se
#: calculan en float64 siempre, dentro del paquete.
DTYPE = None

#: GAMMA con la que se guardo el PNG, para deshacerla: la imagen se eleva a
#: 1/GAMMA_GUARDADO antes de tomar la raiz. 1.0 lo desactiva.
#:
#:   resultados/campos/*/*.png   -> 0.6   (los escribe CamposT/pipeline.py)
#:   un holograma crudo del sensor, o un TIFF/PNG lineal -> 1.0
#:
#: El defecto es 1.0 y no 0.6 a proposito: 1.0 no toca el dato, y una correccion
#: automatica que acierta a veces es peor que ninguna. Si no sabes con que gamma
#: se guardo tu archivo, no lo adivines aqui: vuelve a guardarlo con gamma = 1.
#:
#: Solo aplica a la entrada de imagen. Un .npy trae el campo y no pasa por aqui.
GAMMA_GUARDADO = 1.0

#: True intercambia zonas claras y oscuras antes de tomar la raiz, segun como
#: venga escaneado el holograma. Solo aplica a la entrada de imagen; con .npy
#: se ignora, porque invertir un campo complejo no significa nada.
INVERTIR = False

#: True pinta la fase con la OPACIDAD proporcional a la amplitud; False la
#: pinta cruda, como np.angle() a secas.
#:
#: Aqui el defecto es True y no False a proposito. Reconstruyendo desde
#: intensidad el fondo tiene amplitud DE VERDAD -no es el cero numerico de
#: 1e-16 de una ida y vuelta desde el campo complejo-, asi que su fase
#: significa algo y con False la ves. Pero la gemela desenfocada llena el
#: plano, y sin pesar cuesta separar la estructura del objeto de ella.
#:
#: Es una decision de dibujo, no un filtro sobre los datos: el array es el
#: mismo. En la duda, mira las dos.
PESAR_FASE = True

#: Distancias del barrido de foco, como fraccion de Z. None lo desactiva.
#: A cada z se retropropaga con cada metodo activo y se mide la nitidez, que
#: es lo util cuando te entregan un holograma y la z de enfoque no se conoce.
BARRIDO = np.linspace(0.4, 1.6, 25)

#: Carpeta donde guardar las figuras, o None para solo mostrarlas.
SALIDA = None


#: Nombre legible de cada metodo, para los titulos de la figura.
NOMBRES = {"fft": "FFT-ASM (espectro angular)",
           "blas": "BL-ASM (banda limitada)",
           "mpasm": "MPASM (producto matricial)"}


def a_cpu(a):
    """Baja un array a NumPy. Delega en backend.a_numpy().

    Existe con este nombre porque nitidez() es una copia literal de la de
    scripts/retro_blas.py y la llama asi. Ver la nota de METRICAS.
    """
    return a_numpy(a)


# ════════════════════════════════════════════════════════════════════════════
#  METRICAS  --  copias literales de las de retro_blas.py, retro_mpasm.py y
#  retro_fft_angular.py, que tests/test_nitidez_foco.py y test_pico_de_foco.py
#  comprueban que no diverjan. No las edites solo aqui.
#
#  rms_fase() y sin_piston() NO estan: comparan contra la fase del objeto, y
#  aqui no hay objeto. Por eso este script no entra en test_fases_retro.py.
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


# ════════════════════════════════════════════════════════════════════════════
#  ENTRADA
# ════════════════════════════════════════════════════════════════════════════

def cargar_holograma(ruta, invertir=False, gamma=1.0):
    """Archivo -> (campo complejo del plano del sensor, etiqueta de que es).

    La extension decide, y las dos ramas NO son el mismo experimento:

      .npy   el campo complejo tal cual. Retropropagarlo deshace la
             propagacion y devuelve el objeto exacto. Es lo que hacen los
             otros retro_*.py con campo_sensor, y NO es lo que mide un sensor.

      resto  la imagen es intensidad medida y el campo es sqrt(I), via
             campos.load_field(mode='holograma'). Tomarla como amplitud
             elevaria el campo al cuadrado y falsearia el contraste Y la
             distancia a la que enfoca la reconstruccion. La fase se perdio en
             la medida: de ahi sale la imagen gemela.

             gamma deshace la del archivo antes de la raiz. Como load_field()
             ya devuelve sqrt(t) y t es real y no negativa,

                 sqrt(t^(1/gamma)) = sqrt(t)^(1/gamma)

             asi que se eleva el campo devuelto y no hace falta repetir la
             lectura. Con gamma = 1 no se toca nada.

    La etiqueta se devuelve para imprimirla y para el titulo de la figura,
    porque de que rama viene depende como se lee todo lo demas.
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
        if invertir:
            print("  AVISO: INVERTIR = True se ignora con un .npy. Invertir un "
                  "campo complejo no significa nada.")
        return np.asarray(U), "campo complejo (la vuelta es exacta, sin gemela)"

    if gamma <= 0:
        raise SystemExit(f"GAMMA_GUARDADO = {gamma} y tiene que ser positiva. "
                         f"1.0 la desactiva.")
    U = np.abs(np.asarray(load_field(ruta, mode="holograma", invert=invertir)))
    if gamma == 1.0:
        nota = ""
    else:
        U = U ** (1.0 / gamma)
        nota = f", gamma {gamma:g} deshecha"
    return U.astype(complex), (f"intensidad medida{nota}, campo = sqrt(I) "
                               f"(con gemela)")


# ════════════════════════════════════════════════════════════════════════════
#  RECORTE
# ════════════════════════════════════════════════════════════════════════════

def radio_del_cono(z, lamb, delta):
    """Radio en PIXELES sobre el que un punto del objeto reparte su luz.

    El muestreo a delta acota la frecuencia espacial representable a 1/(2delta),
    o sea el angulo de difraccion a sin(theta) = lambda/(2*delta). Mas alla de
    ahi la malla no puede describir la onda, propague quien propague. La luz de
    un punto llega al sensor sobre un disco de radio z*tan(theta), y en
    pixeles eso es z*tan(theta)/delta.

    Es el numero que dice cuanto cuesta recortar: una ventana mas chica que
    este disco corta el cono de cada punto del objeto que cae dentro de ella.

    Con lambda >= 2*delta no hay angulo propagante que la malla represente: la
    raiz se hace imaginaria y se devuelve inf, que es la respuesta correcta -no
    hay ventana suficientemente grande-.
    """
    sin_max = lamb / (2 * delta)
    if sin_max >= 1:
        return np.inf
    return abs(z) * (sin_max / np.sqrt(1 - sin_max**2)) / delta


def elegir_roi(I, titulo):
    """Abre I, deja arrastrar un rectangulo y devuelve (x0, y0, ancho, alto).

    Bloquea hasta que cierres la ventana. Si la cierras sin haber arrastrado
    nada, aborta: propagar el plano entero en silencio despues de haber pedido
    un recorte seria hacer otra cosa distinta de la que se pidio.

    Las coordenadas salen en pixeles enteros y recortadas a la malla, con
    floor en el origen y ceil en el extremo, de modo que la ventana devuelta
    CONTIENE lo que arrastraste en vez de quedarse por dentro.
    """
    M, N = I.shape
    caja = {}

    def al_soltar(inicio, final):
        caja["xy"] = (inicio.xdata, inicio.ydata, final.xdata, final.ydata)

    alto = 8.0 * M / N
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, min(alto, 9.0))))
    ax.imshow(I, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{titulo}\nArrastra la ventana a retropropagar y cierra "
                 f"esta figura", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    estilo = dict(facecolor="none", edgecolor="red", linewidth=1.5)
    try:
        selector = RectangleSelector(ax, al_soltar, useblit=True, button=[1],
                                     interactive=True, props=estilo)
    except TypeError:
        # matplotlib < 3.5 llamaba rectprops a lo que ahora es props
        selector = RectangleSelector(ax, al_soltar, useblit=True, button=[1],
                                     interactive=True, rectprops=estilo)
    # el selector tiene que sobrevivir a esta funcion mientras la figura este
    # abierta: sin una referencia viva, el recolector se lo lleva y el raton
    # deja de hacer nada
    ax._selector_roi = selector

    plt.show()

    if "xy" not in caja or any(v is None for v in caja["xy"]):
        raise SystemExit(
            "RECORTAR = True pero no arrastraste ninguna ventana.\n"
            "Vuelve a lanzarlo y arrastra sobre el holograma, o pon "
            "RECORTAR = False para retropropagar el plano entero.")

    xa, ya, xb, yb = caja["xy"]
    x0 = int(np.clip(np.floor(min(xa, xb)), 0, N - 1))
    y0 = int(np.clip(np.floor(min(ya, yb)), 0, M - 1))
    x1 = int(np.clip(np.ceil(max(xa, xb)), x0 + 1, N))
    y1 = int(np.clip(np.ceil(max(ya, yb)), y0 + 1, M))

    if (x1 - x0) < 2 or (y1 - y0) < 2:
        raise SystemExit(
            f"La ventana que arrastraste es de {x1 - x0}x{y1 - y0} pixeles y no "
            f"da para propagar nada. Arrastra un rectangulo de verdad.")

    return x0, y0, x1 - x0, y1 - y0


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


def marca_roi(ax, roi):
    """Dibuja la ventana recortada sobre un panel del holograma completo.

    Un Rectangle pertenece a unos ejes y no se puede compartir, asi que cada
    panel necesita el suyo.
    """
    x0, y0, ancho, alto = roi
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), ancho, alto, fill=False,
                           edgecolor="red", linewidth=1.2))


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ruta = pathlib.Path(RUTA)
    if not ruta.is_file():
        raise SystemExit(f"No encuentro el holograma en:\n    {ruta}\n\n"
                         "Edita la constante RUTA al principio del archivo.")
    if Z <= 0:
        raise SystemExit(
            f"Z = {Z} pero es la distancia sensor-objeto y va POSITIVA: el "
            f"menos lo pone la retropropagacion.\nPon Z = {abs(Z)}.")

    activos = [m for m, encendido in (("fft", USAR_ANGULAR),
                                      ("blas", USAR_BLAS),
                                      ("mpasm", USAR_MPASM)) if encendido]
    if not activos:
        raise SystemExit(
            "Los tres propagadores estan en False y no hay nada que correr.\n"
            "Pon en True al menos uno de USAR_ANGULAR, USAR_BLAS o USAR_MPASM.")

    U_h, etiqueta = cargar_holograma(ruta, INVERTIR, GAMMA_GUARDADO)
    M, N = U_h.shape

    print(f"holograma {ruta}")
    print(f"  {etiqueta}")
    print(f"  malla {M}x{N} | lambda {LAMB * 1e6:.1f} nm | "
          f"delta {DELTA * 1e3:.3f} um | vuelta a {-Z:+g} mm")
    print(f"  propagadores: {', '.join(NOMBRES[m] for m in activos)}")
    print("  sin relleno de ceros"
          + (f" | s de MPASM {S_MPASM}" if "mpasm" in activos else ""))
    gpu = info_gpu()
    if gpu is not None and DISPOSITIVO != "cpu":
        print(f"  {gpu['nombre']}, {gpu['VRAM libre [GB]']:.2f} de "
              f"{gpu['VRAM total [GB]']:.2f} GB libres")

    # ---- recorte -----------------------------------------------------------
    radio = radio_del_cono(Z, LAMB, DELTA)
    print(f"  cono de difraccion a z = {Z:g} mm: radio {radio:.0f} px "
          f"(sin theta = lambda/(2 delta) = {LAMB / (2 * DELTA):.4f})")

    roi, U_in = None, U_h
    if RECORTAR:
        I_h = np.abs(U_h) ** 2
        pico = I_h.max()
        roi = elegir_roi(I_h / pico if pico > 0 else I_h, f"{ruta.name}")
        x0, y0, ancho, alto = roi
        U_in = U_h[y0:y0 + alto, x0:x0 + ancho]
        print(f"\nrecorte:  X0 = {x0}  Y0 = {y0}  ANCHO = {ancho}  "
              f"ALTO = {alto}")
        print(f"  {ancho * alto / (M * N):.1%} del holograma "
              f"({ancho}x{alto} de {N}x{M})")
        if min(ancho, alto) / 2 < radio:
            print(f"  AVISO: la ventana tiene semilado "
                  f"{min(ancho, alto) / 2:.0f} px y el cono de un punto del "
                  f"objeto mide {radio:.0f} px de radio.\n  Estas tirando parte "
                  f"del cono de cada punto: la reconstruccion sale con menos "
                  f"resolucion\n  y con anillos en los bordes. Se recorta "
                  f"igual, pero leelo con eso en la cabeza.")
    Mi, Ni = U_in.shape


    if "mpasm" in activos:
        bytes_pico = memoria_mpasm(Mi, Ni, s=S_MPASM, dtype=np.complex64)
        print(f"  MPASM pide ~{bytes_pico / 2**30:.2f} GB de pico por "
              f"distancia (malla {Mi}x{Ni}, s = {S_MPASM})")

    # ---- retropropagacion a Z ---------------------------------------------
    # Se propaga a -Z: Z es la separacion sensor->objeto, positiva, y el signo
    # lo pone la vuelta. Sobre una entrada de intensidad ese signo no se puede
    # comprobar mirando |U|^2 (ver la cabecera del archivo).
    print()
    reconstrucciones = []
    for metodo in activos:
        extra = dict(s=S_MPASM) if metodo == "mpasm" else {}
        U, info = propagar(U_in, DELTA, LAMB, -float(Z), metodo=metodo,
                           pad=1, device=DISPOSITIVO, dtype=DTYPE, **extra)
        reconstrucciones.append((metodo, a_cpu(U)))
        kf = info.get("Kf")
        # Kf sale escalar en malla cuadrada y pareja (Kfy, Kfx) cuando no lo es
        detalle = ("" if kf is None else
                   f", Kf = ({kf[0]:.4f}, {kf[1]:.4f})"
                   if isinstance(kf, (tuple, list)) else f", Kf = {kf:.4f}")
        print(f"  {metodo:6s} {info['device']}, {info['dtype']}{detalle}")
        del U
        liberar_memoria()

    # ---- barrido de foco ---------------------------------------------------
    zs, curvas = None, {}
    if BARRIDO is not None:
        zs = np.asarray(BARRIDO, float) * Z
        total = len(activos) * len(zs)
        hecho = 0
        for metodo in activos:
            extra = dict(s=S_MPASM) if metodo == "mpasm" else {}
            valores = []
            for z in zs:
                U, _ = propagar(U_in, DELTA, LAMB, -float(z), metodo=metodo,
                                pad=1, device=DISPOSITIVO, dtype=DTYPE,
                                **extra)
                valores.append(nitidez(abs(U) ** 2))
                del U
                liberar_memoria()
                hecho += 1
                print(f"    barrido {hecho}/{total}   {metodo:6s} "
                      f"z = {z:8.2f} mm", end="\r")
            curvas[metodo] = np.array(valores)
        print(" " * 56, end="\r")

        print(f"\nbarrido de foco: {len(zs)} distancias de {zs[0]:.1f} a "
              f"{zs[-1]:.1f} mm")
        for metodo in activos:
            z_pico = pico_de_foco(zs, curvas[metodo])
            print(f"    {metodo:6s} nitidez maxima en z = {z_pico:.2f} mm"
                  if z_pico is not None else
                  f"    {metodo:6s} el maximo cae en un extremo del barrido, "
                  f"que NO acota el foco. Ensancha BARRIDO.")
        print(f"    Z del holograma:     z = {Z:.2f} mm")

    # ---- figura ------------------------------------------------------------
    columnas = [(f"holograma (entrada)\n{ruta.name}", U_h, True)]
    columnas += [(f"{NOMBRES[m]}\na {-Z:+g} mm", U, False)
                 for m, U in reconstrucciones]

    n_col = len(columnas)
    fig, ax = plt.subplots(3, n_col, figsize=(4.5 * n_col, 12.6),
                           squeeze=False)
    im = None
    for k, (titulo, U, es_entrada) in enumerate(columnas):
        pinta_intensidad(ax[0, k], U, titulo)
        im = pinta_fase(ax[1, k], U)
        pinta_complejo(ax[2, k], U)
        if es_entrada and roi is not None:
            for fila in range(3):
                marca_roi(ax[fila, k], roi)
    for fila, texto in ((0, "$|U|^2$"), (1, "fase"), (2, "campo complejo")):
        ax[fila, 0].text(-0.04, 0.5, texto, transform=ax[fila, 0].transAxes,
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

    recorte = (f" -- recorte {roi[2]}x{roi[3]} en ({roi[0]}, {roi[1]})"
               if roi is not None else "")
    fig.suptitle(f"Retropropagacion del holograma -- Z = {Z:g} mm, "
                 f"delta = {DELTA * 1e3:.2f} um, "
                 f"lambda = {LAMB * 1e6:.0f} nm{recorte}", fontsize=12)

    figuras = [("retropropagacion.png", fig)]

    if zs is not None:
        fig2, a2 = plt.subplots(figsize=(7.6, 4.4))
        for metodo in activos:
            c = curvas[metodo]
            pico = c.max()
            a2.plot(zs, c / pico if pico > 0 else c, "o-", ms=3,
                    label=NOMBRES[metodo])
        a2.axvline(Z, color="k", ls="--", lw=1, label=f"Z = {Z:g} mm")
        a2.set_xlabel("distancia de reconstruccion [mm]")
        a2.set_ylabel("nitidez normalizada")
        a2.legend(fontsize=9, loc="best")
        a2.set_title("Donde enfoca la retropropagacion" + recorte, fontsize=11)
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
