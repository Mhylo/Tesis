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

UNIDADES: milimetros para todo.  532 nm -> 532e-6    1.85 um -> 1.85e-3
"""

import pathlib

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from CamposT.roi import Roi, elegir, informe

# ════════════════════════════════════════════════════════════════════════════
#  1. PARAMETROS
# ════════════════════════════════════════════════════════════════════════════

#: EL HOLOGRAMA. Se usa su INTENSIDAD tal cual, sin tomar la raiz: la cadena
#: DLHM multiplica la intensidad registrada por la onda de referencia, no su
#: amplitud. Es lo que hace reconstruction_dlhm.py.
RUTA = r"C:\Users\User\Desktop\Tesis\referencia\carlos\DLHM-model-main\DLHM-model-main\data\Simulated_hologram.png"

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
COMPARAR = "fase"

#: EL RECORTE, una ventana por imagen. Mismos cuatro valores que en
#: mi_prueba_angular: None, True (raton sobre ESA imagen), (X0, Y0, ANCHO,
#: ALTO), o "misma" (el rectangulo que resolvio la otra).
#:
#: AQUI CASI NUNCA QUIERES None EN EL HOLOGRAMA: sin recortar, el remuestreo
#: pide 15 GB por paso. Lee "EL ROI NO ES UNA COMODIDAD" arriba.
ROI_HOLOGRAMA = None
ROI_REFERENCIA = None

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
SOBREMUESTREO = 1


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
#  3. RECONSTRUCCION Y BARRIDO
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


def comprobar_memoria(holo, zs, lamb, delta, tope_gb):
    """Aborta ANTES del barrido si la malla remuestreada no cabe.

    Sin esto, el primer paso pide la memoria que pida y muere con un
    MemoryError que no dice que hacer. La tabla dice exactamente cuanto hay que
    recortar, que es la unica palanca que mueve este numero.
    """
    Q, P = holo.shape
    # el peor caso es la z mas chica: s crece con z, y la malla es Wx/s
    if SOBREMUESTREO is not None:
        print(f"  SOBREMUESTREO = {SOBREMUESTREO}: la malla se queda en "
              f"{Q}x{P} ({Q * P / 1e6:.1f} Mpx). Alia, ver ESTADO.")
        return
    N, M = malla_remuestreada(P, Q, min(zs), lamb, delta)
    # seis mallas complejas de pico: h, ref, dos productos y dos espectros
    gb = 6 * N * M * 16 / 2 ** 30
    if gb <= tope_gb:
        print(f"  malla remuestreada: hasta {N}x{M} ({N * M / 1e6:.1f} Mpx, "
              f"~{gb:.2f} GB de pico)")
        return
    lineas = ["  recorte    malla          memoria"]
    for lado in (3000, 2048, 1500, 1024, 750, 512):
        if lado > min(P, Q):
            continue
        n, m = malla_remuestreada(lado, lado, min(zs), lamb, delta)
        lineas.append(f"  {lado:6d}   {n:5d}x{m:<5d}   {6 * n * m * 16 / 2**30:5.2f} GB")
    raise SystemExit(
        f"El remuestreo por geometria pide una malla de {N}x{M} "
        f"({gb:.1f} GB de pico) y el tope es {tope_gb} GB.\n\n"
        f"Eso depende del ANCHO FISICO del sensor, no del paso de pixel: "
        f"submuestrear no ayuda,\nhay que RECORTAR. Con ROI_HOLOGRAMA:\n\n"
        + "\n".join(lineas)
        + f"\n\nO sube MEMORIA_MAX_GB si sabes que tu maquina lo aguanta.")


def reconstruir(holo, z, lamb, delta, L_fuente):
    """Un holograma DLHM -> el campo reconstruido en el plano de la muestra.

    Es la cadena de reconstruction_dlhm.py: remuestrear por geometria,
    multiplicar por la onda esferica, propagar L-z, y dividir la referencia.

    Rec = U * conj(U0) es lo que quita la onda esferica del resultado. Sin ese
    paso, la fase del objeto queda montada sobre la del frente divergente y no
    se parece a nada.
    """
    Q, P = holo.shape
    Wcx, Wcy = P * delta, Q * delta
    k = 2 * np.pi / lamb

    if SOBREMUESTREO is None:
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
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else 0.0


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
    comprobar_memoria(holo, zs, LAMB, DELTA, MEMORIA_MAX_GB)
    print()

    # --- barrido ----------------------------------------------------------
    curva = np.empty(len(zs))
    mejor_Rec, mejor_ref, mejor = None, None, -1
    for i, z in enumerate(zs):
        Rec = reconstruir(holo, z, LAMB, DELTA, L)
        obs = observable(Rec, COMPARAR)
        # La referencia se reescala EN CADA PASO: mover z mueve M = L/z, o sea
        # el foco y la escala a la vez. Asi la correlacion solo sube cuando
        # coinciden las dos.
        r_esc = referencia_a_escala(ref, L / z, obs.shape)
        curva[i] = correlacion(obs, r_esc)
        # No se acumulan las reconstrucciones: son mallas de varios miles de
        # lado. Se guarda solo la mejor hasta ahora.
        if mejor < 0 or abs(curva[i]) > abs(curva[mejor]):
            mejor_Rec, mejor_ref, mejor = obs, r_esc, i
        print(f"  z = {z:6.3f} mm   M = {L / z:5.2f}   corr = {curva[i]:+.4f}")

    print(f"\nenfoca en z = {zs[mejor]:.3f} mm   M = {L / zs[mejor]:.2f}   "
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
