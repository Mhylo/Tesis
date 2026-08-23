"""Ida y vuelta con el espectro angular: objeto -> +z -> holograma -> -z -> objeto.

Propaga la imagen de entrada una distancia Z, y retropropaga el resultado -Z
para ver si vuelve a enfocar donde debe. Un solo metodo, escrito a mano, sin
importar CamposT: sirve de contraste independiente del paquete.

CPU o GPU con el mismo cuerpo de algoritmo (ver DISPOSITIVO). Lo unico que
cambia es si el modulo de arrays es NumPy o CuPy.

EL PROPAGADOR
-------------
angularSpectrum() esta TAL CUAL, sin tocar una linea, y es la referencia.
espectro_angular() calcula lo mismo pero cabe en la tarjeta: la equivalencia
entre las dos se comprueba en cada corrida y se imprime (comprobar_equivalencia).

Dos cosas de angularSpectrum() que conviene saber, medidas, no supuestas:

  - dfx = 1/(dx*M) y dfy = 1/(dy*N) estan CRUZADOS: al eje de N columnas le
    toca el espaciado que corresponde a las M filas. En malla cuadrada da
    igual y no pasa nada. En malla RECTANGULAR el error contra el gaussiano
    analitico es 2.95e-01 frente a 8.41e-06 con los ejes en su sitio. Es
    autoconsistente —pasa la prueba de transponer a 3.5e-16—, asi que solo se
    ve con una referencia externa. BenchmarkTarget.png es 4000x3000: si
    trabajas con el sin recortar a cuadrado, esto te afecta. EJES_CRUZADOS lo
    controla y el script avisa al arrancar.

  - Las ondas evanescentes (lamb^2 f^2 > 1) se dejan decaer en vez de
    anularse. Propagando hacia adelante decaen y esta bien; RETROPROPAGANDO
    el mismo factor crece, y con delta = lamb/4 el campo devuelto llega a
    2.87e+110. Con delta = 3.45 um contra lamb/2 = 0.32 um no hay ninguna en
    la malla y la linea no se activa; con la magnificacion de DLHM si.

PRECISION EN GPU
----------------
El campo va en complex64 y la fase en float64 SIEMPRE, se propague donde se
propague. La fase 2*pi*z*sqrt(1/lamb^2 - f^2) vale ~2e5 rad a z = 20 mm, y en
float32 eso son 0.02 rad de error solo por la mantisa. Calcularla en doble y
bajar al dtype de trabajo unicamente el fasor —que ya esta acotado a modulo
1— cuesta cero y quita el problema.

complex64 no es una concesion: en una GeForce la doble precision va a 1/32 del
ritmo de la simple. Medido a 2048^2, fft2+ifft2: 2.6 ms en complex64, 32.4 ms
en complex128, 454 ms en CPU. Y con PAD = 2 sobre 4000x3000 la malla es
6000x8000, o sea 0.36 GB por array en complex64 y 0.72 en complex128: en una
tarjeta de 4 GB lo segundo no cabe.

UNIDADES: milimetros para todo. Da igual cual sea mientras sea la MISMA en
lambda, delta y z: el espectro angular solo ve lambda*z/delta^2.

    633 nm -> 633e-6 mm     3.45 um -> 3.45e-3 mm     20 mm -> 20.0

LAS DOS VUELTAS, Y POR QUE SON DISTINTAS
----------------------------------------
  A) desde el CAMPO COMPLEJO que sale de la ida. Es propagacion invertida y
     punto: tiene que devolver el objeto. Si esto no cierra, el fallo esta en
     el codigo o en los parametros, no en la fisica.

  B) desde sqrt(|U|^2), que es lo unico que da un sensor real. La fase se
     perdio en la medida y vuelve el objeto con su imagen gemela encima.

Cuanto estropea la gemela depende del objeto. Si es mayormente OPACO (barras
claras sobre fondo negro, como campos.usaf_like) no queda haz sin tocar que
haga de onda de referencia, y sin referencia no hay holograma de Gabor que
reconstruir. Con el target invertido (barras oscuras sobre fondo claro, que es
como se ve un DLHM real) si reconstruye: INVERTIR lo cambia.

EL SIGNO DE Z
-------------
Z es la separacion objeto-sensor, POSITIVA. La ida usa +Z y la vuelta -Z.

Ese signo no se puede comprobar mirando la intensidad: el campo de la vuelta B
es real (sqrt de una intensidad), y para entrada real U(-z) = conj(U(+z)),
luego |U(-z)|^2 = |U(+z)|^2 exactamente. Con el signo cambiado la figura sale
identica. Solo se nota en la fase, y en cuanto se encadene algo no real
(correccion de fuente puntual, filtro complejo, recuperacion de fase).
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
Z = 100.0

# ════════════════════════════════════════════════════════════════════════════
#  Y esto solo si hace falta
# ════════════════════════════════════════════════════════════════════════════

#: "auto" usa la GPU si hay CuPy con CUDA; "cpu" y "gpu" fuerzan.
DISPOSITIVO = "auto"

#: dtype del campo. complex64 en GPU (ver PRECISION EN GPU); la fase va en
#: float64 en los dos casos pase lo que pase.
DTYPE = None                 # None = complex64 en GPU, complex128 en CPU

#: Reduce la imagen a este lado mayor antes de propagar, o None para dejarla.
#: OJO: si lo cambias, DELTA deja de ser el de tu sensor y hay que escalarlo
#: por (lado_original / lado_nuevo). El script lo hace y lo dice.
REDUCIR_A = None

#: True mantiene los ejes cruzados de angularSpectrum() tal como los escribiste.
#: False los pone por eje. Solo cambia algo en malla rectangular.
EJES_CRUZADOS = True

#: Relleno de ceros. La FFT convoluciona de forma circular: lo que sale por un
#: borde reentra por el opuesto. Con 2 ese doblez queda fuera del recorte.
PAD = 2

#: True invierte la imagen (barras oscuras sobre fondo claro). Es lo que le da
#: onda de referencia al holograma y hace que la vuelta B reconstruya.
INVERTIR = False

#: Distancias del barrido de foco, como fraccion de Z. None lo desactiva.
BARRIDO = np.linspace(0.4, 1.6, 25)

#: Carpeta donde escribir las figuras, o None para solo mostrarlas.
SALIDA = None

#: Filas por bloque al evaluar el kernel. Baja si la GPU se queda sin memoria.
FILAS_POR_BLOQUE = 512


# ------------------------------------------------- propagador de referencia
def angularSpectrum(field, z, wavelength, dx, dy, scale_factor=1):
    """
    Propagación angular del frente de onda usando el espectro angular
    field: campo complejo
    z: distancia de propagación
    wavelength: longitud de onda
    dx, dy: pasos espaciales
    """
    # Inputs:
    # field - complex field
    # wavelength - wavelength
    # z - propagation distance
    # dxy - sampling pitches
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
    """
    U = xp.asarray(field, dtype=dtype)
    M, N = U.shape

    if cruzados:
        dfx, dfy = 1 / (dx * M), 1 / (dy * N)
    else:
        dfx, dfy = 1 / (dx * N), 1 / (dy * M)

    # (1, N) y (M, 1): la aritmetica difunde igual que con meshgrid y no
    # materializa dos mallas completas
    fx = ((np.arange(N) - N / 2) * dfx).astype(np.float64)
    fy = ((np.arange(M) - M / 2) * dfy).astype(np.float64)
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


# ------------------------------------------------------------------ utilidades
def cargar_objeto(ruta, invertir=False, reducir_a=None, delta=DELTA):
    """Imagen -> (transmitancia en [0,1] como campo complejo, delta efectivo).

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


def propagar(U, z, delta, lamb, pad, xp, dtype, cruzados):
    """espectro_angular con relleno de ceros y recorte al tamano original."""
    M, N = U.shape
    if pad > 1:
        rel = xp.zeros((M * pad, N * pad), dtype=dtype)
        i, j = (rel.shape[0] - M) // 2, (rel.shape[1] - N) // 2
        rel[i:i + M, j:j + N] = xp.asarray(U, dtype=dtype)
    else:
        rel, i, j = xp.asarray(U, dtype=dtype), 0, 0
    fuera = espectro_angular(rel, z, lamb, delta, delta, xp=xp, dtype=dtype,
                             cruzados=cruzados)
    del rel
    # copia, no vista: una vista mantendria viva la malla rellena entera, que
    # con PAD = 2 sobre 4000x3000 son 0.36 GB por reconstruccion del barrido
    recorte = fuera[i:i + M, j:j + N].copy()
    del fuera
    return recorte


def nitidez(I):
    """Energia del gradiente, normalizada. Maxima en el plano de foco.

    Un objeto de amplitud enfocado tiene bordes duros; desenfocado los tiene
    difuminados. La suma de |grad I|^2 lo mide sin necesitar saber cual era el
    objeto, que es lo que hace falta en un barrido de una medida real.
    """
    I = np.asarray(a_cpu(I), dtype=float)
    I = I / I.max()
    gy, gx = np.gradient(I)
    return float(np.mean(gx**2 + gy**2))


def parecido(a, b):
    """Correlacion entre dos mapas, normalizados por su maximo."""
    a = np.asarray(a_cpu(a), float).ravel()
    b = np.asarray(a_cpu(b), float).ravel()
    return float(np.corrcoef(a / a.max(), b / b.max())[0, 1])


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
    comprobar_memoria(xp, M * PAD, N * PAD, dtype)

    print(f"objeto {RUTA}")
    print(f"  malla {M}x{N}, ventana {N * delta:.3f} x {M * delta:.3f} mm "
          f"(con relleno x{PAD}: {N * PAD * delta:.3f} mm)")
    if delta != DELTA:
        print(f"  imagen reducida a {max(M, N)} px: delta escalado de "
              f"{DELTA * 1e3:.3f} a {delta * 1e3:.3f} um")
    print(f"  lambda {LAMB * 1e6:.1f} nm | delta {delta * 1e3:.3f} um | "
          f"ida +{Z:.3f} mm, vuelta {-Z:+.3f} mm")
    print(f"  dispositivo {dev.upper()} | dtype {np.dtype(dtype).name} | "
          f"fase en float64")
    if dev == "gpu":
        libre, total = cp.cuda.runtime.memGetInfo()
        print(f"  {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, "
              f"{libre / 2**30:.2f} de {total / 2**30:.2f} GB libres")

    eq = comprobar_equivalencia(xp, dtype, EJES_CRUZADOS)
    print(f"  espectro_angular vs angularSpectrum (referencia): {eq:.2e}")

    if M != N and EJES_CRUZADOS:
        print(f"\n  AVISO: la malla es RECTANGULAR ({M}x{N}) y EJES_CRUZADOS "
              f"esta en True.\n  angularSpectrum usa dfx = 1/(dx*M) y "
              f"dfy = 1/(dy*N), o sea cada eje con la\n  longitud del otro. En "
              f"cuadrada da igual; aqui NO. Contra el gaussiano\n  analitico el "
              f"error es 2.95e-01 asi frente a 8.41e-06 con los ejes en su\n"
              f"  sitio. Pon EJES_CRUZADOS = False, o recorta la imagen a "
              f"cuadrada.")

    f_nyq = 1 / (2 * delta)
    Np = N * PAD
    f_util = Np * delta / (LAMB * np.sqrt(4 * Z**2 + (Np * delta) ** 2))
    # la fraccion se satura en 1: f_util > f_Nyquist quiere decir que la malla
    # muestrea la funcion de transferencia entera, no que aproveche un 619 %
    frac = min(f_util / f_nyq, 1.0)
    print(f"\n  f_Nyquist {f_nyq:.1f} 1/mm | banda util {frac * 100:.1f} %"
          + ("  (la malla la muestrea entera)" if frac >= 1.0 else ""))
    if frac < 0.5:
        print("  AVISO: a esta z el espectro angular alia mas de lo que "
              "propaga. Es el limite del metodo: usa BL-ASM o MPASM.")
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

    kw = dict(xp=xp, dtype=dtype, cruzados=EJES_CRUZADOS)
    U0 = xp.asarray(t_obj, dtype=dtype)

    U_h, t_ida = cronometrar(lambda: propagar(U0, +Z, delta, LAMB, PAD, **kw))
    I_h = xp.abs(U_h) ** 2
    U_a, t_a = cronometrar(lambda: propagar(U_h, -Z, delta, LAMB, PAD, **kw))
    U_b, t_b = cronometrar(
        lambda: propagar(xp.sqrt(I_h).astype(dtype), -Z, delta, LAMB, PAD, **kw))

    print(f"\n  ida {t_ida * 1e3:.0f} ms | vuelta A {t_a * 1e3:.0f} ms | "
          f"vuelta B {t_b * 1e3:.0f} ms")
    print(f"  vuelta A (campo complejo)  correlacion con el objeto = "
          f"{parecido(xp.abs(U_a), xp.abs(U0)):+.4f}   <- tiene que ser ~1")
    print(f"  vuelta B (solo intensidad) correlacion con el objeto = "
          f"{parecido(xp.abs(U_b), xp.abs(U0)):+.4f}")

    # --- barrido de foco -----------------------------------------------------
    zs = curva_a = curva_b = None
    if BARRIDO is not None:
        zs = np.asarray(BARRIDO, float) * Z
        raiz_I = xp.sqrt(I_h).astype(dtype)
        curva_a, curva_b = [], []
        t0 = time.perf_counter()
        for k, z in enumerate(zs):
            Ua = propagar(U_h, -z, delta, LAMB, PAD, **kw)
            curva_a.append(nitidez(xp.abs(Ua) ** 2)); del Ua
            Ub = propagar(raiz_I, -z, delta, LAMB, PAD, **kw)
            curva_b.append(nitidez(xp.abs(Ub) ** 2)); del Ub
            liberar(xp)
            print(f"    barrido {k + 1}/{len(zs)}   z = {z:7.2f} mm", end="\r")
        if dev == "gpu":
            cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        del raiz_I
        liberar(xp)
        curva_a, curva_b = np.array(curva_a), np.array(curva_b)
        print(" " * 48, end="\r")
        print(f"\n  barrido de foco: {len(zs)} distancias de {zs[0]:.1f} a "
              f"{zs[-1]:.1f} mm en {dt:.1f} s "
              f"({dt / (2 * len(zs)) * 1e3:.0f} ms por propagacion)")
        print(f"    vuelta A enfoca en z = {zs[curva_a.argmax()]:.2f} mm")
        print(f"    vuelta B enfoca en z = {zs[curva_b.argmax()]:.2f} mm")
        print(f"    esperado:            z = {Z:.2f} mm")

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
    fig.tight_layout()
    figuras = [("ida_y_vuelta.png", fig)]

    if zs is not None:
        fig2, a2 = plt.subplots(figsize=(7.2, 4.2))
        a2.plot(zs, curva_a / curva_a.max(), "o-", ms=3, label="A: campo complejo")
        a2.plot(zs, curva_b / curva_b.max(), "s-", ms=3, label="B: solo intensidad")
        a2.axvline(Z, color="k", ls="--", lw=1, label=f"z real = {Z:g} mm")
        a2.set_xlabel("distancia de reconstruccion [mm]")
        a2.set_ylabel("nitidez (energia del gradiente, normalizada)")
        a2.set_title(f"Donde enfoca la retropropagacion  ({dev.upper()})",
                     fontsize=11)
        a2.legend(fontsize=9)
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
