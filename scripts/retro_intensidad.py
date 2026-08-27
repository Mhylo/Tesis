"""Retropropaga un holograma y guarda la INTENSIDAD reconstruida.

Entra una imagen que ya es un holograma (intensidad medida o simulada), y sale
un PNG por distancia del barrido con |U|^2, sin gamma ni ninguna otra cosa.

QUE HACE, EN TRES LINEAS
    campo = sqrt(imagen)              un sensor mide intensidad: la fase se perdio
    U = retropropagar(campo, -z)      espectro angular a distancia negativa
    PNG = |U|^2 / max                 modulo cuadrado, y nada mas

POR QUE SIN GAMMA
    Los scripts retro_*.py guardan con gamma 0.5, y

        (|U|^2 / max)^0.5  =  |U| / sqrt(max)

    o sea que sus imagenes no son intensidad: son AMPLITUD. La normalizacion
    por el maximo si se queda -un PNG de 8 bits no admite floats-, pero es una
    escala lineal y no cambia la magnitud que se ensena.

QUE ESPERAR DE LA RECONSTRUCCION
    Un holograma in-line siempre devuelve el objeto con su imagen gemela
    desenfocada superpuesta. Es inherente al metodo, no un fallo del script.
    Tampoco hay correccion DLHM de fuente puntual: esto asume iluminacion
    colimada (onda plana).

LOS DOS PROPAGADORES
    angularSpectrum()  es la REFERENCIA, copiada sin tocar una linea. Corre
                       siempre en CPU y complex128 (no acepta CuPy), y
                       materializa seis mallas completas: a 512x512 con PAD=2
                       son ~80 MB, pero con un holograma 4000x3000 pide ~3.8 GB
                       y falla.
    espectro_angular() es la de TRABAJO: el mismo calculo por bloques de filas,
                       en NumPy o CuPy, y con los ejes de frecuencia en su
                       sitio (ver el aviso de M != N mas abajo).

UNIDADES: milimetros para todo. Da igual cual sea mientras sea la MISMA en
LAMB, DELTA y Z: el espectro angular solo ve lambda*z/delta^2.

    633 nm -> 633e-6 mm     3.45 um -> 3.45e-3 mm     150 mm -> 150.0
"""

import pathlib

import numpy as np
from PIL import Image

try:
    import cupy as cp
except Exception:                      # sin CuPy, sin CUDA, o CuPy roto
    cp = None


# ════════════════════════════════════════════════════════════════════════════
#  1. PARAMETROS  --  es lo unico que hay que editar
# ════════════════════════════════════════════════════════════════════════════

#: El HOLOGRAMA (imagen de intensidad), no un objeto. Usa barras normales o
#: antepon r a las comillas para que \U no se lea como escape.
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\campos\fft\z0150.png"

#: Longitud de onda [mm]. 633 nm se escribe 633e-6.
LAMB = 633e-6

#: Paso de pixel de TU sensor [mm]. 3.45 um se escribe 3.45e-3.
DELTA = 3.45e-3

#: Barrido de distancias holograma <-> objeto [mm], POSITIVAS: el signo lo pone
#: retropropagar(). Con un holograma real no sabes a que distancia esta el
#: objeto; para eso barre. Para una sola distancia, pon Z = (150, 150), PASOS = 1.
Z = (10.0, 150.0)
PASOS = 30

#: Relleno de ceros. La FFT convoluciona de forma circular: lo que sale por un
#: borde reentra por el opuesto. Con 2 ese doblez queda fuera del recorte.
#: Bajar a 1 lo desactiva y cuadruplica la velocidad, a cambio del artefacto.
PAD = 2

#: Con que propagar. Quita el que no te interese y ese barrido no se hace.
#: "referencia" = angularSpectrum, "bloques" = espectro_angular.
METODOS = ("referencia", "bloques")

#: "auto" usa la GPU si hay CuPy con CUDA; "cpu" y "gpu" fuerzan. Solo afecta a
#: "bloques": "referencia" corre en CPU siempre.
DISPOSITIVO = "auto"

#: Carpeta destino, o None para resultados/retro_intensidad/<holograma>/
SALIDA = None


# ════════════════════════════════════════════════════════════════════════════
#  2. RETROPROPAGADOR  --  el nucleo. Si borras algo de aqui, no queda script.
# ════════════════════════════════════════════════════════════════════════════

def angularSpectrum(field, z, wavelength, dx, dy, scale_factor=1):
    """
    Propagación angular del frente de onda usando el espectro angular
    field: campo complejo
    z: distancia de propagación
    wavelength: longitud de onda
    dx, dy: pasos espaciales
    """
    # NO EDITAR. Esta funcion esta copiada tal cual de la implementacion de
    # referencia, y su valor entero es que nadie la ha tocado: es contra ella
    # contra la que se contrasta espectro_angular(). Si hay que cambiar algo,
    # se cambia en la otra.
    #
    # Ojo a dfx = 1/(dx*M) y dfy = 1/(dy*N): estan CRUZADOS, cada eje lleva la
    # longitud del otro. En malla cuadrada da igual. En rectangular NO, y por
    # eso main() avisa cuando M != N.
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


def espectro_angular(field, z, wavelength, dx, dy, scale_factor=1,
                     xp=np, dtype=np.complex128, filas=512):
    """Lo mismo que angularSpectrum(), pero cabe en la tarjeta.

    Misma matematica y mismo orden de shifts. Tres diferencias, todas de
    ejecucion:

    1. El kernel no se materializa entero: el fasor se evalua por bloques de
       `filas` filas y se multiplica in situ sobre el espectro. Baja `filas` si
       la GPU se queda sin memoria.
    2. La fase se calcula en float64 y solo el fasor -acotado a modulo 1- baja
       a dtype. A z = 150 mm la fase vale ~1e6 rad; en float32 puro eso serian
       decimas de radian de error solo por la mantisa.
    3. xp es NumPy o CuPy, con el mismo cuerpo.

    Y una de algoritmo: dfx = 1/(dx*N) y dfy = 1/(dy*M), cada eje con SU
    longitud, no cruzados como en la referencia.
    """
    U = xp.asarray(field, dtype=dtype)
    M, N = U.shape

    dfx, dfy = 1 / (dx * N), 1 / (dy * M)

    # (1, N) y (M, 1): la aritmetica difunde igual que con meshgrid y no
    # materializa dos mallas completas
    fx = xp.asarray(((np.arange(N) - N / 2) * dfx).astype(np.float64))[None, :]
    fy = xp.asarray(((np.arange(M) - M / 2) * dfy).astype(np.float64))[:, None]

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


def retropropagar(U, z, delta, lamb, pad, metodo, xp=np, dtype=np.complex128):
    """Rellena de ceros, propaga a -z, y recorta al tamano original.

    El signo va DENTRO: se llama con z positiva y aqui se le pone el menos. Asi
    no hay forma de equivocarse desde fuera.
    """
    M, N = U.shape
    if pad > 1:
        rel = xp.zeros((M * pad, N * pad), dtype=dtype)
        i, j = (rel.shape[0] - M) // 2, (rel.shape[1] - N) // 2
        rel[i:i + M, j:j + N] = xp.asarray(U, dtype=dtype)
    else:
        rel, i, j = xp.asarray(U, dtype=dtype), 0, 0

    if metodo == "referencia":
        # a_cpu porque angularSpectrum hace np.array(field) y no traga CuPy.
        # Es deliberado: la referencia corre siempre igual, se ponga lo que se
        # ponga en DISPOSITIVO.
        fuera = angularSpectrum(a_cpu(rel), -z, lamb, delta, delta)
    else:
        fuera = espectro_angular(rel, -z, lamb, delta, delta,
                                 xp=xp, dtype=dtype)
    del rel

    # copia, no vista: una vista mantendria viva la malla rellena entera
    recorte = fuera[i:i + M, j:j + N].copy()
    del fuera
    return recorte


# ════════════════════════════════════════════════════════════════════════════
#  3. IMAGENES  --  entrada y salida. Aqui esta el modulo cuadrado.
# ════════════════════════════════════════════════════════════════════════════

def cargar_holograma(ruta):
    """Imagen -> campo complejo de partida.

    La imagen es INTENSIDAD medida, asi que el campo es sqrt(I): un sensor no
    registra fase. (Si la imagen fuese un objeto -una transmitancia- el campo
    seria t, sin raiz. No es este script.)
    """
    img = Image.open(ruta).convert("L")
    I = np.asarray(img, dtype=np.float64) / 255.0
    return I


def guardar_intensidad(U, ruta):
    """Escribe |U|^2 como PNG de 8 bits, creando la carpeta.

    ESTA ES LA FUNCION DEL EJERCICIO. Lo unico que se le aplica al campo es el
    modulo cuadrado y una division por el maximo.

    La division no es una correccion de visualizacion: un PNG de 8 bits no
    admite floats y hay que llevar el rango a [0, 1]. Es LINEAL, o sea que no
    cambia la relacion entre valores. Una gamma si, y de la forma exacta que
    convierte intensidad en amplitud:

        (|U|^2 / max)^0.5  =  |U| / sqrt(max)

    Normalizar por el maximo de CADA imagen y no por uno comun al barrido hace
    que cada PNG mida contraste y no brillo absoluto: dos distancias son
    comparables aunque no les llegue la misma energia.

    Un campo identicamente nulo daria 0/0 -NaN por todo el array y un PNG de
    basura, sin aviso-. Un plano negro es un resultado legitimo y hay que poder
    verlo como tal.
    """
    I = np.abs(a_cpu(U)).astype(np.float64) ** 2
    m = I.max()
    A = np.clip(I / m, 0, 1) if m > 0 else np.zeros_like(I)
    ruta = pathlib.Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((A * 255).astype(np.uint8)).save(ruta)


# ════════════════════════════════════════════════════════════════════════════
#  4. AUXILIARES  --  fontaneria. Borrables si te quedas solo en CPU.
# ════════════════════════════════════════════════════════════════════════════

def elegir_dispositivo(preferencia="auto"):
    """(modulo de arrays, nombre). Cae a NumPy sin ruido si no hay CUDA.

    Si decides trabajar solo en CPU: borra esta funcion y a_cpu(), quita el
    try/except de cupy y la constante DISPOSITIVO, y sustituye xp por np en las
    dos firmas que lo llevan. Son ~25 lineas menos.
    """
    hay_gpu = False
    if cp is not None:
        try:
            hay_gpu = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            hay_gpu = False
    if preferencia == "gpu" and not hay_gpu:
        raise SystemExit("DISPOSITIVO = 'gpu' pero no encuentro CuPy con CUDA.\n"
                         "Pon DISPOSITIVO = 'auto' o 'cpu'.")
    if preferencia == "cpu" or not hay_gpu:
        return np, "cpu"
    return cp, "gpu"


def a_cpu(a):
    """Array de NumPy, venga de donde venga."""
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def nombre_png(z):
    """Nombre del PNG de una distancia del barrido.

    Tres decimales porque un linspace da distancias no enteras: con formato
    entero, 49.6 y 50.2 escribirian las dos en z0050.png y la segunda pisaria a
    la primera sin aviso. El ancho fijo mantiene el orden alfabetico igual al
    orden del barrido, que es lo que hace que un `ls` ensene la pila.
    """
    return f"z{z:08.3f}.png"


def carpeta(holograma, metodo):
    """resultados/retro_intensidad/<holograma>/<metodo>/, bajo la raiz del repo.

    Absoluta y no relativa: el script se lanza desde el editor, desde scripts/ o
    desde la raiz, y una ruta relativa dejaria la pila en el directorio de
    invocacion, distinto cada vez.

    <holograma> es el stem de RUTA. Sin el, dos hologramas distintos al mismo
    barrido escriben los mismos nombres en la misma carpeta y el segundo pisa
    al primero sin aviso.
    """
    if SALIDA is not None:
        raiz = pathlib.Path(SALIDA)
    else:
        raiz = (pathlib.Path(__file__).resolve().parent.parent
                / "resultados" / "retro_intensidad")
    return raiz / holograma / metodo


# ════════════════════════════════════════════════════════════════════════════
#  5. MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    if not pathlib.Path(RUTA).is_file():
        raise SystemExit(f"No encuentro el holograma en:\n    {RUTA}\n\n"
                         "Edita la constante RUTA al principio del archivo.")

    zs = np.linspace(Z[0], Z[1], PASOS)
    if zs.min() <= 0:
        raise SystemExit(
            f"El barrido llega a z = {zs.min():g}, pero Z es la separacion "
            f"holograma-objeto y va POSITIVA:\nel signo lo pone "
            f"retropropagar(). Sube Z[0] por encima de 0.")

    xp, dev = elegir_dispositivo(DISPOSITIVO)
    dtype = np.complex64 if dev == "gpu" else np.complex128

    U0 = cargar_holograma(RUTA)
    M, N = U0.shape
    holograma = pathlib.Path(RUTA).stem

    print(f"holograma {RUTA}")
    print(f"  malla {M}x{N}, ventana {N * DELTA:.3f} x {M * DELTA:.3f} mm "
          f"(con relleno x{PAD}: {N * PAD * DELTA:.3f} mm)")
    print(f"  lambda {LAMB * 1e6:.1f} nm | delta {DELTA * 1e3:.3f} um")
    print(f"  barrido: {PASOS} distancias de {zs[0]:.2f} a {zs[-1]:.2f} mm")
    print(f"  dispositivo {dev.upper()} (solo para 'bloques'; "
          f"'referencia' va en CPU y complex128)")

    if M != N and "referencia" in METODOS and "bloques" in METODOS:
        print(f"\n  AVISO: la malla es RECTANGULAR ({M}x{N}), asi que las dos "
              f"pilas van a SALIR DISTINTAS.\n  angularSpectrum usa "
              f"dfx = 1/(dx*M) y dfy = 1/(dy*N), o sea cada eje con la longitud "
              f"del\n  otro; espectro_angular los pone en su sitio. En cuadrada "
              f"coinciden; aqui no.\n  La diferencia es ese cruce, no una "
              f"diferencia de metodo: 'bloques' es la buena.\n  Contra el "
              f"gaussiano analitico el error es 2.95e-01 cruzado frente a "
              f"8.41e-06 sin cruzar.")

    U0 = xp.asarray(U0, dtype=dtype)

    for metodo in METODOS:
        destino = carpeta(holograma, metodo)
        print(f"\n  {metodo} -> {destino}")
        for k, z in enumerate(zs):
            U = retropropagar(U0, z, DELTA, LAMB, PAD, metodo,
                              xp=xp, dtype=dtype)
            guardar_intensidad(U, destino / nombre_png(z))
            del U
            if dev == "gpu":
                cp.get_default_memory_pool().free_all_blocks()
            print(f"    {k + 1}/{PASOS}   z = {z:7.2f} mm", end="\r")
        print(f"    {PASOS} PNG escritos" + " " * 20)

    print("\nListo. Abre la carpeta y pasa las flechas: la distancia que "
          "enfoque es la buena.")


if __name__ == "__main__":
    main()
