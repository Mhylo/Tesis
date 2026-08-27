import pathlib

import numpy as np
from PIL import Image

try:
    import cupy as cp
except Exception:                      # sin CuPy, sin CUDA, o CuPy roto
    cp = None

import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════════════════════════
#  1. PARAMETROS  --  es lo unico que hay que editar
# ════════════════════════════════════════════════════════════════════════════

#: El HOLOGRAMA (imagen de intensidad), no un objeto. Usa barras normales o
#: antepon r a las comillas para que \U no se lea como escape.
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\campos\entrada.png"

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

img = Image.open(RUTA).convert("L")
I = np.asarray(img, dtype=np.float64) / 255.0


plt.imshow(I, cmap='gray')
plt.show()
plt.imshow(np.angle(I), cmap='gray')
plt.show()

holograma = angularSpectrum(I, 10, LAMB, DELTA, DELTA)
plt.imshow(np.abs(holograma)**2, cmap='gray')
plt.show()
plt.imshow(np.angle(holograma), cmap='gray')
plt.show()

retropropagado = angularSpectrum(holograma, -10, LAMB, DELTA, DELTA)
plt.imshow(np.abs(retropropagado)**2, cmap='gray')
plt.show()
plt.imshow(np.angle(retropropagado), cmap='gray')
plt.show()