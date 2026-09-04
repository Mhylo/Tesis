"""Prueba suelta: ida y vuelta con el angularSpectrum de pyDHM, a pelo.

NO ES UNA PRUEBA DE PYTEST, aunque vivio en tests/mi_test.py hasta el 01/09.
Ahi su nombre casaba con *_test.py, asi que pytest lo IMPORTABA en cada
recoleccion; y como el cuerpo del modulo propaga y llama seis veces a
plt.show(), con el backend TkAgg de esta maquina la recoleccion se quedaba
COLGADA esperando a que alguien cerrara las ventanas.

Esta superado por scripts/retro_fft_angular.py, que hace la misma ida y vuelta
con el mismo angularSpectrum copiado tal cual, pero entera: barrido, figura de
3x3 y las dos erratas del original medidas. Esto se queda como quedo.

QUE HACE AHORA. Barre distancias y a cada una correlaciona la reconstruccion
con una IMAGEN DE REFERENCIA -el objeto que produjo el holograma- para decir en
que z enfoca. Eso lo saca del ojimetro: devuelve un numero por distancia, no una
impresion.

El holograma que trae RUTA por defecto lo escribio scripts/retro_fft_angular.py
y viene con un .txt al lado que dice de que objeto salio y a que distancia. Ese
.txt es la verdad de terreno: el pico del barrido TIENE que caer donde el diga.
Si no cae ahi, no se ajusta el barrido, se mira por que.

ENTRADA decide como se lee la imagen, y no es cosmetico:

    "intensidad"  campo = sqrt(I). Lo correcto para un holograma: un sensor
                  registra |U|^2, asi que el campo es su raiz.
    "amplitud"    campo = I tal cual. Es lo que este script hacia antes.

Tomar la intensidad por amplitud eleva el campo al cuadrado, y eso MUEVE la z a
la que enfoca. Con el barrido puntuado se pueden correr las dos y medir cuanto
se mueve; antes era invisible.

POR QUE LA VUELTA CUADRA. El holograma por defecto se hizo con los EJES
CRUZADOS -lo dice su .txt- sobre una malla rectangular de 3000x4000, y este
script usa la MISMA angularSpectrum con los mismos ejes cruzados: por eso la
vuelta deshace la ida. Con un holograma hecho con los ejes en su sitio dejaria
de cuadrar, y el pico se iria de sitio sin que nada avisara.

Las distancias de Z van POSITIVAS: el menos lo pone el barrido.
"""
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
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\hologramas\BenchmarkTarget\fft\z0010.000.png"

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
Z = (5.0, 20.0)
PASOS = 30



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

# ════════════════════════════════════════════════════════════════════════════
#  3. BARRIDO CON REFERENCIA
# ════════════════════════════════════════════════════════════════════════════

def campo_de_entrada(ruta, entrada):
    """Imagen -> (campo complejo, imagen cruda, etiqueta de que es).

    Un sensor registra |U|^2 y tira la fase, asi que de un holograma el campo de
    partida es sqrt(I). Tomarlo como amplitud -lo que hacia este script- eleva
    el campo al cuadrado: no es solo contraste, MUEVE la z a la que enfoca.

    La imagen cruda se devuelve aparte porque es lo que hay que pintar en el
    panel de entrada: con "amplitud", |campo|^2 seria I^2 y no la imagen.

    La etiqueta se devuelve para imprimirla: de que rama viene depende como se
    lee el barrido entero.
    """
    img = np.asarray(Image.open(ruta).convert("L"), dtype=np.float64) / 255.0
    if entrada == "intensidad":
        return np.sqrt(img).astype(complex), img, "intensidad medida, campo = sqrt(I)"
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
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else 0.0


def main():
    ref = np.asarray(Image.open(REFERENCIA).convert("L"), dtype=np.float64) / 255.0
    campo, img, etiqueta = campo_de_entrada(RUTA, ENTRADA)

    # Reventar antes que reescalar. Dos muestreos distintos del mismo objeto dan
    # una correlacion perfectamente calculable y sin sentido, y el pico caeria
    # donde le diera la gana sin que nada avisara.
    if campo.shape != ref.shape:
        raise SystemExit(
            f"La referencia y el holograma no tienen la misma forma:\n"
            f"    holograma  {campo.shape}  {RUTA}\n"
            f"    referencia {ref.shape}  {REFERENCIA}\n"
            f"No se reescala ninguna de las dos a proposito: comparar dos "
            f"muestreos distintos del mismo objeto da un numero que parece "
            f"bueno y no lo es.")

    if min(Z) <= 0:
        raise SystemExit(f"Z = {Z} y son distancias holograma-objeto, "
                         f"POSITIVAS: el menos lo pone el barrido.")
    zs = np.linspace(float(Z[0]), float(Z[1]), PASOS)

    M, N = campo.shape
    print(f"holograma  {RUTA}")
    print(f"  {etiqueta}")
    print(f"referencia {REFERENCIA}")
    print(f"  malla {M}x{N} | lambda {LAMB * 1e6:.1f} nm | "
          f"delta {DELTA * 1e3:.3f} um")
    print(f"  {len(zs)} distancias de {zs[0]:.3f} a {zs[-1]:.3f} mm")
    if M != N:
        print("  malla RECTANGULAR: angularSpectrum lleva los ejes CRUZADOS, "
              "asi que esto solo cuadra\n  con hologramas hechos con esa "
              "misma funcion. Ver el docstring.")
    print()

    curva = np.empty(len(zs))
    mejor_U, mejor = None, -1
    for i, z in enumerate(zs):
        U = angularSpectrum(campo, -z, LAMB, DELTA, DELTA)
        curva[i] = correlacion(np.abs(U) ** 2, ref)
        # No se acumulan los campos: 30 mallas de 3000x4000 en complex128 son
        # 5.7 GB. Se guarda solo el mejor hasta ahora, que son 192 MB.
        if mejor < 0 or curva[i] > curva[mejor]:
            mejor_U, mejor = U, i
        print(f"  z = {z:8.3f} mm   corr = {curva[i]:+.4f}")

    print(f"\nenfoca en z = {zs[mejor]:.3f} mm   corr = {curva[mejor]:+.4f}")
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
