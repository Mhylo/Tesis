"""Retropropaga TU holograma. Edita el bloque de abajo y dale a Run.

Es la misma retropropagación que la CLI de CamposT.retropropagacion —de hecho
llama a su main(), no duplica nada—, pero con los parámetros como constantes
editables en vez de argumentos. Para cuando quieres darle a Run en el editor y
no escribir una línea de órdenes.

Si prefieres la línea de órdenes, es equivalente a:

    python -m CamposT.retropropagacion TU_HOLOGRAMA.png --z 5 50 --pasos 30

Lo que hace: carga la imagen como intensidad medida (el campo es sqrt(I)),
propaga a −z con los tres métodos, y escribe la intensidad reconstruida a cada
distancia del barrido. La distancia de enfoque sale de MIRAR la pila: la que
enfoque es la buena.

Dos límites heredados del módulo, que están en su docstring y conviene tener
presentes al leer el resultado: asume iluminación colimada (sin la corrección
de fuente puntual de DLHM) y la reconstrucción trae la imagen gemela
superpuesta.
"""

import pathlib

from CamposT.retropropagacion import main

# ════════════════════════════════════════════════════════════════════════════
#  EDITA DE AQUÍ...
# ════════════════════════════════════════════════════════════════════════════

#: Ruta a tu holograma. Pega la ruta completa entre las comillas.
#: Ojo con las barras en Windows: usa / o antepón r a las comillas
#: (r"C:\Users\...") para que \U, \n y compañía no se lean como escapes.
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\campos\fft\z0150.png"

#: Paso de píxel de TU sensor, en mm. 3.45 µm se escribe 3.45e-3.
DELTA = 3.45e-3

#: Longitud de onda de TU láser, en mm. 405 nm se escribe 405e-6.
LAMB = 405e-6

#: Barrido de distancias sensor-objeto, en mm, POSITIVAS: el signo lo pone la
#: retropropagación. Dos valores barren entre ellos; uno solo reconstruye a esa
#: distancia y ya. Si no sabes a qué distancia está el objeto, ábrelo: es para
#: eso que esto barre.
Z = (10, 150)
PASOS = 30

#: Propagadores a comparar. Quita los que no te interesen.
METODOS = ("fft", "blas", "mpasm")

#: Redimensionar el holograma a N×N antes de propagar, o None para dejarlo
#: como está. Bajarlo acelera el barrido; OJO: si lo cambias, DELTA deja de
#: ser el de tu sensor y hay que escalarlo por (tamaño_original / N).
N = None

#: Relleno de ceros (2 evita el wrap-around de la convolución circular), gamma
#: de la imagen guardada, dispositivo, y sobremuestreo de MPASM. El defecto de
#: mpasm() es s=10, pero su matriz espectral es (s·N)² POR DISTANCIA: en un
#: barrido no cabe. Súbelo sólo con una única distancia.
PAD = 2
GAMMA = 0.6
DEVICE = "auto"          # "auto" | "cpu" | "gpu"
S = 1

#: True si tu holograma viene invertido (fondo claro donde debería ser oscuro).
INVERTIR = False

#: Carpeta destino, o None para resultados/retropropagacion/<nombre>/
SALIDA = None

# ════════════════════════════════════════════════════════════════════════════
#  ...HASTA AQUÍ. Lo de abajo no hace falta tocarlo.
# ════════════════════════════════════════════════════════════════════════════


def _argumentos():
    """Traduce las constantes de arriba a los argumentos de la CLI.

    Se pasa por la CLI en vez de llamar a retropropagar() directamente para
    que este script y `python -m CamposT.retropropagacion` no puedan divergir:
    hay un solo camino de código, y arreglar algo en uno lo arregla en los dos.
    """
    argv = [RUTA]
    argv += ["--z"] + [str(v) for v in (Z if isinstance(Z, (tuple, list)) else [Z])]
    argv += ["--pasos", str(PASOS)]
    argv += ["--delta", str(DELTA), "--lamb", str(LAMB)]
    argv += ["--metodos"] + list(METODOS)
    argv += ["--pad", str(PAD), "--gamma", str(GAMMA)]
    argv += ["--device", DEVICE, "--s", str(S)]
    if N is not None:
        argv += ["--N", str(N)]
    if INVERTIR:
        argv += ["--invert"]
    if SALIDA is not None:
        argv += ["--salida", str(SALIDA)]
    return argv


if __name__ == "__main__":
    if not pathlib.Path(RUTA).is_file():
        raise SystemExit(
            f"No encuentro el holograma en:\n    {RUTA}\n\n"
            "Edita la constante RUTA al principio de este archivo y pon ahí la "
            "ruta de tu imagen.\nEn Windows, escríbela con barras normales "
            "(C:/Users/...) o antepón r a las comillas.")
    main(_argumentos())
