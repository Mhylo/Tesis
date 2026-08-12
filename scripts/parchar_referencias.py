"""Arregla los repos de referencia de terceros para que corran en este entorno.

referencia/carlos/ está en .gitignore (son ~224 MB de repos ajenos con su
propia historia en GitHub), así que cualquier corrección hecha a mano sobre
esos archivos se pierde al re-clonarlos y no queda registrada en ningún lado.
Este script es el registro: aplica los arreglos de forma idempotente y deja en
el código de terceros un comentario marcado con [parche TG] en cada línea
tocada, para que nunca se confunda lo publicado con lo corregido aquí.

Ninguno de los parches cambia la física. Son incompatibilidades con NumPy 2 y
un refactor que quedó a medias en el repo original.

QUÉ ARREGLA

  pyLHM/myfunctions.py
    rayleigh1Free (la Rayleigh-Sommerfeld completa, sin aproximaciones) pide
    dtype='complex_'. NumPy 2.0 eliminó ese alias, así que la función muere
    antes de calcular nada. Es la única incompatibilidad con NumPy 2 en todo
    el árbol de referencias.

  dlhm.py
    La firma de dlhm() se actualizó a un sensor rectangular (W_cx, W_cy) pero
    el cuerpo quedó usando el W_c anterior, que ya no existe: toda llamada
    moría con NameError. Además el conteo de píxeles del sensor se calcula
    como flotante y np.linspace exige un entero.

  main_dlhm.py
    Llama a dlhm() con la firma vieja de 7 argumentos contra una función que
    ahora pide 8.

  main_dlhm.py, reconstruction_dlhm.py, simulation_reconstruction_asm_dlhm.py
    Cargan sus imágenes con una ruta relativa a data/, que Python resuelve
    contra el directorio desde el que se lanza y no contra el del script.
    Lanzados desde cualquier otra carpeta, cv.imread devuelve None sin lanzar
    excepción y el error aparece más adelante como un desempaquetado imposible
    sobre sample.shape, sin mencionar nunca que faltó un archivo.

QUÉ NO ARREGLA

  - simulation_reconstruction_asm_dlhm.py necesita data/Complete_Benchmark.png,
    que no viene en el repo. Hay que pedírselo a los autores.
  - Las salidas (fig.write_html) siguen cayendo en el directorio desde el que
    se lanza el script. Eso es inofensivo y a veces cómodo, así que se deja.
  - Los scripts de demostración de DLHM-processing-tools (RS1_main.py,
    kreuzer_main.py, main.py, simulate.py) llaman a métodos que no existen en
    el myfunctions.py publicado: rayleigh_convolutional, kreuzer_reconstruct y
    convergentSAASM_full. Están desactualizados frente a su propia librería y
    apuntan a rutas fijas de otra máquina; no son punto de entrada válido.
    Hay que instanciar pyLHM.myfunctions.reconstruct directamente.

DEPENDENCIAS

  pyLHM importa skimage y xarray a nivel de módulo (solo las usa en helpers de
  visualización y autofoco, pero el import falla igual y se lleva todo por
  delante) y sus scripts usan imageio:

      Tesis_env/Scripts/python.exe -m pip install scikit-image xarray imageio

USO

    Tesis_env/Scripts/python.exe -m scripts.parchar_referencias
    Tesis_env/Scripts/python.exe -m scripts.parchar_referencias --revisar

Se puede correr las veces que haga falta: lo ya parcheado se reporta y se deja
igual. Devuelve 1 si algún ancla no aparece, que es la señal de que el archivo
de terceros cambió y el parche necesita revisión.
"""

import argparse
import pathlib
import sys
from dataclasses import dataclass

#: raíz del repo, dos niveles arriba de este archivo
RAIZ = pathlib.Path(__file__).resolve().parent.parent

#: repos de terceros, fuera del control de versiones (ver .gitignore)
MODELO = RAIZ / "referencia" / "carlos" / "DLHM-model-main" / "DLHM-model-main"
HERRAMIENTAS = (RAIZ / "referencia" / "carlos" / "DLHM-processing-tools-main"
                / "DLHM-processing-tools-main")

#: marca que hace idempotente al parche y visible al cambio dentro del código ajeno
MARCA = "[parche TG]"


@dataclass(frozen=True)
class Parche:
    """Una sustitución literal sobre un archivo de terceros.

    `antes` debe aparecer exactamente una vez: si aparece cero veces el archivo
    cambió, y si aparece varias la sustitución sería ambigua. En ambos casos el
    parche se salta en vez de adivinar.
    """

    archivo: pathlib.Path
    motivo: str
    antes: str
    despues: str


PARCHES = (
    Parche(
        archivo=HERRAMIENTAS / "pyLHM" / "myfunctions.py",
        motivo="rayleigh1Free: dtype='complex_' no existe desde NumPy 2.0",
        antes="        U0 = np.zeros(out_shape, dtype='complex_')",
        despues=(
            f"        # {MARCA} era dtype='complex_'. NumPy 2.0 eliminó ese alias y\n"
            "        # la función moría con TypeError antes de calcular nada.\n"
            "        U0 = np.zeros(out_shape, dtype=complex)"
        ),
    ),
    Parche(
        archivo=MODELO / "dlhm.py",
        motivo="dlhm(): el número de píxeles del sensor sale flotante",
        antes=(
            "    Q = W_cy / dx_out\n"
            "    P = W_cx / dx_out"
        ),
        despues=(
            f"    # {MARCA} P y Q son conteos de píxeles y se usaban sin redondear:\n"
            "    # np.linspace(..., num=P) rechaza un flotante desde NumPy 1.18.\n"
            "    Q = round(W_cy / dx_out)\n"
            "    P = round(W_cx / dx_out)"
        ),
    ),
    Parche(
        archivo=MODELO / "dlhm.py",
        motivo="dlhm(): coordenadas del sensor con el W_c huérfano",
        antes=(
            "    x = np.linspace(-W_c / 2, W_c / 2, P)\n"
            "    y = np.linspace(-W_c / 2, W_c / 2, Q)"
        ),
        despues=(
            f"    # {MARCA} la firma se actualizó a W_cx/W_cy pero el cuerpo quedó\n"
            "    # usando W_c, que ya no existe: toda llamada moría con NameError.\n"
            "    # x recorre P = W_cx/dx_out píxeles y y recorre Q = W_cy/dx_out.\n"
            "    x = np.linspace(-W_cx / 2, W_cx / 2, P)\n"
            "    y = np.linspace(-W_cy / 2, W_cy / 2, Q)"
        ),
    ),
    Parche(
        archivo=MODELO / "dlhm.py",
        motivo="dlhm(): magnificación máxima con el W_c huérfano",
        antes="    Mag_max = np.sqrt(W_c ** 2 / 2 + L ** 2) / z",
        despues=(
            f"    # {MARCA} mismo W_c huérfano. sqrt(W_c**2/2 + L**2) es la distancia\n"
            "    # del origen a la esquina de un sensor cuadrado; la forma rectangular\n"
            "    # equivalente es sqrt((W_cx**2 + W_cy**2)/4 + L**2), que coincide con\n"
            "    # la original cuando W_cx == W_cy.\n"
            "    Mag_max = np.sqrt((W_cx ** 2 + W_cy ** 2) / 4 + L ** 2) / z"
        ),
    ),
    Parche(
        archivo=MODELO / "main_dlhm.py",
        motivo="main_dlhm.py: llama a dlhm() con la firma vieja de 7 argumentos",
        antes="holo = dlhm(sample, dx_in, L, z, W_c, dx_in, lambda_, x0=0, y0=0, NA_s=0.1)",
        despues=(
            f"# {MARCA} la llamada usaba la firma vieja de 7 argumentos:\n"
            "# (sample, dx_in, L, z, W_c, dx_out, wavelength). La actual pide el ancho\n"
            "# del sensor en los dos ejes, así que el sensor cuadrado de este script se\n"
            "# escribe W_cx = W_cy = W_c, y dx_out sigue siendo dx_in como antes.\n"
            "holo = dlhm(sample, dx_in, L, z, W_c, W_c, dx_in, lambda_, x0=0, y0=0, NA_s=0.1)"
        ),
    ),
)


def _parche_ruta_datos(script, literal, imagen, linea):
    """Ancla la carga de una imagen a la carpeta del script y no al cwd.

    Los tres scripts de DLHM-model cargan sus imágenes con una ruta relativa,
    que Python resuelve contra el directorio desde el que se lanzó, no contra
    el del script. Correrlos desde cualquier otra carpeta hace que cv.imread
    devuelva None sin lanzar excepción, y el error aparece varios archivos
    después como un desempaquetado imposible sobre `sample.shape`.

    `literal` es la ruta tal como está escrita en el código (incluido el './'
    de reconstruction_dlhm.py) e `imagen` el nombre del archivo dentro de data/.
    """
    return Parche(
        archivo=MODELO / script,
        motivo=f"'{literal}' se resolvía contra el cwd, no contra el script",
        antes=linea,
        despues=(
            f"# {MARCA} la ruta era relativa al directorio desde el que se lanza\n"
            "# Python, no al del script: al correrlo desde otra carpeta cv.imread\n"
            "# devolvía None en silencio y el fallo salía después, disfrazado de\n"
            "# 'not enough values to unpack' sobre sample.shape. pathlib se importa\n"
            "# aquí, y no en la cabecera, para que el parche sea un bloque contiguo\n"
            "# y no pueda quedar aplicado a medias.\n"
            "import pathlib\n"
            "DATOS_TG = pathlib.Path(__file__).resolve().parent / 'data'\n"
            + linea.replace(f"'{literal}'", f"str(DATOS_TG / '{imagen}')")
        ),
    )


#: los tres scripts de DLHM-model comparten el mismo defecto de ruta relativa
PARCHES += (
    _parche_ruta_datos(
        "main_dlhm.py", "data/BenchmarkTarget.png", "BenchmarkTarget.png",
        "intensityImage = np.array(cv.imread('data/BenchmarkTarget.png', "
        "cv.IMREAD_GRAYSCALE)).astype(float) / 255",
    ),
    _parche_ruta_datos(
        "reconstruction_dlhm.py", "./data/Simulated_hologram.png",
        "Simulated_hologram.png",
        "hologram = np.array(cv.imread('./data/Simulated_hologram.png', "
        "cv.IMREAD_GRAYSCALE)).astype(float) / 255  # Load hologram image",
    ),
    # este seguirá fallando hasta que aparezca Complete_Benchmark.png, que no
    # viene en el repo publicado; el parche lo deja listo para cuando llegue
    _parche_ruta_datos(
        "simulation_reconstruction_asm_dlhm.py", "data/Complete_Benchmark.png",
        "Complete_Benchmark.png",
        "intensityImage = np.array(cv.imread('data/Complete_Benchmark.png', "
        "cv.IMREAD_GRAYSCALE)).astype(float) / 255",
    ),
)


def aplicar(parche, escribir=True):
    """Aplica un parche y devuelve su estado como cadena.

    Estados: 'falta archivo', 'ya estaba', 'sin ancla', 'aplicado'.
    """
    if not parche.archivo.exists():
        return "falta archivo"

    texto = parche.archivo.read_text(encoding="utf-8")

    # la marca dentro del bloque de reemplazo es lo que hace idempotente al
    # script: si ya está, el parche corrió antes sobre este archivo
    if parche.despues in texto:
        return "ya estaba"

    apariciones = texto.count(parche.antes)
    if apariciones != 1:
        return "sin ancla"

    if escribir:
        parche.archivo.write_text(texto.replace(parche.antes, parche.despues),
                                  encoding="utf-8")
    return "aplicado"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--revisar", action="store_true",
                   help="informa el estado de cada parche sin escribir nada")
    args = p.parse_args(argv)

    if not MODELO.exists() and not HERRAMIENTAS.exists():
        print(f"Los repos de referencia no están en {RAIZ / 'referencia' / 'carlos'}.")
        print("Se recuperan de github.com/mloper23 (ver README). Nada que parchar.")
        return 0

    if args.revisar:
        print("Revisión (no se escribe nada):\n")

    ancho = max(len(p.archivo.name) for p in PARCHES)
    estados = []
    for parche in PARCHES:
        estado = aplicar(parche, escribir=not args.revisar)
        estados.append(estado)
        # en revisión 'aplicado' significa 'se aplicaría': nada se escribió
        etiqueta = "por aplicar" if (args.revisar and estado == "aplicado") else estado
        print(f"  {etiqueta:14s} {parche.archivo.name:{ancho}s}  {parche.motivo}")

    sin_ancla = estados.count("sin ancla")
    if sin_ancla:
        print(f"\n{sin_ancla} parche(s) sin ancla: el archivo de terceros cambió "
              "respecto\na lo que este script espera. Hay que revisarlos a mano "
              "antes de confiar\nen los resultados.")
        return 1

    if estados.count("falta archivo") == len(PARCHES):
        print("\nNinguno de los archivos esperados está en disco.")
        return 1

    verbo = "se aplicaría(n)" if args.revisar else "aplicado(s)"
    print(f"\nListo. {estados.count('aplicado')} {verbo}, "
          f"{estados.count('ya estaba')} ya estaba(n).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
