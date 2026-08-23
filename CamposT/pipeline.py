"""Orquestación: de un campo de entrada a un resultado en disco.

Es el punto de entrada de alto nivel del paquete y el único sitio donde se
eligen método, dispositivo y precisión a la vez:

    propagar()     despacha al propagador pedido, con relleno de ceros y
                   recorte, y devuelve (campo, info) con lo que se usó.
    diagnostico()  números que conviene mirar ANTES de propagar: fracción de
                   banda útil de FFT-ASM, Kf sugerido, NA máxima registrable.
    intensidad()   |U|², y guardar() el PNG.

El orden de uso es ése: diagnostico() dice si el caso cae dentro del rango de
FFT-ASM o hace falta MPASM, y propagar() lo ejecuta.

Las piezas están en campos.py (qué se propaga), propagadores.py (con qué),
referencias.py (contra qué se contrasta) y metricas.py (cómo se mide).
"""

import pathlib

import numpy as np
from PIL import Image

from CamposT.backend import (a_dispositivo, a_numpy, dtype_por_defecto, get_xp,
                             gpu_disponible)
from CamposT.campos import usaf_like
from CamposT.propagadores import blas, fft_asm, kf_auto, mpasm


# ----------------------------------------------------- relleno y recorte
def pad_field(U0, factor=2):
    """Rodea el campo de ceros. Evita el wrap-around de la convolución circular
    en FFT-ASM y BLAS. MPASM no lo necesita, pero para comparar en igualdad de
    condiciones conviene aplicarlo a todos."""
    M, N = U0.shape
    Mp, Np = int(M * factor), int(N * factor)
    out = np.zeros((Mp, Np), dtype=U0.dtype)
    i, j = (Mp - M) // 2, (Np - N) // 2
    out[i:i + M, j:j + N] = U0
    return out


def crop_center(U, M, N):
    i, j = (U.shape[0] - M) // 2, (U.shape[1] - N) // 2
    return U[i:i + M, j:j + N]

# ------------------------------------------------------------------ propagación
def propagar(U0, delta, lamb, z, metodo="mpasm", pad=2, device="auto", dtype=None,
             **kw):
    """Propaga un campo con el método pedido, con relleno de ceros opcional.

    device: 'gpu' | 'cpu' | 'auto'. dtype: complex64 (por defecto en GPU) o
    complex128 (por defecto en CPU).

    Devuelve (campo, info). El campo queda en el dispositivo usado y recortado
    al tamaño original, salvo cuando r o mag cambian la malla de salida (caso
    MPASM con ventana ampliada), donde recortar no tendría sentido.
    """
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)

    M, N = U0.shape
    Up = pad_field(a_numpy(U0), pad) if pad and pad > 1 else U0
    Up = a_dispositivo(Up, xp, dtype)

    if metodo == "mpasm":
        Uz, Kf = mpasm(Up, delta, lamb, z, device=dev, dtype=dtype, **kw)
        info = {"Kf": Kf, "s": kw.get("s", 10), "r": kw.get("r", 1),
                "mag": kw.get("mag", 1.0)}
    elif metodo == "fft":
        Uz, info = fft_asm(Up, delta, lamb, z, device=dev, dtype=dtype), {}
    elif metodo == "blas":
        Uz, info = blas(Up, delta, lamb, z, device=dev, dtype=dtype), {}
    else:
        raise ValueError(metodo)

    info["device"] = dev
    info["dtype"] = np.dtype(dtype).name

    # sólo recortamos si la malla de salida conserva la escala de entrada
    if kw.get("mag", 1.0) == 1.0 and kw.get("r", 1) == 1 and pad and pad > 1:
        Uz = crop_center(Uz, M, N)
    return Uz, info


def intensidad(U, normalizar=True):
    """|U|². Respeta el dispositivo del campo de entrada."""
    I = abs(U) ** 2
    if normalizar:
        m = I.max()
        if m > 0:
            I = I / m
    return I


def guardar(I, path, gamma=1.0):
    """Guarda un mapa de intensidad como PNG de 8 bits. Baja de la GPU si hace
    falta."""
    A = a_numpy(I).astype(np.float64)
    A = np.clip(A / A.max(), 0, 1) ** gamma
    Image.fromarray((A * 255).astype(np.uint8)).save(path)


# --------------------------------------------------------------------- criterios
def diagnostico(N, delta, lamb, z, s=1):
    """Números que conviene mirar antes de propagar. Todo en las unidades en
    que se den delta y lamb (usa mm para todo, o µm para todo)."""
    f_nyq = 1 / (2 * delta)
    f_fft_max = N * delta / (lamb * np.sqrt(4 * z**2 + N**2 * delta**2))  # Ec. (12)
    Kf = kf_auto(N, delta, lamb, z, s=s)
    return {
        "ventana [mm]": N * delta,
        "f_Nyquist [1/mm]": f_nyq,
        "f_max_FFT [1/mm]": f_fft_max,
        "fraccion_banda_util_FFT": f_fft_max / f_nyq,
        "Kf_sugerido": Kf,
        "NA_maxima_registrable": min(1.0, lamb * f_nyq),
    }

if __name__ == "__main__":
    from CamposT.backend import cronometrar, info_gpu, liberar_memoria

    # Los PNG van siempre a resultados/, se lance el modulo desde donde se
    # lance: una ruta relativa los dejaria en el directorio de invocacion.
    SALIDA = pathlib.Path(__file__).resolve().parent.parent / "resultados"
    CAMPOS = SALIDA / "campos"
    CAMPOS.mkdir(parents=True, exist_ok=True)

    def destino(metodo, nombre):
        """Cada propagador escribe en su propia carpeta, para que el metodo se
        lea en la ruta y no haya que descifrarlo del nombre del archivo. Los
        parametros (s, mag) si van en el nombre: son variantes de un mismo
        propagador, no propagadores distintos."""
        carpeta = CAMPOS / metodo
        carpeta.mkdir(parents=True, exist_ok=True)
        return carpeta / nombre

    # --- parámetros de ejemplo (unidades: mm) ---
    N = 512
    delta = 3.45e-3        # paso de píxel del sensor, 3.45 µm
    lamb = 633e-6          # 633 nm

    if gpu_disponible():
        gi = info_gpu()
        print(f"GPU: {gi['nombre']}, {gi['VRAM total [GB]']:.1f} GB, "
              f"CuPy {gi['cupy']}")
    else:
        print("Sin GPU disponible: todo el benchmark corre en CPU.")

    # Para usar tu propia imagen en vez del target sintético:
    #   U0 = load_field("mi_objeto.png", N=512, mode="transmitancia")
    t = usaf_like(N)
    U0 = t.astype(complex)
    guardar(t, CAMPOS / "entrada.png")

    print("\nDiagnóstico (ventana ya con relleno de ceros ×2):")
    print(f"{'z [mm]':>8} {'frac. banda FFT':>16} {'Kf sugerido':>12}")
    for z in (20, 60, 150, 400):
        d = diagnostico(2 * N, delta, lamb, z)
        print(f"{z:8.0f} {d['fraccion_banda_util_FFT']:16.3f} {d['Kf_sugerido']:12.2f}")

    metodos = [("fft", {}), ("blas", {}), ("mpasm", {"s": 1}), ("mpasm", {"s": 2})]
    dispositivos = ["cpu"] + (["gpu"] if gpu_disponible() else [])

    print("\nPropagación comparada (tiempos sincronizados, sin coste de calentamiento):")
    cab = f"{'z [mm]':>8} {'método':>10} {'Kf':>7}"
    for dev in dispositivos:
        cab += f" {dev + ' [s]':>10}"
    if len(dispositivos) == 2:
        cab += f" {'x':>6} {'SAM':>8}"
    print(cab)

    for z in (20, 60, 150, 400):
        for metodo, kw in metodos:
            etiqueta = metodo + (f" s={kw['s']}" if "s" in kw else "")
            sufijo = f"_s{kw['s']}" if "s" in kw else ""
            campos, tiempos = {}, {}
            for dev in dispositivos:
                (Uz, info), dt = cronometrar(propagar, U0, delta, lamb, z,
                                             metodo=metodo, pad=2, device=dev, **kw)
                campos[dev], tiempos[dev] = Uz, dt
            guardar(intensidad(campos[dispositivos[-1]]),
                    destino(metodo, f"z{z:04.0f}{sufijo}.png"),
                    gamma=0.6)

            fila = f"{z:8.0f} {etiqueta:>10} {info.get('Kf', 1.0):7.2f}"
            for dev in dispositivos:
                fila += f" {tiempos[dev]:10.3f}"
            if len(dispositivos) == 2:
                from CamposT.metricas import rms_amplitud
                fila += (f" {tiempos['cpu'] / tiempos['gpu']:6.1f}"
                         f" {rms_amplitud(campos['gpu'], campos['cpu']):8.2e}")
            print(fila)
            liberar_memoria()

    # Ventana de observación ampliada sin tocar el plano de entrada.
    # Ésta es la característica que interesa para DLHM: el patrón se sale de la
    # ventana de entrada y MPASM lo recupera cambiando sólo el paso de salida.
    print("\nAmpliación de la ventana de observación (z = 400 mm, sin padding):")
    for mag in (1.0, 2.0, 3.0):
        Uz, info = propagar(U0, delta, lamb, 400, metodo="mpasm", pad=1,
                            s=1, mag=mag)
        guardar(intensidad(Uz), destino("mpasm", f"ventana_x{mag:.0f}.png"),
                gamma=0.6)
        print(f"  mag={mag:.0f}  salida {Uz.shape}  "
              f"ventana = {Uz.shape[0] * delta * mag:.2f} mm  ({info['device']})")

    print(f"\nCampos escritos en {CAMPOS}, una carpeta por propagador.")

