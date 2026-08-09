"""
Carga de campos de entrada a partir de imágenes y utilidades de propagación.

Se apoya en propagadores.py. Pensado para propagar un objeto real (un target,
una máscara, un holograma) en vez de un haz gaussiano analítico.

La generación del objeto (imagen, target USAF, relleno de ceros) se hace en
NumPy porque es coste despreciable y se hace una vez; el campo sube a la GPU
en propagar(). Los mapas de intensidad vuelven a CPU sólo al guardarlos.
"""

import numpy as np
from PIL import Image

from CamposT.backend import (a_dispositivo, a_numpy, dtype_por_defecto, get_xp,
                             gpu_disponible)
from CamposT.propagadores import mpasm, fft_asm, blas, kf_auto


# --------------------------------------------------------------- carga de imagen
def load_field(path, N=None, mode="amplitud", phase_depth=np.pi, invert=False):
    """Convierte una imagen en un campo complejo de entrada U0.

    mode:
        'amplitud'      -> objeto de amplitud puro. U0 = t, con t en [0,1].
        'fase'          -> objeto de fase puro.    U0 = exp(i·phase_depth·t).
        'transmitancia' -> amplitud binarizada (útil para targets tipo USAF).

    invert=True intercambia zonas opacas y transparentes (según cómo venga
    escaneado el target).
    """
    img = Image.open(path).convert("L")
    if N is not None:
        img = img.resize((N, N), Image.LANCZOS)
    t = np.asarray(img, dtype=float) / 255.0
    if invert:
        t = 1.0 - t

    if mode == "amplitud":
        return t.astype(complex)
    if mode == "fase":
        return np.exp(1j * phase_depth * t)
    if mode == "transmitancia":
        return (t > 0.5).astype(float).astype(complex)
    raise ValueError(f"mode desconocido: {mode}")


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


# ------------------------------------------------------------- objeto de prueba
def usaf_like(N=512, n_elementos=12, cols=3, p0=None):
    """Target sintético de barras tipo USAF 1951, sin solapamientos.

    Cada elemento son tres barras horizontales y tres verticales, con ancho de
    barra igual al periodo p y longitud 5p. El periodo decrece en factor 2^(1/6)
    por elemento, como en el target real. Devuelve transmitancia en [0,1].

    El periodo en píxeles del elemento i se recupera con periodos_usaf().
    """
    t = np.zeros((N, N))
    rows = int(np.ceil(n_elementos / cols))
    cell_w, cell_h = N // cols, N // rows
    if p0 is None:                      # el elemento mayor debe caber en la celda
        p0 = int(min((cell_w * 0.85) / 11, (cell_h * 0.85) / 5))

    for i in range(n_elementos):
        p = max(2, int(round(p0 * 2 ** (-i / 6))))
        L = 5 * p                                   # longitud de barra
        r, c = divmod(i, cols)
        # esquina superior izquierda del bloque, centrado en su celda
        blk_w, blk_h = 11 * p, 5 * p
        y0 = r * cell_h + (cell_h - blk_h) // 2
        x0 = c * cell_w + (cell_w - blk_w) // 2
        # tres barras horizontales (periodo vertical p, es decir barra p y hueco p)
        for k in range(3):
            y = y0 + 2 * k * p
            t[y:y + p, x0:x0 + L] = 1.0
        # tres barras verticales, desplazadas a la derecha del grupo horizontal
        xv = x0 + L + p
        for k in range(3):
            x = xv + 2 * k * p
            t[y0:y0 + L, x:x + p] = 1.0
    return t


def periodos_usaf(N=512, n_elementos=12, cols=3, p0=None):
    """Periodo en píxeles de cada elemento generado por usaf_like, para poder
    convertir a pares de líneas por mm: lp_mm = 1/(p·delta)."""
    rows = int(np.ceil(n_elementos / cols))
    if p0 is None:
        p0 = int(min((N // cols * 0.85) / 11, (N // rows * 0.85) / 5))
    return [max(2, int(round(p0 * 2 ** (-i / 6)))) for i in range(n_elementos)]


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

    # --- parámetros de ejemplo (unidades: mm) ---
    N = 512
    delta = 3.45e-3        # paso de píxel del sensor, 3.45 µm
    lamb = 405e-6          # 405 nm

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
    guardar(t, "entrada.png")

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
            campos, tiempos = {}, {}
            for dev in dispositivos:
                (Uz, info), dt = cronometrar(propagar, U0, delta, lamb, z,
                                             metodo=metodo, pad=2, device=dev, **kw)
                campos[dev], tiempos[dev] = Uz, dt
            guardar(intensidad(campos[dispositivos[-1]]),
                    f"z{z:04.0f}_{etiqueta.replace(' ', '').replace('=', '')}.png",
                    gamma=0.6)

            fila = f"{z:8.0f} {etiqueta:>10} {info.get('Kf', 1.0):7.2f}"
            for dev in dispositivos:
                fila += f" {tiempos[dev]:10.3f}"
            if len(dispositivos) == 2:
                from CamposT.propagadores import sam
                fila += (f" {tiempos['cpu'] / tiempos['gpu']:6.1f}"
                         f" {sam(campos['gpu'], campos['cpu']):8.2e}")
            print(fila)
            liberar_memoria()

    # Ventana de observación ampliada sin tocar el plano de entrada.
    # Ésta es la característica que interesa para DLHM: el patrón se sale de la
    # ventana de entrada y MPASM lo recupera cambiando sólo el paso de salida.
    print("\nAmpliación de la ventana de observación (z = 400 mm, sin padding):")
    for mag in (1.0, 2.0, 3.0):
        Uz, info = propagar(U0, delta, lamb, 400, metodo="mpasm", pad=1,
                            s=1, mag=mag)
        guardar(intensidad(Uz), f"ventana_x{mag:.0f}.png", gamma=0.6)
        print(f"  mag={mag:.0f}  salida {Uz.shape}  "
              f"ventana = {Uz.shape[0] * delta * mag:.2f} mm  ({info['device']})")

    print("\nPNG escritos en el directorio actual.")
