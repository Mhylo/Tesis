"""Construcción del campo de entrada U0.

Sólo fabrica campos: a partir de una imagen (load_field) o del target
sintético de barras (usaf_like). Quién los propaga y los guarda está en
pipeline.py.

La geometría del target sale toda de una fuente: anchos_barra_usaf() da el
ancho de barra b, periodos_usaf() da el periodo 2b, y lineas_por_mm_usaf() la
resolución en pares de línea por mm. usaf_like dibuja con esos mismos anchos,
así que el dibujo y la medida no pueden divergir.

Se hace en NumPy porque es coste despreciable y ocurre una vez; el campo sube
a la GPU al propagarlo.
"""

import numpy as np
from PIL import Image


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

# ------------------------------------------------------------- objeto de prueba
def usaf_like(N=512, n_elementos=12, cols=3, p0=None):
    """Target sintético de barras tipo USAF 1951, sin solapamientos.

    Cada elemento son tres barras horizontales y tres verticales. Barras y
    huecos miden lo mismo, b píxeles, así que el PERIODO espacial es 2b: eso
    es un par de líneas del USAF 1951. La longitud de barra es 5b. El periodo
    decrece en factor 2^(1/6) por elemento, como en el target real. Devuelve
    transmitancia en [0,1].

    Geometría del elemento i: anchos_barra_usaf() da b, periodos_usaf() da 2b
    y lineas_por_mm_usaf() la resolución. Salen todas de la misma fuente que
    este dibujo, para que no puedan divergir.
    """
    t = np.zeros((N, N))
    rows = int(np.ceil(n_elementos / cols))
    cell_w, cell_h = N // cols, N // rows
    anchos = anchos_barra_usaf(N, n_elementos, cols, p0)

    for i in range(n_elementos):
        p = anchos[i]
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


def _p0_usaf(N, n_elementos, cols):
    """Ancho de barra del elemento mayor, en píxeles: el que hace que su bloque
    (11b de ancho, 5b de alto) quepa en su celda. Fuente única de la geometría;
    usaf_like y las funciones de medida salen todas de aquí."""
    rows = int(np.ceil(n_elementos / cols))
    cell_w, cell_h = N // cols, N // rows
    return int(min((cell_w * 0.85) / 11, (cell_h * 0.85) / 5))


def anchos_barra_usaf(N=512, n_elementos=12, cols=3, p0=None):
    """Ancho de barra b, en píxeles, de cada elemento. El hueco mide lo mismo,
    así que el periodo es 2b: ver periodos_usaf()."""
    if p0 is None:
        p0 = _p0_usaf(N, n_elementos, cols)
    return [max(2, int(round(p0 * 2 ** (-i / 6)))) for i in range(n_elementos)]


def periodos_usaf(N=512, n_elementos=12, cols=3, p0=None):
    """Periodo espacial en píxeles: barra + hueco = 2·b.

    Es lo que corresponde a UN PAR DE LÍNEAS del USAF 1951, que es la unidad en
    la que se reporta la resolución. La versión anterior de esta función
    devolvía el ancho de barra llamándolo periodo, lo que metía un factor 2 en
    toda cifra de resolución.
    """
    return [2 * b for b in anchos_barra_usaf(N, n_elementos, cols, p0)]


def lineas_por_mm_usaf(delta, N=512, n_elementos=12, cols=3, p0=None):
    """Pares de línea por milímetro de cada elemento, dado el paso de píxel.

        lp/mm = 1 / (periodo · delta) = 1 / (2·b·delta)

    Es el par completo, no la barra. Confundirlos duplica la resolución
    reportada, y ésta es la cifra que va al informe (tarea 40, semana 12).
    """
    return [1.0 / (p * delta) for p in periodos_usaf(N, n_elementos, cols, p0)]
