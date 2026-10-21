"""Magnificacion con el USAF 1951: ajuste de tres barras al perfil de un elemento.

Un elemento son tres barras de ancho w separadas por huecos de w, periodo 2w.
En la muestra el periodo es 1/(lp/mm), con lp/mm = 2^(g + (e - 1)/6). En la
reconstruccion (malla del sensor) mide 2w pixeles, asi que
M = 2 w delta / (1000/lp_mm) con delta en um.

Por que un ajuste y no los minimos del perfil: dentro de las barras la
reconstruccion trae las franjas finas de la imagen gemela, y buscar los tres
minimos (o el pico de la FFT) cae en ellas o en armonicos. El modelo de tres
barras con bordes erf ajusta la forma entera.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.special import erf


def lp_mm(g, e):
    return 2 ** (g + (e - 1) / 6)


def barras(x, x0, w, s):
    """1 dentro de las tres barras (desde x0, ancho w, huecos w), bordes erf de sigma s."""
    s = max(abs(s), 0.3)
    y = np.zeros_like(np.asarray(x, float))
    for j in range(3):
        a, b = x0 + 2 * j * w, x0 + 2 * j * w + w
        y += 0.5 * (erf((x - a) / (np.sqrt(2) * s)) - erf((x - b) / (np.sqrt(2) * s)))
    return y


def perfil(A, caja, orient):
    """Perfil medio a lo largo de las barras, en el 60 % central de su largo.

    caja = (x0, x1, y0, y1) en pixeles; orient "v" (barras verticales: perfil
    a lo largo de x) o "h"."""
    x0, x1, y0, y1 = caja
    B = np.asarray(A, float)[y0:y1, x0:x1]
    if orient == "v":
        m = B.shape[0]
        return B[int(0.2 * m):int(0.8 * m)].mean(axis=0)
    if orient == "h":
        m = B.shape[1]
        return B[:, int(0.2 * m):int(0.8 * m)].mean(axis=1)
    raise ValueError(f"orient = {orient!r}: 'v' o 'h'")


def ajustar_barras(p):
    """-> (w, sigma) en pixeles. Varios arranques: el patron (5w) ocupa entre
    el 55 y el 95 % de la caja."""
    p = gaussian_filter1d(np.asarray(p, float), 2.0)
    n = p.size
    x = np.arange(n, dtype=float)
    mejor = None
    for frac in np.linspace(0.55, 0.95, 9):
        w0 = frac * n / 5
        for c in np.linspace(-0.15, 0.15, 7) * n:
            q0 = [(n - 5 * w0) / 2 + c, w0, np.percentile(p, 90), np.ptp(p) * 0.6, 3.0]
            r = least_squares(lambda q: q[2] - q[3] * barras(x, q[0], q[1], q[4]) - p, q0,
                              bounds=([-0.2 * n, 0.08 * n, -np.inf, 0, 0.3],
                                      [0.6 * n, 0.25 * n, np.inf, np.inf, 0.2 * n]))
            if mejor is None or r.cost < mejor.cost:
                mejor = r
    return float(mejor.x[1]), float(abs(mejor.x[4]))


def magnificacion(w_px, g, e, delta_um):
    return 2 * w_px * delta_um / (1000 / lp_mm(g, e))
