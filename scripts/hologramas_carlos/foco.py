"""Retropropagacion paraxial en la malla del sensor y busqueda del foco.

Por el escalado de Fresnel, un holograma DLHM leido en la malla del sensor
(paso delta) es un holograma de Gabor de onda plana a la distancia
D = L(L - z)/z: la misma propagacion que z_eq = z(L - z)/L en la malla delta/M
de la muestra, expresada en pixeles del sensor. D NO es una distancia fisica.
Barrer D no necesita L: da el foco, y despues cada L da su z.

Paraxial es el espectro angular de banda limitada (BLAS) con relleno: el
espectro del holograma se calcula una vez y cada D cuesta un producto y una
IFFT. H se separa en e^{-ikD} (fase global, se tira) por e^{+ikDw}, con
w = 1 - sqrt(1 - (lamb fx)^2 - (lamb fy)^2) calculada una vez en doble: asi el
producto por D va en simple sin perder fase (k D w ~ 4e4 rad; en float32,
~2e-3 rad de error).
"""

import numpy as np
from scipy.ndimage import median_filter

from CamposT.backend import get_xp


def contraste(holo, ref, piso=0.05):
    """I/ref - 1, y 0 donde ref < piso*max(ref): el JPEG tiene esquinas oscuras."""
    holo = np.asarray(holo, np.float64)
    ref = np.asarray(ref, np.float64)
    c = np.zeros_like(holo)
    luz = ref > piso * ref.max()
    c[luz] = holo[luz] / ref[luz] - 1
    return c


class Paraxial:
    """1 + U retropropagado -D en la malla del sensor, para muchas D."""

    def __init__(self, c, delta, lamb, pad=2, xp=None):
        self.xp = xp if xp is not None else get_xp("auto")[0]
        xp = self.xp
        M, N = c.shape
        self.M, self.N = M, N
        self.P, self.Q = pad * M, pad * N
        self.r0, self.c0 = (self.P - M) // 2, (self.Q - N) // 2
        self.delta, self.lamb = float(delta), float(lamb)
        U0 = xp.zeros((self.P, self.Q), np.complex64)
        U0[self.r0:self.r0 + M, self.c0:self.c0 + N] = xp.asarray(c, np.complex64)
        self.F0 = xp.fft.fft2(U0)
        del U0
        fx = xp.asarray(np.fft.fftfreq(self.Q, d=delta))[None, :]
        fy = xp.asarray(np.fft.fftfreq(self.P, d=delta))[:, None]
        arg = 1.0 - (lamb * fx) ** 2 - (lamb * fy) ** 2
        self.prop = arg > 0
        self.w = (1.0 - xp.sqrt(xp.where(self.prop, arg, 0.0))).astype(np.float32)
        self.afx = xp.abs(fx).astype(np.float32)
        self.afy = xp.abs(fy).astype(np.float32)
        self.k = 2 * np.pi / lamb

    def campo(self, D, ventana=None):
        """ventana = (fila, columna, alto, ancho) en pixeles del holograma."""
        xp, lamb, delta = self.xp, self.lamb, self.delta
        flx = 1 / (lamb * np.sqrt((2 * D / (delta * self.Q)) ** 2 + 1))
        fly = 1 / (lamb * np.sqrt((2 * D / (delta * self.P)) ** 2 + 1))
        H = xp.exp(1j * np.float32(self.k * D) * self.w).astype(np.complex64)
        H *= (self.prop & (self.afx < flx) & (self.afy < fly))
        U = xp.fft.ifft2(self.F0 * H)
        f, c, a, b = ventana or (0, 0, self.M, self.N)
        return 1 + U[self.r0 + f:self.r0 + f + a, self.c0 + c:self.c0 + c + b]


def tamura(A, xp=np):
    """sqrt(std/media): la de barrido_foco y mi_prueba_dlhm."""
    m = A.mean()
    return float(xp.sqrt(A.std() / m)) if m > 0 else 0.0


def barrer(P, Ds, ventana, signo):
    """Tamura de |t| en cada D, orientada para maximizar: con signo "min"
    (objeto de fase) va con el signo cambiado."""
    if signo not in ("max", "min"):
        raise ValueError(f"signo = {signo!r}: 'max' (amplitud) o 'min' (fase)")
    s = 1.0 if signo == "max" else -1.0
    xp = P.xp
    return np.array([s * tamura(xp.abs(P.campo(D, ventana)).astype(np.float64), xp)
                     for D in Ds])


def pico(Ds, curva, tendencia_mm=20.0):
    """D del maximo de la curva sin su tendencia lenta (mediana movil de
    tendencia_mm), con interpolacion parabolica."""
    Ds = np.asarray(Ds, float)
    curva = np.asarray(curva, float)
    paso = Ds[1] - Ds[0]
    ancho = max(int(tendencia_mm / paso) | 1, 3)
    r = curva - median_filter(curva, size=ancho, mode="nearest")
    i = int(np.clip(r.argmax(), 1, r.size - 2))
    a, b, c = r[i - 1], r[i], r[i + 1]
    den = a - 2 * b + c
    return float(Ds[i] + (0.5 * (a - c) / den if den < 0 else 0.0) * paso)


def pico_fino(Ds, curva):
    """D del maximo en una ventana fina: se quita la recta que une los
    extremos (la tendencia lenta, que en +-7 mm es casi lineal) y se
    interpola con una parabola."""
    Ds = np.asarray(Ds, float)
    curva = np.asarray(curva, float)
    r = curva - np.interp(Ds, [Ds[0], Ds[-1]], [curva[0], curva[-1]])
    i = int(np.clip(r.argmax(), 1, r.size - 2))
    a, b, c = r[i - 1], r[i], r[i + 1]
    den = a - 2 * b + c
    return float(Ds[i] + (0.5 * (a - c) / den if den < 0 else 0.0) * (Ds[1] - Ds[0]))


def centros_robustos(ks, D, tolerancia=8.0):
    """Centros para el barrido fino: D gruesos, salvo los que se apartan mas
    de tolerancia [mm] de la curva D = a/(k + b) - c ajustada con perdida
    robusta (con la platina a paso fijo, D = L^2/(z1 + (k-1)dz) - L tiene esa
    forma). Esos se reemplazan por la curva."""
    from scipy.optimize import least_squares
    ks = np.asarray(ks, float)
    D = np.asarray(D, float)
    modelo = lambda q: q[0] / (ks + q[1]) - q[2]
    r = least_squares(lambda q: modelo(q) - D, [D[0] * (ks[0] + 10), 10.0, 0.0],
                      loss="soft_l1", f_scale=2.0)
    # segunda pasada sin los que saltan: aun con perdida robusta, un salto en
    # el extremo tuerce la curva justo donde hay que extrapolar
    bien = np.abs(D - modelo(r.x)) <= tolerancia
    r = least_squares(lambda q: (modelo(q) - D)[bien], r.x)
    m = modelo(r.x)
    return np.where(np.abs(D - m) > tolerancia, m, D)
