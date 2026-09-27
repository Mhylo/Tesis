"""DLHM con MPASM: iluminación esférica exacta, sin aproximación paraxial.

La DLHM ilumina con una fuente puntual: la muestra está a z de la fuente y el
sensor a L, así que entre muestra y sensor hay d = L - z. Este módulo propaga
en los dos sentidos con el espectro angular exacto (el mismo H que
propagadores.py) y la esférica exacta e^{ikr}/r, sin aproximar nada por
paraxial:

    holograma()      ida:    objeto t en la muestra  ->  holograma en el sensor
    reconstruir()    vuelta: holograma en el sensor  ->  t en la muestra
    Reconstructor    la vuelta para un barrido de z: el espectro del holograma
                     se calcula una sola vez y cada z cuesta solo la salida

y, para comparar, reconstruir_escalado(), la formulación (i) de la tarea 24:
el holograma leído como uno de onda plana en la malla delta/M a la distancia
z_eq = z(L-z)/L. Es paraxial, y aquí está para medir lo que eso cuesta.

POR QUÉ NO BASTA EL ESCALADO. El teorema de escalado de Fresnel es exacto solo
en paraxial. Fuera de él, la fase que se desprecia tiene dos partes: una
aberración esférica que no depende de la posición y un desenfoque astigmático
que sí: un detalle visto a un ángulo theta de la fuente enfoca a z_eq/cos³theta
en la dirección radial y a z_eq/cos theta en la tangencial. Con detalle fino eso
no es pequeño. Medido con un objeto de ~1.25 um, montaje del laboratorio y
N = 3000 (scripts/verif_dlhm_esferico.py): el escalado reconstruye el campo con
correlación 0.89 en el eje y 0.63 en el borde; este módulo, 1.000 y 0.93. El
documento de la tarea 24 lo daba por bueno hasta 5.7 grados porque lo midió
con objetos lisos (de 6 a 30 um), donde la fase despreciada no se nota.

CÓMO SE METE LA ESFÉRICA EN EL MPASM. Multiplicar el holograma por e^{ikR}/R y
pasarlo al MPASM tal cual falla por muestreo: en el sensor la esférica tiene
frecuencia local X/(lamb R), que con el sensor entero (501 mm^-1) pasa de
largo el Nyquist del píxel (273 mm^-1). La salida está en que el MPASM no
necesita la entrada bien muestreada, sino la transformada de Fourier de la
entrada, y esa se puede calcular sin muestrear el chirp:

    J(x) = sum_n J_n sinc((x - x_n)/delta)       (J = holograma, sí muestreado)
    F(f) = int J(x) e^{i pi x²/(lamb R)} e^{-i 2 pi f x} dx
         = sum_n J_n A[n, f]

A[n, f] tiene forma cerrada con integrales de Fresnel (ventana_chirp): es la
fila de la DFT de siempre, e^{-i 2 pi f x_n}·delta, multiplicada por el chirp
en x_n y por una ventana que deja pasar solo las frecuencias a menos de
1/(2 delta) de la frecuencia local del chirp en esa muestra. Sin ventana, cada
componente reaparece desplazada 1/delta y se propaga como un rayo desviado
~16 grados, que cae dentro o fuera de la ventana del objeto según el periodo
de la rejilla: acierta por geometría, no por construcción. Medido con el
sensor entero (N = 3000) y la rejilla de este módulo: 0.52 contra 0.97 de
correlación con el objeto en toda la ventana; en N = 2048, 0.42 con la
rejilla mínima y 0.946 con una 45 % más holgada. Solo la parte parabólica de
la esférica va en las matrices, porque es la separable; el resto,
e^{ik(r - x²/2R)}/r, varía despacio (a lo sumo ~80 mm^-1 en la esquina de un
sensor de 3000x4000 px) y se multiplica punto a punto.

LA REJILLA DE FRECUENCIAS SALE DE LA GEOMETRÍA, NO DE LA Ec. (14) DE ZHAO. El
Kf de Zhao comprime la banda para que el chirp de H quepa en el periodo de la
salida, y supone onda plana. Aquí la banda la fija la esférica: va de
X_min/(lamb L) - 1/(2 delta) a X_max/(lamb L) + 1/(2 delta), MÁS ancha que
1/delta, o sea Kf < 1: una expansión, no una compresión. Y el periodo P = 1/df
tiene que bastar para que ninguna copia periódica del campo caiga en la
ventana de salida: el campo ocupa la ventana más lo que se abre alrededor de
cada rayo principal, d·tan(asin(lamb/(2 delta))) ~ 1 mm por lado. Con un
sensor de 3000 x 4000 píxeles son 5200 x 7712 frecuencias, un espectro de
306 MB en complex64; construirlo cuesta 9.8 s y cada z después 0.56 s en la
GPU de desarrollo (RTX 3050 de 4 GB). El escalado, en N = 3000, 0.33 s por z:
la exacta cuesta prácticamente lo mismo por z.

POR QUÉ SE RECONSTRUYE c - 1 Y NO c. El holograma de contraste c = I/I_ref
vale 1 + V + V* + |V|², y el 1 es la referencia truncada por el borde del
sensor. Retropropagada, esa truncadura deja difracción del borde en toda la
periferia de la ventana del objeto: medido, arrastraba la correlación con el
objeto de 0.90 a 0.27 en la ventana entera. La cadena de Carlos lo evita
dividiendo por la referencia propagada numéricamente, que lleva la misma
difracción; aquí se evita no propagándola: se retropropaga c - 1 y se suma el
1 en la muestra, donde es exacto.

VALIDADO (scripts/verif_dlhm_esferico.py):
    - la ida contra Rayleigh-Sommerfeld directo (Lopera, Ec. 1), objeto de
      ~1.25 um: error relativo 3.7e-6 en el eje, 3.6e-6 a 5.7 y a 12.4
      grados (el escalado, en los mismos puntos: 0.47, 0.65 y 0.89);
    - la vuelta contra reconstruir() de scripts/mi_prueba_dlhm.py (la cadena
      de Carlos, validada contra reconstruction_dlhm.py) con remuestreo de
      banda limitada y malla par: correlación 0.993 en la ventana entera,
      0.999 en el centro;
    - el uso de verdad: holograma de INTENSIDAD del BenchmarkTarget a z = 3
      mm (N = 2048) y z desconocida. Un barrido por nitidez con el
      Reconstructor encuentra z = 3.0005 mm, 61 planos en 9 s; con el
      escalado el máximo se va al extremo del barrido, porque desenfoca
      tanto que no tiene pico en el foco.

UNIDADES: milímetros para todo, como en montaje.py. Convenio e^{+ikr}: la ida
propaga +d y la vuelta -d. Los ejes van como en el resto del paquete: arrays
(filas = y, columnas = x) con coordenadas (arange(n) - n/2)·paso + centro,
donde centro es la posición del píxel central respecto del eje óptico (el pie
de la perpendicular desde la fuente). Con un recorte del sensor, centro es lo
que dice dónde estaba el recorte: la esférica depende de la posición absoluta.

REQUIERE scipy (scipy.special.fresnel) para las ventanas. Se calculan en la
CPU en doble precisión y solo después suben al dispositivo, una vez por
holograma: dependen del sensor, de L y de lamb, no de z.
"""

import numpy as np

from CamposT.backend import (a_dispositivo, a_numpy, bloques, comprobar_memoria,
                             dtype_por_defecto, get_xp, phasor)
from CamposT.propagadores import mpasm, transfer_function

__all__ = ["magnificacion", "distancia_equivalente", "coordenadas", "esferica",
           "ventana_chirp", "holograma", "Reconstructor", "reconstruir",
           "reconstruir_escalado", "memoria", "memoria_holograma"]


# ------------------------------------------------------------------ geometría
def magnificacion(z, L):
    """M = L/z. Proyección central entre planos paralelos: exacta, no paraxial."""
    return L / z


def distancia_equivalente(z, L):
    """z_eq = z(L-z)/L, la distancia del escalado de Fresnel (tarea 24).

    Solo tiene sentido en la aproximación paraxial: es la que usa
    reconstruir_escalado(). reconstruir() no la necesita.
    """
    return z * (L - z) / L


def _por_eje(v):
    """Escalar o pareja (y, x) -> pareja de floats (y, x)."""
    if np.ndim(v) == 0:
        return float(v), float(v)
    vy, vx = v
    return float(vy), float(vx)


def coordenadas(n, paso, centro=0.0):
    """Las del resto del paquete, (arange(n) - n/2)·paso, desplazadas a centro."""
    return (np.arange(n) - n / 2) * paso + centro


def esferica(y, x, R, lamb, xp=np, dtype=np.complex128, parte="entera"):
    """La onda de la fuente puntual en un plano a distancia R, sobre la malla (y, x).

    parte = 'entera'   e^{ikr}/r, con r = sqrt(x² + y² + R²)
            'residuo'  e^{ik(r - (x²+y²)/(2R))}/r: la esférica sin su parte
                       parabólica, que es la que va dentro de las matrices
            'inversa'  r·e^{-ikr} = 1/(e^{ikr}/r), para dividir sin dividir

    La fase k·r vale ~1e5 rad: se evalúa en float64 y por bloques de filas, y
    solo el fasor baja al dtype de trabajo.
    """
    k = 2 * np.pi / lamb
    y = xp.asarray(y, dtype=np.float64)
    x = xp.asarray(x, dtype=np.float64)
    out = xp.empty((y.size, x.size), dtype=dtype)
    for i0, i1 in bloques(y.size, x.size):
        rho2 = x[None, :] ** 2 + y[i0:i1, None] ** 2
        r = xp.sqrt(rho2 + R * R)
        if parte == "entera":
            v = xp.exp(1j * k * r) / r
        elif parte == "residuo":
            v = xp.exp(1j * k * (r - rho2 / (2 * R))) / r
        elif parte == "inversa":
            v = xp.exp(-1j * k * r) * r
        else:
            raise ValueError(f"parte desconocida: {parte!r}")
        out[i0:i1] = v.astype(dtype, copy=False)
    return out


# ------------------------------------------------------ la DFT con el chirp dentro
def ventana_chirp(pos, f, paso, lamb, R, xp=np, dtype=np.complex128):
    """Matriz A (len(pos) x len(f)) de la DFT de una entrada muestreada por un
    chirp parabólico que NO se muestrea:

        A[n, m] = int sinc((x - x_n)/paso) e^{i pi x²/(lamb R)} e^{-i 2 pi f_m x} dx

    de modo que sum_n J_n A[n, m] es la transformada exacta de J·chirp si J
    está limitada en banda a |f| < 1/(2 paso). Forma cerrada: por el teorema
    de convolución, con B = 1/(2 paso) y g0 = f - x_n/(lamb R),

        A[n, m] = e^{i pi x_n²/(lamb R)} e^{-i 2 pi f_m x_n} · W(g0)
        W(g0)   = paso · sqrt(i lamb R) · int_{-B-g0}^{B-g0} e^{-i pi lamb R s²} ds

    y la integral es una diferencia de integrales de Fresnel. Dentro de la
    banda W ~ paso y A es la fila de DFT de siempre (con el chirp en x_n);
    fuera, W cae a cero con las ondulaciones de Fresnel. Con R -> infinito es
    la DFT de Riemann recortada a |f| < B.

    Se calcula en la CPU en float64 (scipy no corre en la GPU) por bloques de
    filas, y cada bloque sube ya en el dtype de trabajo.
    """
    try:
        from scipy.special import fresnel
    except ImportError as e:                                   # pragma: no cover
        raise ImportError("ventana_chirp necesita scipy (scipy.special.fresnel): "
                          "pip install scipy") from e
    if not R > 0:
        raise ValueError(f"R = {R!r}: el chirp de la fuente puntual va con R > 0")
    pos = np.asarray(pos, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    B = 1 / (2 * paso)
    alfa = lamb * R
    a = np.sqrt(2 * alfa)                     # s -> t = a·s deja e^{-i pi t²/2}
    pref = paso * np.sqrt(1j * alfa) / a

    def G(s):
        S, C = fresnel(a * s)
        return C - 1j * S

    out = xp.empty((pos.size, f.size), dtype=dtype)
    for i0, i1 in bloques(pos.size, f.size, itemsize=48):
        xn = pos[i0:i1, None]
        g0 = f[None, :] - xn / alfa
        W = pref * (G(B - g0) - G(-B - g0))
        bloque = np.exp(1j * (np.pi * xn**2 / alfa - 2 * np.pi * xn * f[None, :])) * W
        out[i0:i1] = a_dispositivo(bloque, xp, dtype)
    return out


# ------------------------------------------------------------- piezas comunes
def _producto_entrada(J, Ay, Ax):
    """Ay^T @ J @ Ax en el orden más barato."""
    ny, nx = J.shape
    nfy, nfx = Ay.shape[1], Ax.shape[1]
    if nfy * ny * nx + nfy * nx * nfx <= ny * nx * nfx + nfy * ny * nfx:
        return (Ay.T @ J) @ Ax
    return Ay.T @ (J @ Ax)


def _salida(F, fy, fx, lamb, D, y, x, xp, dtype):
    """sum_f F(f) H(f; D) e^{i 2 pi f·x} df sobre la malla (y, x), sin tocar F.

    H se aplica por bloques de filas del espectro dentro del primer producto,
    así que ni H ni F·H se materializan enteros: con un Reconstructor el
    espectro se reutiliza en cada z y no se puede multiplicar in situ.
    """
    Ex = phasor(fx, x, +1, xp, dtype)                  # (nfx, nx)
    Ey = phasor(y, fy, +1, xp, dtype)                  # (ny, nfy)
    nfy, nfx = F.shape
    ny, nx = Ey.shape[0], Ex.shape[1]
    if nfy * nfx * nx + ny * nfy * nx <= ny * nfy * nfx + ny * nfx * nx:
        G = xp.empty((nfy, nx), dtype=dtype)
        for i0, i1 in bloques(nfy, nfx):
            G[i0:i1] = (F[i0:i1] * transfer_function(fx, fy[i0:i1], lamb, D, xp, dtype)) @ Ex
        U = Ey @ G
    else:
        T = xp.zeros((ny, nfx), dtype=dtype)
        for i0, i1 in bloques(nfy, nfx):
            T += Ey[:, i0:i1] @ (F[i0:i1] * transfer_function(fx, fy[i0:i1], lamb, D, xp, dtype))
        U = T @ Ex
    return U * ((fx[1] - fx[0]) * (fy[1] - fy[0]))


def _rejilla(f_lo, f_hi, P):
    """Frecuencias de paso 1/P que cubren [f_lo, f_hi], centradas en el tramo."""
    n = int(np.ceil((f_hi - f_lo) * P)) + 1
    return (f_lo + f_hi) / 2 + (np.arange(n) - (n - 1) / 2) / P


def _aterrizaje(pos, fx, fy, lamb, D):
    """Dónde cae, a distancia D, el rayo de frecuencia (fx, fy) que sale de pos
    en el eje x: pos + D·lamb·fx / sqrt(1 - lamb²(fx² + fy²))."""
    w = np.sqrt(np.maximum(1 - lamb**2 * (fx**2 + fy**2), 1e-12))
    return pos + D * lamb * fx / w


def _soporte(bordes_x, bordes_y, bx, by, centro_f, lamb, D):
    """Intervalo en x donde cae el campo propagado D, con la banda local de cada
    punto (centro_f(p) ± b). Las esquinas de la caja (x, y, fx, fy) bastan: el
    aterrizaje es monótono en las cuatro. 2-D a propósito: con fy grande el
    rayo se tumba más en x que lo que diría la fórmula de un eje (un 22 % en
    la esquina de un sensor de 3000x4000 px)."""
    vals = []
    for p in bordes_x:
        for q in bordes_y:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    fx = np.clip(centro_f(p) + sx * bx, -0.999 / lamb, 0.999 / lamb)
                    fy = np.clip(centro_f(q) + sy * by, -0.999 / lamb, 0.999 / lamb)
                    vals.append(_aterrizaje(p, fx, fy, lamb, D))
    return min(vals), max(vals)


def _comprobar_periodo(P, soporte, ventana, eje):
    """Ninguna copia periódica del campo (periodo P) puede caer en la ventana.

    El análogo de propagadores._comprobar_ventana: si no se cumple, la salida
    sale con copias superpuestas sin avisar. Aquí se aborta.
    """
    u0, u1 = soporte
    w0, w1 = ventana
    necesario = max(u1 - w0, w1 - u0)
    if necesario > P * (1 + 1e-9):
        raise ValueError(
            f"eje {eje}: la ventana de salida [{w0:.4g}, {w1:.4g}] mm no cabe en el "
            f"periodo de la rejilla de frecuencias, P = {P:.4g} mm: el campo ocupa "
            f"[{u0:.4g}, {u1:.4g}] mm y haría falta P >= {necesario:.4g} mm. Saldría "
            f"con copias periódicas superpuestas. Construye el Reconstructor con el "
            f"rango de z y la ventana que vas a pedir, o sube margen.")


def memoria(forma_entrada, n_f, forma_salida, dtype=np.complex64):
    """Bytes de pico estimados de una ida o una vuelta. Para comprobar en GPU."""
    (ny, nx), (nfy, nfx), (my, mx) = forma_entrada, n_f, forma_salida
    elementos = (ny * nx + ny * nfy + nx * nfx      # entrada y matrices de entrada
                 + max(nfy * nx, ny * nfx)          # intermedio
                 + nfy * nfx                        # espectro
                 + my * nfy + nfx * mx              # matrices de salida
                 + max(nfy * mx, my * nfx)          # intermedio de salida
                 + my * mx)
    return elementos * np.dtype(dtype).itemsize


# ----------------------------------------------------------------- la vuelta
def _contraste(holo, fondo):
    """c - 1, con c = holo/fondo.

    fondo = None     holo YA es de contraste: su media tiene que andar por 1,
                     y si no, aborta. Un holograma crudo (cuentas de la
                     cámara) reconstruido así lleva la referencia entera
                     dentro, multiplicada por su nivel, y sale un resultado
                     que parece una reconstrucción y no lo es.
    fondo = 'media'  divide por la media del propio holograma: para cuando no
                     hay imagen de referencia. Deja la envolvente de la
                     iluminación (el cono del pinhole) dentro de c.
    fondo = array o escalar: divide por él (la imagen sin muestra).

    Siempre en coma flotante: una imagen de 8 bits restada en enteros da la
    vuelta en los ceros.
    """
    holo = a_numpy(holo)
    holo = holo.astype(np.result_type(holo.dtype, np.float64), copy=False)
    if fondo is None:
        media = abs(holo.mean())
        if not 0.5 < media < 1.5:
            raise ValueError(
                f"con fondo=None el holograma tiene que ser de contraste (I/I_ref, media "
                f"~1) y su media es {media:.4g}. Pasa fondo=<imagen sin muestra>, o "
                f"fondo='media' si no la tienes.")
        c = holo
    elif isinstance(fondo, str):
        if fondo != "media":
            raise ValueError(f"fondo={fondo!r}: los valores de texto admitidos son 'media'")
        c = holo / holo.mean()
    else:
        c = holo / a_numpy(fondo)
    return c - 1


class Reconstructor:
    """Del holograma DLHM al plano de la muestra, para una o muchas z.

    Construirlo cuesta el espectro del holograma con la esférica dentro, una
    sola vez; cada campo(z) cuesta solo la función de transferencia y los dos
    productos de salida. Es lo que abarata un barrido de enfoque: con L fijo y
    z variable el espectro del sensor no depende de z.

    holo     holograma (ny, nx). Real (intensidad) o complejo (U/S_L, para
             pruebas sin imagen gemela).
    delta    paso del sensor, escalar o (delta_y, delta_x), en mm.
    lamb, L  longitud de onda y fuente -> sensor, en mm.
    z        fuente -> muestra, o (z_min, z_max) si se va a barrer. La rejilla de
             frecuencias se dimensiona para ese rango y campo() fuera de él
             aborta.
    fondo    None si holo ya es de contraste (I/I_ref, media ~1; si no lo es,
             aborta); la imagen sin muestra o un escalar por el que dividir;
             o 'media' para dividir por la media del propio holograma.
    centro   (y, x) del píxel central del holograma respecto del eje óptico,
             en mm. (0, 0) si el sensor está centrado en la fuente.
    margen   factor sobre el periodo mínimo de la rejilla de frecuencias.

    La rejilla de la vuelta no lleva la guarda de la ida (ver holograma()):
    aquí la banda de cada muestra ya la cierran las ventanas de Fresnel, y lo
    que queda fuera son sus colas. Medido en N = 1024: ensancharla con una
    rampa cambia la reconstrucción en 2.4e-4 relativo, por debajo de lo que
    imponen la imagen gemela y la NA, al precio de doblar las frecuencias.
    """

    def __init__(self, holo, delta, lamb, L, z, *, fondo=None, centro=(0.0, 0.0),
                 margen=1.1, device="auto", dtype=None):
        self.xp, dev = get_xp(device)
        self.dtype = dtype or dtype_por_defecto(dev)
        xp, dtype = self.xp, self.dtype
        c1 = _contraste(holo, fondo)
        if c1.ndim != 2:
            raise ValueError(f"el holograma tiene que ser 2-D, llegó {c1.shape}")
        self.forma = c1.shape
        self.delta = _por_eje(delta)
        self.centro = _por_eje(centro)
        self.lamb, self.L = float(lamb), float(L)
        zr = (float(z), float(z)) if np.ndim(z) == 0 else tuple(sorted(map(float, z)))
        if not 0 < zr[0] <= zr[1] < L:
            raise ValueError(f"z = {z!r}: hace falta 0 < z < L = {L}")
        self.z_rango = zr

        (ny, nx), (dy, dx), (cy, cx) = self.forma, self.delta, self.centro
        self.Y, self.X = coordenadas(ny, dy, cy), coordenadas(nx, dx, cx)
        Y, X = self.Y, self.X
        # frecuencia local del residuo en la esquina más lejana: se suma a la
        # banda, porque J·residuo ya no está limitada a 1/(2 delta) exactos
        Rmax = np.sqrt(np.abs(X).max() ** 2 + np.abs(Y).max() ** 2 + L**2)
        self.banda = (1 / (2 * dy) + np.abs(Y).max() * (1 / L - 1 / Rmax) / lamb,
                      1 / (2 * dx) + np.abs(X).max() * (1 / L - 1 / Rmax) / lamb)
        by, bx = self.banda
        ffy = (Y[0] / (lamb * L) - by, Y[-1] / (lamb * L) + by)
        ffx = (X[0] / (lamb * L) - bx, X[-1] / (lamb * L) + bx)

        # el periodo: el peor z del rango, con la ventana por defecto
        Py = Px = 0.0
        for zz in np.linspace(zr[0], zr[1], 9):
            sx, sy, wx, wy = self._soportes(zz)
            Px = max(Px, sx[1] - wx[0], wx[1] - sx[0])
            Py = max(Py, sy[1] - wy[0], wy[1] - sy[0])
        self.P = (margen * Py, margen * Px)
        self.fy = _rejilla(*ffy, self.P[0])
        self.fx = _rejilla(*ffx, self.P[1])

        if dev == "gpu":
            comprobar_memoria(memoria(self.forma, (self.fy.size, self.fx.size), self.forma,
                                      dtype), f"Reconstructor({ny}x{nx})", dtype=dtype)
        Ax = ventana_chirp(X, self.fx, dx, lamb, L, xp, dtype)
        Ay = ventana_chirp(Y, self.fy, dy, lamb, L, xp, dtype)
        J = a_dispositivo(c1, xp, dtype) * esferica(Y, X, L, lamb, xp, dtype, parte="residuo")
        self.F = _producto_entrada(J, Ay, Ax)

    def _soportes(self, z, ventana_y=None, ventana_x=None):
        """(soporte_x, soporte_y, ventana_x, ventana_y) en la muestra a z."""
        lamb, L = self.lamb, self.L
        d = L - z
        Y, X = self.Y, self.X
        by, bx = self.banda
        centro_f = lambda p: p / (lamb * L)
        sx = _soporte((X[0], X[-1]), (Y[0], Y[-1]), bx, by, centro_f, lamb, -d)
        sy = _soporte((Y[0], Y[-1]), (X[0], X[-1]), by, bx, centro_f, lamb, -d)
        wx = ventana_x or (X[0] * z / L, X[-1] * z / L)
        wy = ventana_y or (Y[0] * z / L, Y[-1] * z / L)
        return sx, sy, wx, wy

    @property
    def n_frecuencias(self):
        """(nfy, nfx): el tamaño del espectro que se guarda."""
        return self.fy.size, self.fx.size

    def campo(self, z, paso=None, forma=None, centro=None):
        """t en la muestra a distancia z de la fuente, en el dispositivo.

        Por defecto en la malla que el sensor proyecta desde la fuente: paso
        delta·z/L, la forma del holograma y centrada en centro·z/L. Se puede
        pedir otra (más fina, un trozo, desplazada) mientras quepa en el
        periodo de la rejilla; si no, aborta.
        """
        z = float(z)
        if not self.z_rango[0] * (1 - 1e-9) <= z <= self.z_rango[1] * (1 + 1e-9):
            raise ValueError(f"z = {z} fuera del rango {self.z_rango} con que se "
                             f"construyó el Reconstructor")
        L, lamb = self.L, self.lamb
        py, px = _por_eje(paso) if paso is not None else (self.delta[0] * z / L,
                                                          self.delta[1] * z / L)
        my, mx = forma if forma is not None else self.forma
        cy, cx = _por_eje(centro) if centro is not None else (self.centro[0] * z / L,
                                                             self.centro[1] * z / L)
        y, x = coordenadas(my, py, cy), coordenadas(mx, px, cx)
        sx, sy, wx, wy = self._soportes(z, (y[0], y[-1]), (x[0], x[-1]))
        _comprobar_periodo(self.P[1], sx, wx, "x")
        _comprobar_periodo(self.P[0], sy, wy, "y")
        U = _salida(self.F, self.fy, self.fx, lamb, -(L - z), y, x, self.xp, self.dtype)
        return 1 + U * esferica(y, x, z, lamb, self.xp, self.dtype, parte="inversa")


def reconstruir(holo, delta, lamb, z, L, **kw):
    """t en la muestra a z de la fuente. Para una sola z; para barrer, usa
    Reconstructor, que reutiliza el espectro. Los **kw son los suyos."""
    return Reconstructor(holo, delta, lamb, L, z, **kw).campo(z)


def reconstruir_escalado(holo, delta, lamb, z, L, *, fondo=None, s=1,
                         device="auto", dtype=None):
    """La formulación (i) de la tarea 24: onda plana en la malla delta·z/L a
    -z_eq. Paraxial. Para comparar con reconstruir(), no para producir.

    Reconstruye c - 1 por lo mismo que reconstruir(). El factor e^{ik z_eq}
    es la fase del teorema de escalado: sin él, t sale girado una constante.
    """
    dy, dx = _por_eje(delta)
    if dy != dx:
        raise ValueError("el escalado va con propagadores.mpasm(), que toma un solo "
                         "paso: delta_y y delta_x tienen que coincidir")
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)
    c1 = _contraste(holo, fondo)
    zeq = distancia_equivalente(z, L)
    U, _ = mpasm(c1, dx * z / L, lamb, -zeq, s=s, device=device, dtype=dtype)
    return 1 + U * np.exp(2j * np.pi / lamb * zeq)


# -------------------------------------------------------------------- la ida
def _rampa(f, lo, hi, ancho):
    """1 en [lo, hi] y un coseno que baja a 0 en `ancho` por fuera."""
    if ancho <= 0:
        return ((f >= lo) & (f <= hi)).astype(np.float64)
    u = np.clip(np.maximum(lo - f, f - hi) / ancho, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * u))


def holograma(t, paso, lamb, z, L, forma, delta, *, centro_objeto=(0.0, 0.0),
              centro=(0.0, 0.0), banda=None, guarda=4.0, margen=1.1, complejo=False,
              device="auto", dtype=None):
    """El holograma DLHM de un objeto delgado, sin aproximación paraxial.

    t        transmitancia compleja (ny, nx) en la muestra, a z de la fuente.
             Fuera de la ventana el objeto es transparente: se propaga t - 1,
             así que conviene que t valga 1 en el borde.
    paso     paso de t, escalar o (y, x), en mm. Típicamente delta·z/L.
    forma    (ny, nx) del sensor; delta su paso; centro el del sensor respecto
             del eje, y centro_objeto el de t.
    banda    semibanda propia del objeto en mm^-1, si se sabe menor que el
             Nyquist de su malla: estrecha la rejilla de frecuencias.
    guarda   la banda se recorta a las frecuencias que pueden llegar al sensor
             desde algún punto del objeto (el resto cae fuera, y con el sensor
             entero y la banda completa pasaría de 10000 frecuencias por eje),
             pero NO en seco ni justo en el límite geométrico. Un rayo que cae
             en el borde del sensor recibe frecuencias de un entorno de anchura
             w = 1/sqrt(lamb d) alrededor de la suya (la zona de Fresnel en
             frecuencia, ~16 mm^-1 en el montaje), y un corte abrupto donde el
             espectro aún es grande deja contribuciones de extremo en todo el
             sensor. Por eso: guarda·w de banda plana más allá del alcance, y
             una rampa coseno de otras guarda·w. Medido contra RS directo con
             objeto de ~1 um: en un trozo de 24x24 px, guarda 1 -> 6e-3, 2 ->
             2.5e-4, 4 -> 1.8e-5, 6 -> 2.7e-6; en campo completo N = 1024 el
             corte en seco daba 8.3e-3 y guarda 4, 6.6e-6.
    complejo False: el holograma de contraste |1 + V|² = I/I_ref, que es lo
             que mide el sensor dividido por el fondo. True: V = U_obj/S_L,
             el campo del objeto relativo a la referencia, sin imagen gemela.

    Devuelve el array en el dispositivo.
    """
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)
    t = a_numpy(t)
    (py, px), (dy, dx) = _por_eje(paso), _por_eje(delta)
    (oy, ox), (cy, cx) = _por_eje(centro_objeto), _por_eje(centro)
    ny, nx = forma
    d = L - z
    if not 0 < z < L:
        raise ValueError(f"z = {z!r}: hace falta 0 < z < L = {L}")
    yo, xo = coordenadas(t.shape[0], py, oy), coordenadas(t.shape[1], px, ox)
    Ys, Xs = coordenadas(ny, dy, cy), coordenadas(nx, dx, cx)

    rho_max = np.sqrt(np.abs(xo).max() ** 2 + np.abs(yo).max() ** 2 + z**2)
    bo = (min(1 / (2 * py), banda or np.inf) + np.abs(yo).max() * (1 / z - 1 / rho_max) / lamb,
          min(1 / (2 * px), banda or np.inf) + np.abs(xo).max() * (1 / z - 1 / rho_max) / lamb)

    w_fresnel = 1 / np.sqrt(lamb * d)
    rampa = guarda * w_fresnel

    def tramo(o, S, b):
        """Banda del eje: la del objeto con su chirp, recortada a lo que puede
        llegar al sensor desde algún punto del objeto más la guarda plana, y
        la rampa por fuera. Devuelve ((lo, hi) de la rejilla, (lo, hi) plano)."""
        lo, hi = o[0] / (lamb * z) - b, o[-1] / (lamb * z) + b
        plano = (np.sin(np.arctan((S[0] - o[-1]) / d)) / lamb - rampa,
                 np.sin(np.arctan((S[-1] - o[0]) / d)) / lamb + rampa)
        lo, hi = max(lo, plano[0] - rampa), min(hi, plano[1] + rampa)
        if lo >= hi:
            raise ValueError("ninguna frecuencia del objeto llega al sensor")
        return (lo, hi), plano

    (ffy, alc_y), (ffx, alc_x) = tramo(yo, Ys, bo[0]), tramo(xo, Xs, bo[1])

    def soporte(o, otro, b, bo_otro, ff):
        vals = []
        for p in (o[0], o[-1]):
            for q in (otro[0], otro[-1]):
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        fx = np.clip(p / (lamb * z) + sx * b, *ff)
                        fy = np.clip(q / (lamb * z) + sy * bo_otro, -0.999 / lamb, 0.999 / lamb)
                        vals.append(_aterrizaje(p, fx, fy, lamb, d))
        return min(vals), max(vals)

    sx = soporte(xo, yo, bo[1], bo[0], ffx)
    sy = soporte(yo, xo, bo[0], bo[1], ffy)
    Px = margen * max(sx[1] - Xs[0], Xs[-1] - sx[0])
    Py = margen * max(sy[1] - Ys[0], Ys[-1] - sy[0])
    fx, fy = _rejilla(*ffx, Px), _rejilla(*ffy, Py)

    # El espectro NO se materializa: con el sensor entero son ~17000² frecuencias
    # (2.4 GB en complex64). Se recorre por bloques de filas de fy, y cada
    # bloque se propaga y se acumula en T = sum_bloques Ey_b @ (F_b · H_b):
    #
    #     JA = J @ Ax                      (ny_o, nfx)   una vez
    #     F_b = Ay_b^T @ JA                (nb, nfx)     por bloque
    #     T  += Ey_b @ (F_b · H_b · rampas)
    #     U   = T @ Ex
    #
    # Son los mismos productos que con el espectro entero, en el mismo orden.
    # Ax y Ex tampoco se materializan: JA se llena y T @ Ex se acumula por
    # bloques de columnas de fx. Lo único entero son JA y T.
    if dev == "gpu":
        comprobar_memoria(memoria_holograma(t.shape, (fy.size, fx.size), (ny, nx), dtype),
                          f"holograma({ny}x{nx})", dtype=dtype)
    real = np.zeros(0, dtype=dtype).real.dtype
    J = a_dispositivo(t - 1, xp, dtype) * esferica(yo, xo, z, lamb, xp, dtype, parte="residuo")
    JA = xp.empty((J.shape[0], fx.size), dtype=dtype)
    for j0, j1 in bloques(fx.size, max(J.shape[1], nx), itemsize=48):
        JA[:, j0:j1] = J @ ventana_chirp(xo, fx[j0:j1], px, lamb, z, xp, dtype)
    del J
    JA *= xp.asarray(_rampa(fx, *alc_x, rampa), dtype=real)[None, :]
    rampa_y = _rampa(fy, *alc_y, rampa)
    T = xp.zeros((ny, fx.size), dtype=dtype)
    for i0, i1 in bloques(fy.size, fx.size):
        Fb = ventana_chirp(yo, fy[i0:i1], py, lamb, z, xp, dtype).T @ JA
        Fb *= transfer_function(fx, fy[i0:i1], lamb, d, xp, dtype)
        Fb *= xp.asarray(rampa_y[i0:i1], dtype=real)[:, None]
        T += phasor(Ys, fy[i0:i1], +1, xp, dtype) @ Fb
    del JA, Fb
    V = xp.zeros((ny, nx), dtype=dtype)
    for j0, j1 in bloques(fx.size, nx):
        V += T[:, j0:j1] @ phasor(fx[j0:j1], Xs, +1, xp, dtype)
    del T
    V *= (fx[1] - fx[0]) * (fy[1] - fy[0])
    V *= esferica(Ys, Xs, L, lamb, xp, dtype, parte="inversa")
    return V if complejo else xp.abs(1 + V) ** 2


def memoria_holograma(forma_objeto, n_f, forma_sensor, dtype=np.complex64):
    """Bytes de pico estimados de holograma(), que recorre el espectro y las
    matrices por bloques: lo único entero son JA (ny_o, nfx) y T (ny, nfx)."""
    (my, mx), (nfy, nfx), (ny, nx) = forma_objeto, n_f, forma_sensor
    elementos = (2 * my * mx                 # objeto y J
                 + my * nfx                  # JA
                 + ny * nfx                  # T
                 + ny * nx                   # la salida
                 + 4 * (64 << 20) // 16)     # el bloque, su H, su Ey y la ventana
    return elementos * np.dtype(dtype).itemsize
