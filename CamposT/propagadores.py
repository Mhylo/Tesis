"""Propagadores de campo óptico bajo una interfaz común, en CPU o en GPU.

Port del código de Zhao et al., Opt. Lett. 45, 5937 (2020), más FFT-ASM y
BL-ASM para comparación. Punto de partida del Objetivo 1.

Sólo los métodos. Contra qué se contrastan está en referencias.py, cómo se
mide la discrepancia en metricas.py, y quién los orquesta en pipeline.py.

Todas las funciones aceptan device='gpu'|'cpu'|'auto' y un dtype de trabajo.
El cuerpo del algoritmo es el mismo en ambos casos (ver backend.py): lo único
que cambia es si `xp` es CuPy o NumPy, de modo que los tiempos comparados
miden el dispositivo y no dos implementaciones distintas.

Correspondencia con la notación del paper y del código original:
    s   <-> k1 en el código, S1=S2 en el paper. Sobremuestreo en frecuencia.
    Kf  <-> k2 en el código, K_fx=K_fy en el paper. Compresión del intervalo espectral.
    r   <-> k3 en el código, r1=r2 en el paper. Nº de puntos del plano de salida.
    mag <-> k4 en el código. Razón Δx'/Δx (tamaño de la ventana de observación).

OJO: el código original calcula Kf con `A**2 * (8*N*lamb*z)**2` donde la Ec. (14)
del paper dice `A**2 + (8*N*lamb*z)**2`. Es un error de tipeo (`*+` en vez de `+`):
la versión con producto no es dimensionalmente consistente. Aquí se usa la suma.
Ver kf_auto(..., formula='codigo') para reproducir el comportamiento original.
"""

import numpy as np

from CamposT import backend as bk
from CamposT.backend import (a_dispositivo, bloques, dtype_por_defecto,
                             fase_a_complejo, get_xp, phasor)


# ---------------------------------------------------------------- Kf automático
def kf_auto(N, delta, lamb, z, s=1, formula="paper"):
    """Coeficiente de compresión frecuencial K_f, Ec. (14) del paper.

    Es aritmética escalar: se hace siempre en doble precisión en CPU, sin
    importar el backend con el que luego se propague.

    Depende de |z|, no de z. La compresión existe para que el intervalo
    espectral cubra el ancho de banda de la función de transferencia, y ese
    ancho es el mismo propagando hacia adelante que hacia atrás: H(-z) =
    conj(H(z)). La Ec. (14) está escrita para z > 0 y al aplicarla tal cual a
    z < 0 salía fmax negativo, con lo que el max(1.0, ...) devolvía 1 y
    apagaba la compresión sin avisar. Como retropropagacion.py propaga
    siempre a -z, eso dejaba a MPASM corriendo como un FFT-ASM rellenado de
    ceros en TODA reconstrucción (medido: RMS 2.3e-1 contra 1.7e-3 con el Kf
    correcto).
    """
    if z == 0:
        return 1.0
    z = abs(z)
    Ns = s * N
    A = (Ns * lamb) ** 2
    B = (8 * Ns * lamb * z) ** 2
    inner = A**2 + B if formula == "paper" else A**2 * B   # 'codigo' = bug original
    fmax = np.sqrt(np.sqrt(inner) - A) / (4 * np.sqrt(2) * z * lamb)
    return max(1.0, (1 / (2 * delta)) / fmax)


# ---------------------------------------------------------------------- MPASM
def memoria_mpasm(M, N, s=10, r=1, dtype=np.complex64):
    """Bytes de pico estimados para mpasm(). Útil antes de lanzar en GPU."""
    itemsize = np.dtype(dtype).itemsize
    Ms, Ns = s * M, s * N
    elementos = (Ms * Ns          # espectro F
                 + M * Ns         # intermedio My @ U0
                 + N * Ns         # Mx
                 + Ms * M         # My
                 + Ns * (r * N)   # Mx1
                 + (r * M) * Ms   # My1
                 + (r * M) * Ns)  # intermedio My1 @ F
    return elementos * itemsize + bk.PRESUPUESTO_BLOQUE


def _comprobar_ventana(r, mag, s, Kfy, Kfx):
    """La ventana de salida no puede exceder el periodo espacial del espectro.

    El espectro se muestrea cada 1/(s·N·delta·Kf), así que el campo que
    describe se repite en el espacio cada s·N·delta·Kf. La malla de salida
    abarca r·N·delta·mag. Si lo segundo supera a lo primero, la salida sale
    con copias periódicas del campo superpuestas: no falla, no avisa,
    simplemente devuelve otra cosa (medido contra el gaussiano analítico:
    error relativo 1.0, es decir, nada que ver).

    N y M se cancelan en la desigualdad, así que la condición es r·mag <= s·Kf
    por eje, y manda el eje con menos compresión. La igualdad se admite: ahí
    la ventana cubre exactamente un periodo y no hay solape.

    Es el único acoplamiento entre los cuatro parámetros de la Tabla 1 que no
    se ve leyendo sus definiciones por separado, y el más fácil de violar sin
    enterarse al subir mag para ampliar la ventana de observación.
    """
    holgura = s * min(Kfy, Kfx)
    if r * mag > holgura * (1 + 1e-12):
        raise ValueError(
            f"la ventana de salida no cabe en el periodo del espectro: "
            f"r·mag = {r * mag:g} > s·Kf = {holgura:g}. El campo saldría con "
            f"copias periódicas superpuestas. Sube s a >= "
            f"{int(np.ceil(r * mag / min(Kfy, Kfx)))}, o baja r o mag.")


def mpasm(U0, delta, lamb, z, s=10, Kf=None, r=1, mag=1.0, formula="paper",
          device="auto", dtype=None):
    """Espectro angular por producto matricial.

    Devuelve (campo_salida, Kf_usado). Con s=Kf=r=mag=1 es equivalente a FFT-ASM.

    Kf se calcula por eje (K_fy del numero de filas, K_fx del de columnas) y
    se devuelve como escalar cuando los dos coinciden -- toda malla cuadrada,
    que es el caso habitual -- y como pareja (Kfy, Kfx) cuando no. El
    parametro Kf acepta las dos formas.

    r y mag no son libres: la ventana de salida tiene que caber en el periodo
    espacial que impone el muestreo del espectro, o sea r·mag <= s·Kf. Se
    comprueba y se aborta si no se cumple (ver _comprobar_ventana). Ampliar la
    ventana de observación exige, por tanto, subir s o dejar que Kf crezca con
    z: no basta con subir mag.

    El campo devuelto vive en el dispositivo pedido; usa backend.a_numpy() para
    bajarlo.

    Es el método que más gana con CUDA: el grueso del coste son dos productos
    matriciales densos encadenados, justo lo que cuBLAS hace bien.
    """
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)
    U0 = a_dispositivo(U0, xp, dtype)

    M, N = U0.shape
    Ms, Ns = s * M, s * N
    # Kf por eje: la Tabla 1 del paper lista K_fx y K_fy por separado, y Kf va
    # como ~1/sqrt(N), asi que en malla rectangular el eje corto necesita mas
    # compresion que el largo. Aplicar el de un solo eje a los dos submuestrea
    # el corto sin avisar (medido: RMS 1.3e-2 frente a 4.8e-4).
    if Kf is None:
        Kfy = kf_auto(M, delta, lamb, z, s=s, formula=formula)
        Kfx = kf_auto(N, delta, lamb, z, s=s, formula=formula)
    elif isinstance(Kf, (tuple, list)):
        Kfy, Kfx = (float(v) for v in Kf)
    else:
        Kfy = Kfx = float(Kf)
    _comprobar_ventana(r, mag, s, Kfy, Kfx)

    if dev == "gpu":
        bk.comprobar_memoria(memoria_mpasm(M, N, s, r, dtype),
                             f"mpasm(N={N}, s={s}, r={r}, {np.dtype(dtype).name})",
                             dtype=dtype)

    # mallas en doble precisión: son coordenadas, no datos de campo
    x = (np.arange(N) - N / 2) * delta
    y = (np.arange(M) - M / 2) * delta
    fx = (np.arange(Ns) - Ns / 2) / (s * delta * N) / Kfx
    fy = (np.arange(Ms) - Ms / 2) / (s * delta * M) / Kfy

    # DFT como triple producto matricial, Ec. (3)
    Mx = phasor(x, fx, -1, xp, dtype)                    # (N, Ns)
    My = phasor(fy, y, -1, xp, dtype)                    # (Ms, M)
    norma = float(s**2 * M * N * Kfx * Kfy)
    F = ((My @ U0 @ Mx) / norma).astype(dtype, copy=False)
    del Mx, My

    # H se aplica por bloques de filas y nunca se materializa entera: a s alto
    # duplicaría el pico de memoria del espectro sin necesidad
    aplicar_transferencia(F, fx, fy, lamb, z, xp)

    # DFT inversa con muestreo espacial de salida independiente, Ecs. (6)-(8)
    x1 = (np.arange(r * N) - r * N / 2) * delta * mag
    y1 = (np.arange(r * M) - r * M / 2) * delta * mag
    Mx1 = phasor(fx, x1, +1, xp, dtype)                  # (Ns, rN)
    My1 = phasor(y1, fy, +1, xp, dtype)                  # (rM, Ms)
    return My1 @ F @ Mx1, (Kfy if Kfy == Kfx else (Kfy, Kfx))


# ------------------------------------------------------- función de transferencia
def transfer_function(fx, fy, lamb, z, xp=np, dtype=np.complex128):
    """H(fx,fy;z) para propagación en espacio libre. Ondas evanescentes a cero.

    La fase k·z·sqrt(1-λ²f²) puede valer ~1e12 rad, así que se evalúa siempre
    en float64 y sólo se convierte el fasor resultante al dtype de trabajo.
    """
    H = xp.empty((len(fy), len(fx)), dtype=dtype)
    _llenar_transferencia(H, fx, fy, lamb, z, xp)
    return H


def aplicar_transferencia(F, fx, fy, lamb, z, xp):
    """Multiplica F por H(fx,fy;z) in situ, bloque de filas a bloque de filas."""
    _llenar_transferencia(F, fx, fy, lamb, z, xp, multiplicar=True)


def _llenar_transferencia(dst, fx, fy, lamb, z, xp, multiplicar=False):
    fx = xp.asarray(fx, dtype=np.float64)
    fy = xp.asarray(fy, dtype=np.float64)
    k = 2 * np.pi / lamb
    uu = (lamb * fx)[None, :]
    for i0, i1 in bloques(fy.size, fx.size):
        vv = (lamb * fy[i0:i1])[:, None]
        arg = 1.0 - uu**2 - vv**2
        propagante = arg > 0
        H = xp.where(propagante,
                     fase_a_complejo(k * z * xp.sqrt(xp.where(propagante, arg, 0.0)),
                                     dst.dtype, xp),
                     0)
        if multiplicar:
            dst[i0:i1] *= H
        else:
            dst[i0:i1] = H


# -------------------------------------------------------------------- FFT-ASM
def fft_asm(U0, delta, lamb, z, device="auto", dtype=None):
    """Espectro angular por FFT. El caso de referencia, sin control de muestreo.

    H se aplica sobre el espectro in situ, igual que en mpasm(): materializarla
    y multiplicar por ella pedía dos arrays del tamaño del plano —la propia H y
    el producto— que aquí no hacen falta. El resultado es bit a bit el mismo.
    """
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)
    U0 = a_dispositivo(U0, xp, dtype)

    M, N = U0.shape
    fx = (np.arange(N) - N / 2) / (delta * N)
    fy = (np.arange(M) - M / 2) / (delta * M)
    F = xp.fft.fftshift(xp.fft.fft2(U0))
    aplicar_transferencia(F, fx, fy, lamb, z, xp)
    return xp.fft.ifft2(xp.fft.ifftshift(F))


# ----------------------------------------------------------------------- BLAS
def blas(U0, delta, lamb, z, device="auto", dtype=None):
    """Espectro angular de banda limitada, Matsushima & Shimobaba (2009).

    El límite de banda es un rectángulo centrado, y fx y fy van en orden
    creciente: las frecuencias que sobreviven son un tramo contiguo, así que
    basta anular las cuatro bandas de fuera por rodajas. La máscara completa
    costaba dos meshgrid en float64, un booleano del tamaño del plano y una
    pasada más sobre todo él. Bit a bit da lo mismo.
    """
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)
    U0 = a_dispositivo(U0, xp, dtype)

    M, N = U0.shape
    fx = (np.arange(N) - N / 2) / (delta * N)
    fy = (np.arange(M) - M / 2) / (delta * M)
    flim_x = 1 / (lamb * np.sqrt((2 * z / (delta * N)) ** 2 + 1))
    flim_y = 1 / (lamb * np.sqrt((2 * z / (delta * M)) ** 2 + 1))

    F = xp.fft.fftshift(xp.fft.fft2(U0))
    aplicar_transferencia(F, fx, fy, lamb, z, xp)
    # side='right' en el extremo negativo y 'left' en el positivo reproduce la
    # desigualdad estricta |f| < flim de la formulación original.
    F[:, :np.searchsorted(fx, -flim_x, side="right")] = 0
    F[:, np.searchsorted(fx, flim_x, side="left"):] = 0
    F[:np.searchsorted(fy, -flim_y, side="right")] = 0
    F[np.searchsorted(fy, flim_y, side="left"):] = 0
    return xp.fft.ifft2(xp.fft.ifftshift(F))

