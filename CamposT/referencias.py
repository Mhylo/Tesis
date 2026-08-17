"""Soluciones de referencia contra las que se mide la exactitud del Objetivo 1.

Dos referencias independientes, y la diferencia entre ellas es un resultado en
si misma:

    gauss_analytic  solucion cerrada del haz gaussiano. Exacta, instantanea y
                    PARAXIAL. Devuelve amplitud, no campo complejo.
    rs1_radial      integral de Rayleigh-Sommerfeld I, NO paraxial, resuelta
                    por cuadratura. Es la referencia que usa el paper para su
                    Figura 4.

Donde las dos valen coinciden; donde discrepan, lo que se esta midiendo es
hasta donde llega la aproximacion paraxial.

La R-S completa cuesta O(N^4) y la malla 512x512 de la Tabla 1 tardaria mas de
una hora. Por eso el paper "reduce el calculo 2-D a 1-D": bajo simetria de
revolucion la integral se reescribe en polares sin aproximar nada. Ver
rs1_radial() y tests/test_rs1.py.
"""

import numpy as np

from CamposT.backend import (a_dispositivo, a_numpy, bloques, dtype_por_defecto,
                             fase_a_complejo, get_xp)


# ------------------------------------------------------- referencias analíticas
def gauss_beam(N, delta, w0, device="auto", dtype=None):
    """Devuelve (U0 en el dispositivo, X, Y en NumPy para las referencias)."""
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)
    x = (np.arange(N) - N / 2) * delta
    X, Y = np.meshgrid(x, x)
    U0 = np.exp(-(X**2 + Y**2) / w0**2)
    return a_dispositivo(U0, xp, dtype), X, Y


def n_phi_auto(rho_max, lamb, z, muestras_por_ciclo=16, minimo=256):
    """Nodos de la cuadratura angular de rs1_radial.

    El integrando oscila en phi como e^{ikr}. La excursion maxima de r sobre
    phi se da con rho = rho' = rho_max, entre phi=0 y phi=pi:

        dr = sqrt(4*rho_max^2 + z^2) - z

    que son dr/lamb ciclos completos de fase. Se piden varias muestras por
    ciclo.

    ES DELIBERADAMENTE CONSERVADORA. El criterio es de tipo Nyquist, pero la
    regla del punto medio sobre un integrando periodico y suave converge
    espectralmente, no algebraicamente: medido con el gaussiano de las pruebas
    a z = 500 mm, el automatico pide 1348 nodos y con 128 el error ya es 1e-11.
    O sea que en los regimenes medidos el suelo de 256 basta por si solo y la
    regla solo actua de red de seguridad, previsiblemente para campos anchos a
    z corto (caso que no se ha llegado a caracterizar: el barrido necesario
    excedia el tiempo disponible).

    Consecuencia para la suite: sustituir esta regla por su suelo NO se detecta
    por mutacion, porque en ningun regimen probado es determinante. Lo que si
    esta fijado es que el valor por defecto da un resultado convergido, que es
    la propiedad que importa (test_converge_al_refinar_la_cuadratura_angular).
    """
    ciclos = (np.sqrt(4 * rho_max**2 + z**2) - abs(z)) / lamb
    return int(max(minimo, np.ceil(ciclos * muestras_por_ciclo)))


def n_rho_auto(rho_max, delta, lamb, z, paso_fase=0.5):
    """Nodos de la cuadratura radial de rs1_radial.

    A diferencia del ASM, que trabaja en frecuencia sobre un campo ya limitado
    en banda, la R-S suma en el espacio un nucleo que oscila como e^{ikr}. El
    paso de fase entre nodos consecutivos vale

        k * (dr/drho') * drho'  ~  k * (rho_max/z) * drho'

    y el muestreo del sensor no tiene por que satisfacerlo: con la malla de
    N=32 y z=500 mm de las pruebas salen 8 rad por muestra, muy por encima de
    Nyquist. Por eso la cuadratura radial se refina por encima del paso de
    pixel cuando hace falta; el campo de entrada es suave y esta limitado en
    banda, asi que interpolarlo mas fino es legitimo.

    Devuelve el numero de nodos en [0, rho_max_entrada], nunca menos que uno
    por pixel.
    """
    n_pixel = max(1, int(np.ceil(rho_max / delta)))
    d_rho = paso_fase * z / ((2 * np.pi / lamb) * max(rho_max, delta))
    return int(max(n_pixel, np.ceil(rho_max / d_rho)))


def _perfil_radial(U0, delta, tol_simetria):
    """(radios, perfil complejo) del campo de entrada, y control de simetria.

    La malla del paquete es x = (arange(N) - N/2)*delta, que no es simetrica
    respecto al origen (sobra una muestra en el lado negativo), asi que no
    vale comprobar la simetria con volteos del array.

    Se agrupan los pixeles que estan EXACTAMENTE al mismo radio. Con esta
    malla r/delta = sqrt(m^2 + n^2) con m, n enteros, asi que permutar o
    cambiar de signo las coordenadas da el mismo radio bit a bit: agrupar por
    ese valor junta los pixeles que un campo con simetria de revolucion tiene
    obligados a valer lo mismo. Agrupar por anillos de anchura finita no
    serviria: dentro del anillo el campo varia porque los radios difieren, y
    eso no es asimetria.

    Es una condicion mas fuerte que la simetria de cuatro pliegues: una
    apertura cuadrada la pasa y esta no.
    """
    N = U0.shape[0]
    x = (np.arange(N) - N / 2) * delta
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y).ravel()
    orden = np.argsort(R)
    R, U = R[orden], U0.ravel()[orden]

    _, inv = np.unique(np.round(R / delta, 9), return_inverse=True)
    cuenta = np.bincount(inv)
    media = (np.bincount(inv, weights=U.real)
             + 1j * np.bincount(inv, weights=U.imag)) / cuenta
    escala = np.abs(U).max()
    if escala > 0:
        desvio = float(np.abs(U - media[inv]).max() / escala)
        if desvio > tol_simetria:
            raise ValueError(
                f"el campo no tiene simetría circular: los píxeles de un mismo "
                f"anillo difieren hasta {desvio:.2e} (tolerancia "
                f"{tol_simetria:.0e}). La reducción a 1-D sólo vale bajo "
                f"simetría de revolución; usa fft_asm o mpasm para el caso "
                f"general.")
    return R, U


def nucleo_rs1(r, k, z, dtype=np.complex128, xp=np):
    """Nucleo de la Rayleigh-Sommerfeld I:  (ik + 1/r) e^{ikr} (z/r^2).

    El factor (ik + 1/r) sale de derivar la funcion de Green; z/r es el factor
    de oblicuidad y el 1/r restante, la caida de la onda esferica.

    El termino 1/r es lo que separa la R-S exacta de su forma de campo lejano
    (ik e^{ikr} z/r^2): pesa 1/(kr) frente a ik, o sea que solo cuenta cuando
    r es del orden de la longitud de onda. En DLHM, con z de milimetros a
    centimetros, esta entre cinco y ocho ordenes por debajo. Se conserva
    porque no cuesta nada y hace exacta la implementacion.

    La fase kr llega a ~1e9 rad, asi que se evalua en float64 y solo el fasor
    ya acotado baja al dtype de trabajo (ver backend.fase_a_complejo).
    """
    return (1j * k + 1 / r) * fase_a_complejo(k * r, dtype, xp) * (z / r**2)


def rs1_radial(U0, delta, lamb, z, n_phi=None, n_rho=None, n_rho_in=None,
               tol_simetria=1e-6, device="auto", dtype=None):
    """Rayleigh-Sommerfeld I exacta, reducida a 1-D por simetria circular.

        U(P) = -(1/2pi) * integral U0(Q) (ik + 1/r) (e^{ikr}/r) (z/r) dS

    escrita en polares. Para un campo con simetria de revolucion el integrando
    solo depende de rho, rho' y del angulo entre ellos, asi que

        U(rho) = -(1/2pi) integral U0(rho') rho' F(rho,rho') drho'
        F      = integral_0^{2pi} (ik + 1/r)(e^{ikr}/r)(z/r) dphi
        r      = sqrt(rho^2 + rho'^2 - 2 rho rho' cos phi + z^2)

    Cuesta O(N_rho^2 N_phi) en vez de los O(N^4) de la integral 2-D directa.
    No es una aproximacion: es la misma integral en las coordenadas que
    respetan la simetria del problema. Contrastada contra la fuerza bruta de
    pyLHM en tests/test_rs1.py.

    Es la referencia con la que el paper construye su Figura 4, y a diferencia
    de gauss_analytic NO es paraxial: la diferencia entre las dos mide donde
    deja de valer la aproximacion paraxial.

    Devuelve el campo en la misma malla que la entrada. Se integra sobre el
    disco inscrito en la ventana, no sobre el cuadrado; para un campo confinado
    en ese disco es lo mismo, y si no lo esta el control de simetria avisa
    antes.
    """
    if z <= 0:
        raise ValueError(f"rs1_radial necesita z > 0 (el nucleo diverge en 0); "
                         f"se pidio z={z}")
    xp, dev = get_xp(device)
    dtype = dtype or dtype_por_defecto(dev)
    U0 = a_numpy(U0)
    if U0.shape[0] != U0.shape[1]:
        raise ValueError(f"malla no cuadrada: {U0.shape}")

    N = U0.shape[0]
    R_pix, U_pix = _perfil_radial(U0, delta, tol_simetria)

    # --- malla de cuadratura en el plano de entrada (disco inscrito) ---------
    rho_max_entrada = (N // 2) * delta
    n_in = n_rho_in or n_rho_auto(rho_max_entrada, delta, lamb, z)
    d_rho = rho_max_entrada / n_in
    rho_in = (np.arange(n_in) + 0.5) * d_rho          # regla del punto medio
    perfil = (np.interp(rho_in, R_pix, U_pix.real)
              + 1j * np.interp(rho_in, R_pix, U_pix.imag))

    # --- malla radial de salida: hasta la esquina de la ventana --------------
    rho_max_salida = float(np.hypot(*((np.array([N / 2, N / 2])) * delta)))
    n_rho = n_rho or 2 * N
    rho_out = np.linspace(0.0, rho_max_salida, n_rho)

    # --- cuadratura angular --------------------------------------------------
    # cos(phi) es par respecto a pi, asi que basta integrar en [0, pi] y doblar
    n_phi = n_phi or n_phi_auto(max(rho_max_salida, rho_in[-1]), lamb, z)
    phi = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)
    d_phi = np.pi / n_phi

    k = 2 * np.pi / lamb
    cos_phi = xp.asarray(np.cos(phi), dtype=np.float64)[None, None, :]
    rho_i = xp.asarray(rho_in, dtype=np.float64)[None, :, None]
    pesos = xp.asarray(perfil * rho_in * d_rho, dtype=dtype)[None, :]

    salida = xp.empty(n_rho, dtype=dtype)
    for i0, i1 in bloques(n_rho, n_in * n_phi):
        rho_o = xp.asarray(rho_out[i0:i1], dtype=np.float64)[:, None, None]
        r = xp.sqrt(rho_o**2 + rho_i**2 - 2 * rho_o * rho_i * cos_phi + z**2)
        G = nucleo_rs1(r, k, z, dtype, xp)
        salida[i0:i1] = xp.sum(pesos * (2 * d_phi) * xp.sum(G, axis=2), axis=1)
    salida *= -1 / (2 * np.pi)

    # --- del perfil radial de vuelta al plano ---------------------------------
    x = (np.arange(N) - N / 2) * delta
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    s_cpu = a_numpy(salida)
    U = (np.interp(R, rho_out, s_cpu.real)
         + 1j * np.interp(R, rho_out, s_cpu.imag))
    return a_dispositivo(U, xp, dtype)


def gauss_analytic(X, Y, w0, lamb, z):
    """Amplitud de un haz gaussiano propagado (solución exacta)."""
    zR = np.pi * w0**2 / lamb
    wz = w0 * np.sqrt(1 + (z / zR) ** 2)
    return (w0 / wz) * np.exp(-(X**2 + Y**2) / wz**2)
