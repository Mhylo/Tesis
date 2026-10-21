"""Tarea 86: ¿rinde más el MPASM en montajes de distancia larga, como la
interferometría holográfica digital (DHI)?

El manuscrito (commit 8a3c4a3) dice que el K_f automático crece como sqrt(z)
cuando el campo desborda la ventana y que por eso el MPASM es más útil en
montajes largos. Aquí se contrasta en una geometría de DHI simulada.

LA GEOMETRÍA. Sensor de N x N píxeles de DELTA a LAMB, y el objeto en una
ventana del mismo tamaño, coaxial. La distancia se da por el cociente de la
ventana, q = z / (N DELTA^2 / LAMB): con N = 1024 y 3.45 um, q = 20 son
0.46 m; con 2048 píxeles el mismo q cae al doble de z (0.92 m). Lo que cuenta
es q y el ángulo del difusor, no N por separado.

EL OBJETO. Un difusor de banda limitada: ruido complejo filtrado a
|f| <= THETA/LAMB (speckle de ~LAMB/THETA), por un disco de borde suave de
radio A_FRAC*N*DELTA. Dos estados, como en la DHI: el segundo lleva la
deformación de una placa circular empotrada, dphi(r) = DPHI (1 - r^2/a^2)^2
(fase 4 pi w / lambda). La banda limitada permite saber hasta dónde se abre la
luz, y con eso dimensionar la referencia. Un difusor real se abre más: eso
solo adelanta el desborde de la ventana.

LA REFERENCIA. El espectro angular en una malla rellenada de P x P, con P tal
que la luz que sale de la ventana no reentra (P DELTA > N DELTA + d) y que H
quede bien muestreada en toda la banda (P DELTA > 2 d), con d = z tan(1.5
THETA). Es el mismo cálculo que FFT-ASM, pero sin aliasing por construcción;
se verifica repitiéndolo con 1.3 P (err_ref).

LOS MÉTODOS, ida (objeto -> sensor) y vuelta (sensor -> objeto), todos sobre
la malla N x N de paso DELTA salvo Fresnel:
    fft_p1, fft_p2, fft_p4   CamposT.propagadores.fft_asm con relleno x1, x2, x4
    blas_p2                  CamposT.propagadores.blas con relleno x2
    mpasm_s1, _s2, _s4       scripts/mpasm_minimo.py, K_f automático
    fresnel                  transformada de Fresnel de una FFT (solo vuelta):
                             paraxial, malla de salida LAMB z / (N DELTA)

LAS MEDIDAS.
    err_ida     ||U - U_ref|| / ||U_ref|| en el sensor (simular el holograma)
    err_vuelta  lo mismo frente a la reconstrucción ideal: la referencia
                devuelta desde el sensor, que ya trae el límite de su apertura
    dphi_rms    RMS de arg(U2 U1*) - dphi, pesado por |U1||U2|, en r < 0.8 a:
                la medida de la DHI. dphi_ideal es la de la reconstrucción
                ideal, el piso que pone la apertura del sensor
    t_s         una vuelta, en CPU y complex128
    mb          el mayor arreglo complejo del método

Salida: resultados/interferometria_larga/{resumen.csv, mapas_q20.npz}.

    Tesis_env/Scripts/python.exe -m scripts.interferometria_larga
"""

import csv
import gc
import pathlib
import time

import numpy as np
import scipy.fft as sfft
from scipy.signal.windows import tukey
from scipy.special import erf

from CamposT.pipeline import crop_center, pad_field
from CamposT.propagadores import blas, fft_asm
from scripts.mpasm_minimo import kf, mpasm

RAIZ = pathlib.Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "resultados" / "interferometria_larga"

LAMB = 532e-6          #: mm
DELTA = 3.45e-3        #: mm
N = 1024
THETA = 0.01           #: rad, banda del difusor
A_FRAC = 0.45          #: radio del objeto, en fracción de la ventana
#: píxeles del borde suave. Con 5 el espectro del borde abría la banda más allá
#: de 1.5 THETA y la referencia no convergía (err_ref 1.5e-4 a q = 20)
BORDE = 15
DPHI = 3 * np.pi       #: deformación máxima, rad
Q = (0.5, 1, 2, 3, 5, 10, 20)
#: la luz se abre hasta z tan(ABRE THETA) al dimensionar la referencia
ABRE = 2.5
#: apodización del holograma antes de devolverlo (Tukey, igual para todos los
#: métodos). Sin ella, el borde duro del sensor difracta hasta Nyquist y a
#: q = 20 esa luz se abre +-35 mm: ni la referencia la contiene
TUKEY = 0.2
SEMILLA = 86
Z_LIM = N * DELTA**2 / LAMB

METODOS_IDA = ("fft_p1", "fft_p2", "fft_p4", "blas_p2", "mpasm_s1", "mpasm_s2",
               "mpasm_s4")
METODOS_VUELTA = METODOS_IDA + ("fresnel",)


# ------------------------------------------------------------------ objeto
def radio():
    return A_FRAC * N * DELTA


def amplitud(r):
    return 0.5 * (1 - erf((r - radio()) / (BORDE * DELTA)))


def deformacion(r):
    a = radio()
    return np.where(r < a, DPHI * (1 - (r / a) ** 2) ** 2, 0.0)


def objeto():
    """Los dos estados del difusor, en la malla N x N de paso DELTA."""
    x = (np.arange(N) - N / 2) * DELTA
    R = np.hypot(*np.meshgrid(x, x))
    rng = np.random.default_rng(SEMILLA)
    ruido = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    f = sfft.fftfreq(N, DELTA)
    banda = np.hypot(*np.meshgrid(f, f)) <= THETA / LAMB
    S = sfft.ifft2(sfft.fft2(ruido) * banda)
    S /= np.sqrt(np.mean(np.abs(S) ** 2))
    U1 = amplitud(R) * S
    return U1, U1 * np.exp(1j * deformacion(R)), R


# -------------------------------------------------------------- referencia
def lado_referencia(z, factor=1.0):
    d = abs(z) * np.tan(ABRE * THETA)
    lado = max(N * DELTA + d, 2 * d) + 0.25 * N * DELTA
    return sfft.next_fast_len(int(np.ceil(factor * lado / DELTA / 2)) * 2)


def asm_relleno(U, z, P):
    """Espectro angular de U (N x N, centrado) en una malla de P x P, recortado
    a N x N. La H se aplica por bloques de filas, en su sitio, con la fase en
    float64: 2 pi z / lambda llega a 5e6 rad."""
    G = np.zeros((P, P), np.complex128)
    i0 = (P - N) // 2
    G[i0:i0 + N, i0:i0 + N] = U
    G = sfft.fft2(G, workers=-1, overwrite_x=True)
    kx2 = (LAMB * sfft.fftfreq(P, DELTA)) ** 2
    paso = max(1, 2**22 // P)
    for i in range(0, P, paso):
        arg = 1 - kx2[None, :] - kx2[i:i + paso, None]
        G[i:i + paso] *= np.exp(2j * np.pi / LAMB * z * np.sqrt(np.maximum(arg, 0))) * (arg > 0)
    G = sfft.ifft2(G, workers=-1, overwrite_x=True)
    out = G[i0:i0 + N, i0:i0 + N].copy()
    del G
    gc.collect()
    return out


# ----------------------------------------------------------------- métodos
def fresnel_vuelta(U, z):
    """Transformada de Fresnel de una FFT, a -z: paso de salida LAMB z/(N DELTA).
    Las fases globales y la cuadrática de salida se cancelan en U2 U1*, pero
    se dejan para que el campo sea el de Fresnel."""
    x = (np.arange(N) - N / 2) * DELTA
    X, Y = np.meshgrid(x, x)
    g = U * np.exp(-1j * np.pi * (X**2 + Y**2) / (LAMB * z))
    G = sfft.fftshift(sfft.ifft2(sfft.ifftshift(g), workers=-1))
    xi = (np.arange(N) - N / 2) * LAMB * z / (N * DELTA)
    XI, YI = np.meshgrid(xi, xi)
    return G * np.exp(-1j * np.pi * (XI**2 + YI**2) / (LAMB * z)), np.hypot(XI, YI)


def propagar(metodo, U, z):
    if metodo.startswith("fft_p") or metodo.startswith("blas_p"):
        p = int(metodo[-1])
        funcion = fft_asm if metodo.startswith("fft") else blas
        G = funcion(pad_field(U, p) if p > 1 else U, DELTA, LAMB, z, device="cpu",
                    dtype=np.complex128)
        return crop_center(np.asarray(G), N, N)
    if metodo.startswith("mpasm_s"):
        return mpasm(U, DELTA, LAMB, z, s=int(metodo[-1]))
    raise ValueError(metodo)


def megabytes(metodo, P=None):
    """El mayor arreglo complejo del método, en MB (complex128)."""
    if metodo.startswith(("fft_p", "blas_p")):
        lado = int(metodo[-1]) * N
    elif metodo.startswith("mpasm_s"):
        lado = int(metodo[-1]) * N
    elif metodo == "fresnel":
        lado = N
    else:
        lado = P
    return lado**2 * 16 / 1e6


# ---------------------------------------------------------------- medidas
def relativo(U, ref):
    return float(np.linalg.norm(U - ref) / np.linalg.norm(ref))


def dphi_rms(U1, U2, R, I1=None, I2=None):
    """RMS de la diferencia de fase, pesada por |U1||U2|, en r < 0.8 a: frente a
    la deformación, o con I1 e I2 frente a la de la reconstrucción ideal (que
    aísla el error del propagador del de la apertura)."""
    m = R < 0.8 * radio()
    verdad = (np.exp(1j * deformacion(R[m])) if I1 is None
              else np.exp(1j * np.angle(I2[m] * np.conj(I1[m]))))
    e = np.angle(U2[m] * np.conj(U1[m]) * np.conj(verdad))
    w = np.abs(U1[m]) * np.abs(U2[m])
    return float(np.sqrt(np.sum(w * e**2) / np.sum(w)))


# ------------------------------------------------------------------ correr
def correr():
    SALIDA.mkdir(parents=True, exist_ok=True)
    U1, U2, R = objeto()
    print(f"N = {N}, DELTA = {DELTA * 1e3} um, LAMB = {LAMB * 1e6} nm, z_lim = {Z_LIM:.1f} mm, "
          f"difusor de {THETA * 1e3:g} mrad, objeto de {2 * radio():.2f} mm")
    filas = []
    for q in Q:
        z = q * Z_LIM
        P = lado_referencia(z)
        t0 = time.time()
        S1, S2 = asm_relleno(U1, z, P), asm_relleno(U2, z, P)
        err_ref = relativo(asm_relleno(U1, z, lado_referencia(z, 1.3)), S1)
        ventana = np.outer(tukey(N, TUKEY), tukey(N, TUKEY))
        H1, H2 = S1 * ventana, S2 * ventana
        I1, I2 = asm_relleno(H1, -z, P), asm_relleno(H2, -z, P)
        err_ref_vuelta = relativo(asm_relleno(H1, -z, lado_referencia(z, 1.3)), I1)
        energia = float(np.sum(np.abs(S1) ** 2) / np.sum(np.abs(U1) ** 2))
        piso = dphi_rms(I1, I2, R)
        print(f"\nq = {q:g}, z = {z:.1f} mm, P = {P}, err_ref = {err_ref:.1e} (ida), "
              f"{err_ref_vuelta:.1e} (vuelta), energía en el sensor {energia:.2f}, "
              f"dphi ideal {piso:.3f} rad ({time.time() - t0:.0f} s)")
        print(f"  {'método':9s} {'Kf':>5s} {'err_ida':>9s} {'err_vuelta':>10s} "
              f"{'dphi_rms':>9s} {'vs ideal':>9s} {'t [s]':>7s} {'MB':>7s}")
        for metodo in METODOS_VUELTA:
            fila = dict(q=q, z_mm=f"{z:.2f}", metodo=metodo, P_ref=P,
                        err_ref=f"{err_ref:.2e}", err_ref_vuelta=f"{err_ref_vuelta:.2e}",
                        energia_sensor=f"{energia:.3f}",
                        dphi_ideal=f"{piso:.4f}", mb=f"{megabytes(metodo):.0f}")
            kfv = kf(N, DELTA, LAMB, z, int(metodo[-1])) if metodo.startswith("mpasm") else np.nan
            fila["kf"] = f"{kfv:.3f}"
            if metodo in METODOS_IDA:
                fila["err_ida"] = f"{relativo(propagar(metodo, U1, z), S1):.3e}"
                t = time.perf_counter()
                V1 = propagar(metodo, H1, -z)
                fila["t_s"] = f"{time.perf_counter() - t:.3f}"
                V2 = propagar(metodo, H2, -z)
                fila["err_vuelta"] = f"{relativo(V1, I1):.3e}"
                fila["dphi_rms"] = f"{dphi_rms(V1, V2, R):.4f}"
                fila["dphi_vs_ideal"] = f"{dphi_rms(V1, V2, R, I1, I2):.4f}"
            else:
                t = time.perf_counter()
                V1, Rf = fresnel_vuelta(H1, z)
                fila["t_s"] = f"{time.perf_counter() - t:.3f}"
                V2, _ = fresnel_vuelta(H2, z)
                # su malla no es la del objeto: solo frente a la deformación
                fila["err_ida"] = fila["err_vuelta"] = fila["dphi_vs_ideal"] = ""
                fila["dphi_rms"] = f"{dphi_rms(V1, V2, Rf):.4f}"
            if q == Q[-1] and metodo in ("fft_p2", "mpasm_s1", "fresnel"):
                np.save(SALIDA / f"dphi_q{q:g}_{metodo}.npy",
                        np.angle(V2 * np.conj(V1)).astype(np.float32))
            print(f"  {metodo:9s} {kfv:5.2f} {fila['err_ida'] or '-':>9s} "
                  f"{fila['err_vuelta'] or '-':>10s} {fila['dphi_rms']:>9s} "
                  f"{fila['dphi_vs_ideal'] or '-':>9s} {fila['t_s']:>7s} {fila['mb']:>7s}")
            filas.append(fila)
            del V1, V2
            gc.collect()
        if q == Q[-1]:
            np.save(SALIDA / f"dphi_q{q:g}_ideal.npy", np.angle(I2 * np.conj(I1)).astype(np.float32))
            np.save(SALIDA / f"dphi_q{q:g}_verdad.npy", deformacion(R).astype(np.float32))
        del S1, S2, H1, H2, I1, I2
        gc.collect()
    campos = ["q", "z_mm", "metodo", "kf", "err_ida", "err_vuelta", "dphi_rms", "dphi_vs_ideal",
              "t_s", "mb", "P_ref", "err_ref", "err_ref_vuelta", "energia_sensor", "dphi_ideal"]
    with open(SALIDA / "resumen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"\n-> {SALIDA / 'resumen.csv'}")


if __name__ == "__main__":
    correr()
