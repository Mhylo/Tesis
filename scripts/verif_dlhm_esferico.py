"""Verificación de CamposT.dlhm: de dónde salen las cifras que cita el módulo.

Tres partes, cada una con su orden en la línea de comandos:

    ida      La ida (holograma) contra Rayleigh-Sommerfeld directo (Lopera,
             Ec. 1), con un objeto de detalle de ~1.25 um, en el eje y a 5.7 y
             12.4 grados. Junto a ella, el escalado paraxial de la tarea 24
             (malla delta/M, onda plana a z_eq) sobre los mismos puntos.

    campo    Campo completo: el holograma exacto de un mosaico del mismo objeto
             y tres reconstrucciones sobre la MISMA rejilla de frecuencias:
                 exacto        CamposT.dlhm, con las ventanas de Fresnel
                 sin ventanas  la esférica muestreada en el sensor y DFT de
                               Riemann: lo que sale de multiplicar por
                               e^{ikR}/R y pasar el MPASM tal cual
                 escalado      la formulación (i) de la tarea 24
             Correlación con el objeto por teselas, en función del ángulo
             desde la fuente. Con --n 3000 es el sensor entero (en GPU).

             "Sin ventanas" no es una versión peor por grados: sus réplicas
             (cada componente repetida a 1/delta) se propagan como rayos
             desviados ~16 grados y caen dentro o fuera de la ventana del
             objeto según el periodo de la rejilla. Medido en N = 2048: con la
             rejilla mínima de CamposT.dlhm (P = 2.40 mm) 0.42; con una 45 %
             más holgada, 0.946, igual que con ventanas. Acierta por geometría,
             no por construcción.

    carlos   La vuelta exacta contra reconstruir() de scripts/mi_prueba_dlhm.py
             (la cadena de Carlos, validada contra reconstruction_dlhm.py) en
             N = 1024, tal cual y con sus dos detalles cambiados por separado:
             el remuestreo lineal de cv.resize y la malla de tamaño impar que
             devolvía malla_remuestreada antes de redondear al par (2027 aquí;
             ahora 2028). La impar es la que pesa: su rejilla de frecuencias,
             linspace(-P/2, P/2 - 1, P)·dfx, no pasa por f = 0 con P impar, H
             se evalúa medio paso corrida y la reconstrucción sale desplazada
             lambda·d/(2W) ~ 1 um, del orden del detalle (el mismo fallo que
             documenta frecuencias_fft() en propagadores.py). Medido: 0.875
             con la malla impar, 0.990 tal cual (la par), 0.993 además con
             banda limitada.

Las tres reconstruyen c - 1 (el holograma de contraste menos el fondo): ver
el docstring de CamposT/dlhm.py.

    Tesis_env/Scripts/python.exe -m scripts.verif_dlhm_esferico
    Tesis_env/Scripts/python.exe -m scripts.verif_dlhm_esferico campo --n 3000

Escribe en resultados/dlhm_esferico/: campo_N<n>.csv (la tabla por ángulo) y
campo_N<n>.png (la figura).

Los números del montaje salen de CamposT/montaje.py: lambda = 528 nm,
delta = 1.83 um, z = 3 mm, L = 10 mm. El detalle del objeto está elegido
para que se note la fase que desprecia el escalado: con objetos lisos, como
los de la verificación de la tarea 24 (6 a 30 um), no se nota.
"""

import argparse
import pathlib
import sys
import time

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from CamposT import dlhm                                                    # noqa: E402
from CamposT.backend import (a_numpy, gpu_disponible, liberar_memoria,      # noqa: E402
                             phasor, sincronizar)
from CamposT.montaje import MONTAJE                                        # noqa: E402

LAMB, DELTA, Z, L = MONTAJE.lamb, MONTAJE.delta_x, MONTAJE.z, MONTAJE.L
M, D = L / Z, L - Z
ZEQ = dlhm.distancia_equivalente(Z, L)
K = 2 * np.pi / LAMB
DO = DELTA / M
UM = 1e-3
SALIDA = RAIZ / "resultados" / "dlhm_esferico"


# ----------------------------------------------------------------- el objeto
def objeto_fino(n, paso, f0=250.0):
    """Absorbente de barras de 1.25 um de ancho y 2.5 um de periodo, puntos de
    1.6 um y una L, pasado por un paso bajo gaussiano exp(-(f/f0)²/2): banda
    efectiva por debajo de 910 mm^-1, el Nyquist de la malla delta/M. a en
    (0, 0.8); la transmitancia es 1 - a."""
    g = (np.arange(n) - n // 2) * paso
    X, Y = np.meshgrid(g, g)
    a = np.zeros_like(X)
    for i in range(4):
        a += (np.abs(X + 6 * UM - i * 2.5 * UM) < 0.625 * UM) * (np.abs(Y - 3 * UM) < 4 * UM)
    a += np.hypot(X - 5 * UM, Y - 4 * UM) < 0.8 * UM
    a += np.hypot(X - 7.5 * UM, Y - 4 * UM) < 0.8 * UM
    a += (np.abs(X - 3 * UM) < 4 * UM) * (np.abs(Y + 5 * UM) < 0.6 * UM)
    a += (np.abs(X + 1 * UM) < 0.6 * UM) * (np.abs(Y + 3 * UM) < 2.5 * UM)
    f = np.fft.fftfreq(n, paso)
    FX, FY = np.meshgrid(f, f)
    a = np.fft.ifft2(np.fft.fft2(np.clip(a, 0, 1)) * np.exp(-(FX**2 + FY**2) / (2 * f0**2))).real
    return 0.8 * a


def mosaico(N, paso_red=128, lado=48):
    """El objeto fino repetido en una red, en la malla delta/M. -> (a, esquinas)."""
    pieza = objeto_fino(2 * lado, DO / 2)[::2, ::2]
    a = np.zeros((N, N))
    esquinas = np.arange(paso_red // 2, N - lado, paso_red)
    for i in esquinas:
        for j in esquinas:
            a[i:i + lado, j:j + lado] = pieza
    return a, esquinas


def correlacion(u, v):
    # en doble: con los campos complex64 de la GPU la cuenta en simple daba
    # 1.000001, que parece un fallo y es redondeo
    u = np.ravel(u).astype(np.complex128)
    v = np.ravel(v).astype(np.complex128)
    return abs(np.vdot(u, v)) / np.sqrt(np.vdot(u, u).real * np.vdot(v, v).real)


# ------------------------------------------------- Rayleigh-Sommerfeld directo
def rs_directo(a, paso, centro_obj, Ys, Xs):
    """V = U_obj/S_L en la malla (Ys, Xs) del sensor, por suma directa de la
    Ec. 1 de Lopera con el núcleo RS-I exacto. En GPU si la hay."""
    xp = np
    if gpu_disponible():
        import cupy as xp
    n = a.shape[0]
    g = (np.arange(n) - n / 2) * paso
    XS, YS = np.meshgrid(g + centro_obj[1], g + centro_obj[0])
    m = np.abs(a) > 1e-9
    rho = np.sqrt(XS[m] ** 2 + YS[m] ** 2 + Z**2)
    fuente = xp.asarray(-a[m] * np.exp(1j * K * rho) / rho * paso**2)
    xs, ys = xp.asarray(XS[m]), xp.asarray(YS[m])
    XX, YY = np.meshgrid(Xs, Ys)
    xo, yo = XX.ravel(), YY.ravel()
    U = np.empty(xo.size, complex)
    lote = max(1, (1 << 24) // xs.size)          # ~16 M elementos por lote
    for i in range(0, xo.size, lote):
        bx, by = xp.asarray(xo[i:i + lote]), xp.asarray(yo[i:i + lote])
        r = xp.sqrt((bx[:, None] - xs[None, :]) ** 2 + (by[:, None] - ys[None, :]) ** 2 + D * D)
        h = D / (2 * np.pi * r * r) * (1 / r - 1j * K) * xp.exp(1j * K * r)
        U[i:i + lote] = a_numpy(h @ fuente)
    R = np.sqrt(XX**2 + YY**2 + L**2)
    return U.reshape(XX.shape) / (np.exp(1j * K * R) / R)


def ida_escalada(a, paso, centro_obj, Ys, Xs, banda=910.0):
    """V por la formulación (i): -a en la malla delta/M, onda plana a z_eq,
    evaluado en (Ys, Xs)/M. DFT matricial general, CPU, complex128."""
    n = a.shape[0]
    g = (np.arange(n) - n / 2) * paso
    ext = (n * paso + ZEQ * np.tan(np.arcsin(LAMB * banda))) * 2 + (Xs[-1] - Xs[0]) / M
    f = np.arange(-banda, banda, 1 / ext)
    Ax = np.exp(-2j * np.pi * np.outer(g + centro_obj[1], f)) * paso
    Ay = np.exp(-2j * np.pi * np.outer(g + centro_obj[0], f)) * paso
    F = Ay.T @ (-a) @ Ax
    arg = 1 / LAMB**2 - f[None, :] ** 2 - f[:, None] ** 2
    F *= np.where(arg > 0, np.exp(2j * np.pi * ZEQ * np.sqrt(np.maximum(arg, 0))), 0)
    Ex = np.exp(2j * np.pi * np.outer(f, Xs / M))
    Ey = np.exp(2j * np.pi * np.outer(Ys / M, f))
    return (Ey @ F @ Ex) * (f[1] - f[0]) ** 2 * np.exp(-1j * K * ZEQ)


def parte_ida():
    print("== ida: holograma contra Rayleigh-Sommerfeld directo ==")
    print(f"   objeto de ~1.25 um, trozo de 256x256 puntos cada 8 px alrededor de su "
          f"proyeccion; dispositivo {'GPU' if gpu_disponible() else 'CPU'}")
    af = objeto_fino(96, DO / 2)
    ac = af[::2, ::2]
    ne, pe = 256, 8 * DELTA
    for X0 in (0.0, 1.0, 2.2):
        centro_obj = (0.0, X0 / M)
        Ys = (np.arange(ne) - ne / 2) * pe
        Xs = Ys + X0
        t0 = time.perf_counter()
        ref = rs_directo(af, DO / 2, centro_obj, Ys, Xs)
        t1 = time.perf_counter()
        # la ida exacta evaluada en esos puntos: sensor de ne x ne con paso pe
        V = a_numpy(dlhm.holograma(1 - ac, DO, LAMB, Z, L, (ne, ne), pe, centro_objeto=centro_obj,
                                   centro=(0.0, X0), complejo=True))
        t2 = time.perf_counter()
        Ve = ida_escalada(ac, DO, centro_obj, Ys, Xs)
        th = np.degrees(np.arctan(X0 / L))
        err = lambda u: np.linalg.norm(u - ref) / np.linalg.norm(ref)
        print(f"   X0 = {X0:3.1f} mm ({th:4.1f} grados): exacto err {err(V):.1e} corr "
              f"{correlacion(V, ref):.6f} | escalado err {err(Ve):.2f} corr "
              f"{correlacion(Ve, ref):.4f}   (RS {t1 - t0:.0f} s, exacto {t2 - t1:.1f} s)")


# ------------------------------------------------------------- campo completo
def reconstruir_ingenuo(R, c1):
    """Lo mismo que R.campo(Z), en la misma rejilla, pero sin ventanas: la
    esférica entera muestreada en el sensor y la DFT de Riemann."""
    xp, dtype = R.xp, R.dtype
    (dy, dx) = R.delta
    Ay = phasor(R.Y, R.fy, -1, xp, dtype) * dy
    Ax = phasor(R.X, R.fx, -1, xp, dtype) * dx
    J = xp.asarray(c1, dtype=dtype) * dlhm.esferica(R.Y, R.X, L, LAMB, xp, dtype)
    F = dlhm._producto_entrada(J, Ay, Ax)
    del J, Ax, Ay
    y = dlhm.coordenadas(R.forma[0], dy * Z / L)
    x = dlhm.coordenadas(R.forma[1], dx * Z / L)
    U = dlhm._salida(F, R.fy, R.fx, LAMB, -D, y, x, xp, dtype)
    return 1 + U * dlhm.esferica(y, x, Z, LAMB, xp, dtype, parte="inversa")


def por_angulo(t, a, esquinas, lado=48):
    N = a.shape[0]
    filas = []
    for i in esquinas:
        for j in esquinas:
            s = np.s_[i:i + lado, j:j + lado]
            r = np.hypot(i + lado / 2 - N / 2, j + lado / 2 - N / 2) * DELTA
            filas.append((np.degrees(np.arctan(r / L)), correlacion(t[s] - 1, -a[s])))
    return np.array(filas)


def parte_campo(N):
    print(f"== campo completo, N = {N}: borde a {np.degrees(np.arctan(N * DELTA / 2 / L)):.1f} "
          f"grados, esquina a {np.degrees(np.arctan(N * DELTA / np.sqrt(2) / L)):.1f} ==")
    a, esquinas = mosaico(N)
    t0 = time.perf_counter()
    V = a_numpy(dlhm.holograma(1 - a, DO, LAMB, Z, L, (N, N), DELTA, complejo=True))
    print(f"   holograma exacto: {time.perf_counter() - t0:.1f} s")
    liberar_memoria()
    bordes = [0, 3, 5, 7.5, 10, 12.5, 15, 25]
    tabla = {}
    for entrada, c in (("campo complejo", 1 + V), ("intensidad", np.abs(1 + V) ** 2)):
        t0 = time.perf_counter()
        R = dlhm.Reconstructor(c, DELTA, LAMB, L, Z)
        t1 = time.perf_counter()
        R.campo(Z)
        sincronizar()
        t2 = time.perf_counter()
        te = a_numpy(R.campo(Z))
        sincronizar()
        t3 = time.perf_counter()
        ti = a_numpy(reconstruir_ingenuo(R, c - 1))
        del R
        liberar_memoria()
        dlhm.reconstruir_escalado(c, DELTA, LAMB, Z, L)
        sincronizar()
        t4 = time.perf_counter()
        ts = a_numpy(dlhm.reconstruir_escalado(c, DELTA, LAMB, Z, L))
        sincronizar()
        t5 = time.perf_counter()
        print(f"   {entrada}: Reconstructor {t1 - t0:.1f} s una vez (ventanas + espectro), "
              f"{t3 - t2:.2f} s por z; escalado {t5 - t4:.2f} s por z")
        print(f"      ventana entera: exacto {correlacion(te - 1, -a):.4f}  sin ventanas "
              f"{correlacion(ti - 1, -a):.4f}  escalado {correlacion(ts - 1, -a):.4f}")
        res = {m: por_angulo(t, a, esquinas) for m, t in
               (("exacto", te), ("ingenuo", ti), ("escalado", ts))}
        filas = []
        for lo, hi in zip(bordes[:-1], bordes[1:]):
            sel = (res["exacto"][:, 0] >= lo) & (res["exacto"][:, 0] < hi)
            if sel.any():
                filas.append((lo, hi, int(sel.sum()),
                               *(res[m][sel, 1].mean() for m in ("exacto", "ingenuo", "escalado"))))
                print(f"      {lo:4.1f}-{hi:4.1f} grados ({sel.sum():3d} teselas): exacto "
                      f"{filas[-1][3]:.4f}  sin ventanas {filas[-1][4]:.4f}  escalado {filas[-1][5]:.4f}")
        tabla[entrada] = filas
        liberar_memoria()
    guardar_campo(N, tabla)


def guardar_campo(N, tabla):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SALIDA.mkdir(parents=True, exist_ok=True)
    with open(SALIDA / f"campo_N{N}.csv", "w", encoding="utf-8") as fh:
        fh.write("entrada,grados_desde,grados_hasta,teselas,exacto,sin_ventanas,escalado\n")
        for entrada, filas in tabla.items():
            for f in filas:
                fh.write(f"{entrada},{f[0]},{f[1]},{f[2]},{f[3]:.6f},{f[4]:.6f},{f[5]:.6f}\n")

    # tres series: las tres primeras ranuras categóricas, validadas por pares
    # (ΔE CVD 9.2, normal 24.0); la aguamarina queda bajo 3:1 sobre el fondo,
    # así que cada serie lleva además su etiqueta al final y su marcador
    estilo = {"exacto": ("#2a78d6", "o"), "sin ventanas": ("#eb6834", "s"),
              "escalado": ("#1baf7a", "^")}
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for eje, (entrada, filas) in zip(ejes, tabla.items()):
        x = [(f[0] + f[1]) / 2 for f in filas]
        finales = []
        for q, (metodo, (color, marca)) in enumerate(estilo.items()):
            y = [f[3 + q] for f in filas]
            eje.plot(x, y, color=color, lw=2, marker=marca, ms=6, label=metodo)
            finales.append([y[-1], metodo])
        # etiquetas al final de cada línea, separadas si caen encima
        finales.sort()
        for q in range(1, len(finales)):
            finales[q][0] = max(finales[q][0], finales[q - 1][0] + 0.03)
        for y_et, metodo in finales:
            eje.annotate(metodo, (x[-1], y_et), xytext=(8, 0), textcoords="offset points",
                         va="center", fontsize=9, color="#52514e")
        eje.set_facecolor("#fcfcfb")
        eje.set_title(f"entrada: {entrada}", fontsize=10, color="#0b0b0b")
        eje.set_xlabel("ángulo desde la fuente [grados]", color="#52514e")
        eje.grid(color="#e4e3df", lw=0.8)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
        eje.set_xlim(0, max(x) * 1.18)
    ejes[0].set_ylabel("correlación con el objeto (por tesela)", color="#52514e")
    ejes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(f"Reconstrucción DLHM, N = {N}: MPASM exacto, sin ventanas y escalado paraxial",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(SALIDA / f"campo_N{N}.png", dpi=150, facecolor=fig.get_facecolor())
    print(f"   -> {SALIDA / f'campo_N{N}.csv'} y .png")


# ------------------------------------------------------------ cadena de Carlos
def parte_carlos(N=1024):
    sys.path.insert(0, str(RAIZ / "scripts"))
    import mi_prueba_dlhm as mp

    print(f"== contra reconstruir() de mi_prueba_dlhm.py (cadena de Carlos), N = {N} ==")
    a, _ = mosaico(N)
    V = a_numpy(dlhm.holograma(1 - a, DO, LAMB, Z, L, (N, N), DELTA, complejo=True))
    X = dlhm.coordenadas(N, DELTA)
    fN = (np.arange(N) - N / 2) / (N * DELTA)
    # malla_remuestreada ya redondea al par; la impar es la que daba antes,
    # int(N·of) = 2027 en N = 1024, y se rehace aquí para medir lo que costaba
    par = mp.malla_remuestreada(N, N, Z, LAMB, DELTA)[0]
    impar = par - 1
    for entrada, c in (("campo complejo", 1 + V), ("intensidad", np.abs(1 + V) ** 2)):
        R = dlhm.Reconstructor(c, DELTA, LAMB, L, Z)
        print(f"   {entrada}:")
        for nombre, Nr, remuestreo in (("tal cual (cv.resize lineal, malla par)", par, None),
                                       ("cv.resize lineal, malla impar", impar, "lineal"),
                                       ("banda limitada, malla impar", impar, "banda"),
                                       ("banda limitada, malla par", par, "banda")):
            p = DELTA * N / Nr
            xn = (np.arange(Nr) - Nr / 2) * p
            if remuestreo is None:
                Rec = mp.reconstruir(c, Z, LAMB, DELTA, L, sobremuestreo=None)
            elif remuestreo == "lineal":
                cu = mp.resize(c.astype(complex), Nr, Nr)
                cu = cu if np.iscomplexobj(c) else cu.real
                Rec = mp.reconstruir(cu, Z, LAMB, p, L, sobremuestreo=1)
            else:
                Dm = np.exp(2j * np.pi * np.outer(xn, fN)) @ np.exp(-2j * np.pi * np.outer(fN, X)) / N
                cu = Dm @ c @ Dm.T
                cu = cu if np.iscomplexobj(c) else cu.real
                Rec = mp.reconstruir(cu, Z, LAMB, p, L, sobremuestreo=1)
            U0 = mp.espectro_angular(mp.fuente_puntual(Nr, Nr, L, 0, 0, LAMB, p),
                                     N * DELTA, N * DELTA, K, L - Z)
            dentro = np.flatnonzero(np.abs(xn) < N * DO / 2)
            n0, k = dentro[0], dentro.size
            tc = (Rec / np.abs(U0) ** 2)[np.ix_(dentro, dentro)]
            te = a_numpy(R.campo(Z, paso=p, forma=(k, k), centro=xn[n0] + k / 2 * p))
            print(f"      {nombre:42s}: corr con la exacta {correlacion(tc - 1, te - 1):.4f}")


# --------------------------------------------------- barrido de enfoque
def nitidez(I):
    """Energía del gradiente normalizada por la media (la de
    retro_holograma.py), sobre el cuarto central de la malla, EN PÍXELES.

    En píxeles y no en milímetros porque la malla de la muestra es delta·z/L:
    reconstruido a otra z, el objeto sale con otro tamaño físico pero ocupa
    los mismos píxeles (xi/paso = X/delta). Un recorte de tamaño físico fijo
    abarca más objeto a z menor y la métrica gana el extremo bajo del
    barrido; la ventana entera mete la periferia, donde se acumulan la
    gemela y el borde, y gana los dos extremos. Medido en N = 2048 con paso
    0.05 mm: el cuarto central tiene el máximo en z = 3.00 (3359, frente a
    <= 3119 en el resto); la ventana entera, en 3.30."""
    n = I.shape[0]
    I = I[3 * n // 8:5 * n // 8, 3 * n // 8:5 * n // 8]
    gy, gx = np.gradient(I)
    return float((gx**2 + gy**2).sum() / max(I.mean(), 1e-300) ** 2)


def parte_barrido(N=2048):
    """El uso de verdad: un holograma de intensidad y z desconocida. Se busca el
    foco por nitidez, sin mirar el objeto, con el Reconstructor (espectro una
    vez) y con el escalado. El objeto es el BenchmarkTarget del modelo de
    Carlos, absorbente: t = 1 - 0.7·img."""
    import cv2 as cv

    ruta = (RAIZ / "referencia" / "carlos" / "DLHM-model-main" / "DLHM-model-main" / "data"
            / "BenchmarkTarget.png")
    if not ruta.exists():
        print(f"== barrido: no esta {ruta}, se salta ==")
        return
    img = cv.imread(str(ruta), cv.IMREAD_GRAYSCALE).astype(float) / 255
    lado = min(img.shape)
    img = cv.resize(img[:lado, :lado], (N, N), interpolation=cv.INTER_AREA)
    t = 1 - 0.7 * img
    print(f"== barrido de enfoque, N = {N}: BenchmarkTarget a z = {Z} mm, holograma de intensidad ==")
    c = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (N, N), DELTA))
    liberar_memoria()
    zs = np.round(np.arange(2.70, 3.3001, 0.01), 4)
    t0 = time.perf_counter()
    R = dlhm.Reconstructor(c, DELTA, LAMB, L, (zs[0], zs[-1]))
    sincronizar()
    t1 = time.perf_counter()
    ne = [nitidez(np.abs(a_numpy(R.campo(z))) ** 2) for z in zs]
    t2 = time.perf_counter()
    en_foco = a_numpy(R.campo(Z)).real
    del R
    liberar_memoria()
    t3 = time.perf_counter()
    ns = [nitidez(np.abs(a_numpy(dlhm.reconstruir_escalado(c, DELTA, LAMB, z, L))) ** 2)
          for z in zs]
    t4 = time.perf_counter()
    for nombre, curva, tt in (("exacto", ne, (t1 - t0, t2 - t1)), ("escalado", ns, (0.0, t4 - t3))):
        i = int(np.argmax(curva))
        # vértice de la parábola por los tres puntos del pico: resolución por debajo del paso
        if 0 < i < len(zs) - 1:
            y0, y1, y2 = curva[i - 1:i + 2]
            dz = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (zs[1] - zs[0])
        else:
            dz = 0.0
        print(f"   {nombre:8s}: foco en z = {zs[i] + dz:.4f} mm (verdad {Z}); {len(zs)} planos en "
              f"{tt[1]:.1f} s" + (f" + {tt[0]:.1f} s de construir" if tt[0] else ""))
    print(f"   exacto en el foco: correlacion de Re(t) con el objeto {correlacion(en_foco - en_foco.mean(), t - t.mean()):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("partes", nargs="*", default=["ida", "campo", "carlos", "barrido"],
                    choices=["ida", "campo", "carlos", "barrido"])
    ap.add_argument("--n", type=int, default=2048, help="lado del sensor en campo y barrido")
    args = ap.parse_args()
    for parte in args.partes:
        {"ida": parte_ida, "campo": lambda: parte_campo(args.n), "carlos": parte_carlos,
         "barrido": lambda: parte_barrido(args.n)}[parte]()
