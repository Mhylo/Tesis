"""Barrido del objetivo 2 (tarea 30): el error de cada propagador en la malla de
la tarea 29 (CamposT/barrido.py), contra Rayleigh-Sommerfeld radial.

QUÉ SE MIDE. En cada punto (z, NA) de la malla, el problema de onda plana
equivalente de DLHM: N x N puntos de paso delta·z/L propagados a
z_eq = z(L - z)/L. Cada propagador lo resuelve y se compara con la RS-I en
toda la ventana N x N, que es el sensor.

EL OBJETO. Una mancha gaussiana de cintura w0 = 2.5 píxeles de la malla
delta/M, CINTURA_PX. Es casi el detalle más fino que esa malla representa sin
aliasing (su espectro vale e^{-(1.25 pi)^2} = 2e-7 en Nyquist), así que
difracta hasta el borde de la banda en cualquier punto: es casi el peor caso
que el holograma puede codificar. Con 2 píxeles (5e-5 en Nyquist) el aliasing
del objeto muestreado ponía un piso común de 3.9e-5 en z = 8, N = 1420; con
2.5 baja a 1.8e-5 y con 3 ya no baja más, porque ahí manda la referencia.
Y es radial, que es lo que la RS radial exige. Se propaga el campo
dispersado, sin el fondo: un fondo constante en una ventana finita difracta
en los bordes, y FFT-ASM, que la trata como periódica, no tiene bordes.

LA REFERENCIA. rs1_perfil con el perfil ANALÍTICO de la gaussiana, integrado
hasta SOPORTE·w0 (e^{-49}): sin pasar por la interpolación de píxeles, que
con un objeto de dos píxeles se come varios por ciento. Los nodos los fija la
fase del núcleo (n_rho_soporte, n_phi_soporte) con un suelo radial de
N_RHO_MIN, que son ~150 nodos por cintura. La salida radial (2N radios hasta la
esquina) se lleva al plano con un spline cúbico DESPUÉS de quitarle la fase de
la onda esférica que sale del centro: lo que queda varía despacio, y el spline
converge a 1e-11. En cada punto se repite la referencia con el doble de nodos;
su diferencia, err_ref, es el piso de ese punto: por debajo no se mide nada.

Esta referencia destapó el 2026-10-07 un signo invertido en el término 1/r de
referencias.nucleo_rs1 (ver su docstring): un piso de 2/(k·z_eq), 8e-5 en el
banco, en todos los propagadores a la vez.

LOS PROPAGADORES. FFT-ASM y BLAS con relleno de ceros x PAD, como los usa
pipeline.propagar(); MPASM sin relleno, que no lo necesita, con cada (s, K_f)
de la malla. Los puntos con s·K_f < 1 no se calculan (mpasm() los rechaza) y
van al CSV con valido = False. Todo en GPU en complex64, el dtype por defecto:
medido en el banco, complex128 da el mismo error a tres cifras.

LAS MÉTRICAS. err_rel = ||U - U_ref|| / ||U_ref|| sobre el campo COMPLEJO, que
ve la fase, y sam_dB (metricas.sam, la del paper), que solo ve la forma de la
intensidad. t_s es el tiempo del propagador, sin la referencia.

SALIDA. resultados/barrido_obj2/barrido.csv, una fila por cálculo, escrita en
cuanto termina. Relanzar retoma donde se quedó. --figuras escribe
mapas_metodos y mapas_kf (PNG y PDF); --resumen UMBRAL, la tabla por
configuración.

RESULTADO (2026-10-07, 1078 cálculos en 73 min en la RTX 3050). El piso de la
referencia, err_ref, va de 3.0e-6 (z = 1.5) a 7.4e-5 (z = 8, N = 2822).
Puntos de 98 con error <= 1e-3 (<= 1e-2 entre paréntesis), K_f automático:

    FFT-ASM x2   79 (84)       MPASM s = 1   78 (92)
    BLAS x2      78 (85)       MPASM s = 2   90 (97)
                               MPASM s = 3   95 (98)

- La frontera de FFT-ASM va en z·NA: baja de 1e-3 desde z = 5.5 mm con NA
  0.03, 3.5 con 0.06, 2.5 con la del pinhole y 1.5 con 0.25. Donde el campo
  difractado se sale de la ventana rellenada, sus réplicas se superponen.
- De los 19 puntos en que FFT-ASM pasa de 1e-3, MPASM rescata 11 con s = 2 y
  16 con s = 3. Los 3 que s = 3 no rescata son NA 0.03 con z <= 2.5 mm.
- K_f: el automático es el bueno. Con 0.5x o 2x ninguna configuración baja de
  1e-3 en ningún punto (salvo 2 de s = 3 con 0.5x). Por debajo del automático
  el espectro se evalúa más allá de lo que la ventana resuelve y aparecen
  réplicas; por encima, se recorta la banda.
- EN EL BANCO (z = 3, NA del pinhole) todos valen: FFT-ASM 2.2e-6, BLAS
  1.2e-5, MPASM 1.7e-4 / 2.9e-6 / 3.0e-6 con s = 1 / 2 / 3. El 38 % de banda
  de FFT-ASM que da CamposT.barrido (la Ec. 12 de Zhao) NO predice el fallo
  con este objeto: con el relleno x2 el campo cabe en la ventana.
- t_s incluye el plan de cuFFT en la primera llamada de cada tamaño: no sirve
  para comparar costes (para eso, scripts/comparacion.py).
- LA FRONTERA ES EL CRITERIO DE VENTANA (--frontera, 2026-10-21). FFT-ASM x2
  se aparta del piso en 26 celdas, y son exactamente las de cociente
  z_eq/[N(delta/M)^2/lambda] >= 3.17; el resto queda en <= 3.20. Con relleno
  x2, la luz del borde de la banda que sale del centro reentra al sensor
  cuando el cociente llega a 2·2 - 1 = 3. Igual de nítida con 2x, 3x o 5x el
  piso como umbral, y con el cociente sin aproximar (cae en 3.9).

    Tesis_env/Scripts/python.exe -m scripts.barrido_obj2             # todo
    Tesis_env/Scripts/python.exe -m scripts.barrido_obj2 --rapido    # 4 puntos
    Tesis_env/Scripts/python.exe -m scripts.barrido_obj2 --frontera  # la 4.2.2
"""

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from CamposT.backend import (a_dispositivo, a_numpy, get_xp, liberar_memoria,  # noqa: E402
                             sincronizar)
from CamposT.barrido import MALLA                                              # noqa: E402
from CamposT.metricas import sam                                               # noqa: E402
from CamposT.pipeline import crop_center, pad_field                            # noqa: E402
from CamposT.propagadores import blas, fft_asm, mpasm                          # noqa: E402
from CamposT.referencias import n_phi_soporte, n_rho_soporte, rs1_perfil      # noqa: E402

SALIDA = RAIZ / "resultados" / "barrido_obj2"
CSV = SALIDA / "barrido.csv"

#: cintura del objeto, en píxeles de la malla delta/M
CINTURA_PX = 2.5
#: el perfil se integra hasta SOPORTE·w0
SOPORTE = 7.0
#: suelo de nodos radiales de la referencia
N_RHO_MIN = 1024
#: relleno de FFT-ASM y BLAS, el de pipeline.propagar()
PAD = 2

COLUMNAS = ["z", "N", "NA", "M", "paso_um", "z_eq", "banda_fft", "metodo", "s",
            "kf_factor", "kf", "valido", "err_rel", "sam_dB", "t_s", "err_ref"]


def objeto(n, paso):
    """La gaussiana de cintura CINTURA_PX píxeles, en la malla del paquete."""
    w0 = CINTURA_PX * paso
    x = (np.arange(n) - n / 2) * paso
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / w0**2).astype(np.complex128), w0


def referencia(n, paso, lamb, z, w0, refinar=1, device="auto"):
    """La RS-I del objeto en la ventana n x n, en complex128 y en CPU."""
    k = 2 * np.pi / lamb
    soporte = SOPORTE * w0
    rho_out = np.linspace(0.0, np.hypot(n / 2, n / 2) * paso, 2 * n)
    n_rho = refinar * max(N_RHO_MIN, n_rho_soporte(rho_out[-1], soporte, lamb, z))
    n_phi = refinar * n_phi_soporte(rho_out[-1], soporte, lamb, z)
    u = a_numpy(rs1_perfil(lambda r: np.exp(-r**2 / w0**2), soporte, rho_out, lamb, z,
                           n_rho_in=n_rho, n_phi=n_phi, device=device,
                           dtype=np.complex128))
    # sin la fase de la esférica que sale del centro lo que queda va despacio
    v = u * np.exp(-1j * k * np.hypot(rho_out, z))
    re, im = CubicSpline(rho_out, v.real), CubicSpline(rho_out, v.imag)
    x = (np.arange(n) - n / 2) * paso
    R = np.hypot(*np.meshgrid(x, x))
    return (re(R) + 1j * im(R)) * np.exp(1j * k * np.hypot(R, z))


def propagar(punto, U0, lamb, device="auto"):
    """El campo del propagador del punto, en NumPy."""
    n, paso, z = punto["N"], punto["paso"], punto["z_eq"]
    if punto["metodo"] == "mpasm":
        U, _ = mpasm(U0, paso, lamb, z, s=punto["s"], Kf=punto["kf"], device=device)
        return U
    xp, _ = get_xp(device)
    funcion = fft_asm if punto["metodo"] == "fft_asm" else blas
    U = funcion(pad_field(a_dispositivo(U0, xp), PAD, xp), paso, lamb, z, device=device)
    return crop_center(U, n, n)


def error_relativo(U, ref):
    """||U - ref|| / ||ref|| sobre el campo complejo, en float64."""
    U = a_numpy(U).astype(np.complex128)
    return float(np.linalg.norm(U - ref) / np.linalg.norm(ref))


def _clave(fila):
    """Lo que identifica un cálculo, para retomar sin repetir. Vale para un
    punto de la malla y para una fila leída del CSV, donde None es ""."""
    texto = lambda v: "" if v is None or v == "" else str(float(v))
    return (float(fila["z"]), int(fila["N"]), fila["metodo"], texto(fila.get("s")),
            texto(fila.get("kf_factor")))


def hechos():
    """Las claves que ya están en el CSV."""
    if not CSV.exists():
        return set()
    with CSV.open(newline="") as f:
        return {_clave(fila) for fila in csv.DictReader(f)}


def puntos(rapido):
    """Los cálculos de la malla; con rapido, solo 2 z x 2 NA."""
    todos = list(MALLA.puntos())
    if not rapido:
        return todos
    zs, ns = (MALLA.z_banco, max(MALLA.z)), (min(MALLA.n), MALLA.n[MALLA.na.index(MALLA.na_banco)])
    return [p for p in todos if p["z"] in zs and p["N"] in ns]


def correr(rapido=False, device="auto"):
    SALIDA.mkdir(parents=True, exist_ok=True)
    ya = hechos()
    pendientes = [p for p in puntos(rapido) if _clave(p) not in ya]
    print(f"{len(pendientes)} cálculos pendientes ({len(ya)} ya en {CSV.name})")
    nuevo = not CSV.exists()
    with CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo:
            w.writeheader()
        actual, t0 = None, time.time()
        for i, p in enumerate(pendientes):
            if (p["z"], p["N"]) != actual:
                actual = (p["z"], p["N"])
                prob = MALLA.problema(p["z"], p["N"])
                U0, w0 = objeto(p["N"], p["paso"])
                tr = time.time()
                ref = referencia(p["N"], p["paso"], MALLA.lamb, p["z_eq"], w0, device=device)
                fina = referencia(p["N"], p["paso"], MALLA.lamb, p["z_eq"], w0, refinar=2,
                                  device=device)
                err_ref = float(np.linalg.norm(fina - ref) / np.linalg.norm(fina))
                ref = fina
                print(f"[{time.time() - t0:7.0f} s] z = {p['z']:.1f}  N = {p['N']}: "
                      f"referencia en {time.time() - tr:.1f} s, err_ref = {err_ref:.1e}",
                      flush=True)
            fila = {"z": p["z"], "N": p["N"], "NA": prob["NA"], "M": prob["M"],
                    "paso_um": p["paso"] * 1e3, "z_eq": p["z_eq"],
                    "banda_fft": prob["fraccion_banda_fft"], "metodo": p["metodo"],
                    "s": p["s"], "kf_factor": p.get("kf_factor"), "kf": p["kf"],
                    "valido": p["valido"], "err_ref": err_ref}
            if p["valido"]:
                sincronizar()
                ti = time.perf_counter()
                U = propagar(p, U0, MALLA.lamb, device=device)
                sincronizar()
                fila["t_s"] = time.perf_counter() - ti
                fila["err_rel"] = error_relativo(U, ref)
                fila["sam_dB"] = sam(U, ref)
                del U
                liberar_memoria()
            w.writerow(fila)
            f.flush()
    print(f"hecho en {time.time() - t0:.0f} s: {CSV}")


# ------------------------------------------------------------------- figuras
#: rampa secuencial de un solo tono, claro = error bajo, oscuro = error alto
RAMPA = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
#: log10 del error: de 1e-6 (el piso de la referencia) a 1 (nada que ver)
ESCALA = (-6.0, 0.0)
#: dos umbrales candidatos de "precisión aceptable"; el de la tesis lo fija el
#: asesor, así que se dibujan los dos y no se elige ninguno
UMBRALES = (1e-3, 1e-2)


def leer():
    with CSV.open(newline="") as f:
        return list(csv.DictReader(f))


#: un error a menos de PISO veces err_ref no se distingue de la referencia
PISO = 2.0


def _mapa(filas, metodo, s=None, kf_factor=None):
    """log10(err_rel) como matriz NA x z, con NaN donde no hay dato, la
    máscara de los puntos fuera por construcción y la de los que están en el
    piso de la referencia."""
    zs, nas = list(MALLA.z), list(MALLA.n)
    E = np.full((len(nas), len(zs)), np.nan)
    fuera = np.zeros_like(E, dtype=bool)
    piso = np.zeros_like(E, dtype=bool)
    for r in filas:
        if r["metodo"] != metodo:
            continue
        if s is not None and (r["s"] != str(s) or float(r["kf_factor"]) != kf_factor):
            continue
        i, j = nas.index(int(r["N"])), zs.index(float(r["z"]))
        if r["valido"] == "False":
            fuera[i, j] = True
        elif r["err_rel"]:
            E[i, j] = np.log10(max(float(r["err_rel"]), 10 ** ESCALA[0]))
            piso[i, j] = float(r["err_rel"]) <= PISO * float(r["err_ref"])
    return E, fuera, piso


def _panel(ax, E, fuera, piso, titulo, cmap):
    from matplotlib.patches import Rectangle
    zs = np.array(MALLA.z)
    im = ax.imshow(E, origin="lower", aspect="auto", cmap=cmap, vmin=ESCALA[0],
                   vmax=ESCALA[1], extent=(zs[0] - 0.25, zs[-1] + 0.25, -0.5, len(MALLA.na) - 0.5),
                   interpolation="nearest")
    # fuera por construccion: gris con trama, no un color de la rampa
    for i, j in zip(*np.nonzero(fuera)):
        ax.add_patch(Rectangle((zs[j] - 0.25, i - 0.5), 0.5, 1, facecolor="#e8e7e3",
                               edgecolor="#b8b7b2", hatch="///", linewidth=0))
    # en el piso de la referencia: el error medido es el de la referencia
    filas_p, cols_p = np.nonzero(piso)
    ax.plot(zs[cols_p], filas_p, linestyle="none", marker="o", markersize=2.5,
            color="#5f5e5a")
    Z, A = np.meshgrid(zs, np.arange(len(MALLA.na)))
    if np.isfinite(E).sum() > 3:
        cs = ax.contour(Z, A, np.where(np.isfinite(E), E, ESCALA[1]),
                        levels=[np.log10(u) for u in UMBRALES],
                        colors=["#1f1f1d", "#1f1f1d"], linestyles=["-", "--"], linewidths=1.2)
        ax.clabel(cs, fmt={np.log10(u): f"{u:.0e}" for u in UMBRALES}, fontsize=7)
    ib = MALLA.na.index(MALLA.na_banco)
    ax.plot(MALLA.z_banco, ib, marker="o", markersize=9, markerfacecolor="none",
            markeredgecolor="#c2410c", markeredgewidth=2)
    ax.set_title(titulo, fontsize=10, color="#1f1f1d")
    ax.set_yticks(range(len(MALLA.na)))
    ax.set_yticklabels([f"{a:.3g}" for a in MALLA.na], fontsize=8, color="#5f5e5a")
    ax.tick_params(axis="x", labelsize=8, colors="#5f5e5a")
    for lado in ax.spines.values():
        lado.set_color("#d3d1c7")
    return im


def figuras():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", RAMPA)
    cmap.set_bad("#ffffff")
    filas = leer()

    paneles = [("FFT-ASM (relleno x2)", "fft_asm", None, None),
               ("BLAS (relleno x2)", "blas", None, None)]
    paneles += [(f"MPASM, s = {s}, $K_f$ automático", "mpasm", s, 1.0) for s in MALLA.s]
    fig, ejes = plt.subplots(1, len(paneles), figsize=(3.2 * len(paneles), 3.6), sharey=True,
                             constrained_layout=True)
    for ax, (titulo, m, s, f) in zip(ejes, paneles):
        im = _panel(ax, *_mapa(filas, m, s, f), titulo, cmap)
        ax.set_xlabel("z, fuente a muestra (mm)", fontsize=9, color="#5f5e5a")
    ejes[0].set_ylabel("NA de la ventana", fontsize=9, color="#5f5e5a")
    barra = fig.colorbar(im, ax=ejes, shrink=0.9, pad=0.01)
    barra.set_label("log$_{10}$ error relativo del campo", fontsize=9, color="#5f5e5a")
    fig.suptitle("Error frente a Rayleigh-Sommerfeld en la malla del banco DLHM "
                 "(línea continua 1e-3, discontinua 1e-2; círculo: el banco; "
                 "punto: en el piso de la referencia)",
                 fontsize=10, color="#1f1f1d")
    for ext in ("png", "pdf"):
        fig.savefig(SALIDA / f"mapas_metodos.{ext}", dpi=200)
    plt.close(fig)

    fig, ejes = plt.subplots(len(MALLA.s), len(MALLA.kf_factor),
                             figsize=(3.2 * len(MALLA.kf_factor), 2.9 * len(MALLA.s)),
                             sharex=True, sharey=True, constrained_layout=True)
    for a, s in enumerate(MALLA.s):
        for b, f in enumerate(MALLA.kf_factor):
            im = _panel(ejes[a, b], *_mapa(filas, "mpasm", s, f),
                        f"MPASM, s = {s}, $K_f$ = {f:g} x automático", cmap)
        ejes[a, 0].set_ylabel("NA de la ventana", fontsize=9, color="#5f5e5a")
    for b in range(len(MALLA.kf_factor)):
        ejes[-1, b].set_xlabel("z, fuente a muestra (mm)", fontsize=9, color="#5f5e5a")
    barra = fig.colorbar(im, ax=ejes, shrink=0.6, pad=0.01)
    barra.set_label("log$_{10}$ error relativo del campo", fontsize=9, color="#5f5e5a")
    fig.suptitle("MPASM: el efecto de s y de $K_f$ (gris con trama: s·$K_f$ < 1, "
                 "fuera por construcción)", fontsize=10, color="#1f1f1d")
    for ext in ("png", "pdf"):
        fig.savefig(SALIDA / f"mapas_kf.{ext}", dpi=200)
    plt.close(fig)
    print(f"figuras en {SALIDA}")


def resumen(umbral):
    """Por configuración: en cuántos de los 98 puntos el error queda por
    debajo de umbral, y el error en el banco."""
    filas = leer()
    grupos = {}
    for r in filas:
        clave = (r["metodo"], r["s"], r["kf_factor"])
        grupos.setdefault(clave, []).append(r)
    print(f"umbral {umbral:g}: puntos que lo cumplen de {len(MALLA.z) * len(MALLA.na)}, "
          f"y el error en el banco (z = {MALLA.z_banco}, NA = {MALLA.na_banco})")
    nb = MALLA.n[MALLA.na.index(MALLA.na_banco)]
    for (m, s, f), rs in sorted(grupos.items()):
        ok = sum(1 for r in rs if r["err_rel"] and float(r["err_rel"]) <= umbral)
        banco = [r for r in rs if float(r["z"]) == MALLA.z_banco and int(r["N"]) == nb]
        eb = banco[0]["err_rel"] if banco and banco[0]["err_rel"] else "-"
        eb = f"{float(eb):.1e}" if eb != "-" else eb
        print(f"  {m:8s} s={s or '-':2s} Kf x{f or '-':4s}  {ok:3d}   banco {eb}")


def cociente(z, n, exacto=False):
    """z_eq / [N (delta/M)^2 / lambda], el cociente del criterio de ventana,
    del punto (z, n). Paraxial, como en el texto; con exacto, dividido por la
    raíz de la Ec. (zlim) del manuscrito."""
    paso = MALLA.delta * z / MALLA.L
    q = z * (MALLA.L - z) / MALLA.L / (n * paso**2 / MALLA.lamb)
    return q / np.sqrt(1 - (MALLA.lamb / (2 * paso)) ** 2) if exacto else q


#: FFT-ASM x2 "se sale" de la ventana donde su error pasa de SALE veces el
#: piso de la referencia (con 2 o 5 salen las mismas celdas)
SALE = 3.0


def frontera():
    """El criterio de ventana sobre el mapa. Con relleno x2 la ventana mide 2N:
    la luz que sale del centro del objeto recorre 1.5 ventanas antes de
    reentrar al sensor por el borde opuesto, y la del borde de la banda, que se
    desvía lambda z_eq/(2 delta/M), lo hace con cociente 2·2 - 1 = 3. Imprime
    dónde se aparta FFT-ASM x2 del piso, qué cociente tienen esas celdas y qué
    hacen ahí s y K_f. Los números de la Sección 4.2.2."""
    filas = leer()
    err = {(r["metodo"], r["s"], r["kf_factor"], float(r["z"]), int(r["N"])):
           float(r["err_rel"]) for r in filas if r["err_rel"]}
    piso = {(float(r["z"]), int(r["N"])): float(r["err_ref"]) for r in filas}
    na = dict(zip(MALLA.n, MALLA.na))
    fft = lambda c: err[("fft_asm", "", "", *c)]
    mp = lambda s, c, f="1.0": err.get(("mpasm", str(s), f, *c), np.nan)
    sale = sorted(c for c in piso if fft(c) > SALE * piso[c])
    queda = [c for c in piso if c not in sale]

    print(f"piso de la referencia: {min(piso.values()):.1e} a {max(piso.values()):.1e}")
    for exacto in (False, True):
        q = lambda c: cociente(*c, exacto=exacto)
        print(f"FFT-ASM x2 > {SALE:g} x piso en {len(sale)} de {len(piso)} celdas; cociente "
              f"{'sin aproximar' if exacto else 'paraxial'}: >= {min(map(q, sale)):.2f} "
              f"en ellas, <= {max(map(q, queda)):.2f} en el resto")
    print(f"  error / piso: <= {max(fft(c) / piso[c] for c in queda):.2f} en el resto, "
          f">= {min(fft(c) / piso[c] for c in sale):.1f} en ellas; error máximo "
          f"{max(map(fft, sale)):.2f}")
    for n in MALLA.n:
        zs = [z for z, nn in sale if nn == n]
        print(f"  NA {na[n]:.3f}: z = {', '.join(f'{z:g}' for z in zs) or '-'}")

    for s in (2, 3):
        r = [fft(c) / mp(s, c) for c in sale]
        print(f"FFT-ASM x2 / MPASM s = {s} en esas celdas: {min(r):.1f} a {max(r):.0f} "
              f"(mediana {np.median(r):.0f})")
    print("MPASM s = 3 por encima de 1e-3 en (z, NA):",
          [(z, round(na[n], 3)) for z, n in sorted(piso) if mp(3, (z, n)) > 1e-3])
    print(f"MPASM s = 1 peor que FFT-ASM x2 en {sum(mp(1, c) > fft(c) for c in piso)} "
          f"de {len(piso)} celdas")
    for z, n in ((3.0, MALLA.n[MALLA.na.index(0.06)]),):
        print(f"  z = {z:g}, NA 0.06: FFT-ASM x2 {fft((z, n)):.1e}, MPASM s = 1/2/3 "
              + " / ".join(f"{mp(s, (z, n)):.1e}" for s in MALLA.s))

    for s in MALLA.s:
        for f in ("0.5", "2.0"):
            r = np.array([mp(s, c, f) / mp(s, c) for c in piso])
            r = r[np.isfinite(r)]
            print(f"K_f = {f} x automático, s = {s}: {r.size} celdas válidas; empeora entre "
                  f"{r.min():.1f} y {r.max():.0e} veces (mediana {np.median(r):.0f}), "
                  f"mejora en {(r < 1).sum()}")
    kf = {(r["s"], float(r["z"]), int(r["N"])): float(r["kf"]) for r in filas
          if r["metodo"] == "mpasm" and r["kf_factor"] == "1.0"}
    nb = MALLA.n[MALLA.na.index(MALLA.na_banco)]
    print(f"K_f automático en el banco (z = {MALLA.z_banco:g}, NA {MALLA.na_banco}): "
          + ", ".join(f"s = {s}: {kf[(str(s), MALLA.z_banco, nb)]:.2f}" for s in MALLA.s)
          + f"; con NA {MALLA.na[-1]}: {kf[('1', MALLA.z_banco, MALLA.n[-1])]:.2f} (s = 1)")
    (s, z, n), v = max(kf.items(), key=lambda t: t[1])
    print(f"K_f automático máximo: {v:.2f} (s = {s}, z = {z:g}, NA {na[n]:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rapido", action="store_true", help="solo 2 z x 2 NA, para probar")
    ap.add_argument("--device", default="auto", help="gpu | cpu | auto")
    ap.add_argument("--figuras", action="store_true", help="solo las figuras, desde el CSV")
    ap.add_argument("--resumen", type=float, metavar="UMBRAL",
                    help="solo la tabla de puntos bajo UMBRAL, desde el CSV")
    ap.add_argument("--frontera", action="store_true",
                    help="solo el criterio de ventana sobre el mapa, desde el CSV")
    args = ap.parse_args()
    if args.figuras:
        figuras()
    elif args.frontera:
        frontera()
    elif args.resumen:
        resumen(args.resumen)
    else:
        correr(args.rapido, args.device)
