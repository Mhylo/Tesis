"""El MPASM esferico (CamposT.dlhm) frente a la cadena de Carlos, sobre los hologramas de Carlos.

    python -m scripts.hologramas_carlos.comparar focos      # las 17 posiciones x 2 conjuntos
    python -m scripts.hologramas_carlos.comparar metricas   # z1, z9, z17 de amplitud
    python -m scripts.hologramas_carlos.comparar tiempos    # tiempo por plano y memoria

Los dos motores sobre el MISMO recorte central cuadrado del sensor, con la
geometria adoptada (correr.DELTA_EXP, correr.L_EXP) y la misma normalizacion por
amplitud con ref.jpg (piso 0.05):
    exacto  CamposT.dlhm.Reconstructor sobre el recorte.
    Carlos  scripts.mi_prueba_dlhm.MotorCarlos con recorte = el mismo lado,
            holograma 1 + c1 y raiz del fondo, como en mi_prueba_dlhm.main().
El lado lo fija la memoria de Carlos: su malla remuestreada crece con el ANCHO
del recorte (6 arrays de n_r^2, ver mi_prueba_dlhm.comprobar_memoria), y con el
pixel de 3.8 um y z = 0.41 mm un recorte de 1024 pide mas que la GPU de 4 GB.

focos     LA COMPARACION ES EN EL FOCO COMUN: el z del exacto en el recorte grande
          (focos_exactos.csv), que cuadra con la platina a paso fijo con 4 um
          rms. El foco fisico es uno, y la Tamura de un recorte de 512 px no es
          fiable para buscarlo: a M = 44 el recorte ve ~44 um de muestra, y el
          propio exacto se iba +14 %. En ese plano se miden la Tamura de cada
          motor, su tiempo por plano y, en amp z1, z9 y z17, se guarda t (.npy).
          En amplitud, ademas, el foco PROPIO de cada motor por Tamura (9 planos
          en +-10 % y 5 alrededor del maximo), marcado si cae en el borde. Los
          planos son pocos porque cada uno le cuesta ~11-14 s a la cadena del
          laboratorio en la GPU de 4 GB: evalua la esferica en float64 sobre su
          malla remuestreada (5000^2 a z = 0.4 mm), y esa tarjeta es lenta en
          doble. -> carlos_vs_exacto.csv y, para
          amp z1, z9 y z17, t en el foco de cada motor (.npy).
metricas  En amp z1, z9 y z17, la misma zona del sensor reconstruida tres veces:
          el exacto y la cadena del laboratorio en el recorte comun, y el exacto
          en su recorte grande (2048x2560, imagen_amp_z<k>.npy) recortado a esa
          zona. Por cada elemento del USAF (cajas_metricas.json, en pixeles del
          recorte comun) el contraste de Michelson en las tres, y su frecuencia
          en la muestra con el ajuste de tres barras sobre la imagen nitida (la
          grande). Ademas, la correlacion de |t| entre las dos cadenas en el
          recorte comun y el SNR de una caja de fondo. -> metricas_carlos.csv
tiempos   Tiempo por plano de cada cadena en el recorte comun y del exacto en el
          recorte grande (2048x2560), con un plano de calentamiento -los planes
          de FFT- y la GPU sincronizada; mediana de 3. Y la memoria que pediria
          la cadena del laboratorio con recortes mayores. -> tiempos_carlos.csv
"""

import argparse
import csv
import json
import pathlib
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from CamposT.backend import a_numpy, dtype_por_defecto, get_xp, memoria_gpu, sincronizar
from CamposT.dlhm import Reconstructor, contraste_amplitud
from scripts import mi_prueba_dlhm as mp
from scripts.hologramas_carlos import correr, datos, foco, geometria, usaf

SALIDA = correr.SALIDA
CAJAS = pathlib.Path(__file__).with_name("cajas_metricas.json")
CANDIDATOS = (1024, 768, 512, 384)
PISO = 0.05
PLANOS, ANCHO_REL = 9, 0.10
GUARDAR = (1, 9, 17)


def michelson(perfil, suave=2.0):
    """(max - min)/(max + min) del perfil suavizado."""
    p = gaussian_filter1d(np.asarray(perfil, float), suave)
    return float((p.max() - p.min()) / (p.max() + p.min()))


def snr(A, caja):
    """media/desviacion de |A| en caja = (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = caja
    B = np.abs(np.asarray(A))[y0:y1, x0:x1]
    return float(B.mean() / B.std())


def memoria_carlos(lado, z, delta, lamb, itemsize=8):
    """GB de la cadena de Carlos para un recorte de lado px a la distancia z."""
    n = max(mp.malla_remuestreada(lado, lado, z, lamb, delta))
    return 6 * n * n * itemsize / 2**30


def recorte_carlos(z_min, delta, lamb, presupuesto_gb, candidatos=CANDIDATOS):
    """El mayor lado de candidatos que cabe a la z mas chica."""
    for lado in candidatos:
        if memoria_carlos(lado, z_min, delta, lamb) <= presupuesto_gb:
            return lado
    raise MemoryError(f"ni {candidatos[-1]} px caben en {presupuesto_gb:.2f} GB")


def _recorte(a, lado):
    M, N = a.shape
    f0, c0 = (M - lado) // 2, (N - lado) // 2
    return a[f0:f0 + lado, c0:c0 + lado]


def motores(conjunto, k, captura, z_rango, lado):
    """-> (exacto(z), carlos(z)), funciones que devuelven t en el recorte."""
    h, r = datos.leer(conjunto, k, captura)
    d, lamb, L = correr.DELTA_EXP, correr.LAMB, correr.L_EXP
    xp, dev = get_xp("auto")
    dtype = dtype_por_defecto(dev)
    R = Reconstructor(_recorte(h, lado), d, lamb, L, z_rango, fondo=_recorte(r, lado), piso=PISO)
    c1, raiz, raiz_min = contraste_amplitud(h, r, PISO)
    M, N = h.shape
    ventana = ((M - lado) // 2, (N - lado) // 2, lado, lado)
    C = mp.MotorCarlos(1 + c1, ventana, lamb, d, L, xp, dtype, None, recorte=lado,
                       raiz=raiz, raiz_min=raiz_min)
    return (lambda z: R.campo(z)), (lambda z: C.campo(z, ventana)), xp


def barrer(f, zs, signo, xp):
    s = 1.0 if signo == "max" else -1.0
    t0 = time.perf_counter()
    c = np.array([s * foco.tamura(xp.abs(f(z)).astype(np.float64), xp) for z in zs])
    return c, (time.perf_counter() - t0) / len(zs)


def barrer_grueso_fino(f, z0, signo, xp, planos=PLANOS, ancho=ANCHO_REL, finos=5):
    """-> (z del foco, Tamura en el foco, segundos por plano, borde). Grueso de
    `planos` en z0(1 +- ancho) y fino de `finos` en +- un paso grueso alrededor
    de su maximo. borde: el maximo grueso cayo en un extremo, o sea que el foco
    puede estar fuera del rango; la z devuelta nunca sale de lo barrido."""
    zg = z0 * (1 + np.linspace(-ancho, ancho, planos))
    cg, dg = barrer(f, zg, signo, xp)
    i = int(cg.argmax())
    paso = zg[1] - zg[0]
    zf = np.linspace(max(zg[i] - paso, zg[0]), min(zg[i] + paso, zg[-1]), finos)
    cf, df = barrer(f, zf, signo, xp)
    # maximo y parabola, sin quitar rampa: la ventana fina es del orden del
    # ancho del pico y una recta entre sus extremos lo correria
    j = int(np.clip(cf.argmax(), 1, cf.size - 2))
    a, b, c = cf[j - 1], cf[j], cf[j + 1]
    den = a - 2 * b + c
    z = zf[j] + (0.5 * (a - c) / den if den < 0 else 0.0) * (zf[1] - zf[0])
    z = float(np.clip(z, zf[0], zf[-1]))
    return z, float(abs(cf.max())), (dg + df) / 2, i in (0, planos - 1)


def liberar_todo(xp):
    """La memoria de un motor no vuelve sola: CuPy guarda los planes de FFT de
    la malla remuestreada de Carlos (5000^2) y el pool sus bloques."""
    import gc
    gc.collect()
    if xp is not np:
        xp.fft.config.get_plan_cache().clear()
        xp.get_default_memory_pool().free_all_blocks()


def leer_exactos():
    with open(SALIDA / "focos_exactos.csv") as fh:
        return {(f["conjunto"], int(f["k"])): float(f["z"]) for f in csv.DictReader(fh)}


def correr_focos():
    zex = leer_exactos()
    libre = memoria_gpu()[0] / 2**30 if get_xp("auto")[1] == "gpu" else 8.0
    lado = recorte_carlos(min(zex.values()) * (1 - ANCHO_REL), correr.DELTA_EXP, correr.LAMB,
                          0.8 * libre)
    print(f"recorte comun: {lado} px", flush=True)
    filas = []
    for conjunto in ("amp", "phase"):
        for k in correr.KS:
            z0 = zex[(conjunto, k)]
            margen = z0 * (ANCHO_REL + 2 * ANCHO_REL / (PLANOS - 1))
            ex, ca, xp = motores(conjunto, k, "1", (z0 - margen, z0 + margen), lado)
            signo = datos.SIGNO[conjunto]
            fila = dict(conjunto=conjunto, k=k, lado=lado, z=f"{z0:.5f}")
            for nombre, f in (("exacto", ex), ("carlos", ca)):
                t0 = time.perf_counter()
                t = f(z0)
                tam = foco.tamura(xp.abs(t).astype(np.float64), xp)
                fila[f"ms_{nombre}"] = f"{1e3 * (time.perf_counter() - t0):.1f}"
                fila[f"tamura_{nombre}"] = f"{tam:.4f}"
                if conjunto == "amp" and k in GUARDAR:
                    np.save(SALIDA / f"t_{nombre}_z{k}.npy", a_numpy(t).astype(np.complex64))
                del t
            for nombre, f in (("exacto", ex), ("carlos", ca)):
                if conjunto == "amp":
                    zp, _, _, borde = barrer_grueso_fino(f, z0, signo, xp)
                    fila[f"z_propio_{nombre}"] = f"{zp:.5f}"
                    fila[f"borde_{nombre}"] = int(borde)
                else:
                    fila[f"z_propio_{nombre}"], fila[f"borde_{nombre}"] = "", ""
            fila["cociente_recorte"] = f"{geometria.cociente_ventana(z0, correr.L_EXP, correr.DELTA_EXP, lado, correr.LAMB):.3f}"
            fila["cociente_sensor"] = f"{geometria.cociente_ventana(z0, correr.L_EXP, correr.DELTA_EXP, 4640, correr.LAMB):.3f}"
            filas.append(fila)
            print(filas[-1], flush=True)
            del ex, ca
            liberar_todo(xp)
            with open(SALIDA / "carlos_vs_exacto.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(filas[0]))
                w.writeheader()
                w.writerows(filas)


def zona_grande(k, lado):
    """La zona del recorte comun dentro de la imagen del recorte grande: los
    dos recortes estan centrados en el sensor y en la misma malla delta z/L."""
    G = np.load(SALIDA / f"imagen_amp_z{k}.npy")
    f0 = (G.shape[0] - lado) // 2
    c0 = (G.shape[1] - lado) // 2
    return G[f0:f0 + lado, c0:c0 + lado]


def correlacion(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def correr_metricas():
    cajas = json.loads(CAJAS.read_text())
    zex = leer_exactos()
    filas = []
    for k in GUARDAR:
        v = cajas[str(k)]
        M = correr.L_EXP / zex[("amp", k)]
        A = {m: np.abs(np.load(SALIDA / f"t_{m}_z{k}.npy")) for m in ("exacto", "carlos")}
        lado = A["exacto"].shape[0]
        A["grande"] = np.abs(zona_grande(k, lado))
        r = correlacion(A["exacto"], A["carlos"])
        for o, *caja in v["elementos"]:
            w, _ = usaf.ajustar_barras(usaf.perfil(A["grande"], caja, o))
            lp = M / (2 * w * correr.DELTA_EXP)       # lineas por mm en la muestra
            fila = dict(k=k, M=f"{M:.2f}", orient=o, caja=" ".join(map(str, caja)),
                        lp_mm=f"{lp:.1f}", periodo_um=f"{1e3 / lp:.2f}",
                        corr_exacto_carlos=f"{r:.4f}")
            for m in ("exacto", "carlos", "grande"):
                fila[f"contraste_{m}"] = f"{michelson(usaf.perfil(A[m], caja, o)):.4f}"
                fila[f"snr_{m}"] = f"{snr(A[m], v['fondo']):.3f}"
            filas.append(fila)
            print(fila, flush=True)
    with open(SALIDA / "metricas_carlos.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def _cronometro(f, z, xp, veces=3):
    f(z)                                    # calentamiento: planes de FFT
    sincronizar(xp)
    ts = []
    for _ in range(veces):
        t0 = time.perf_counter()
        f(z)
        sincronizar(xp)
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def correr_tiempos():
    zex = leer_exactos()
    with open(SALIDA / "carlos_vs_exacto.csv") as fh:
        lado = int(next(csv.DictReader(fh))["lado"])
    filas = []
    for k in GUARDAR:
        z0 = zex[("amp", k)]
        ex, ca, xp = motores("amp", k, "1", (z0, z0), lado)
        fila = dict(k=k, z=f"{z0:.5f}", lado=lado,
                    ms_exacto=f"{_cronometro(ex, z0, xp):.1f}",
                    ms_carlos=f"{_cronometro(ca, z0, xp):.1f}")
        del ex, ca
        liberar_todo(xp)
        h, r = datos.leer("amp", k, "1")
        R = Reconstructor(correr.recorte(h), correr.DELTA_EXP, correr.LAMB, correr.L_EXP, z0,
                          fondo=correr.recorte(r), piso=PISO)
        fila["ms_exacto_grande"] = f"{_cronometro(lambda z: R.campo(z), z0, xp):.1f}"
        fila["px_grande"] = correr.RECORTE[0] * correr.RECORTE[1]
        del R
        liberar_todo(xp)
        for n in (lado, 1024, 2048):
            fila[f"gb_carlos_{n}"] = f"{memoria_carlos(n, z0, correr.DELTA_EXP, correr.LAMB):.2f}"
        filas.append(fila)
        print(fila, flush=True)
    with open(SALIDA / "tiempos_carlos.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("que", choices=("focos", "metricas", "tiempos"))
    {"focos": correr_focos, "metricas": correr_metricas,
     "tiempos": correr_tiempos}[a.parse_args().que]()
