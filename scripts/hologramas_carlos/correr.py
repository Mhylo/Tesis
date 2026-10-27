"""Las corridas sobre los hologramas de Carlos. Escriben en resultados/hologramas_carlos/.

    python -m scripts.hologramas_carlos.correr foco
    python -m scripts.hologramas_carlos.correr calibracion
    python -m scripts.hologramas_carlos.correr grupo5
    python -m scripts.hologramas_carlos.correr exacto
    python -m scripts.hologramas_carlos.correr zL

foco         Barrido grueso de D (5 a 200 mm, paso 0.5) en la captura 1 de cada
             carpeta y fino (paso 0.1) alrededor: +-7 mm en las 16 capturas de
             amplitud, +-9 mm en las impares de fase. Recorte central de
             2048x2560 y Tamura en su cuadrado central de 1024. D va en la
             malla del sensor con el pixel SUPUESTO de 1.83 um: con otro pixel
             delta, el mismo foco es D (delta/1.83)^2.
             -> focos_D.csv, curvas_<conjunto>.npz
calibracion  M con el USAF (cajas_usaf.json) y el ajuste conjunto del pixel, L y
             la platina. -> calibracion.json, L_implicito.csv
grupo5       El grupo 5 del USAF de amplitud, que no entra en la calibracion,
             medido igual en z9/1 y z17/5 como comprobacion de la escala
             (cajas_usaf.json["amp_grupo5"]). -> grupo5.csv
exacto       El motor exacto del repo (CamposT.dlhm.Reconstructor) con la
             geometria adoptada (DELTA_EXP, L_EXP) en las 17 posiciones de los
             dos conjuntos, captura 1. -> focos_exactos.csv
zL           El foco en z del Reconstructor para varias L, sobre amp z9/1: la
             cresta z = L^2/(D + L). -> z_frente_a_L.csv
imagenes     t en el foco del exacto para las figuras. -> imagen_<conjunto>_z<k>.npy

La geometria adoptada es INFERIDA (ver __init__): pixel 3.80 um y L = 18.229 mm.
"""

import argparse
import csv
import json
import pathlib
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from CamposT.backend import a_numpy
from CamposT.dlhm import Reconstructor
from scripts.hologramas_carlos import datos, foco, geometria, usaf

RAIZ = pathlib.Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "resultados" / "hologramas_carlos"
CAJAS = pathlib.Path(__file__).with_name("cajas_usaf.json")

LAMB = 528e-6               #: mm, la de Carlos (CamposT/montaje.py)
DELTA_SUPUESTO = 1.83e-3    #: mm, el pixel con que se expresan D y M medidos
DELTA_EXP = 3.80e-3         #: mm, el pixel adoptado (inferido)
L_EXP = 18.229              #: mm, L con ese pixel (ajuste conjunto)
KS = range(1, 18)
RECORTE = (2048, 2560)
ROI = 1024
CAPTURAS = {"amp": [str(i) for i in range(1, 17)], "phase": [str(i) for i in range(1, 17, 2)]}
SEMI_FINO = {"amp": 7.0, "phase": 9.0}
SIGMA_MAX = 10.0            #: px: ajustes de barras con borde mas ancho no cuentan
EM_MIN = 0.01               #: error relativo minimo de M (dispersion entre grupos)


def recorte(a):
    M, N = a.shape
    f0, c0 = M // 2 - RECORTE[0] // 2, N // 2 - RECORTE[1] // 2
    return a[f0:f0 + RECORTE[0], c0:c0 + RECORTE[1]]


def ventana_roi():
    return (RECORTE[0] // 2 - ROI // 2, RECORTE[1] // 2 - ROI // 2, ROI, ROI)


def liberar(xp):
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()


def curva_D(conjunto, k, cap, Ds):
    h, r = datos.leer(conjunto, k, cap)
    P = foco.Paraxial(recorte(foco.contraste(h, r)), DELTA_SUPUESTO, LAMB)
    c = foco.barrer(P, Ds, ventana_roi(), datos.SIGNO[conjunto])
    xp = P.xp
    del P
    liberar(xp)
    return c


def leer_focos():
    with open(SALIDA / "focos_D.csv") as f:
        return [dict(fila) for fila in csv.DictReader(f)]


def D_de(focos, conjunto, k, cap):
    return next(float(f["D"]) for f in focos if f["conjunto"] == conjunto
                and int(f["k"]) == int(k) and f["captura"] == str(cap))


def por_carpeta(focos, conjunto):
    """-> ks, mediana de D y su error (1.2533 sd/sqrt(n), piso 0.05 mm)."""
    ks, med, err = [], [], []
    for k in KS:
        D = np.array([float(f["D"]) for f in focos if f["conjunto"] == conjunto
                      and int(f["k"]) == k])
        ks.append(k)
        med.append(np.median(D))
        err.append(max(1.2533 * D.std(ddof=1) / np.sqrt(D.size), 0.05))
    return np.array(ks), np.array(med), np.array(err)


# ------------------------------------------------------------------ foco
def correr_foco():
    SALIDA.mkdir(parents=True, exist_ok=True)
    filas = []
    for conjunto in ("amp", "phase"):
        Dg = np.arange(5.0, 200.01, 0.5)
        guardar = {"D_grueso": Dg}
        D0 = []
        for k in KS:
            c = curva_D(conjunto, k, "1", Dg)
            guardar[f"grueso_z{k}"] = c
            D0.append(foco.pico(Dg, c))
            print(f"{conjunto} z{k}: grueso {D0[-1]:.1f}", flush=True)
        centros = foco.centros_robustos(np.array(list(KS)), np.array(D0), tolerancia=8.0)
        for k, d0, cen in zip(KS, D0, centros):
            semi = SEMI_FINO[conjunto]
            Df = np.arange(cen - semi, cen + semi + 0.05, 0.1)
            guardar[f"D_fino_z{k}"] = Df
            for cap in CAPTURAS[conjunto]:
                c = curva_D(conjunto, k, cap, Df)
                guardar[f"fino_z{k}_{cap}"] = c
                D = foco.pico_fino(Df, c)
                filas.append(dict(conjunto=conjunto, k=k, captura=cap, D=f"{D:.3f}",
                                  D_grueso=f"{d0:.2f}", centro=f"{cen:.2f}"))
            Ds = [float(f["D"]) for f in filas if f["conjunto"] == conjunto and f["k"] == k]
            print(f"{conjunto} z{k}: centro {cen:.1f}, D mediana {np.median(Ds):.2f}", flush=True)
        np.savez(SALIDA / f"curvas_{conjunto}.npz", **guardar)
        with open(SALIDA / "focos_D.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)


# ------------------------------------------------------------------ calibracion
def medir_M(conjunto, k, cap, elementos, D):
    h, r = datos.leer(conjunto, k, cap)
    P = foco.Paraxial(foco.contraste(h, r), DELTA_SUPUESTO, LAMB)
    A = gaussian_filter(a_numpy(P.xp.abs(P.campo(D))).astype(np.float64), 1.0)
    xp = P.xp
    del P
    liberar(xp)
    Ms = []
    for g, e, o, *caja in elementos:
        w, s = usaf.ajustar_barras(usaf.perfil(A, caja, o))
        M = usaf.magnificacion(w, g, e, DELTA_SUPUESTO * 1e3)
        Ms.append((g, e, o, w, s, M))
    return Ms


def correr_calibracion():
    focos = leer_focos()
    cajas = json.loads(CAJAS.read_text())["amp"]
    kM, MM, eMM, DM, detalle = [], [], [], [], {}
    for k, v in sorted(cajas.items(), key=lambda kv: int(kv[0])):
        D = D_de(focos, "amp", k, v["captura"])
        Ms = medir_M("amp", k, v["captura"], v["elementos"], D)
        nitidos = np.array([m[5] for m in Ms if m[4] <= SIGMA_MAX])
        M, eM = nitidos.mean(), max(nitidos.std(ddof=1) / np.sqrt(nitidos.size),
                                     EM_MIN * nitidos.mean())
        kM.append(int(k))
        MM.append(M)
        eMM.append(eM)
        DM.append(D)
        detalle[k] = dict(captura=v["captura"], D=D, M=M, eM=eM, n=int(nitidos.size),
                          elementos=[dict(g=g, e=e, o=o, w=w, sigma=s, M=m)
                                     for g, e, o, w, s, m in Ms])
        print(f"z{k}/{v['captura']}: D = {D:.2f}, M = {M:.3f} +- {eM:.3f} "
              f"({nitidos.size} elementos nitidos)", flush=True)
    ks, Dmed, eD = por_carpeta(focos, "amp")
    args = (ks, Dmed, eD, kM, MM, eMM, DM, DELTA_SUPUESTO)
    libre = geometria.ajuste_conjunto(*args)
    p183 = geometria.ajuste_conjunto(*args, p_fijo=1.0)
    p380 = geometria.ajuste_conjunto(*args, p_fijo=DELTA_EXP / DELTA_SUPUESTO)
    for nom, r in (("libre", libre), ("pixel 1.83", p183), ("pixel 3.80", p380)):
        print(f"{nom}: p = {r['p']:.4f} +- {r['e_p']:.4f} (delta {r['delta'] * 1e3:.3f} um), "
              f"L = {r['L']:.3f} +- {r['e_L']:.3f}, z1 = {r['z1']:.4f}, dz = {r['dz']:.5f} "
              f"+- {r['e_dz']:.5f}, chi2/dof = {r['chi2_dof']:.2f}")
    json.dump(dict(magnificaciones=detalle, libre=libre, pixel_183=p183, pixel_380=p380),
              open(SALIDA / "calibracion.json", "w"), indent=1)
    with open(SALIDA / "L_implicito.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "D", "M", "eM", "L_183", "eL_183", "L_380", "eL_380"])
        for k, D, M, eM in zip(kM, DM, MM, eMM):
            fila = [k, f"{D:.3f}", f"{M:.4f}", f"{eM:.4f}"]
            for p in (1.0, DELTA_EXP / DELTA_SUPUESTO):
                Dr, Mr = p**2 * D, p * M
                fila += [f"{Dr / (Mr - 1):.4f}", f"{Dr / (Mr - 1) ** 2 * p * eM:.4f}"]
            w.writerow(fila)


# ------------------------------------------------------------------ grupo 5
def correr_grupo5():
    """Comprobacion de la escala con el grupo 5 del USAF de amplitud, que no
    entra en la calibracion (lo pidio Carlos al revisar, 26/10). Se mide como
    en correr_calibracion (medir_M y el mismo criterio de borde) y se compara
    con la M de los otros grupos en esa posicion (calibracion.json) y con la
    que predice la geometria adoptada con el pixel nominal, 1/p + p D/L
    (Ec. M-D del manuscrito)."""
    focos = leer_focos()
    cajas = json.loads(CAJAS.read_text())["amp_grupo5"]
    cal = json.load(open(SALIDA / "calibracion.json"))["magnificaciones"]
    p = DELTA_EXP / DELTA_SUPUESTO
    filas = []
    for k, v in sorted(cajas.items(), key=lambda kv: int(kv[0])):
        D = D_de(focos, "amp", k, v["captura"])
        Ms = medir_M("amp", k, v["captura"], v["elementos"], D)
        for g, e, o, w, s, m in Ms:
            filas.append(dict(k=k, captura=v["captura"], g=g, e=e, o=o, w=f"{w:.2f}",
                              sigma=f"{s:.2f}", M=f"{m:.4f}", nitido=int(s <= SIGMA_MAX)))
        nitidos = np.array([m[5] for m in Ms if m[4] <= SIGMA_MAX])
        if nitidos.size < 2:
            print(f"z{k}/{v['captura']}: {nitidos.size} de {len(Ms)} con borde nitido; "
                  f"no se promedia", flush=True)
            continue
        M, eM = nitidos.mean(), nitidos.std(ddof=1) / np.sqrt(nitidos.size)
        M_geo = 1 / p + p * D / L_EXP
        print(f"z{k}/{v['captura']}: grupo 5 M = {M:.3f} +- {eM:.3f} ({nitidos.size} de "
              f"{len(Ms)} con borde nitido); otros grupos {cal[k]['M']:.3f} "
              f"({100 * (M / cal[k]['M'] - 1):+.1f} %); geometria adoptada {M_geo:.3f} "
              f"({100 * (M / M_geo - 1):+.1f} %)", flush=True)
    with open(SALIDA / "grupo5.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


# ------------------------------------------------------------------ motor exacto
def barrido_exacto(conjunto, k, cap, D, L, delta, semi, n):
    """-> (zs, curva orientada, segundos por plano). D en pixel de 1.83 um."""
    h, r = datos.leer(conjunto, k, cap)
    esc = (delta / DELTA_SUPUESTO) ** 2
    zs = np.sort(geometria.z_de(np.linspace(D - semi, D + semi, n) * esc, L))
    R = Reconstructor(recorte(h), delta, LAMB, L, (zs[0], zs[-1]), fondo=recorte(r), piso=0.05)
    xp = R.xp
    s = 1.0 if datos.SIGNO[conjunto] == "max" else -1.0
    t0 = time.perf_counter()
    curva = np.array([s * foco.tamura(xp.abs(R.campo(z, forma=(ROI, ROI))).astype(np.float64), xp)
                      for z in zs])
    dt = (time.perf_counter() - t0) / n
    del R
    liberar(xp)
    return zs, curva, dt


def correr_exacto():
    focos = leer_focos()
    filas = []
    for conjunto, semi, n in (("amp", 2.5, 51), ("phase", 6.0, 61)):
        for k in KS:
            D = D_de(focos, conjunto, k, "1")
            zs, c, dt = barrido_exacto(conjunto, k, "1", D, L_EXP, DELTA_EXP, semi, n)
            z = foco.pico_fino(zs, c)
            filas.append(dict(conjunto=conjunto, k=k, z=f"{z:.5f}",
                              M=f"{L_EXP / z:.3f}", tamura=f"{abs(c.max()):.4f}",
                              cociente=f"{geometria.cociente_ventana(z, L_EXP, DELTA_EXP, 4640, LAMB):.3f}",
                              ms_por_plano=f"{1e3 * dt:.1f}"))
            print(f"{conjunto} z{k}: z = {z:.4f} mm", flush=True)
    with open(SALIDA / "focos_exactos.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


IMAGENES = {"amp": (1, 5, 9, 17), "phase": (1, 9, 17)}


def correr_imagenes():
    """t en el foco del motor exacto (recorte de 2048x2560, captura 1) para las
    figuras: imagen_<conjunto>_z<k>.npy, complex64."""
    zex = {}
    with open(SALIDA / "focos_exactos.csv") as f:
        for fila in csv.DictReader(f):
            zex[(fila["conjunto"], int(fila["k"]))] = float(fila["z"])
    for conjunto, ks in IMAGENES.items():
        for k in ks:
            h, r = datos.leer(conjunto, k, "1")
            z = zex[(conjunto, k)]
            R = Reconstructor(recorte(h), DELTA_EXP, LAMB, L_EXP, z, fondo=recorte(r), piso=0.05)
            np.save(SALIDA / f"imagen_{conjunto}_z{k}.npy", a_numpy(R.campo(z)).astype(np.complex64))
            xp = R.xp
            del R
            liberar(xp)
            print(f"{conjunto} z{k}: imagen a z = {z:.4f} mm", flush=True)


def correr_zL():
    focos = leer_focos()
    D = D_de(focos, "amp", 9, "1")
    filas = []
    for delta, Ls in ((DELTA_EXP, (12, 15, L_EXP, 22, 26)),
                      (DELTA_SUPUESTO, (5, 6, 8, 10, 12, 15, 20, 30))):
        for L in Ls:
            zs, c, _ = barrido_exacto("amp", 9, "1", D, float(L), delta, 3.0, 41)
            z = foco.pico_fino(zs, c)
            filas.append(dict(delta_um=f"{delta * 1e3:.2f}", L=f"{L:.3f}", z=f"{z:.5f}",
                              D=f"{geometria.D_de(z, L):.2f}", tamura=f"{c.max():.4f}"))
            print(filas[-1], flush=True)
    with open(SALIDA / "z_frente_a_L.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("que", choices=("foco", "calibracion", "grupo5", "exacto", "zL", "imagenes"))
    {"foco": correr_foco, "calibracion": correr_calibracion, "grupo5": correr_grupo5,
     "exacto": correr_exacto, "zL": correr_zL,
     "imagenes": correr_imagenes}[a.parse_args().que]()
