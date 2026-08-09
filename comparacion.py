"""Comparación sistemática CPU vs GPU de los propagadores.

Genera las tablas y figuras del Objetivo 1: cómo escala el tiempo de cómputo
con el tamaño de malla N y con el sobremuestreo s, en cada dispositivo, y qué
parte de la ganancia viene del hardware y qué parte de la precisión.

Tres cuidados para que la comparación sea honesta:

1. Se miden las cuatro combinaciones dispositivo × precisión. El salto de
   CPU-complex128 a GPU-complex64 mezcla dos efectos distintos: la GPU y el
   cambio de precisión. La columna 'x_iso' compara complex64 contra complex64,
   que es la aceleración atribuible sólo al dispositivo.

2. Se separa el cómputo de la transferencia. La columna 'gpu+tr' incluye subir
   el campo por PCIe; 'gpu' parte de un campo que ya está en la tarjeta. En una
   cadena de reconstrucción donde el campo se queda en GPU entre pasos, la
   cifra relevante es la segunda.

3. Todos los tiempos se toman con sincronización explícita y descartando una
   pasada de calentamiento (inicialización de cuBLAS y planes FFT).

Uso:
    python -m CamposT.comparacion                 # barridos por defecto
    python -m CamposT.comparacion --n 256 512 1024 --smax 8
    python -m CamposT.comparacion --sin-figuras
"""

import argparse
import csv
import os
import platform
import time

import numpy as np

from CamposT import backend as bk
from CamposT.backend import a_dispositivo, gpu_disponible, liberar_memoria
from CamposT.campos import usaf_like
from CamposT.propagadores import blas, fft_asm, memoria_mpasm, mpasm, sam

#: parámetros del montaje DLHM (mm)
DELTA = 3.45e-3
LAMB = 405e-6
Z = 150.0

#: por encima de esto se deja de medir la CPU para ese método: el barrido
#: crecería a horas sin aportar nada que la extrapolación no diga ya
LIMITE_CPU = 90.0

CONFIGS = [("cpu", np.complex128), ("cpu", np.complex64),
           ("gpu", np.complex128), ("gpu", np.complex64)]


# ------------------------------------------------------------------- medición
def medir(fn, *args, minimo=0.05, max_rep=10, **kw):
    """Tiempo medio de fn, repitiendo hasta acumular al menos `minimo` segundos.

    Las operaciones en GPU bajan a pocos milisegundos, donde una sola medida es
    puro ruido de reloj y de lanzamiento de kernel.
    """
    _, t = bk.cronometrar(fn, *args, **kw)
    if t < minimo:
        rep = min(max_rep, max(2, int(minimo / max(t, 1e-6))))
        _, t = bk.cronometrar(fn, *args, repeticiones=rep, calentar=False, **kw)
    return t


def propagador(metodo, s=1):
    """Devuelve una función homogénea f(U0, device, dtype) -> campo."""
    if metodo == "mpasm":
        return lambda U0, device, dtype: mpasm(U0, DELTA, LAMB, Z, s=s,
                                               device=device, dtype=dtype)[0]
    fn = fft_asm if metodo == "fft" else blas
    return lambda U0, device, dtype: fn(U0, DELTA, LAMB, Z,
                                        device=device, dtype=dtype)


def cabe_en_gpu(metodo, N, s, dtype):
    if metodo != "mpasm":
        return True
    mem = bk.memoria_gpu()
    if mem is None:
        return False
    return memoria_mpasm(N, N, s, 1, dtype) < 0.85 * mem[0]


def punto(metodo, N, s, saltar_cpu):
    """Mide un (método, N, s) en las cuatro combinaciones. Devuelve una fila.

    `saltar_cpu` es el conjunto de dtypes de CPU ya descartados por lentos; se
    actualiza in situ, porque el coste crece monótonamente con N y con s.
    """
    fn = propagador(metodo, s)
    U0 = usaf_like(N).astype(np.complex128)
    fila = {"metodo": metodo, "N": N, "s": s}
    campos, tiempos = {}, {}

    for device, dtype in CONFIGS:
        clave = f"{device}_{8 * np.dtype(dtype).itemsize}"   # complex64 -> 8 B -> 64
        if device == "cpu" and dtype in saltar_cpu:
            continue
        if device == "gpu" and (not gpu_disponible()
                                or not cabe_en_gpu(metodo, N, s, dtype)):
            continue
        try:
            U = a_dispositivo(U0, bk.get_xp(device)[0], dtype)
            tiempos[clave] = medir(fn, U, device, dtype)
            campos[clave] = fn(U, device, dtype)
        except MemoryError:
            continue
        finally:
            liberar_memoria()
        if device == "cpu" and tiempos[clave] > LIMITE_CPU:
            saltar_cpu.add(dtype)

    # extremo a extremo: incluye subir el campo por PCIe
    if gpu_disponible() and cabe_en_gpu(metodo, N, s, np.complex64):
        U_host = U0.astype(np.complex64)
        try:
            tiempos["gpu_64_tr"] = medir(fn, U_host, "gpu", np.complex64)
        except MemoryError:
            pass
        liberar_memoria()

    fila.update({k: round(v, 6) for k, v in tiempos.items()})
    if "cpu_128" in tiempos and "gpu_64" in tiempos:
        fila["x"] = round(tiempos["cpu_128"] / tiempos["gpu_64"], 1)
    if "cpu_64" in tiempos and "gpu_64" in tiempos:
        fila["x_iso"] = round(tiempos["cpu_64"] / tiempos["gpu_64"], 1)
    if "cpu_128" in campos and "gpu_64" in campos:
        fila["sam"] = sam(campos["gpu_64"], campos["cpu_128"])
    if "gpu_64" in tiempos and "gpu_64_tr" in tiempos:
        fila["frac_transf"] = round(
            1 - tiempos["gpu_64"] / tiempos["gpu_64_tr"], 3)
    campos.clear()
    liberar_memoria()
    return fila


# -------------------------------------------------------------------- barridos
def barrido_N(lista_N, metodos=("fft", "blas", "mpasm")):
    """Escalado con el tamaño de malla, a sobremuestreo fijo s=1."""
    filas = []
    for metodo in metodos:
        saltar = set()
        for N in lista_N:
            filas.append(punto(metodo, N, 1, saltar))
            imprimir_fila(filas[-1])
    return filas


def barrido_s(lista_s, N=512):
    """Escalado con el sobremuestreo de MPASM, a malla fija.

    Es el eje que de verdad importa: s controla la exactitud del método y su
    coste crece como s², así que es donde la GPU decide si s alto es viable.
    """
    filas, saltar = [], set()
    for s in lista_s:
        filas.append(punto("mpasm", N, s, saltar))
        imprimir_fila(filas[-1])
    return filas


# -------------------------------------------------------------------- salidas
COLUMNAS = ["metodo", "N", "s", "cpu_128", "cpu_64", "gpu_128", "gpu_64",
            "gpu_64_tr", "x", "x_iso", "frac_transf", "sam"]


def imprimir_cabecera():
    print(f"{'método':>7} {'N':>5} {'s':>3} {'CPU c128':>9} {'CPU c64':>9} "
          f"{'GPU c128':>9} {'GPU c64':>9} {'+transf':>9} {'x':>6} "
          f"{'x_iso':>6} {'SAM':>9}")


def imprimir_fila(f):
    """Una raya donde no hubo medida: CPU descartada por lenta, o la
    configuración no cabe en la tarjeta."""
    def celda(clave, ancho, fmt):
        return f"{f[clave]:{ancho}{fmt}}" if clave in f else f"{'—':>{ancho}}"

    print(f"{f['metodo']:>7} {f['N']:5d} {f['s']:3d} "
          + " ".join(celda(k, 9, ".4f") for k in
                     ("cpu_128", "cpu_64", "gpu_128", "gpu_64", "gpu_64_tr"))
          + " " + celda("x", 6, ".1f") + " " + celda("x_iso", 6, ".1f")
          + " " + celda("sam", 9, ".2e"))


def guardar_csv(filas, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    print(f"  -> {path}")


def graficar(filas_N, filas_s, carpeta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [("cpu_128", "CPU complex128", "o-", "#1f77b4"),
              ("cpu_64", "CPU complex64", "s--", "#7fb1e0"),
              ("gpu_128", "GPU complex128", "^-", "#d62728"),
              ("gpu_64", "GPU complex64", "v--", "#ff9896")]

    def panel(ax, filas, eje, titulo):
        for clave, etiqueta, estilo, color in series:
            pts = [(f[eje], f[clave]) for f in filas if clave in f]
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, estilo, color=color, label=etiqueta, ms=5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(eje)
        ax.set_ylabel("tiempo [s]")
        ax.set_title(titulo)
        ax.grid(True, which="both", alpha=0.3)

    metodos = sorted({f["metodo"] for f in filas_N})
    fig, axes = plt.subplots(1, len(metodos) + 1,
                             figsize=(4.2 * (len(metodos) + 1), 3.8))
    for ax, metodo in zip(axes, metodos):
        panel(ax, [f for f in filas_N if f["metodo"] == metodo], "N",
              f"{metodo.upper()} — escalado con N (s=1)")
    panel(axes[-1], filas_s, "s", f"MPASM — escalado con s (N={filas_s[0]['N']})")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(carpeta, "tiempos.png")
    fig.savefig(p, dpi=150)
    print(f"  -> {p}")

    # aceleración: hardware solo (x_iso) frente a hardware + precisión (x)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for metodo in metodos:
        sub = [f for f in filas_N if f["metodo"] == metodo]
        for clave, estilo in (("x", "o-"), ("x_iso", "s--")):
            pts = [(f["N"], f[clave]) for f in sub if clave in f]
            if pts:
                x, y = zip(*pts)
                axes[0].plot(x, y, estilo, ms=5,
                             label=f"{metodo} — {'c128→c64' if clave == 'x' else 'c64→c64'}")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("N")
    axes[0].set_title("Aceleración GPU/CPU vs N (s=1)")

    for clave, estilo in (("x", "o-"), ("x_iso", "s--")):
        pts = [(f["s"], f[clave]) for f in filas_s if clave in f]
        if pts:
            x, y = zip(*pts)
            axes[1].plot(x, y, estilo, ms=5,
                         label="c128→c64" if clave == "x" else "c64→c64")
    axes[1].set_xlabel("s")
    axes[1].set_title(f"MPASM — aceleración vs s (N={filas_s[0]['N']})")

    for ax in axes:
        ax.axhline(1, color="k", lw=0.8, ls=":")
        ax.set_ylabel("× respecto a CPU")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(carpeta, "aceleracion.png")
    fig.savefig(p, dpi=150)
    print(f"  -> {p}")


def entorno():
    """Contexto sin el cual las cifras no son reproducibles ni comparables."""
    info = {
        "cpu": platform.processor() or platform.machine(),
        "nucleos": os.cpu_count(),
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    gi = bk.info_gpu()
    if gi:
        info.update({"gpu": gi["nombre"], "vram_gb": round(gi["VRAM total [GB]"], 1),
                     "cupy": gi["cupy"]})
    return info


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[256, 512, 1024, 2048],
                    help="tamaños de malla del barrido en N")
    ap.add_argument("--smax", type=int, default=8,
                    help="sobremuestreo máximo del barrido en s (potencias de 2)")
    ap.add_argument("--ns", type=int, default=512, help="N fijo del barrido en s")
    ap.add_argument("--salida", default="resultados", help="carpeta de salida")
    ap.add_argument("--sin-figuras", action="store_true")
    args = ap.parse_args()

    print("Entorno:")
    for k, v in entorno().items():
        print(f"  {k:>8}: {v}")
    if not gpu_disponible():
        print("\nSin GPU: sólo se medirá la CPU y no habrá comparación.")
    print(f"\nParámetros: delta = {DELTA * 1e3:.2f} µm, lamb = {LAMB * 1e6:.0f} nm, "
          f"z = {Z:.0f} mm, sin relleno de ceros.")
    print(f"Se deja de medir la CPU de un método al superar {LIMITE_CPU:.0f} s.\n")

    t0 = time.perf_counter()
    print("Escalado con el tamaño de malla (s=1):")
    imprimir_cabecera()
    filas_N = barrido_N(args.n)

    lista_s = [2**i for i in range(int(np.log2(args.smax)) + 1)]
    print(f"\nEscalado con el sobremuestreo de MPASM (N={args.ns}):")
    imprimir_cabecera()
    filas_s = barrido_s(lista_s, N=args.ns)

    print(f"\nTotal: {time.perf_counter() - t0:.0f} s. Escribiendo resultados:")
    guardar_csv(filas_N, os.path.join(args.salida, "escalado_N.csv"))
    guardar_csv(filas_s, os.path.join(args.salida, "escalado_s.csv"))
    if not args.sin_figuras:
        graficar(filas_N, filas_s, args.salida)


if __name__ == "__main__":
    main()
