"""Propaga una imagen con scripts/mpasm_minimo.py y guarda entrada y salida."""

import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mpasm_minimo import cp, kf, mpasm

RAIZ = pathlib.Path(__file__).resolve().parents[1]

RUTA = pathlib.Path(r"C:\Users\User\Desktop\Tesis\referencia\carlos\DLHM-model-main\DLHM-model-main\data\BenchmarkTarget.png")   # la imagen es la AMPLITUD, fase 0
LAMB = 633e-6       # mm
DELTA = 3.45e-3     # mm, paso de pixel
Z = 200.0           # mm
S = 1
GPU = cp is not None   # True/False para forzar; con True y sin CuPy aborta
SALIDA = RAIZ / "resultados" / "propagar_imagen" / f"z{Z:g}mm_s{S}.png"

img = plt.imread(RUTA).astype(float)
if img.ndim == 3:
    img = img[..., :3].mean(axis=2)
U0 = (img / img.max()).astype(complex)
M, N = U0.shape

t0 = time.perf_counter()
U = mpasm(U0, DELTA, LAMB, Z, s=S, gpu=GPU)
U = U.get() if GPU else U
t = time.perf_counter() - t0
Kfy, Kfx = kf(M, DELTA, LAMB, Z, S), kf(N, DELTA, LAMB, Z, S)
print(f"{RUTA.name}: {M}x{N}, lambda {LAMB * 1e6:.0f} nm, delta {DELTA * 1e3:.2f} um, "
      f"z {Z:g} mm, s {S}, Kf (y, x) = ({Kfy:.3f}, {Kfx:.3f})")
print(f"{'GPU' if GPU else 'CPU'}: {t:.2f} s")

ext = [-N / 2 * DELTA, N / 2 * DELTA, -M / 2 * DELTA, M / 2 * DELTA]
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
for a, dato, titulo in zip(
        ax,
        (np.abs(U0) ** 2, np.abs(U) ** 2, np.angle(U)),
        ("|U0|^2  (z = 0)", f"|U|^2  (z = {Z:g} mm)", f"fase de U  (z = {Z:g} mm)")):
    im = a.imshow(dato, extent=ext, cmap="twilight" if "fase" in titulo else "gray")
    a.set_title(titulo)
    a.set_xlabel("x [mm]")
    fig.colorbar(im, ax=a, fraction=0.046)
ax[0].set_ylabel("y [mm]")
fig.suptitle(f"MPASM minimo  ·  lambda {LAMB * 1e6:.0f} nm  ·  delta {DELTA * 1e3:.2f} um"
             f"  ·  s = {S}  ·  Kf = ({Kfy:.3f}, {Kfx:.3f})")
fig.tight_layout()
SALIDA.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SALIDA, dpi=120)
print("->", SALIDA)
