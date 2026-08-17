"""Exactitud de los propagadores contra la solución analítica: la tabla de la
Figura 4 del paper.

Barre z sobre siete órdenes de magnitud con los parámetros de la Tabla 1 y
mide SAM [dB] de cada método contra el gaussiano analítico. Más alto es mejor.

Es donde se ve el argumento del Objetivo 1: MPASM se mantiene entre 31 y 48 dB
en todo el rango, FFT-ASM se desploma a números negativos en cuanto el haz deja
de caber en la ventana, y BL-ASM se degrada y luego se recupera cuando su
máscara de banda empieza a actuar. A z corto los tres empatan, o FFT-ASM gana:
cada método tiene su régimen, y ése es el resultado.

Imprime además el efecto del dtype en GPU y los tiempos, para que la exactitud
y el coste se lean juntos.

Uso:
    python -m scripts.exactitud

Pendiente (tarea 3): volcar la tabla a CSV y graficarla contra z para cerrar
la reproducción de la Figura 4.
"""

import numpy as np

from CamposT.backend import (cronometrar, gpu_disponible, info_gpu,
                             liberar_memoria)
from CamposT.metricas import sam
from CamposT.propagadores import blas, fft_asm, mpasm
from CamposT.referencias import gauss_analytic, gauss_beam


def main():

    # Parámetros de la Tabla 1 del paper (sin la lente)
    L0, N, w0, lamb = 5.0, 512, 1.0, 632.8e-6      # mm
    delta = L0 / (N - 1)
    Z = (500, 2000, 6000, 12000, 30000, 80000, 200000)

    if gpu_disponible():
        gi = info_gpu()
        print(f"GPU: {gi['nombre']} (cc {gi['capacidad']}), "
              f"{gi['VRAM total [GB]']:.1f} GB, CuPy {gi['cupy']}")
    else:
        print("Sin GPU: todo en CPU.")

    # --- exactitud contra la solución analítica -------------------------------
    U0_cpu, X, Y = gauss_beam(N, delta, w0, device="cpu")
    print("\nExactitud: SAM [dB] contra el gaussiano analítico, CPU complex128.")
    print("Más alto es mejor. Ec. (15) con la (16) corregida; ver alfa_sam().")
    print(f"{'z [mm]':>8} {'Kf':>7} {'MPASM':>9} {'FFT-ASM':>9} {'BL-ASM':>9}")
    for z in Z:
        ref = gauss_analytic(X, Y, w0, lamb, z)
        Um, Kf = mpasm(U0_cpu, delta, lamb, z, device="cpu")
        print(f"{z:8d} {Kf:7.2f} {sam(Um, ref):9.2f} "
              f"{sam(fft_asm(U0_cpu, delta, lamb, z, device='cpu'), ref):9.2f} "
              f"{sam(blas(U0_cpu, delta, lamb, z, device='cpu'), ref):9.2f}")

    # --- coste de la precisión simple en GPU ----------------------------------
    if gpu_disponible():
        print("\nEfecto del dtype en GPU (SAM [dB] contra el analítico):")
        print(f"{'z [mm]':>8} {'c64':>9} {'c128':>9} {'CPU c128':>9}")
        for z in Z:
            ref = gauss_analytic(X, Y, w0, lamb, z)
            U64, _ = mpasm(U0_cpu, delta, lamb, z, device="gpu", dtype=np.complex64)
            U128, _ = mpasm(U0_cpu, delta, lamb, z, device="gpu", dtype=np.complex128)
            Ucpu, _ = mpasm(U0_cpu, delta, lamb, z, device="cpu")
            print(f"{z:8d} {sam(U64, ref):9.2f} {sam(U128, ref):9.2f} "
                  f"{sam(Ucpu, ref):9.2f}")
            liberar_memoria()

        # --- tiempos ----------------------------------------------------------
        z = 12000
        print(f"\nTiempos a z = {z} mm, N = {N}, s = 10 (con sincronización):")
        print(f"{'método':>16} {'CPU c128 [s]':>13} {'GPU c64 [s]':>12} {'x':>7}")
        for nombre, fn in (("MPASM", mpasm), ("FFT-ASM", fft_asm), ("BL-ASM", blas)):
            _, t_cpu = cronometrar(fn, U0_cpu, delta, lamb, z, device="cpu")
            _, t_gpu = cronometrar(fn, U0_cpu, delta, lamb, z, device="gpu",
                                   dtype=np.complex64)
            print(f"{nombre:>16} {t_cpu:13.3f} {t_gpu:12.3f} {t_cpu / t_gpu:7.1f}")
            liberar_memoria()



if __name__ == "__main__":
    main()
