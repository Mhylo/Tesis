"""Contraste de CamposT contra los propagadores publicados de pyDHM.

Tarea 32 del cronograma. pyDHM (Trujillo et al.) es código publicado y
revisado, así que sirve de validación externa: si CamposT y pyDHM coinciden
sobre el mismo campo, el error de implementación propio queda acotado.

El árbitro no es ninguno de los dos, sino la solución analítica del haz
gaussiano propagado, que se conoce en forma cerrada. Así el contraste no
depende de cuál de las dos implementaciones se tome por buena.

pyDHM vive en referencia/carlos/, que está en .gitignore (ver README). Si no
está, el script lo dice y termina sin error.

Uso:
    Tesis_env/Scripts/python.exe -m scripts.contraste_referencias
"""

import argparse
import pathlib
import sys

import numpy as np

# La tabla se dibuja con caracteres de caja (│ ─) y la consola de Windows
# arranca en cp1252, que no sabe codificarlos: sin esto el script muere con
# UnicodeEncodeError a mitad de la primera fila. Las consolas que ya trabajan
# en UTF-8 no se ven afectadas.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from CamposT.propagadores import blas, fft_asm, mpasm
from CamposT.referencias import gauss_analytic, gauss_beam

#: ruta del paquete pyDHM dentro de referencia/carlos/
PYDHM = (pathlib.Path(__file__).resolve().parent.parent
         / "referencia" / "carlos" / "pyDHM-master" / "pyDHM-master")


def cargar_pydhm():
    """Importa pyDHM.numericalPropagation, o None si no está en disco."""
    if not (PYDHM / "pyDHM" / "numericalPropagation.py").exists():
        return None
    sys.path.insert(0, str(PYDHM))
    from pyDHM import numericalPropagation
    return numericalPropagation


def error_rms(U, referencia):
    """RMS entre amplitudes normalizadas al máximo. Adimensional."""
    a = np.abs(np.asarray(U)).astype(np.float64)
    b = np.abs(np.asarray(referencia)).astype(np.float64)
    return float(np.sqrt(np.mean((a / a.max() - b / b.max()) ** 2)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=256, help="lado de la malla")
    p.add_argument("--s", type=int, default=4, help="sobremuestreo de MPASM")
    args = p.parse_args(argv)

    npr = cargar_pydhm()
    if npr is None:
        print(f"pyDHM no está en {PYDHM}.")
        print("Se recupera de https://github.com/catrujilla/pyDHM (ver README).")
        return 0

    # Tabla 1 del paper de Zhao, en mm. Las mismas unidades para todo: pyDHM no
    # impone ninguna, sólo exige que lamb, dx y z sean coherentes entre sí.
    N, L0, w0, lamb = args.n, 5.0, 1.0, 632.8e-6
    delta = L0 / (N - 1)
    Z = (500, 2000, 6000, 12000, 30000)

    U0, X, Y = gauss_beam(N, delta, w0, device="cpu")
    zR = np.pi * w0 ** 2 / lamb

    print(f"N = {N}, ventana {L0} mm, delta = {delta:.5f} mm, "
          f"lamb = {lamb * 1e6:.1f} nm, zR = {zR:.0f} mm")
    print("Error RMS de amplitud contra el gaussiano analítico "
          "(más bajo = mejor):\n")
    print(f"{'z [mm]':>8} {'z/zR':>6} {'cabe':>5} │ "
          f"{'CamposT':>9} {'CamposT':>9} {'CamposT':>9} │ {'pyDHM':>9} {'pyDHM':>9}")
    print(f"{'':>8} {'':>6} {'':>5} │ {'fft_asm':>9} {'blas':>9} "
          f"{'mpasm s=' + str(args.s):>9} │ {'angSpec':>9} {'bluestein':>9}")
    print("─" * 82)

    for z in Z:
        ref = gauss_analytic(X, Y, w0, lamb, z)
        wz = w0 * np.sqrt(1 + (z / zR) ** 2)
        cabe = "sí" if wz < L0 / 2 else "no"

        e_fft = error_rms(fft_asm(U0, delta, lamb, z, device="cpu"), ref)
        e_blas = error_rms(blas(U0, delta, lamb, z, device="cpu"), ref)
        e_mp = error_rms(mpasm(U0, delta, lamb, z, s=args.s, device="cpu")[0], ref)

        e_as = error_rms(npr.angularSpectrum(U0, z, lamb, delta, delta), ref)
        # bluestein necesita el paso de salida; con dxout = dx la ventana de
        # observación coincide con la de entrada y la comparación es válida.
        # (pyDHM.fresnel no se incluye: devuelve el campo en una malla propia,
        # dxout = lamb·z/(M·dx), unas 13 veces más gruesa a z = 2000 mm, así que
        # compararlo píxel a píxel contra la analítica de entrada no mide nada.)
        e_bs = error_rms(npr.bluestein(U0, z, lamb, delta, delta, delta, delta), ref)

        print(f"{z:8d} {z / zR:6.2f} {cabe:>5} │ {e_fft:9.5f} {e_blas:9.5f} "
              f"{e_mp:9.5f} │ {e_as:9.5f} {e_bs:9.5f}")

    # --- acuerdo directo entre las dos implementaciones de ASM ----------------
    print("\nAcuerdo CamposT.fft_asm ↔ pyDHM.angularSpectrum "
          "(error relativo máximo del campo complejo):")
    for z in Z:
        a = np.asarray(fft_asm(U0, delta, lamb, z, device="cpu"))
        b = np.asarray(npr.angularSpectrum(U0, z, lamb, delta, delta))
        rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
        print(f"  z = {z:6d} mm   {rel:.3e}")

    # --- efecto del exp2 de pyDHM sobre bluestein -----------------------------
    # pyDHM.fresnel y pyDHM.bluestein construyen sus factores de fase con
    # np.exp2 (2^x) donde la física pide np.exp (e^x); angularSpectrum sí usa
    # np.exp. Sustituir uno por otro en caliente cuantifica cuánto cuesta.
    exp2_original = np.exp2
    print("\nEfecto del np.exp2 de pyDHM.bluestein (RMS contra la analítica):")
    print(f"{'z [mm]':>8} {'publicado':>10} {'con np.exp':>11}")
    for z in Z:
        ref = gauss_analytic(X, Y, w0, lamb, z)
        np.exp2 = exp2_original
        antes = error_rms(npr.bluestein(U0, z, lamb, delta, delta, delta, delta), ref)
        np.exp2 = np.exp
        despues = error_rms(npr.bluestein(U0, z, lamb, delta, delta, delta, delta), ref)
        np.exp2 = exp2_original
        print(f"{z:8d} {antes:10.5f} {despues:11.5f}")

    print("\nEl exp2 degrada bluestein de forma medible (unas 4 veces a "
          "z = 6000 mm),\npero aun corregido queda por encima de ASM: "
          "bluestein es paraxial por\nconstrucción y esta es la zona donde esa "
          "aproximación no aplica.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
