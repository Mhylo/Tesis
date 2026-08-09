"""Propagación de campos ópticos en CPU o GPU con el mismo código.

Tres módulos:

    backend       elige CuPy o NumPy y fija la política de precisión
    propagadores  MPASM, FFT-ASM, BLAS y SAM bajo una interfaz común
    campos        construye los campos de entrada (imagen, target USAF)

No se importa nada aquí para que `import CamposT` no arrastre CuPy: la
selección de backend ocurre al importar CamposT.backend, y en una máquina
sin CUDA eso debe poder fallar a NumPy sin ruido.
"""
