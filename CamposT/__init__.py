"""Propagación de campos ópticos en CPU o GPU con el mismo código.

Seis módulos, cada uno con una responsabilidad. Leídos en este orden se sigue
el camino que recorre un campo:

    campos        construye el campo de entrada U0: una imagen, o el target
                  sintético de barras con su geometría (ancho de barra,
                  periodo, pares de línea por mm).

    propagadores  lo propaga. MPASM, FFT-ASM y BL-ASM bajo una firma común
                  (U0, delta, lamb, z, device, dtype), más kf_auto y la
                  función de transferencia que comparten.

    referencias   contra qué se contrasta el resultado: el gaussiano analítico
                  (cerrado y paraxial) y la Rayleigh-Sommerfeld I (cuadratura,
                  no paraxial). Donde discrepan se mide la paraxialidad.

    metricas      cómo se mide la discrepancia: SAM en dB (más alto mejor) y
                  rms_amplitud (más bajo mejor). Convenios opuestos, nombres
                  separados a propósito.

    pipeline      orquesta lo anterior: diagnostico() dice si el caso cae en
                  el rango de FFT-ASM o hace falta MPASM, propagar() lo
                  ejecuta con relleno y recorte, guardar() lo escribe.

    backend       debajo de todo: elige CuPy o NumPy y fija la política de
                  precisión (fases en float64 siempre, campos en complex64 en
                  GPU).

Para empezar por algún sitio: `python -m CamposT.pipeline` propaga el target
por los tres métodos y escribe los PNG en resultados/campos/.

No se importa nada aquí para que `import CamposT` no arrastre CuPy: la
selección de backend ocurre al importar CamposT.backend, y en una máquina sin
CUDA eso debe poder caer a NumPy sin ruido.
"""
