# CamposT/

El paquete. Todo el código propio de cálculo vive aquí; los scripts y los tests
lo importan (`pip install -e .` desde la raíz, ver el [README](../README.md#instalación)).

Los módulos, en el orden que recorre un campo:

| Módulo | Qué hace |
|---|---|
| [`montaje.py`](montaje.py) | Los números del montaje: los del banco DLHM del laboratorio (`MONTAJE`) y los del escenario de Zhao et al. (`TABLA1`), separados. |
| [`campos.py`](campos.py) | Construye el campo de entrada: una imagen, o el target sintético tipo USAF con su geometría (ancho de barra, periodo, pares de línea por mm). |
| [`roi.py`](roi.py) | Recorta una ventana de la imagen antes de propagar; `radio_del_cono()` dice lo que cuesta. |
| [`propagadores.py`](propagadores.py) | Los tres propagadores bajo una firma común: `fft_asm` (el ASM de la charla), `blas` y `mpasm`, más `kf_auto` y la función de transferencia. |
| [`dlhm.py`](dlhm.py) | La DLHM con fuente puntual: el MPASM con la onda esférica exacta dentro de las matrices de la DFT, sin aproximación paraxial. `holograma()` simula; `Reconstructor` reconstruye un barrido de z reutilizando el espectro del sensor; `reconstruir_escalado` es el escalado paraxial con el que se compara. |
| [`referencias.py`](referencias.py) | Contra qué se mide la exactitud: el gaussiano analítico y la integral de Rayleigh–Sommerfeld I reducida a una dimensión. |
| [`metricas.py`](metricas.py) | La discrepancia: SAM en dB (con la Ec. (16) de Zhao et al. corregida) y `rms_amplitud`. |
| [`barrido.py`](barrido.py) | La malla z × NA del barrido de validez del objetivo 2, con los números del banco. |
| [`pipeline.py`](pipeline.py) | Orquesta lo anterior: `diagnostico()` dice si hace falta el MPASM, `propagar()` lo ejecuta y `guardar()` lo escribe. |
| [`propagacion.py`](propagacion.py) | La ida desde la línea de órdenes: `python -m CamposT.propagacion imagen.png --z 20`. |
| [`retropropagacion.py`](retropropagacion.py) | La vuelta: de un holograma medido al objeto, con los tres métodos sobre un barrido de z (iluminación colimada). |
| [`backend.py`](backend.py) | Elige CuPy (GPU) o NumPy (CPU) y fija la precisión: fases en float64 siempre, campos en complex64 en GPU. |

Dónde se usa cada uno en el manuscrito y en la charla: [INDICE.md](../INDICE.md).
Las pruebas, una por módulo, están en [`tests/`](../tests/README.md).
