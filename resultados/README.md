# resultados/

Lo que escriben los scripts. Casi todo se regenera y no está en el
repositorio: el [`.gitignore`](../.gitignore) ignora `resultados/*` y reabre,
con una lista blanca, solo lo que va.

## Qué está en el repositorio

Los CSV y JSON chicos que respaldan las figuras y tablas del manuscrito y de la
presentación. Qué figura lee cada uno: [INDICE.md](../INDICE.md).

| Ruta | Qué es | Lo escribe |
|---|---|---|
| [`exactitud/sam_vs_z.csv`](exactitud/sam_vs_z.csv) | SAM de cada método contra z, escenario de Zhao et al. | `scripts/exactitud.py` |
| [`escalado_N.csv`](escalado_N.csv), [`escalado_s.csv`](escalado_s.csv) | Tiempos en CPU y GPU contra N y contra s | `scripts/comparacion.py` |
| [`dlhm_esferico/campo_N3000.csv`](dlhm_esferico/campo_N3000.csv) | Correlación con el objeto por tesela, MPASM esférico frente al escalado | `scripts/verif_dlhm_esferico.py campo --n 3000` |
| [`contraste_czt/`](contraste_czt/) | Acuerdo y tiempos del MPASM frente a la CZT | `scripts/contraste_czt.py` |
| [`barrido_obj2/barrido.csv`](barrido_obj2/barrido.csv) | El barrido de validez: error de cada propagador en la malla z × NA | `scripts/barrido_obj2.py` |
| [`comparacion_cruzada/*/`](comparacion_cruzada/) | `resumen.csv` y `parametros.txt` de cada corrida de la ida y vuelta con onda plana | `scripts/comparacion_cruzada.py` |
| [`cruzada_dlhm/resumen.csv`](cruzada_dlhm/resumen.csv) | La ida y vuelta en el banco DLHM con onda esférica | `scripts/cruzada_dlhm.py` |
| [`interferometria_larga/resumen.csv`](interferometria_larga/resumen.csv) | Interferometría holográfica simulada (Anexo A) | `scripts/interferometria_larga.py` |
| [`hologramas_carlos/`](hologramas_carlos/) | Las medidas sobre los hologramas del laboratorio: focos, calibración de la geometría, métricas, tiempos y las curvas de enfoque (`curvas_*.npz`). Nunca las imágenes. | `scripts/hologramas_carlos/correr.py` y `comparar.py` |
| [`campos/`](campos/) | Los campos propagados de la demostración, una carpeta por propagador (`mpasm/z0020_s2.png` es MPASM a z = 20 mm con s = 2), y el objeto `entrada.png` que usan las idas y vueltas | `python -m CamposT.pipeline` |

## Qué no está

Los `.npy` (de 2 a 64 MB cada uno), las figuras de trabajo de cada script y las
pilas de los barridos de foco. Se regeneran con el script de la tabla de
arriba; la orden de cada uno está en
[INDICE.md](../INDICE.md#lo-que-no-está-en-el-repositorio).

Para meter al repositorio un resultado curado nuevo, se agrega su ruta a la
lista blanca del `.gitignore` con un `!` delante.
