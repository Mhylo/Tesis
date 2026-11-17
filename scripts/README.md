# scripts/

Los cálculos del trabajo. Se corren desde la raíz como módulos
(`python -m scripts.exactitud`) y escriben en [`resultados/`](../resultados/README.md).
Qué figura, tabla o diapositiva sale de cada uno: [INDICE.md](../INDICE.md).

## Objetivo 1: implementar y verificar

| Script | Qué hace |
|---|---|
| [`exactitud.py`](exactitud.py) | SAM de cada método contra el gaussiano analítico sobre siete órdenes de z, en el escenario de Zhao et al. Dibuja la Fig. 4.1. |
| [`contraste_referencias.py`](contraste_referencias.py) | CamposT contra los propagadores de pyDHM, con el gaussiano de árbitro. Necesita `referencia/carlos/` (ver el [README](../README.md#referenciacarlos)). |
| [`comparacion.py`](comparacion.py) | Tiempo en CPU y GPU contra el tamaño de malla N y el sobremuestreo s (Tabla 4.1, Fig. 4.3). |
| [`kf_evolucion.py`](kf_evolucion.py) | Cómo evoluciona K_f, Ec. (14) de Zhao et al. |
| [`figura_escenario.py`](figura_escenario.py) | Las figuras del escenario de Zhao et al.: la apertura, el frente propagado y el corte axial del haz. |
| [`mpasm_minimo.py`](mpasm_minimo.py) | El MPASM en una sola pieza, sin importar CamposT; lo usan varios scripts de abajo. |
| [`retro_fft_angular.py`](retro_fft_angular.py), [`retro_blas.py`](retro_blas.py), [`retro_mpasm.py`](retro_mpasm.py) | La ida y vuelta de un objeto, un script por propagador y sin CamposT: contraste independiente del paquete. `retro_mpasm.py` trae el `MatrixDftCPU` de Zhao et al. tal cual y compara con él su MPASM en `comprobar_equivalencia()` (la §4.1.1). |
| [`parchar_referencias.py`](parchar_referencias.py) | Deja corriendo en este entorno los repos de terceros de `referencia/carlos/`. |

## Objetivo 2: validez bajo el muestreo de la DLHM

| Script | Qué hace |
|---|---|
| [`verif_dlhm_esferico.py`](verif_dlhm_esferico.py) | Las cifras de `CamposT/dlhm.py`: la ida contra Rayleigh–Sommerfeld, el campo completo con el escalado paraxial (Fig. 4.4, con `campo --n 3000`) y el contraste con la cadena del laboratorio. |
| [`contraste_czt.py`](contraste_czt.py) | El MPASM calculado con sus matrices y con la CZT (Tabla 4.2). |
| [`barrido_obj2.py`](barrido_obj2.py) | El barrido de validez: error de cada propagador contra Rayleigh–Sommerfeld en la malla z × NA del banco (Fig. 4.5 y 4.6, Tabla 4.3). `--resumen` y `--frontera` dan los números del texto. |
| [`comparacion_cruzada.py`](comparacion_cruzada.py) | Ida y vuelta con fase y onda plana, cruzando FFT-ASM y MPASM (Fig. 4.7 a 4.9, Tablas 4.4 y 4.5). Las distancias y la ventana (`ZS`, `R`, `S`) son constantes al principio del archivo; la §3.2.4 dice cuáles se usaron para cada carpeta de `resultados/comparacion_cruzada/`. |
| [`cruzada_dlhm.py`](cruzada_dlhm.py) | La misma ida y vuelta en el banco DLHM con onda esférica: la cadena del laboratorio y el MPASM esférico (Fig. 4.7b). |
| [`interferometria_larga.py`](interferometria_larga.py) | Interferometría holográfica simulada a distancias largas (Anexo A). |

## Objetivo 3: hologramas experimentales

| Script | Qué hace |
|---|---|
| [`hologramas_carlos/`](hologramas_carlos/) | Todo lo de los hologramas del laboratorio. `datos.py` los lee, `foco.py` busca el foco, `geometria.py` lleva la geometría DLHM y el criterio de ventana, y `usaf.py` mide la magnificación con el USAF. Las corridas: `correr.py` (`foco`, `calibracion`, `grupo5`, `exacto`, `zL`, `imagenes`) y `comparar.py` (`focos`, `metricas`, `tiempos`). Las cajas de los elementos del USAF están en `cajas_usaf.json` y `cajas_metricas.json`. |
| [`mi_prueba_dlhm.py`](mi_prueba_dlhm.py) | Reconstrucción DLHM de fuente puntual con selección del foco. Aquí está `MotorCarlos`, el método del laboratorio con el que se compara el MPASM esférico. |

Los hologramas no están en el repositorio: ver [INDICE.md](../INDICE.md#lo-que-no-está-en-el-repositorio).

## Figuras

| Script | Qué hace |
|---|---|
| [`figuras_manuscrito.py`](figuras_manuscrito.py) | Todas las figuras del manuscrito menos la 4.1, desde los datos ya guardados. No recalcula nada. |

Las de la presentación las dibuja [`presentacion/figuras_charla.py`](../presentacion/figuras_charla.py)
con estas mismas funciones.

## Herramientas y pruebas sueltas

| Script | Qué hace |
|---|---|
| [`mi_holograma.py`](mi_holograma.py) | Retropropaga tu holograma: las constantes se editan y se le da a Run. Llama a `CamposT.retropropagacion`. |
| [`retro_holograma.py`](retro_holograma.py) | Retropropaga un holograma ya grabado con `CamposT.pipeline.propagar()`, con uno o varios métodos. |
| [`propagar_imagen.py`](propagar_imagen.py) | Propaga una imagen con `mpasm_minimo.py` y guarda la entrada y la salida. |
| [`barrido_foco.py`](barrido_foco.py) | Holograma de una imagen con `mpasm_minimo.py` y barrido de foco. |
| [`mi_prueba_angular.py`](mi_prueba_angular.py), [`mi_prueba_blas.py`](mi_prueba_blas.py), [`mi_prueba_mpasm.py`](mi_prueba_mpasm.py) | Pruebas sueltas: ida y vuelta con el `angularSpectrum` de pyDHM, y barridos puntuados contra una referencia con BL-ASM y con MPASM. |
