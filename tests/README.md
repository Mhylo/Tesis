# tests/

La suite de pytest. Desde la raíz:

```bash
Tesis_env/Scripts/python.exe -m pytest
```

Las pruebas de propiedad corren dos veces, en CPU (complex128) y en GPU
(complex64); las de GPU se saltan solas si no hay CUDA, y el contraste con pyLHM
de `test_rs1.py`, si no está `referencia/carlos/`. Ninguna necesita los
hologramas del laboratorio. Qué se verifica y por qué, en el
[README](../README.md#tests) y en el docstring de cada archivo.

## Del paquete (`CamposT/`)

| Archivo | Qué verifica |
|---|---|
| [`test_propagadores.py`](test_propagadores.py) | Identidad en z = 0, energía, reversibilidad, el MPASM en su régimen y la paridad CPU/GPU. |
| [`test_rs1.py`](test_rs1.py) | La Rayleigh–Sommerfeld I reducida a una dimensión como referencia: contra la fuente puntual compleja y contra pyLHM (§4.1.1). |
| [`test_metricas.py`](test_metricas.py) | El SAM del artículo y la errata de su Ec. (16). |
| [`test_dlhm.py`](test_dlhm.py) | El MPASM con iluminación esférica exacta: la ventana de Fresnel, la ida contra Rayleigh–Sommerfeld y la vuelta. |
| [`test_campos.py`](test_campos.py) | El target USAF sintético y su conversión a pares de línea por mm. |
| [`test_montaje.py`](test_montaje.py) | Los números del montaje y del escenario de Zhao et al. |
| [`test_barrido.py`](test_barrido.py) | La malla del barrido del objetivo 2. |
| [`test_pipeline.py`](test_pipeline.py) | La orquestación: relleno, recorte, despacho y escritura. |
| [`test_retropropagacion.py`](test_retropropagacion.py) | La retropropagación de hologramas: dónde queda el objeto, el signo y la imagen gemela. |
| [`test_roi.py`](test_roi.py) | El recorte de una ventana de la imagen. |
| [`test_intensidad_pura.py`](test_intensidad_pura.py) | Qué sale de la retropropagación cuando la intensidad se deja tal cual. |
| [`test_objeto_mixto.py`](test_objeto_mixto.py) | El modo `mixto` de la carga del objeto. |

## De los scripts (`scripts/`)

| Archivo | Qué verifica |
|---|---|
| [`test_contraste_czt.py`](test_contraste_czt.py) | `contraste_czt.py`: el MPASM con la CZT da el mismo campo (Tabla 4.2). |
| [`test_cruzada_dlhm.py`](test_cruzada_dlhm.py) | `cruzada_dlhm.py`: la ida y vuelta DLHM con onda esférica. |
| [`test_hologramas_carlos.py`](test_hologramas_carlos.py) | `hologramas_carlos/`: foco paraxial, geometría DLHM y USAF. |
| [`test_mi_prueba_dlhm_seleccion.py`](test_mi_prueba_dlhm_seleccion.py), [`test_malla_remuestreada.py`](test_malla_remuestreada.py) | `mi_prueba_dlhm.py`: la selección de foco y la malla remuestreada del método del laboratorio. |
| [`test_gpu_mi_prueba.py`](test_gpu_mi_prueba.py) | Los cuatro `mi_prueba_*.py` en CPU y en GPU, y el MPASM contra el `MatrixDftCPU` de Zhao et al. |
| [`test_kf_por_eje_scripts.py`](test_kf_por_eje_scripts.py) | Que los scripts calculen K_f por eje, igual que el paquete. |
| [`test_fases_retro.py`](test_fases_retro.py), [`test_guardar_holograma.py`](test_guardar_holograma.py), [`test_sidecar_ejes.py`](test_sidecar_ejes.py) | Lo que comparten los tres `retro_*.py`: las fases, el guardado del holograma y el `.txt` con λ, δ y la convención de ejes. |
| [`test_nitidez_foco.py`](test_nitidez_foco.py), [`test_pico_de_foco.py`](test_pico_de_foco.py) | La nitidez y el pico de la curva de foco. |

[`conftest.py`](conftest.py) define los fixtures comunes; ejecutarlo no hace nada.
