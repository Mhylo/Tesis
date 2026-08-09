# Tesis — propagación de campos ópticos en GPU

Comparación de propagadores de campo óptico ejecutados en CPU y en GPU con el
mismo código, para cuantificar qué parte de la aceleración viene del hardware
y qué parte del cambio de precisión.

## Estructura

    CamposT/            El paquete. Todo el código propio vive aquí.
      backend.py        Elige CuPy (GPU) o NumPy (CPU) y fija la política de
                        precisión: fases en float64 siempre, campos en
                        complex64 en GPU.
      propagadores.py   MPASM, FFT-ASM, BLAS y SAM bajo una interfaz común.
      campos.py         Construye los campos de entrada (imagen, target USAF).

    scripts/
      comparacion.py    Benchmark CPU vs GPU del Objetivo 1: escalado con el
                        tamaño de malla N y con el sobremuestreo s.

    resultados/         Salidas del benchmark: figuras y CSV. Se regeneran.

    referencia/
      zhao2020/         Código original de Zhao et al., Opt. Lett. 45, 5937
                        (2020), del que parten los propagadores. No se modifica.
      carlos/           Tres repos de terceros consultados (pyDHM, DLHM-model,
                        DLHM-processing-tools). NO están en git: son ~224 MB y
                        tienen su propia historia. Ver más abajo.

    docs/               El paper de la tesis.

## Cómo correr

Desde la raíz del repo, con el entorno activado:

```bash
Tesis_env/Scripts/python.exe -m scripts.comparacion
```

Los scripts se invocan **con `-m` y desde la raíz**, no como
`python scripts/comparacion.py`. En la segunda forma Python pone `scripts/` en
`sys.path` en vez de la raíz, y `import CamposT` falla.

Barridos concretos:

```bash
Tesis_env/Scripts/python.exe -m scripts.comparacion --n 256 512 1024 --smax 8
```

Los módulos del paquete también corren solos, con una demostración cada uno:

```bash
Tesis_env/Scripts/python.exe -m CamposT.propagadores
```

## referencia/carlos/

Está en `.gitignore`. Son clones de tres repos públicos que se consultan pero
no se modifican, y pesan 224 MB — de los cuales 172 MB son salidas de plotly
(`manual_focus.html`, 70 MB) que no aportan nada al repo. Si la carpeta no
está, se recupera de:

- <https://github.com/catrujilla/pyDHM>
- DLHM-model y DLHM-processing-tools, del grupo de Óptica Aplicada de EAFIT

De ahí, lo relevante: `pyDHM/numericalPropagation.py` (propagadores de
referencia ya publicados y validados) y `pyLHM/myfunctions.py` (versión
adaptada a holografía sin lente, con Kreuzer y autofoco).
