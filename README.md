# Tesis — propagadores con muestreo controlado para la DLHM

Código del trabajo de grado *Implementación y validación de propagadores
numéricos con muestreo controlado para la reconstrucción de hologramas en
microscopía holográfica digital sin lentes* (Ingeniería Física, Universidad
EAFIT, 2026). Tres propagadores del espectro angular —FFT-ASM, BLAS y el MPASM
de Zhao et al. (2020)— en un mismo paquete que corre en CPU y en GPU con el
mismo código: verificados contra referencias independientes, validados bajo el
muestreo de la DLHM y aplicados a hologramas registrados en el laboratorio.

**¿De dónde sale una figura, una tabla o una diapositiva de la defensa? Está en
[INDICE.md](INDICE.md):** cada una lleva al script que la dibuja y a los datos
que lee.

## Estructura

| Carpeta | Qué hay |
|---|---|
| [`CamposT/`](CamposT/README.md) | El paquete. Todo el código propio de cálculo: los propagadores, la DLHM con la onda esférica exacta, las referencias y las métricas. |
| [`scripts/`](scripts/README.md) | Los cálculos de cada objetivo, que escriben en `resultados/`, y las figuras del manuscrito. |
| [`tests/`](tests/README.md) | La suite de pytest. |
| [`manuscrito/`](manuscrito/README.md) | El documento en LaTeX, en la versión entregada, y sus figuras. |
| [`presentacion/`](presentacion/README.md) | Las figuras que se proyectaron en la defensa y el script que las dibuja. |
| [`resultados/`](resultados/README.md) | Los datos detrás de las figuras y las tablas: los CSV y JSON chicos. Lo grande se regenera. |
| [`referencia/`](referencia/README.md) | El código original de Zhao et al., del que parten los propagadores. |
| [`docs/`](docs/README.md) | Notas de trabajo: la hoja de parámetros del montaje, la errata de K_f y las figuras de la guía de lectura. |

## Instalación

Una sola vez por entorno, desde la raíz del repo:

```bash
Tesis_env/Scripts/python.exe -m pip install -e ".[dev]"
```

El `-e` (editable) registra la carpeta del repo en el entorno, así que
`import CamposT` funciona desde cualquier directorio y con cualquier forma de
invocación, y los cambios en el código se ven sin reinstalar. Sin esto, Python
sólo pone en `sys.path` la carpeta del fichero que ejecutas — `tests/` o
`scripts/`, nunca la raíz — y el import falla.

CuPy no está entre las dependencias porque el paquete correcto depende de tu
versión de CUDA (`cupy-cuda12x`, `cupy-cuda11x`...). Instálalo aparte; sin él
todo corre en CPU.

## Cómo correr

```bash
Tesis_env/Scripts/python.exe -m scripts.comparacion
```

Barridos concretos:

```bash
Tesis_env/Scripts/python.exe -m scripts.comparacion --n 256 512 1024 --smax 8
```

El pipeline corre solo, con una demostración de extremo a extremo que propaga
el target por los tres métodos y escribe los PNG:

```bash
Tesis_env/Scripts/python.exe -m CamposT.pipeline
```

Y la exactitud del Objetivo 1: la tabla por pantalla, y en
`resultados/exactitud/` el CSV y la curva SAM vs z que reproduce la Figura 4
del paper. Tarda un par de minutos: el barrido son 30 distancias por tres
métodos en CPU y doble precisión, que es la fila de referencia.

```bash
Tesis_env/Scripts/python.exe -m scripts.exactitud
```

### Retropropagar un holograma

La imagen la eliges tú en cada corrida. Como la distancia de enfoque no se
conoce de antemano, `--z` acepta dos valores y barre entre ellos:

```bash
Tesis_env/Scripts/python.exe -m CamposT.retropropagacion holograma.png --z 20 40 --pasos 25
```

Los PNG salen en `resultados/retropropagacion/holograma/{fft,blas,mpasm}/`,
uno por distancia. `--delta` y `--lamb` fijan el montaje (por defecto 3.45 µm
y 405 nm), `--metodos` acota los propagadores y `--s` sube el sobremuestreo de
MPASM — con cuidado: su matriz espectral es (s·N)² *por distancia*, así que en
un barrido el defecto es `s=1`.

Para no propagar la imagen entera, `--roi X0 Y0 ANCHO ALTO` recorta antes de
propagar, o `--roi-interactivo` para elegir la ventana con el ratón y cerrar
la figura. El recorte es seco —sin margen de guarda— y la consola avisa de
cuánto cuesta.

Si prefieres no escribir la línea de órdenes, `scripts/mi_holograma.py` hace lo
mismo con los parámetros como constantes editables: pegas la ruta en `RUTA`,
ajustas `DELTA`, `LAMB` y `Z`, y le das a Run. Llama al `main()` de la CLI, así
que los dos caminos no pueden divergir.

Dos límites que conviene conocer antes de leer una reconstrucción, ambos
documentados en el módulo y fijados en su suite: asume iluminación colimada
(sin la corrección de fuente puntual del DLHM), y la reconstrucción trae la
imagen gemela superpuesta, que no se suprime. Para un holograma DLHM de
verdad, con fuente puntual, está la sección siguiente.

### DLHM: la fuente puntual

`CamposT/dlhm.py` reconstruye con la onda esférica exacta y sin aproximación
paraxial. Pide el holograma de contraste (dividido por el fondo), el paso del
sensor y la geometría del banco en milímetros: `z` de la fuente a la muestra y
`L` de la fuente al sensor.

```python
from CamposT import dlhm
from CamposT.backend import a_numpy
from CamposT.montaje import MONTAJE

# una sola z
t = dlhm.reconstruir(holo, MONTAJE.delta_x, MONTAJE.lamb, MONTAJE.z, MONTAJE.L,
                     fondo=referencia)

# un barrido de enfoque: el espectro del sensor se calcula una vez
R = dlhm.Reconstructor(holo, MONTAJE.delta_x, MONTAJE.lamb, MONTAJE.L, (2.5, 3.5),
                       fondo=referencia)
campos = {z: a_numpy(R.campo(z)) for z in (2.8, 2.9, 3.0, 3.1)}
```

Con un sensor de 3000 x 4000 px el Reconstructor tarda ~10 s en construirse y
0.56 s por z en una GPU de 4 GB, prácticamente lo mismo que el escalado
paraxial de la tarea 24 (`dlhm.reconstruir_escalado`), que con detalle fino
se equivoca: 0.89 de correlación con el objeto en el eje y 0.63 en el borde,
frente a 1.000 y 0.93. Las cifras y su verificación:

```bash
Tesis_env/Scripts/python.exe -m scripts.verif_dlhm_esferico
```

### Propagar un objeto hacia adelante

El camino de ida: de una imagen del objeto a su campo propagado a una
distancia `--z` positiva, con los mismos tres propagadores.

```bash
Tesis_env/Scripts/python.exe -m CamposT.propagacion objeto.png --z 20 --modo transmitancia
```

Los PNG salen en `resultados/propagacion/objeto/{fft,blas,mpasm}/`. Comparte
`--delta`, `--lamb`, `--metodos`, `--pad`, `--roi` y `--roi-interactivo` con
`CamposT.retropropagacion`; `--modo` fija cómo se interpreta la imagen
(amplitud, fase, mixta, transmitancia binarizada u holograma).

## Tests

```bash
Tesis_env/Scripts/python.exe -m pytest
```

También vale ejecutar el fichero de pruebas directamente, o darle al botón de
Run del editor: trae un bloque `__main__` que lanza pytest sobre sí mismo.
`conftest.py` no: sólo define fixtures, así que ejecutarlo no hace nada (ni
debe).

Cada prueba de propiedad se ejecuta dos veces, en CPU (complex128, tolerancia
1e-12) y en GPU (complex64, tolerancia 1e-5); las de GPU se saltan solas si no
hay CUDA. Lo que se verifica y por qué está en el docstring de
`tests/test_propagadores.py`; en resumen:

- **z = 0** devuelve el campo de entrada, con y sin sobremuestreo.
- **Energía**: FFT-ASM es unitario y se le exige conservación exacta. BLAS y
  MPASM pierden energía a propósito (máscara de banda limitada y compresión
  por Kf), así que se les exige no *ganarla* nunca y ser exactos donde su
  aproximación está inactiva.
- **Reversibilidad**: propagar z y volver -z. Exacta en FFT-ASM; en BLAS sólo
  por debajo de z_lim = δ²N/λ, y la prueba complementaria fija que por encima
  deja de serlo.
- **MPASM en su régimen**: con s = 4 y Kf automático sigue al gaussiano
  analítico con RMS < 1e-3 a z = 12000 y 80000 mm, donde FFT-ASM alías y da
  0.16 y 0.88. Es la comprobación que justifica el método.
- **Paridad CPU/GPU**: el mismo código en los dos dispositivos difiere sólo lo
  que impone complex64 frente a complex128.

Las tolerancias no están ajustadas hasta que la suite pase: salen de la
mantisa del dtype. La suite se validó por mutación: se inyectan bugs a mano en
`CamposT/` y se comprueba que la suite los detecta, porque un test escrito
después del código y que pasa a la primera no demuestra nada por sí solo. Los
huecos encontrados así están anotados donde salieron; el único que sigue
abierto a propósito está documentado en `referencias.n_phi_auto`.

## referencia/carlos/

Está en `.gitignore`. Son clones de tres repos públicos, y pesan 224 MB — de
los cuales 172 MB son salidas de plotly (`manual_focus.html`, 70 MB) que no
aportan nada al repo. Si la carpeta no está, se recupera de:

- <https://github.com/catrujilla/pyDHM>
- <https://github.com/mloper23/DLHM-model>
- <https://github.com/mloper23/DLHM-processing-tools> (pyLHM)

De ahí, lo relevante: `pyDHM/numericalPropagation.py` (propagadores de
referencia ya publicados y validados) y `pyLHM/myfunctions.py` (versión
adaptada a holografía sin lente, con Kreuzer, autofoco y la
Rayleigh-Sommerfeld completa).

### Cómo dejarlos corriendo

Recién clonados no corren en este entorno. Dos pasos, una sola vez:

```bash
Tesis_env/Scripts/python.exe -m pip install -e ".[referencia]"
Tesis_env/Scripts/python.exe -m scripts.parchar_referencias
```

El segundo aplica ocho arreglos y deja un comentario `[parche TG]` en cada
línea que toca, para que nunca se confunda el código publicado con el
corregido aquí. Es idempotente: correrlo de nuevo informa y no reescribe nada.
Con `--revisar` sólo informa. Existe como script, y no como ediciones a mano,
porque `referencia/carlos/` no está en git: cualquier corrección hecha
directamente sobre esos archivos se perdería al re-clonarlos y no quedaría
registrada en ninguna parte.

Ninguno de los ocho cambia la física. Son dos incompatibilidades con NumPy 2
(`dtype='complex_'`, que rompía la Rayleigh-Sommerfeld completa, y un conteo de
píxeles en coma flotante), un refactor a medias en `dlhm.py` (la firma pasó a
`W_cx`/`W_cy` pero el cuerpo siguió usando el `W_c` anterior, así que toda
llamada moría con `NameError`) y las rutas relativas de los tres scripts de
DLHM-model, que se resolvían contra el directorio desde el que se lanzaban.
El detalle de cada uno está en el docstring del script.

Quedan dos cosas fuera de su alcance, anotadas ahí mismo:
`simulation_reconstruction_asm_dlhm.py` necesita `data/Complete_Benchmark.png`,
que no viene en el repo publicado y hay que pedírselo a los autores; y los
scripts de demostración de DLHM-processing-tools (`RS1_main.py`,
`kreuzer_main.py`, `main.py`, `simulate.py`) llaman a métodos que no existen en
su propio `myfunctions.py` — `rayleigh_convolutional`, `kreuzer_reconstruct`,
`convergentSAASM_full` — y apuntan a rutas fijas de otra máquina. No son punto
de entrada válido: hay que instanciar `pyLHM.myfunctions.reconstruct` directo.

### Contraste contra CamposT

```bash
Tesis_env/Scripts/python.exe -m scripts.contraste_referencias
```

Compara `fft_asm`, `blas` y `mpasm` contra `angularSpectrum` y `bluestein` de
pyDHM, con el gaussiano analítico propagado como árbitro para que el contraste
no dependa de cuál implementación se dé por buena. Si la carpeta no está, lo
dice y termina sin error.

## Licencia

El código propio se distribuye con licencia MIT (ver `LICENSE`).
`referencia/zhao2020/` es el código de Zhao et al., publicado en figshare
(doi:10.6084/m9.figshare.12972872) con licencia CC BY 4.0, y la conserva.
