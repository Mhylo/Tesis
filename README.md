# Tesis — propagación de campos ópticos en GPU

Comparación de propagadores de campo óptico ejecutados en CPU y en GPU con el
mismo código, para cuantificar qué parte de la aceleración viene del hardware
y qué parte del cambio de precisión.

## Estructura

    CamposT/            El paquete. Todo el código propio vive aquí. Los
                        módulos, en el orden que recorre un campo:
      campos.py         Construye el campo de entrada: una imagen, o el target
                        sintético de barras con su geometría (ancho de barra,
                        periodo, pares de línea por mm).
      propagadores.py   Lo propaga. MPASM, FFT-ASM y BL-ASM bajo una firma
                        común, más kf_auto y la función de transferencia.
      referencias.py    Contra qué se contrasta: el gaussiano analítico
                        (cerrado, paraxial) y la Rayleigh-Sommerfeld I
                        (cuadratura, no paraxial).
      metricas.py       Cómo se mide la discrepancia: SAM en dB (más alto
                        mejor) y rms_amplitud (más bajo mejor).
      pipeline.py       Orquesta lo anterior: diagnostico() dice si hace falta
                        MPASM, propagar() lo ejecuta, guardar() lo escribe.
      retropropagacion.py
                        El camino de vuelta: de un holograma medido al objeto.
                        Toma la imagen que le des y la retropropaga con los
                        tres métodos sobre un barrido de distancias.
      backend.py        Debajo de todo: elige CuPy (GPU) o NumPy (CPU) y fija
                        la política de precisión: fases en float64 siempre,
                        campos en complex64 en GPU.

    scripts/
      comparacion.py    Benchmark CPU vs GPU del Objetivo 1: escalado con el
                        tamaño de malla N y con el sobremuestreo s.
      exactitud.py      SAM [dB] de cada método contra el gaussiano analítico
                        sobre siete órdenes de z: la tabla de la Figura 4.
      contraste_referencias.py
                        Contrasta CamposT contra pyDHM sobre el gaussiano
                        analítico, que hace de árbitro entre las dos.
      parchar_referencias.py
                        Arregla los repos de terceros para que corran en este
                        entorno. Ver más abajo.
      mi_holograma.py   Retropropaga TU holograma con los parámetros como
                        constantes editables: pegas la ruta y le das a Run.
                        Llama al main() de CamposT.retropropagacion, no
                        duplica nada.
      retro_fft_angular.py
      retro_blas.py
      retro_mpasm.py    La misma ida y vuelta en una sola pieza, un script por
                        propagador, sin importar CamposT: contraste
                        independiente del paquete. Para leer el algoritmo, no
                        para producir. Cada uno enseña lo suyo:
                        · fft: el angularSpectrum de pyDHM tal cual, con dos
                          defectos suyos medidos (ejes cruzados, evanescentes).
                        · blas: sin referencia de terceros -no hay ningún
                          BL-ASM en referencia/- pero mide cuánta banda
                          descarta la máscara a cada z, y comprueba que corta
                          todas las evanescentes, que es lo que hace que no
                          reviente al retropropagar.
                        · mpasm: el MatrixDftCPU de Zhao copiado tal cual como
                          referencia. Enseña las dos erratas del original
                          corriendo: el A²·B de la Ec. (14) y el Kf que se
                          apaga a z < 0.
      retro_holograma.py
                        Retropropaga un holograma ya grabado: se lo entregas,
                        no lo genera. A diferencia de los tres de arriba no
                        lleva copia propia -llama a CamposT.pipeline.propagar()-
                        y corre uno o varios a la vez sobre el mismo dato.
      figura_escenario.py
                        Dibuja el escenario de la Tabla 1 del paper en tres
                        figuras: la apertura, el frente de onda propagado y el
                        corte axial x-z del haz. Este ultimo comprueba que el
                        radio 1/e sigue a w0(R+z)/R, o sea que la lente del
                        paper es un pinhole virtual.

    tests/              Verificación (pytest). Un fichero por módulo:
      test_propagadores.py   propiedades de los propagadores, Kf por eje
      test_rs1.py            la Rayleigh-Sommerfeld y su reducción a 1-D
      test_metricas.py       el SAM del paper y la errata de su Ec. (16)
      test_campos.py         geometría del target USAF y su conversión a lp/mm
      test_retropropagacion.py
                             re-localización del objeto, convención de signo y
                             la ambigüedad de la imagen gemela

    resultados/         Salidas de los scripts: figuras y CSV. Se regeneran.
      escenario/        Las tres figuras del escenario de la Tabla 1.
      campos/           Los campos propagados, una carpeta por propagador
                        (fft/, blas/, mpasm/), para que el método se lea en
                        la ruta. Los parámetros van en el nombre:
                        mpasm/z0020_s2.png es MPASM a z = 20 mm con s = 2.
      retropropagacion/ Las reconstrucciones, una subcarpeta por holograma y
                        dentro una por propagador.
      exactitud/        sam_vs_z.csv y sam_vs_z.png: la reproducción de la
                        Figura 4, con la Ec. (16) corregida.

    referencia/
      zhao2020/         Código original de Zhao et al., Opt. Lett. 45, 5937
                        (2020), del que parten los propagadores. No se modifica.
      carlos/           Tres repos de terceros consultados (pyDHM, DLHM-model,
                        DLHM-processing-tools). NO están en git: son ~224 MB y
                        tienen su propia historia. Ver más abajo.

    docs/               El paper de la tesis y el cronograma del trabajo de
                        grado.

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

Si prefieres no escribir la línea de órdenes, `scripts/mi_holograma.py` hace lo
mismo con los parámetros como constantes editables: pegas la ruta en `RUTA`,
ajustas `DELTA`, `LAMB` y `Z`, y le das a Run. Llama al `main()` de la CLI, así
que los dos caminos no pueden divergir.

Dos límites que conviene conocer antes de leer una reconstrucción, ambos
documentados en el módulo y fijados en su suite: asume iluminación colimada
(sin la corrección de fuente puntual del DLHM), y la reconstrucción trae la
imagen gemela superpuesta, que no se suprime.

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
- DLHM-model y DLHM-processing-tools, del grupo de Óptica Aplicada de EAFIT

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
