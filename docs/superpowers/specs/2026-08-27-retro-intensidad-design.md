# Retropropagación en intensidad, mínima — diseño

Fecha: 2026-08-27
Rama: referencias-y-versionado

## Problema

Los tres scripts `scripts/retro_*.py` hacen ida y vuelta (objeto → +Z →
holograma → −Z → objeto) y, alrededor de eso, acumulan comprobaciones de
equivalencia, de memoria, de evanescentes, métricas de nitidez, detección del
pico de foco, correlaciones, figuras de fase y marcas de procedencia. Entre los
tres suman 2715 líneas. De ellas, el propagador —lo único que hace el trabajo—
ocupa unas 60.

Para trabajar sobre hologramas reales hace falta lo contrario: un script que se
lea de una sentada, donde esté claro qué constante tocar y qué línea borrar.

Además hay un defecto de fondo, ya documentado en `tests/test_intensidad_pura.py`
y no corregido: `guardar_png` aplica `gamma=0.5` y las figuras pintan
`(im / im.max()) ** 0.5`. Como

    (|U|² / max)^0.5  =  |U| / sqrt(max)

ninguna imagen que sale hoy del barrido es intensidad: **son amplitud**. La
reconstrucción en intensidad hay que escribirla sin gamma para que sea lo que
dice ser.

## Alcance

Dentro:

- Un archivo nuevo, `scripts/retro_intensidad.py`, autocontenido: no importa
  `CamposT`, igual que los `retro_*.py` actuales.
- Entrada: una imagen que **ya es un holograma** (intensidad medida o
  simulada). El campo es `sqrt(I)`.
- Dos propagadores, y se retropropaga con los dos: `angularSpectrum()` verbatim
  como referencia, y `espectro_angular()` por bloques como versión de trabajo.
- Barrido de distancias de reconstrucción (pila de foco).
- Salida: un PNG por distancia y por método, con **|U|² y sin gamma**.
- CPU o GPU en la versión de trabajo, vía la constante `DISPOSITIVO`.

Fuera, a propósito:

- **La ida.** No se simula ningún holograma desde un objeto. Si hace falta uno,
  lo fabrican los `retro_*.py` que se quedan intactos.
- **Toda figura de matplotlib.** El barrido produce una pila de PNGs y se mira
  con el explorador de archivos pasando flechas; 30 distancias no caben en una
  figura. Se va también el `import matplotlib`.
- **Métricas.** Ni nitidez, ni pico de foco, ni correlación, ni RMS de fase. La
  distancia que enfoca la decide quien mira la pila.
- **Comprobaciones de arranque.** Ni equivalencia entre propagadores, ni
  memoria, ni evanescentes, ni simetría del límite, ni aviso de Nyquist, ni
  fondo del objeto.
- **Guardar el campo complejo o la intensidad en float** (`.npy`). Sería útil
  para medir sobre la reconstrucción sin pasar por 8 bits, pero es otro
  requisito y añade una constante más. Se deja escrito aquí para que conste que
  es una decisión y no un olvido.
- **Corrección DLHM de fuente puntual** y **supresión de la imagen gemela**.
  Fuera por las mismas razones que en
  `2026-08-20-retropropagacion-design.md`: el script asume iluminación colimada
  y muestra la reconstrucción cruda.

Los tres `retro_*.py` actuales **no se tocan**, ni sus tests.

## Decisiones

### D1 — La entrada es un holograma, no un objeto

El campo de partida es `sqrt(I)`, donde `I` es la imagen normalizada a `[0,1]`.
Un sensor mide intensidad; la fase se perdió en la medida. Esto es la "vuelta
B" de los scripts viejos, y es el único camino que existe con datos reales.

Consecuencia asumida y escrita en el docstring: la reconstrucción trae la imagen
gemela desenfocada superpuesta. Es inherente al holograma in-line, no un fallo.

### D2 — Se conservan los dos propagadores

`angularSpectrum()` se copia **verbatim, sin editar una línea**. Su valor entero
es que nadie la ha tocado: es la referencia contra la que se contrasta.

Dos propiedades suyas que el diseño acepta en vez de esconder:

- **Siempre corre en CPU y en `complex128`.** Su primera línea es
  `field = np.array(field)`, que no acepta CuPy. Con `DISPOSITIVO = "gpu"`, el
  barrido de `"referencia"` baja a CPU mediante `a_cpu()`, explícitamente y con
  un comentario. Para una referencia eso es deseable.
- **Es la que se queda sin memoria primero.** Materializa `X`, `Y`, `kernel`,
  `phase`, `field_spec` y `tmp` enteros: seis mallas completas. A 512×512 con
  `PAD = 2` son ~80 MB. Con un holograma de 4000×3000 y `PAD = 2` la malla
  padded es 8000×6000 y pide ~3.8 GB: ahí falla, y `"bloques"` no.

`espectro_angular()` mantiene lo que ya tiene: evaluación del fasor por bloques
de filas, fase en `float64` y solo el fasor —acotado a módulo 1— bajado al
dtype de trabajo, y `xp` como NumPy o CuPy.

### D3 — Ejes: la referencia cruzada, la de trabajo correcta

`angularSpectrum()` usa `dfx = 1/(dx*M)` y `dfy = 1/(dy*N)`: cada eje con la
longitud del otro. Al conservarla verbatim, se queda así.

`espectro_angular()` usa los ejes en su sitio: `dfx = 1/(dx*N)`,
`dfy = 1/(dy*M)`. Desaparece la constante `EJES_CRUZADOS` que hasta ahora
permitía reproducir el cruce: existía para poder comparar bit a bit con la
referencia, y esa comparación ya no se hace.

Efecto, y es intencionado:

- **Holograma cuadrado** → los dos métodos dan lo mismo. Comparar sirve de
  control de que la versión por bloques está bien implementada.
- **Holograma rectangular** → dan resultados distintos, y la diferencia es el
  bug de los ejes, no una diferencia de método. Contra el gaussiano analítico el
  error es 2.95e-01 con los ejes cruzados frente a 8.41e-06 con los ejes en su
  sitio.

El script **avisa por consola cuando `M != N`**, diciendo qué significa la
diferencia. Es un `print`, no una constante: no hay nada que elegir, solo algo
que saber al mirar las dos pilas.

### D4 — La salida es |U|² sin gamma

`guardar_intensidad()` escribe `|U|²` normalizado por su máximo, y nada más.

La normalización por el máximo se queda porque un PNG de 8 bits no admite
floats; es una escala **lineal** y no cambia la relación entre valores. La gamma
sí la cambia, y de la forma exacta que convierte intensidad en amplitud. La
identidad va escrita como comentario en la función, junto a la línea que la
evita.

Normalizar por el máximo de cada imagen —y no por un máximo común al barrido—
significa que cada PNG mide **contraste y no brillo absoluto**: dos distancias
son comparables entre sí aunque no les llegue la misma energía. Es lo que ya
hacían los scripts viejos y no hay razón para cambiarlo.

### D5 — Barrido siempre, aunque sea de una distancia

`Z = (z_min, z_max)` y `PASOS` definen un `linspace`. Con un holograma real no
se sabe a qué distancia está el objeto: el barrido es cómo se averigua.

`Z` va en **milímetros y POSITIVA**. El signo lo pone `retropropagar()`, que
llama al propagador con `-z` dentro de la función. Hoy el signo lo pone quien
llama y por eso se puede equivocar; meterlo dentro lo hace imposible.

### D6 — Nombre de archivo y carpeta

Nombre: `z{z:08.3f}.png`, el mismo formato que ya usan los cuatro sitios que
escriben pilas de foco en el repo (`tests/test_guardado_barrido.py` vigila que
no diverjan). Tres decimales porque un `linspace` da distancias no enteras, y
ancho fijo para que el orden alfabético sea el del barrido.

Carpeta, absoluta y bajo la raíz del repo:

    resultados/retro_intensidad/<holograma>/<referencia|bloques>/z0010.000.png

`<holograma>` es el stem de `RUTA`: sin él, dos hologramas distintos al mismo
barrido escriben los mismos nombres en la misma carpeta y el segundo pisa al
primero sin aviso.

## Estructura del archivo

Cinco bloques con cabecera visible, en este orden. El objetivo declarado es
que se pueda leer de arriba abajo y que cada bloque se pueda borrar entero sin
romper los de arriba.

```
"""Docstring: qué hace, qué entra, qué sale, los tres avisos. ~20 líneas."""

imports  (pathlib, numpy, PIL.Image, cupy opcional — NO matplotlib)

# ═══ 1. PARÁMETROS ═══════════════════════════════════════════════
RUTA         el holograma
LAMB         longitud de onda [mm]              633e-6
DELTA        paso de píxel del sensor [mm]      3.45e-3
Z            (z_min, z_max) del barrido [mm], POSITIVAS
PASOS        cuántas distancias                 30
PAD          relleno de ceros                   2
METODOS      ("referencia", "bloques")
DISPOSITIVO  "auto" | "cpu" | "gpu"
SALIDA       carpeta destino, o None

# ═══ 2. RETROPROPAGADOR ══════════════════════════════════════════
angularSpectrum(field, z, wavelength, dx, dy, scale_factor=1)   verbatim
espectro_angular(field, z, wavelength, dx, dy, xp, dtype, filas)
retropropagar(U, z, delta, lamb, pad, metodo, xp, dtype)

# ═══ 3. IMÁGENES ═════════════════════════════════════════════════
cargar_holograma(ruta)      ->  campo = sqrt(I)
guardar_intensidad(I, ruta) ->  |U|² / max, SIN gamma

# ═══ 4. AUXILIARES ═══════════════════════════════════════════════
elegir_dispositivo(preferencia)
a_cpu(a)
nombre_png(z)
carpeta(holograma, metodo)

# ═══ 5. MAIN ═════════════════════════════════════════════════════
main()
```

Cada constante lleva **una línea** de comentario diciendo qué pasa si la
cambias, no un párrafo. Cada bloque lleva una línea diciendo qué se pierde si lo
borras. Ese comentario es parte del entregable, no adorno: el requisito era
poder saber qué se puede eliminar.

Tamaño previsto: ~180 líneas, frente a las 830 de `retro_fft_angular.py`.

## Flujo de datos

```
RUTA ──> cargar_holograma ──> U0 = sqrt(I)  [M×N, float]
                                    │
                    para cada z de linspace(Z[0], Z[1], PASOS):
                        para cada metodo de METODOS:
                                    │
                                    ├─> retropropagar(U0, z, ..., metodo)
                                    │       pad de ceros a (M·PAD, N·PAD)
                                    │       propagador(rel, -z, ...)
                                    │       recorte central a M×N  (copia)
                                    │
                                    ├─> I = |U|²
                                    │
                                    └─> guardar_intensidad
                                            resultados/retro_intensidad/
                                              <holograma>/<metodo>/z0010.000.png
```

## Errores

Solo dos comprobaciones, las dos al arrancar `main()` y las dos con mensaje que
dice qué editar:

1. `RUTA` no existe → `SystemExit` nombrando la constante.
2. Alguna distancia del barrido es ≤ 0 → `SystemExit` explicando que `Z` es la
   separación holograma-objeto y va positiva.

Un aviso, que no detiene nada: si `M != N`, un `print` diciendo que los dos
métodos van a diferir y por qué (D3).

Lo que **no** se comprueba, y es deliberado: memoria disponible, ondas
evanescentes, banda útil contra Nyquist, equivalencia entre los dos
propagadores. Si `"referencia"` no cabe en memoria, el error de NumPy lo dice; la
alternativa era ~60 líneas de diagnóstico en un script cuyo objetivo es no
tenerlas.

## Verificación

No lleva tests automáticos nuevos: es un script de exploración, no un módulo del
paquete, y los `retro_*.py` de los que sale tampoco los tienen (los tests que
los tocan comprueban funciones compartidas, no los scripts).

Se comprueba a mano, y estas dos son la definición de "terminado":

1. Correr sobre `resultados/campos/fft/z0150.png` con `LAMB = 633e-6`,
   `DELTA = 3.45e-3`, `Z = (10, 150)`, `PASOS = 30`. Tienen que salir 60 PNG, 30
   por método, y las dos pilas tienen que ser **indistinguibles** entre sí: la
   imagen es 512×512, cuadrada, y en malla cuadrada los ejes cruzados no cambian
   nada, así que cualquier diferencia visible delata un fallo en la versión por
   bloques.
2. La reconstrucción tiene que enfocar cerca de z = 150 mm.

Esa imagen sirve de holograma de prueba porque es exactamente eso: el demo de
`CamposT/pipeline.py` propaga `usaf_like(512)` con `lamb = 633e-6` y
`delta = 3.45e-3` a z = 20, 60, 150 y 400 mm, y escribe `intensidad(Uz)` con
`guardar(...)`, cuya gamma por defecto es **1.0**. Es |U|² lineal cuantizado a 8
bits, no una amplitud disfrazada. Los parámetros de la corrida de prueba tienen
que ser esos y no otros: cambiar `LAMB` o `DELTA` mueve el foco y la
comprobación deja de significar nada.

Y una comprobación de que el punto del ejercicio se cumple: la misma
reconstrucción guardada por el script viejo y por el nuevo **no** debe
coincidir. Si coinciden, la gamma sigue puesta.
