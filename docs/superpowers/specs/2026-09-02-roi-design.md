# ROI: recortar una ventana de la imagen — diseño

Fecha: 2026-09-02
Rama: referencias-y-versionado

## Problema

El repo ya tiene un ROI, pero está atrapado en un archivo.
`scripts/retro_holograma.py` lleva `elegir_roi()` (un `RectangleSelector` de
matplotlib), `radio_del_cono()` y `marca_roi()`, gobernados por la constante
`RECORTAR`. Cuatro límites, y ninguno es de comodidad:

1. **Vive en un script.** Ni `pipeline.propagar()`, ni
   `retropropagacion.retropropagar()`, ni la CLI
   `python -m CamposT.retropropagacion` saben que existe. El paquete es donde
   vive todo lo reutilizable y esto se quedó fuera.
2. **Sólo sirve para la vuelta.** No hay forma de recortar una ventana y
   propagarla hacia adelante.
3. **Sólo es interactivo.** Un recorte hecho arrastrando el ratón no se puede
   anotar en el cronograma ni repetir mañana: la ventana de mañana no es la de
   hoy. Para una tesis eso es un resultado que no se puede reproducir, que es
   como decir que no es un resultado.
4. **No hay ninguna prueba.** Es matplotlib bloqueante dentro de un script, y
   la geometría del recorte —qué píxeles entran, qué pasa si te sales de la
   malla— no la vigila nada.

Y debajo hay una asimetría que esta tarea tiene que resolver para cumplirse:
**la vuelta tiene CLI y la ida no.** `python -m CamposT.retropropagacion
holograma.png --z 20 60` existe; el equivalente hacia adelante no, porque
`python -m CamposT.pipeline` es un benchmark con `usaf_like(512)` cableado que
no acepta una imagen. Hoy, propagar tu propia imagen hacia adelante sólo se
puede escribiendo Python.

## Alcance

Dentro:

- **`CamposT/roi.py`**, módulo nuevo: la clase `Roi`, `radio_del_cono()`,
  `informe()` y el selector `elegir()`.
- **`CamposT/propagacion.py`**, módulo nuevo: la CLI de la ida, espejo exacto
  de `retropropagacion.py`.
- **`--roi` y `--roi-interactivo`** en la CLI de `retropropagacion.py`.
- **`scripts/retro_holograma.py`** importa el módulo y borra su copia
  (`elegir_roi` y `radio_del_cono`, ~90 líneas), y gana una constante `ROI`
  para fijar la ventana a mano sin ratón.
- **`tests/test_roi.py`**.
- `README.md` y el docstring de `CamposT/__init__.py` al día: los dos listan
  los módulos y aquí entran dos.

Fuera, a propósito. Cada una es una decisión, no un olvido:

- **El margen de guarda.** Ver D2. Es lo más caro de la lista y lo que más
  cambiaría el resultado; se descarta con los ojos abiertos.
- **Apodizar el borde del recorte** (ventana de Hann o similar para suavizar
  los dos bordes duros que crea la ventana). Reduciría los anillos, pero
  modifica el dato medido y eso es otra decisión distinta de recortar.
- **ROI no rectangular** (círculo, polígono, máscara arbitraria).
- **ROI en el plano de SALIDA**: pedir «quiero esta región de la
  reconstrucción» en vez de «recorta esta región de la entrada». Es un
  problema distinto y necesita el margen de D2 para significar algo.
- **Medir la nitidez sólo dentro de la ROI** en el barrido de foco.
- **Guardar la ROI en un JSON o un sidecar.** `Roi.como_argumento()` ya la
  reproduce en una línea de texto que cabe en el cronograma; un formato de
  archivo sería una cosa más que versionar.
- **Tocar los propagadores.** Ni una línea de `propagadores.py`. La ROI ocurre
  antes de que el campo llegue a ellos (D1).
- Los tres `retro_*.py` autocontenidos (`retro_fft_angular`, `retro_blas`,
  `retro_mpasm`) **no se tocan**, ni sus pruebas.

## Decisiones

### D1 — La ROI es un recorte previo, no un parámetro del propagador

`Roi.recortar(U)` devuelve un array más chico. Eso es todo. No entra en la
firma de `propagar()`, ni de `retropropagar()`, ni de `mpasm()`.

Dos cosas se siguen de esto, y son la razón de la decisión:

- **Sirve en los dos sentidos por construcción.** «Propagarla» y
  «retropropagarla» no son dos implementaciones: son el mismo recorte seguido
  de la llamada que ya existía. No hay forma de que la ida y la vuelta
  diverjan, porque no hay dos códigos.
- **El diff sobre el paquete existente es casi cero.** `retropropagar()` es un
  generador con validación de signo y filtrado de kwargs por método; meterle un
  recorte dentro le añadiría un estado que no le corresponde.

El precio: quien llame a `retropropagar()` desde un script propio tiene que
escribir `roi.recortar(U_h)` él mismo. Es una línea, y explícita.

### D2 — Recorte seco: no hay margen de guarda

El muestreo a `delta` acota el ángulo de difracción que la malla representa a
`sin(theta) = lambda / (2*delta)`. La luz de **un** punto del objeto llega al
sensor repartida sobre un disco de radio

    r = z * tan(theta) / delta   [px]

Con λ = 633 nm, δ = 3.45 µm eso da **534 px a z = 20 mm**, 1602 px a 60 mm y
4006 px a 150 mm. Una ventana de 256×256 recortada de un holograma tomado a
z = 20 mm deja fuera la mayor parte del cono de cada punto que contiene. La
consecuencia es real y conocida: la reconstrucción sale con menos resolución y
con anillos en los bordes.

La alternativa era recortar `ROI + r` por cada lado, propagar eso y recortar al
final a la ROI pedida, que da el resultado correcto dentro de la ventana. **Se
descarta**: el propósito declarado de esta ROI es coste (que quepa en memoria y
vaya rápido) y detalle local, y con `r = 534 px` una ROI de 256 px pediría una
malla de 1324 px de lado, o sea más de lo que se estaba intentando evitar. Un
margen que casi siempre es más grande que la imagen no es una optimización.

Lo que **sí** se conserva es el número. `radio_del_cono()` se muda a `roi.py` y
sigue siendo lo que ya era en el script: **información, no guarda**. Se imprime
siempre, se avisa cuando `min(ancho, alto) / 2 < r`, y se recorta igual. Es tu
decisión y el diseño la respeta; lo que no hace es dejarte tomarla sin el
número delante.

Nota de paso: el docstring de `retro_holograma.py` dice hoy «~532 px» para ese
caso y el valor es 534.1. Se corrige al reescribir esa sección.

### D3 — Una ROI que se sale de la malla revienta

`recortar(U)` comprueba `x0 + ancho <= U.shape[1]` y `y0 + alto <= U.shape[0]`,
y lanza `ValueError` con las dos formas en el mensaje. **No ajusta al borde.**

Ajustar en silencio devolvería una ventana distinta de la que pediste, con la
forma correcta y el contenido equivocado: un resultado creíble y falso. Es
exactamente el fallo que `SinMedir` existe para impedir en `montaje.py`, el que
`_comprobar_ventana()` impide en `propagadores.py`, y el que motivó el
endurecimiento de `sam()`. Misma familia, misma respuesta: fallar fuerte antes
que devolver un número que parece bueno.

### D4 — `como_argumento()` es la reproducibilidad, y hace round-trip

    >>> Roi(312, 208, 256, 256).como_argumento()
    '--roi 312 208 256 256'

Ésa es la línea que se pega en el cronograma, en el pie de una figura o en el
siguiente comando. La prueba que la protege no comprueba el texto: **parsea esa
cadena con el parser real de la CLI y verifica que sale la misma `Roi`.** Un
formato que sólo se comprueba contra sí mismo no garantiza nada.

Cuando eliges con el ratón, las dos CLIs imprimen esa línea. `elegir()` no
imprime: devuelve la `Roi` y quien la use decide. Separar las dos cosas es lo
que la hace comprobable sin capturar stdout.

### D5 — La ROI se aplica DESPUÉS de `--N`

`load_field(..., N=...)` redimensiona la imagen antes de nada. Las coordenadas
de la ROI son píxeles **de la imagen ya redimensionada**, no del archivo
original.

Con el ratón esto no se puede equivocar: el selector enseña esa misma imagen.
A mano sí, y por eso va escrito en el `--help`, en el docstring del módulo y en
el spec. El orden es: cargar (con `--N`) → recortar → propagar.

### D6 — matplotlib se importa dentro de `elegir()`

`import CamposT.roi` no debe arrastrar matplotlib ni abrir una ventana. El
`import matplotlib.pyplot` y el `RectangleSelector` van **dentro** de la
función.

Es la misma política que ya fija `CamposT/__init__.py`, que no importa nada
para que `import CamposT` no arrastre CuPy. Un módulo del paquete que al
importarse exige un backend gráfico deja de poder usarse desde un test, desde
un servidor sin pantalla o desde un cuaderno.

### D7 — La ida vive en `CamposT/propagacion.py`, no en `pipeline.py`

`pipeline.py` es a la vez el orquestador que usa todo el paquete y el benchmark
que el README documenta: `python -m CamposT.pipeline` propaga `usaf_like` por
los tres métodos y escribe `resultados/campos/`. Meterle un argparse haría que
el mismo comando hiciera dos cosas distintas según le pases algo o no.

`propagacion.py` es un módulo delgado, hermano exacto de `retropropagacion.py`:
lee la imagen con `load_field`, la recorta si hay ROI, llama a `propagar()` y
escribe con `guardar()`. No reimplementa nada.

El nombre se parece a `propagadores.py` y hay que distinguirlos en el
docstring, en una línea: **`propagadores.py` son los algoritmos;
`propagacion.py` es la corrida de ida de punta a punta**, igual que
`retropropagacion.py` es la de vuelta.

### D8 — `barrido_z` y `nombre_png` se importan de `retropropagacion.py`

Las dos son puras y no tienen dirección: un `linspace` de distancias y un nombre
de archivo con ancho fijo. `propagacion.py` las importa de donde ya están.

Importar del módulo de la vuelta desde el de la ida es direccionalmente raro, y
la alternativa era moverlas a `pipeline.py` y re-exportarlas. Se descarta porque
`tests/test_retropropagacion.py:43` las importa de `CamposT.retropropagacion` y
las prueba allí: mover código probado para arreglar una incomodidad estética es
churn. La tercera opción, copiarlas, es exactamente cómo los tres `retro_*.py`
llegaron a sumar 2715 líneas.

Si algún día hay un tercer consumidor, el sitio correcto será `pipeline.py` y
este párrafo dice por qué no se hizo ahora.

### D9 — `informe()` devuelve texto, no imprime

Devuelve el bloque completo —tamaño, fracción de la imagen, radio del cono y el
aviso cuando toca— como una cadena. Las CLIs lo imprimen.

Así se puede comprobar en una prueba que el aviso sale cuando la ventana es más
chica que el cono y **no sale** cuando no lo es, sin capturar stdout.

### D10 — El radio del cono se lee distinto en cada sentido

Es el mismo número y las dos lecturas van en el docstring:

- **En la vuelta** (holograma → objeto) dice **cuánta resolución pierdes**: el
  cono de cada punto del objeto que cae en la ventana está recortado.
- **En la ida** (objeto → sensor) dice **cuánto `--pad` necesitas**: es el
  radio sobre el que se va a extender la luz de cada punto de tu ROI, y si la
  malla rellenada no lo cubre, la convolución circular de la FFT lo devuelve
  por el borde opuesto.

Sin esta nota, el mismo aviso en la CLI de la ida se leería como si estuvieras
perdiendo algo, cuando lo que te está diciendo es que amplíes el relleno.

## Interfaz

```python
# CamposT/roi.py

@dataclass(frozen=True)
class Roi:
    x0: int
    y0: int
    ancho: int
    alto: int
    # __post_init__: enteros, x0/y0 >= 0, ancho/alto >= 2

    def recortar(self, U):        ...   # -> array (alto, ancho); ValueError si no cabe
    def como_argumento(self):     ...   # -> '--roi 312 208 256 256'

def radio_del_cono(z, lamb, delta):      ...  # -> px, o inf si lamb >= 2*delta
def informe(roi, forma, zs, lamb, delta): ... # -> str; forma es (M, N) ANTES
                                              # de recortar. zs escalar o
                                              # secuencia: informa del cono en
                                              # los dos extremos del barrido
def elegir(I, titulo=""):                ...  # -> Roi; matplotlib dentro

def anadir_argumentos(parser):           ...  # --roi / --roi-interactivo
def desde_argumentos(args, U=None, titulo=""): ...  # -> Roi | None
```

Los dos últimos existen porque las dos CLIs piden **los mismos** dos argumentos
con **la misma** ayuda, y definirlos por separado en cada una es como divergen
dos textos que deberían decir lo mismo.

Las dos CLIs, con los mismos dos argumentos y excluyentes entre sí:

```
--roi X0 Y0 ANCHO ALTO     ventana en pixeles de la imagen ya cargada (ver D5)
--roi-interactivo          arrastrala con el raton; imprime la linea --roi
```

```bash
# la vuelta (ya existia; gana --roi)
python -m CamposT.retropropagacion holograma.png --z 10 60 --pasos 25 \
       --roi 312 208 256 256

# la ida (nueva)
python -m CamposT.propagacion objeto.png --z 20 --modo transmitancia \
       --roi-interactivo
```

`propagacion.py` acepta además `--modo`, con los cinco de `load_field`
(`amplitud`, `fase`, `mixto`, `transmitancia`, `holograma`) y **`amplitud` por
defecto**, que es el defecto de `load_field`. Por lo demás repite la lista de
la vuelta: `--delta`, `--lamb`, `--metodos`, `--device`, `--N`, `--pad`,
`--s`, `--gamma`, `--invert`, `--salida`, `--pasos`.

**Dónde se evalúa el informe del cono.** `r` crece con `z`, así que en un
barrido un solo número miente. Las dos CLIs imprimen `informe()` en los **dos
extremos** del barrido (y una sola vez si `--z` trae un único valor).

**El signo, simétrico y explícito**: `--z` es positiva en las dos. La vuelta le
pone el menos por dentro (contrato ya fijado en
`tests/test_retropropagacion.py`); la ida propaga a `+z`. Ninguna de las dos
acepta una distancia negativa.

## Salidas

    resultados/propagacion/<imagen>/<metodo>/z0020.000.png

Espejo de `resultados/retropropagacion/<holograma>/<metodo>/`, con el mismo
`nombre_png(z)` de ancho fijo y tres decimales que ya usan los cuatro sitios
que escriben pilas de foco.

## Verificación

`tests/test_roi.py`, en el estilo de la suite: contratos, no aritmética.

Geometría:

1. `recortar` devuelve exactamente la ventana pedida, contra el slicing hecho a
   mano.
2. Una ROI que se sale de la malla lanza `ValueError` en vez de ajustarse al
   borde (D3).
3. `ancho` o `alto` menores que 2 revientan al construir.
4. La `Roi` es inmutable, como `TABLA1`.

Reproducibilidad:

5. **Round-trip**: `como_argumento()` parseado por el parser real de la CLI
   devuelve la misma `Roi` (D4).
6. La ROI se aplica después de `--N`: con `--N` distinto del tamaño del
   archivo, las coordenadas son de la imagen redimensionada (D5).

El cono:

7. `radio_del_cono` devuelve `inf` cuando `lamb >= 2*delta`.
8. Crece con `|z|` y es simétrico en el signo.
9. `informe` incluye el aviso cuando `min(ancho, alto)/2 < r` y no lo incluye
   cuando no.

Y la que justifica todo el resto:

10. **Recortar y propagar no es propagar y recortar.**
    `propagar(roi.recortar(U), z)[0]` contra `roi.recortar(propagar(U, z)[0])`
    —`propagar()` devuelve `(campo, info)`— sobre el mismo campo, la misma z y
    el mismo método difieren, y la prueba fija cuánto. Es el coste de D2 medido y clavado en la suite, con un número al
    lado en vez de un párrafo. Si algún día alguien implementa el margen de
    guarda, esta prueba es la que dirá si funcionó.

Las dos CLIs se prueban llamando a `main(argv)` sobre un PNG diminuto escrito
en `tmp_path`, comprobando que el recorte llega al resultado (la forma del PNG
de salida es la de la ROI) y que `--roi` y `--roi-interactivo` juntos son un
error de argparse. El selector de ratón no se prueba: es una ventana bloqueante
y esa parte se comprueba a mano.

Comprobación manual, y esto es la definición de terminado:

    python -m CamposT.propagacion resultados/campos/entrada.png --z 20 \
           --roi 128 128 256 256 --lamb 633e-6 --delta 3.45e-3

Tiene que escribir un PNG de 256×256 —la forma de la ROI, con `--pad` valga lo
que valga, porque `propagar()` recorta al tamaño de entrada— y el informe tiene
que decir que el cono a 20 mm mide 534 px, o sea que la ventana se queda corta.

Esa imagen sirve de entrada de prueba porque es el target `usaf_like(512)` que
escribe el demo de `pipeline.py`, y porque **está versionada**: `.gitignore`
ignora `resultados/*` pero exceptúa `resultados/campos/`. La comprobación no
depende de haber corrido nada antes.

## Estado

Diseño aprobado el 2026-09-02. Pendiente el plan de implementación.
