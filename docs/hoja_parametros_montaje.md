# Hoja de parámetros del montaje DLHM

Entregable de la **tarea 17** del cronograma. Llévatelo al laboratorio y vuelve
con las casillas llenas: es lo que desbloquea 11 tareas aguas abajo (24–27,
29–31, 34–35, 39–40).

> **TODO EN MILÍMETROS.** El paquete trabaja en mm para longitudes de onda,
> pasos de píxel y distancias, sin excepción. Anota en las unidades del
> instrumento y convierte después — la conversión está en cada fila.
>
> | Lees | Escribes |
> |---|---|
> | 633 nm | `633e-6` |
> | 3.45 µm | `3.45e-3` |
> | 20 mm | `20.0` |
> | 1.5 cm | `15.0` |

---

## 1. Fuente

| Magnitud | Símbolo | Valor | Por qué la necesita el código |
|---|---|---|---|
| Longitud de onda | λ | ☐ ________ nm | Entra en `z_lim`, en `Kf`, en el radio del cono de difracción y en la distancia a la que enfoca la reconstrucción |
| Tipo de láser | — | ☐ ________ | Para citarlo. HeNe son 632.8 nm exactos, no 633 |
| ¿Pinhole? Diámetro | — | ☐ ________ µm | De aquí sale NA si no está especificada |
| Potencia / atenuación | — | ☐ ________ | Solo para la bitácora |

**Ojo:** ahora mismo el repo tiene tres λ distintas conviviendo (632.8, 633 y
405 nm). Entre 633 y 405 hay un 56 %. Necesito la tuya, no una plausible.

## 2. Sensor

| Magnitud | Símbolo | Valor | Por qué |
|---|---|---|---|
| Modelo | — | ☐ ________ | Para citarlo y para buscar la hoja de datos |
| Paso de píxel horizontal | δx | ☐ ________ µm | El muestreo del plano. Todo el criterio de validez cuelga de esto |
| Paso de píxel vertical | δy | ☐ ________ µm | **Anótalo aparte aunque parezca igual** — ver el aviso de abajo |
| Resolución | M × N | ☐ ______ × ______ px | Define la malla y, con δ, el tamaño físico de la ventana |
| Área activa | — | ☐ ______ × ______ mm | Comprobación cruzada: debe dar M·δx × N·δy |
| Profundidad de bits | — | ☐ ______ bits | 8 bits satura antes; cambia el contraste utilizable |

> ### ⚠ Si el sensor no es cuadrado
> La tarea 61 corrigió `Kf` para que se calcule **por eje**, porque `Kf` va como
> ~1/√N y con un solo valor el eje corto quedaba submuestreado en silencio. En
> una malla 64×128 el error medido fue de 5× a 63× mayor. Los sensores de DLHM
> rara vez son cuadrados: **si δx ≠ δy o M ≠ N, anótalo explícitamente.**

## 3. Geometría DLHM

| Magnitud | Símbolo | Valor | Por qué |
|---|---|---|---|
| Fuente → sensor | L | ☐ ________ mm | Con z da la magnificación M = L/z |
| Fuente → muestra | z | ☐ ________ mm | La distancia de reconstrucción |
| **Rango ajustable de z** | z_min, z_max | ☐ ____ a ____ mm | **Define la malla del barrido de la tarea 29.** Sin esto el barrido se diseña a ciegas |
| Apertura numérica | NA | ☐ ________ | Junto con λ, el límite de resolución teórico contra el que se compara la tarea 40 |
| Incertidumbre de L y z | — | ☐ ± ______ mm | Cómo mediste: regla, tornillo micrométrico, platina graduada |

**Magnificación esperada:** M = L/z. Con los valores que anotes, calcula M ahí
mismo y comprueba que es del orden que esperas ver en el holograma. Si sale
absurda, hay un error de medida y es mejor descubrirlo estando allí.

## 4. Adquisición — la trampa de la gamma

> ### ⚠ Esto ya nos mordió una vez
> La tarea 69 encontró que una gamma de 0.5 en el guardado hacía que los paneles
> mostraran **amplitud etiquetada como intensidad**, y que la gamma **mueve la
> distancia a la que enfoca la reconstrucción**. Nivel medio medido: 35 sin
> gamma frente a 89 con ella.

| Pregunta | Respuesta |
|---|---|
| ¿En qué formato guarda la cámara? | ☐ ________ (PNG, TIFF, RAW…) |
| ¿Aplica gamma o alguna curva? | ☐ Sí / ☐ No / ☐ No se sabe |
| ¿Se puede desactivar? | ☐ Sí / ☐ No |
| ¿Hay control de ganancia o autoexposición? | ☐ ________ |
| Tiempo de exposición usado | ☐ ________ ms |

Si la respuesta a "¿aplica gamma?" es "no se sabe", **captura la misma escena a
dos exposiciones conocidas** (por ejemplo 10 ms y 20 ms). Si la respuesta del
sensor es lineal, el nivel medio se duplica. Es la forma de averiguarlo sin
depender de la documentación.

## 5. Antes de irte

- ☐ Una **captura de referencia sin muestra** (adelanta la tarea 37)
- ☐ Una **captura a oscuras** para el ruido de fondo
- ☐ **Foto del montaje** con la regla o la platina visible, para la bitácora
- ☐ M = L/z calculado y comprobado *in situ*
- ☐ Si sobra tiempo: un holograma cualquiera, aunque sea malo — para tener con
  qué probar `scripts/retro_holograma.py` antes de la semana 11

---

## Al volver

Los números van a `CamposT/montaje.py`, en el bloque `MONTAJE`. Ese módulo es
la única fuente: mientras un parámetro siga sin medir, cualquier cuenta que lo
use falla en el acto con el nombre de la magnitud, en vez de propagar un valor
inventado y devolver un resultado creíble que habría que tirar.
