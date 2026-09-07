# Reunión con el asesor — qué pedir

Preparado el 2026-09-07. Ordenado por lo que más desbloquea.

---

## 1. Los 15 parámetros del banco

**Lo que más rinde de toda la lista.** `CamposT/montaje.py` los tiene los quince
como `PENDIENTE`, y `docs/hoja_parametros_montaje.md` —la hoja de laboratorio de
la tarea 17— dice que destraban **11 tareas** aguas abajo (24-27, 29-31, 34-35,
39-40).

Llevar esa hoja impresa. Los tres que más pesan, porque hoy el repo trabaja con
valores inventados:

| | por qué no puede esperar |
|---|---|
| **λ real** | Conviven 632.8, 633 y 405 nm en distintos archivos. Entre 633 y 405 hay un 56 %, y λ entra en la fase del propagador: con la equivocada el barrido devuelve una `z` creíble y falsa. Medido. |
| **δx y δy por separado** | Aunque el sensor parezca cuadrado. `Kf` se calcula por eje desde la tarea 61, y con un solo valor el eje corto queda submuestreado en silencio. |
| **L, z, y el recorrido de la platina** | Sin `L` no hay magnificación. Sin `z_min`/`z_max` no se sabe qué barrido es alcanzable en el banco. |

---

## 2. Tarea 24: la distancia efectiva

**Éste es el bloqueo técnico real.** El repo ya decidió *cómo* corregir la
fuente puntual, y decidió **no tocar los propagadores**:

> `CamposT/retropropagacion.py`: «la corrección de fuente puntual divergente…
> entra como un **cambio de coordenadas ANTES** de llamar aquí, sin tocar nada
> de este módulo».

Y señala exactamente qué falta:

> `CamposT/montaje.py`: «la **distancia efectiva** no está aquí a propósito:
> sale del desarrollo de la **tarea 24**, que aún no está escrito, y ponerle una
> fórmula ahora sería inventarla».

### Lo que ya está verificado

**El paso de píxel del cambio de coordenadas.** De `main_Kreuzer.py`:

    kreuzer3F(z=0.68e-3, ..., pixel_pitch_in=3.6e-6, pixel_pitch_out=3.06e-7, L=8e-3)

    M = L/z          = 11.7647
    dx_in / dx_out   = 11.7647      <- coinciden al 0.00 %

o sea **δ_out = δ_in · z/L = δ_in/M**. Verificado con sus propios números, no
derivado.

### Lo que NO está, y es la pregunta

Hay **tres formulaciones** en el material disponible, y no está claro cuál va a
la tesis:

| | qué hace | estado en el repo |
|---|---|---|
| **`reconstruction_dlhm.py`** | onda esférica × holograma, propaga `L−z`, divide `Rec = U·conj(U0)`. Sin cambio de coordenadas. | implementada y **validada** en `scripts/mi_prueba_dlhm.py` (`corr = 1.000000` contra el original) |
| **`kreuzer3F`** | cambio de coordenadas: `δ_in ≠ δ_out`, toma `z` y `L` y resuelve la geometría dentro | referencia disponible, sin usar |
| **El paper (Optics Express 32(27), 2024)** | descompone en difracción de un frente esférico + magnificación no homogénea | es la derivación autorizada, sin escribir |

**Preguntas concretas:**

1. ¿Cuál de las tres es la que debe escribirse como tarea 24?
2. En la formulación de cambio de coordenadas, **¿cuál es la distancia
   efectiva?** El paso ya está (`δ/M`); la distancia no.
3. ¿Las tres son equivalentes, o cada una vale en un régimen distinto?

---

## 3. El holograma de `data/Simulated_hologram.png`

**No es reproducible con el código actual**, y eso lo bloquea como dato de
tesis.

Medido: se comporta como un **recorte central magnificado ×4** (barriendo
escalas contra el objeto, pico limpio en M ≈ 4). Pero el `main_dlhm.py` de hoy
no puede producir eso: toma la otra rama y devuelve el objeto entero con las
columnas aplastadas ×0.75.

**Pedir:** la versión del código o los parámetros exactos con que se generó.

**Preguntar también:** sus dos scripts no concuerdan entre sí —`main_dlhm.py`
usa `L = 8 mm, z = 2 mm`; `reconstruction_dlhm.py` usa `L = 11 mm, z = 4.95 mm`—.
¿Cuál corresponde a ese archivo?

---

## 4. Dos cosas de su código, en tono de pregunta

Es su repositorio. Plantearlas como «me pasó esto, ¿lo estoy usando mal?».

**El recorte de `dlhm()` devuelve un array vacío.** La rama
`if W_provided > W_s` usa los `N, M` **originales** en vez de los del sample ya
remuestreado:

```python
sample = resize(sample, int(M/res_d), int(N/res_d))   # pasa a 750x1000
N_s, M_s = sample.shape                                # 750, 1000  <- se calcula
...
start_x = int(N/2 - Q/2 + x0)                          # ...pero usa N = 3000
sample = sample[1000:2000, 1500:2500]                  # sobre 750x1000: VACIO
```

De ahí sale un `ZeroDivisionError` en cuanto se pide cualquier sensor distinto
del de `main_dlhm.py`. Ahí no salta porque `W_provided` y `W_s` valen
`1.3875e-3` **exactos** y la comparación sale `False` por un empate de flotantes.
**Consecuencia: `dlhm()` sólo genera en esa única configuración.**

**El sentido del remuestreo.** `res_d = dx_in·Mag/dx_out` y luego submuestrea,
dejando el paso del sample en `dx_in·Mag = 7.4 µm`, cuando el píxel del sensor
retroproyectado al plano de la muestra son `dx_out/Mag = 0.4625 µm`. Parece
invertido — pero puede ser que yo esté leyendo mal la convención.

**Mencionar** que el repo ya le aplica **tres parches** (`[parche TG]`, ver
`scripts/parchar_referencias.py`) por incompatibilidades con NumPy 2 y un
refactor a medias. Por si los quiere aguas arriba.

---

## 5. Método

**¿Cómo manejan el sobremuestreo?** El paper exige respetar el criterio de
muestreo para no aliar. Con el sensor de 5.55 mm eso pide una malla de
**13144² ≈ 15 GB** por reconstrucción. Con `of = 1` cabe, pero **alía**: la
reconstrucción sale replicada en una rejilla 3×3 (visto). ¿Recortan, GPU, sensor
más pequeño?

**¿Qué métrica usan sin verdad de terreno?** Con un holograma de laboratorio no
hay objeto conocido. La conclusión de este repo es que hay que puntuar por
**nitidez**, no por correlación contra el objeto. Confirmarlo y preguntar cuál
usan ellos.

---

## 6. Lo más importante

**Un holograma experimental de su banco, con sus parámetros anotados.**

Todo lo construido hasta ahora funciona sobre simulaciones propias, y funciona:
los tres barridos (`mi_prueba_angular`, `mi_prueba_blas`, `mi_prueba_mpasm`)
encuentran la distancia exacta que dice el `.txt` de cada holograma. El salto a
datos reales es el que decide la tesis, y **un solo archivo suyo con procedencia
conocida vale más que un mes de simulación**.
