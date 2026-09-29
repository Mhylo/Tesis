# La emisión de la fuente en la ida DLHM — diseño

Fecha: 2026-09-29
Rama: fuente-dlhm

## Problema

El usuario notó que los hologramas DLHM «ya no generan el característico patrón
de frente de onda esférico». Lo que se investigó el 2026-09-29:

- **No es una regresión.** Con el código de `main` (56f3694) la ida
  regenera el holograma exacto guardado el 2026-09-28 a las 23:08 con
  diferencia relativa 1.8e-4, y `tests/test_dlhm.py` pasa entero (23
  pruebas, Rayleigh-Sommerfeld a 3.7e-6).
- **El patrón venía de otro modelo.** Hasta 03d77a5, `mi_prueba_dlhm.py`
  leía `Simulated_hologram.png` (Carlos) o `resultados/dlhm_propio/holo.png`
  (`dlhm()` de Lopera). Los dos modelos, DESPUÉS de propagar, multiplican
  por una envolvente de la fuente —`exp(-2r/(L·tan(asin NA)))` en `dlhm()`,
  `(1 - r/r.max())` en `realisticDLHM` de pyLHM— y aplican Brown-Conrady.
  Desde dfe6fe7 el script simula con `CamposT.dlhm.holograma()`, que supone
  una fuente puntual ideal (isótropa) y devuelve I/I_ref. Por el escalado de
  Fresnel eso es un holograma de Gabor de onda plana del objeto magnificado,
  sin firma global.
- **Medido** (media radial centro/borde, BenchmarkTarget a z = 3 mm,
  L = 10 mm, 3000x4000):

  | holograma | centro/borde |
  |---|---|
  | exacto (fase) | 0.98 |
  | exacto × (L/R)², la esférica ideal sin dividir | 1.17 |
  | exacto × la envolvente de `dlhm()`, NA 0.1 | 58.6 |
  | modelo de Lopera entero | 54.7 |

  El aspecto «esférico» es la envolvente de la fuente. Las franjas de cada
  detalle ya están en la ida exacta.

**Lo que falta es física del banco, no un adorno.** El pinhole de 5 µm de
diámetro (`MONTAJE.pinhole`) emite un patrón de Airy: primer cero en
sin θ = 1.2197·λ/d = 0.1288, que en el sensor a L = 10 mm cae a 710 px del
centro. El sensor simulado de 3000x4000 llega a sin θ = 0.416 en la esquina.
Si el pinhole fija la emisión, el holograma del laboratorio es un disco
brillante con anillos de Airy, y la ida exacta ilumina el sensor entero por
igual. La NA real sigue sin medir (`MONTAJE.NA` es `SinMedir`): la puede fijar
también el objetivo que enfoca el láser. Por eso la fuente es una opción y va
apagada por defecto.

## Alcance

Dentro:

- `CamposT/dlhm.py`: `holograma(..., fuente=None)`, `fuente_pinhole()`,
  `iluminacion()` y `Reconstructor(..., piso=0.0)`.
- `scripts/mi_prueba_dlhm.py`: las constantes `FUENTE` y `PISO`, el fondo
  simulado para los dos motores y lo que se guarda para releer la corrida.
- El ROI por defecto de `mi_prueba_dlhm.py`, que pasa del cuarto central al
  grupo USAF central del BenchmarkTarget (D8), pedido por el usuario el
  2026-09-29.
- Pruebas nuevas en `tests/test_dlhm.py` y
  `tests/test_mi_prueba_dlhm_seleccion.py`.
- La corrida a escala real, su resultado en la cabecera del script y una tarea
  nueva del cronograma.

Fuera, a propósito:

- **La envolvente de Lopera como opción con nombre.** Se enchufa por el
  gancho `fuente` si hace falta. La suya se normaliza con el mínimo y el
  máximo del sensor, así que depende del tamaño del sensor y no solo del
  ángulo.
- **Una imagen de fondo medida como fuente.** No hay ninguna todavía.
  Además, al ser intensidad pierde el signo de los anillos de Airy.
- **Brown-Conrady.** Rayleigh-Sommerfeld no la muestra (tarea 24): la
  proyección central entre planos paralelos es lineal.
- **`barrido_foco.py`.** Sigue con la tarea 77 sin commitear.

## Decisiones

### D1 — La fuente es una amplitud a(sin θ) y apodiza la ida por dentro

Una fuente real emite e^{ikr}/r · a(θ), con a la amplitud angular. En
`holograma()` eso entra en los dos extremos de la propagación:

    en la muestra:  J = (t - 1) · a(s_z) · esferica(residuo)     s_z = ρ/sqrt(ρ² + z²)
    en el sensor:   la referencia es a(s_L)·S_L                   s_L = ρ/sqrt(ρ² + L²)

y como hoy V = U_obj/S_L, el holograma de intensidad es

    I/|S_L|² = |a_L + V|²

Es exacto mientras la emisión sea ya su campo lejano en la muestra, porque
entonces el patrón angular viaja pegado a los rayos. Para el pinhole lo es:
su número de Fresnel a z = 3 mm es (2.5 µm)²/(λ·3 mm) = 0.004. Se descartó
multiplicar |a_L|² a la salida: coincide con esto solo donde a varía despacio
frente a la mancha de cada detalle (~60 px), y se equivoca cerca de los ceros
de Airy y en los anillos, donde a cambia de signo.

Convenio: a es una amplitud (real o compleja) con a(0) = 1, así que en el eje
el holograma no cambia de nivel. a tiene que variar despacio frente a la banda
del objeto. El Airy de 5 µm oscila en la muestra a d/(2λz) = 1.6 mm⁻¹, frente
a una banda de 1/(2·0.549 µm) = 911 mm⁻¹. Queda escrito en el docstring y no
se comprueba en tiempo de ejecución.

### D2 — `fuente=None` es la ida de hoy, sin tocar un bit

Con `None` no se evalúa nada nuevo ni se multiplica por 1: el camino es el
actual. Con `complejo=True` se sigue devolviendo V. Lo que cambia con una
fuente es que la referencia deja de ser 1, así que el campo total relativo a
S_L es a_L + V, y el docstring lo dice.

### D3 — Guardas de la función fuente

Antes de propagar se evalúa `fuente(np.zeros(1))` y `fuente` sobre las dos
mallas. Aborta con `ValueError` si |a(0) - 1| > 1e-6, si devuelve algo con
otra forma que su entrada, o si hay valores no finitos. No se verifica la
suavidad (D1).

### D4 — El fondo simulado y el piso del Reconstructor

`iluminacion(forma, delta, lamb, L, fuente, centro=(0, 0))` devuelve |a_L|²
en el sensor: la imagen sin muestra en la misma normalización que el
holograma (dividida por |S_L|²). Con `fuente=None` son unos.

Por qué basta dividir por ese fondo para reconstruir: con c = I/|a_L|²,
c - 1 ≈ V/a_L + c.c. El V de cada punto del objeto lleva su a_z, y a_z ≈ a_L
en su proyección, porque es el mismo rayo con el mismo ángulo. El cociente
deshace la envolvente en la muestra y en el sensor a la vez, y la referencia
de las métricas sigue siendo t_ref.

Esa aproximación se rompe cerca de los ceros de a. `Reconstructor(...,
piso=0.0)` pone c - 1 = 0 donde `fondo < piso·max(fondo)`, sin dividir ahí,
como lo que cae fuera del sensor: sin información. Se descartó dividir por
max(fondo, piso): en lo oscuro daría c - 1 ≈ -1, una zona opaca falsa cuyo
borde difracta dentro del ROI. `piso > 0` exige un fondo que sea un array
(con `None`, `'media'` o un escalar aborta), y `piso` va en [0, 1). Con 0
todo queda como hoy.

### D5 — mi_prueba_dlhm: FUENTE y PISO

- **`FUENTE`** acepta tres valores:
  - `None`, el defecto: lo de hoy;
  - `"pinhole"`: `fuente_pinhole(MONTAJE.pinhole, LAMB)`;
  - una función a(s).
- **`PISO = 0.05`.** Con el pinhole de 5 µm el disco con |a|² ≥ 0.05 tiene
  555 px de radio; con 0.01 serían 632 px y con 0.1, 504.
- **Combinaciones que abortan en `comprobar_configuracion()`,** con un
  mensaje que dice qué cambiar:
  - `FUENTE` con `RUTA` (el holograma grabado ya trae su fuente);
  - `FUENTE` con `SIMULADOR = "lopera"` (trae su propia envolvente);
  - `FUENTE` con `CAMPO = "complejo"`;
  - un texto distinto de `"pinhole"`;
  - `PISO` fuera de [0, 1).
- **El motor exacto** recibe `fondo = iluminacion(...)` y `piso = PISO`.
  `MotorExacto` pasa `piso` al `Reconstructor`.
- **El motor de Carlos** recibe c = holo/fondo donde el fondo supera el piso,
  y 1 donde no, normalizado a [0, 1] como hoy.
- **`PISO` se aplica siempre que se divide por una imagen:** la simulada, o
  `FONDO` con un holograma grabado, que en el laboratorio también tiene
  esquinas oscuras. Con `FONDO = None` (`'media'`) no se usa.

### D6 — Aviso si la región medida sale de la zona iluminada

Con fuente o con `FONDO`, `main()` mide qué fracción de la ventana medida
cae bajo el piso. Si es mayor que cero imprime un AVISO con esa fracción y
con el lado del mayor cuadrado centrado que queda entero dentro de la zona
iluminada, para usarlo de ROI. No aborta: es una elección del usuario. Con el
ROI por defecto (D8) no salta; con `ROI = None` —el cuarto central, 750x1750
px, ±875 px en x— y el pinhole, sí.

### D8 — El ROI por defecto es el grupo USAF central

`ROI = (1884, 1306, 256, 256)`: un cuadrado de 256 px centrado en el grupo
USAF del centro del BenchmarkTarget. El grupo, medido sobre la imagen
(píxeles > 0.5), ocupa las filas 1357-1512 y las columnas 1932-2091, con
centro en (1434.5, 2011.5). El cuadrado le deja 48-51 px de margen por lado
para las franjas, no toca ningún otro grupo del target y su esquina más lejana
queda a 239 px del centro del sensor, dentro del disco iluminado del pinhole
(555 px con `PISO = 0.05`). Es cuadrado porque el motor de Carlos lo exige.

No está centrado en la malla: queda 65.5 px arriba y 11.5 px a la derecha del
centro (1500, 2000). El motor exacto no lo nota, porque el ROI solo elige qué
se evalúa. El de Carlos pone la fuente en el centro del recorte, así que la
supone 0.12 mm corrida en el sensor, e imprime el aviso que ya tiene. Se mide
en la Verificación y no se le cambia nada a ese motor.

El valor describe el caso por defecto: el BenchmarkTarget simulado a
3000x4000. Con otro holograma en `RUTA`, el ROI se cambia a mano o se pone en
`None` o `True`, como hoy. Las filas de «LO QUE SALIÓ» medidas con el cuarto
central siguen valiendo, pero dejan de llamarse «el defecto».

### D7 — Salida: nombre y fondo guardado para releer

- El nombre del holograma simulado lleva la fuente: `_pinhole5um`, con el
  diámetro en µm y formato `:g`, o `_fuente` si es una función.
- Junto al `<nombre>.npy` se guarda `<nombre>_fondo.npy`, y el aviso de
  «ponlo en RUTA» dice también `FONDO = <nombre>_fondo.npy`. Releer la
  corrida con `RUTA` y `FONDO` usa el mismo fondo y el mismo piso (D5).

## Interfaz

```python
# CamposT/dlhm.py
def fuente_pinhole(diametro, lamb):
    """a(s) = 2·J1(x)/x con x = pi·diametro·s/lamb; a(0) = 1. Amplitud con signo."""

def iluminacion(forma, delta, lamb, L, fuente, centro=(0.0, 0.0)):
    """|a(s_L)|² en el sensor (float64, NumPy); unos con fuente=None."""

def holograma(t, paso, lamb, z, L, forma, delta, *, centro_objeto=(0.0, 0.0),
              centro=(0.0, 0.0), banda=None, guarda=4.0, margen=1.1, complejo=False,
              fuente=None, device="auto", dtype=None): ...

class Reconstructor:
    def __init__(self, holo, delta, lamb, L, z, *, fondo=None, piso=0.0,
                 centro=(0.0, 0.0), margen=1.1, device="auto", dtype=None): ...
```

```python
# scripts/mi_prueba_dlhm.py, en 1b (el holograma y el objeto)
FUENTE = None          # None | "pinhole" | funcion a(sin theta) con a(0) = 1
PISO = 0.05            # fraccion del maximo del fondo bajo la que c - 1 = 0

# en 1d (la region medida)
ROI = (1884, 1306, 256, 256)   # el grupo USAF central del BenchmarkTarget (D8)
```

`reconstruir(holo, ..., **kw)` hereda `piso` por `**kw`.

## Coste

- La ida con fuente: dos evaluaciones de a y un producto punto a punto más
  sobre el objeto y sobre el sensor. Frente a los ~40 s de la ida entera en
  3000x4000, no se nota.
- `iluminacion()` y la máscara del piso: una pasada sobre el sensor.

## Verificación

1. La suite entera sigue verde. Con `fuente=None` y `piso=0` nada cambia.
2. `tests/test_dlhm.py`, en CPU y mallas chicas:
   - `fuente_pinhole`: a(0) = 1, primer cero en s = 1.2197·λ/d, y coincide
     con 2J1(x)/x de `scipy.special.j1` en puntos sueltos.
   - **La ida con fuente contra Rayleigh-Sommerfeld** con la fuente dentro:
     `rs_directo` gana `fuente`, que pesa cada punto del objeto con
     a(ρ⊥/ρ). Mismos casos que la prueba de RS actual, (0, 0) y
     (0.7, 1.9) mm. El segundo cae en sin θ = 0.199, más allá del primer
     cero, donde a es negativa. Tolerancia: la de hoy, 1e-4.
   - La intensidad con fuente es |a_L + V|² con el V de `complejo=True`;
     `holograma(t ≡ 1, fuente)` es `iluminacion()`; una fuente a ≡ 1 da lo
     mismo que `None` (rtol 1e-12).
   - Las guardas de D3 abortan con `ValueError`.
   - `piso`: donde el fondo cae bajo el piso, c - 1 = 0 sin dividir (un fondo
     con ceros no da NaN); con `piso = 0`, igual que hoy; `piso > 0` sin
     fondo array aborta.
   - Ida y vuelta con el Airy: se simula con la fuente y se reconstruye con
     `fondo = iluminacion()` y `piso = 0.05`. Dentro del disco iluminado la
     correlación con el objeto queda a 0.02 o menos de la que da sin fuente.
   - GPU igual que CPU con la fuente (marcada `gpu`).
3. `tests/test_mi_prueba_dlhm_seleccion.py`:
   - el ROI por defecto es un cuadrado que contiene el grupo USAF central de
     `BenchmarkTarget.png` con 48 px de margen como mínimo, sin tocar otro
     grupo (se salta si `referencia/` no está);
   - las combinaciones de D5 abortan;
   - el sufijo `_pinhole5um` en el nombre y el `_fondo.npy` (D7);
   - el aviso de D6, con el lado del cuadrado sugerido;
   - con la fuente, c = holo/fondo tiene media ~1 dentro del disco.
4. **A escala real, fuera de la suite.** El BenchmarkTarget simulado entero
   (3000x4000, z = 3 mm, amplitud, intensidad) con el ROI por defecto (el
   USAF central, D8), los dos motores, sin fuente y con
   `FUENTE = "pinhole"`: cuatro corridas. Se exige la tolerancia de
   2026-09-28: el exacto a medio paso grueso y el de Carlos a un paso
   grueso. Se anotan foco, FWHM y alt/riz de las cuatro, y el aspecto del
   holograma con fuente. Si falla, no se toca ningún umbral: se busca por
   qué. En particular, si el de Carlos falla con el ROI descentrado, se
   repite con un cuadrado centrado que contenga el USAF, y se anotan las dos.

## Documentación

- Una sección «LA FUENTE» en el docstring de `CamposT/dlhm.py` con D1 y D4.
- En la cabecera de `mi_prueba_dlhm.py`: `FUENTE` y `PISO`, y lo que salió en
  el punto 4 de la Verificación.
- Una tarea nueva en `docs/Cronograma_TG_DLHM.xlsx`, con la convención de
  siempre: retroactiva, notas con los números medidos, en ASCII.
