# La emisión de la fuente en la ida DLHM — diseño

Fecha: 2026-09-29
Rama: fuente-dlhm

## Problema

El usuario notó que los hologramas DLHM «ya no generan el característico patrón
de frente de onda esférico». Lo que se investigó el 2026-09-29:

- **No es una regresión.** Con el código de `main` (27636b5) la ida
  regenera el holograma exacto guardado el 2026-09-28 a las 23:08 con
  diferencia relativa 1.8e-4, y `tests/test_dlhm.py` pasa entero (23
  pruebas, Rayleigh-Sommerfeld a 3.7e-6).
- **El patrón venía de otro modelo.** Hasta a2fd810, `mi_prueba_dlhm.py`
  leía `Simulated_hologram.png` (Carlos) o `resultados/dlhm_propio/holo.png`
  (`dlhm()` de Lopera). Los dos modelos, DESPUÉS de propagar, multiplican
  por una envolvente de la fuente —`exp(-2r/(L·tan(asin NA)))` en `dlhm()`,
  `(1 - r/r.max())` en `realisticDLHM` de pyLHM— y aplican Brown-Conrady.
  Desde fa871bd el script simula con `CamposT.dlhm.holograma()`, que supone
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

> **Sustituida la división por |a_L|² por la división por amplitud:** ver
> «Revisión tras la verificación a escala real», D4bis. `iluminacion()` y el
> piso siguen; cambia cómo se usa el fondo.

`iluminacion(forma, delta, L, fuente, centro=(0, 0))` devuelve |a_L|² en el
sensor: la imagen sin muestra en la misma normalización que el holograma
(dividida por |S_L|²). Con `fuente=None` son unos. No lleva `lamb`: la
función fuente ya lo trae dentro.

Por qué basta dividir por ese fondo para reconstruir: con c = I/|a_L|²,
c - 1 ≈ V/a_L + c.c. El V de cada punto del objeto lleva su a_z, y a_z ≈ a_L
en su proyección, porque es el mismo rayo con el mismo ángulo. El cociente
deshace la envolvente en la muestra y en el sensor a la vez, y la referencia
de las métricas sigue siendo t_ref.

Esa aproximación se rompe cerca de los ceros de a. `Reconstructor(...,
piso=0.0)` pone c - 1 = 0 donde `|fondo| < piso·max|fondo|`, sin dividir ahí
(el módulo, para que valga también con un fondo complejo de amplitud),
como lo que cae fuera del sensor: sin información. Se descartó dividir por
max(fondo, piso): en lo oscuro daría c - 1 ≈ -1, una zona opaca falsa cuyo
borde difracta dentro del ROI. `piso > 0` exige un fondo que sea un array
(con `None`, `'media'` o un escalar aborta), y `piso` va en [0, 1). Con 0
todo queda como hoy.

**Comparada con dividir por amplitud (2026-09-29).** La alternativa es
c1 = (I − fondo)/√fondo en el sensor y la salida dividida por √fondo
proyectado en la muestra. Con campo complejo es exacta y la D4 no: error
6.7e-4 en 256 px con NA 0.17. Con intensidad, que es lo que mide el sensor,
manda la imagen gemela, y la división de la salida la amplifica hacia el borde
de la luz. Medido en el BenchmarkTarget entero (z = 3 mm, pinhole de 5 µm),
correlación de t − 1 con el objeto:

| ventana | sin fuente | D4 | amplitud |
|---|---|---|---|
| ROI del USAF, 256 px | 0.7990 | 0.7900 | 0.7906 |
| centro, 512 px | 0.8231 | 0.7875 | 0.7874 |

En 256 px con NA 0.17 la amplitud empata en el centro y queda peor en la
periferia (0.52 frente a 0.70). Una revisión anterior había medido lo
contrario en parches de otro objeto (N = 1536); con este objeto no se
reproduce. Se queda la D4: da lo mismo donde se mide y es mucho menos código.

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
- **El motor de Carlos** recibe c = holo/fondo donde el fondo supera el
  piso, y 1 donde no, siempre que haya una imagen de fondo: la simulada o
  `FONDO`. `MotorCarlos` lo normaliza a [0, 1] como hoy. Así releer una
  corrida simulada con `RUTA` y `FONDO` da el mismo foco que simularla, con
  los dos motores (decidido por el usuario el 2026-09-29, tras la revisión de
  la Task 5; antes Carlos ignoraba `FONDO`).

  > **Sustituido por D5bis:** ver «Revisión tras la verificación a escala
  > real», D5bis. Con un fondo imagen Carlos recibe 1 + c1 sin normalizar y
  > su salida se corrige por √fondo; ya no recibe c = holo/fondo en [0, 1].

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

> **Sustituido por D9:** ver «Revisión tras la verificación a escala real»,
> D9. Carlos reconstruye su propio recorte centrado, así que el ROI
> descentrado ya no le importa y su aviso no salta.

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

def iluminacion(forma, delta, L, fuente, centro=(0.0, 0.0)):
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

- Una sección «LA FUENTE» en el docstring de `CamposT/dlhm.py` con D1 y D4bis.
- En la cabecera de `mi_prueba_dlhm.py`: `FUENTE` y `PISO`, y lo que salió en
  el punto 4 de la Verificación.
- Una tarea nueva en `docs/Cronograma_TG_DLHM.xlsx`, con la convención de
  siempre: retroactiva, notas con los números medidos, en ASCII.

## Revisión tras la verificación a escala real (2026-09-29)

La primera corrida de la Verificación 4 (Task 8) se bloqueó con dos fallos
reales. Los dos se midieron y el usuario decidió cómo resolverlos.

### Lo que falló

1. **Motor exacto con fuente: SSIM se va −0.035 mm.** Tamura y Pearson
   enfocan en 3.000. La causa, medida con los mismos hologramas cambiando solo
   la normalización (ROI del USAF, barrido de 2.90 a 3.10 mm):

   | normalización | foco Tamura | foco Pearson | foco SSIM | SSIM en 3.000 / máximo |
   |---|---|---|---|---|
   | sin fuente | 3.000 | 3.000 | 3.000 | 0.394 / 0.394 |
   | D4, \|a_L\|² | 3.000 | 3.000 | 2.965 | 0.517 / 0.612 |
   | amplitud | 3.000 | 3.000 | 3.000 | 0.609 / 0.609 |

   La D4 retropropaga V/a_L, que deforma el término lineal cuando la mancha de
   un detalle fino abarca píxeles con distinta iluminación. La deformación
   cambia con z y SSIM la premia a ±35 µm. La comparación de la D4 medía solo
   correlación y Pearson, que no la ven.
2. **Motor de Carlos con el ROI del USAF (256 px): falla incluso sin
   fuente** (Tamura +0.10 mm, SSIM −0.50 mm; con 384 centrado, igual). El
   recorte chico no tiene NA. Con un recorte centrado de 1024 px acierta a
   2 µm con y sin fuente, igual que el 2026-09-28. El riesgo que la D8
   anticipaba, el descentrado, era el menor.

### D4bis — La reconstrucción divide por amplitud

Con un fondo que es una imagen (`fondo` array 2-D, simulado o medido):

- **En el sensor:** c1 = (holo − fondo)/√|fondo| donde √|fondo| ≥ √(piso·max|fondo|)
  y √|fondo| > 0; 0 donde no. La parte lineal es sgn(a_L)(V + V*), que se
  retropropaga sin error a (t − 1)·|a_z| dentro del lóbulo.
- **En la muestra:** la salida se divide por √|fondo| proyectado: el punto
  (y, x) del plano z cae en el sensor en (y, x)·L/z, el mismo rayo, así que
  √fondo ahí es |a_z|. Se interpola bilineal sobre la malla del sensor (en la
  malla por defecto es píxel a píxel). Donde el proyectado cae bajo el piso, o
  fuera del sensor, t = 1.
- Con `fondo` None, `'media'` o escalar, todo sigue como hoy (`_contraste`
  sin piso). `piso > 0` sin fondo imagen aborta. Con un fondo constante da lo
  mismo que dividir. Un holograma complejo con un fondo que varía aborta: el
  fondo es una intensidad.
- El precio, medido: hacia el borde de la luz amplifica la gemela por 1/|a_z|.
  En 256 px con un pinhole de 20 µm la correlación con el objeto en el centro
  del disco (|a|² ≥ 0.5) queda en 0.888 contra 0.893 sin fuente, y en todo el
  disco (|a|² ≥ 0.05) cae a 0.656 contra 0.817. Por eso la prueba de la ida y
  vuelta mide en el centro.

`contraste_amplitud(holo, fondo, piso=0.0) -> (c1, raiz, raiz_min)` es
pública en `CamposT.dlhm`, porque el motor de Carlos la usa igual.

### D5bis — El motor de Carlos, igual

Con un fondo imagen, Carlos recibe h = 1 + c1 **sin normalizar** (su cadena
reconstruye L{1} = 1 exactamente, porque divide por la referencia propagada),
y su salida t se corrige como la del exacto: t = 1 + (t − 1)/√fondo en cada
píxel del recorte (su salida va en la malla del sensor proyectada, píxel a
píxel), t = 1 bajo el piso. Sin fondo imagen, como hoy: el holograma
normalizado a [0, 1]. `mi_prueba_dlhm.contraste_enmascarado()` desaparece.

### D9 — Carlos reconstruye su propio recorte centrado

Se separa el recorte de Carlos de la región medida. `CARLOS_RECORTE = 1024`:
Carlos reconstruye el cuadrado de ese lado CENTRADO en la malla —así su fuente
está en el eje y el aviso de descentrado sobra— y las métricas se miden solo
en la ventana del ROI, dentro de él. La ventana tiene que caber en el recorte,
y el recorte en la malla; si no, aborta y dice qué cambiar. Con
`CARLOS_RECORTE = None`, el recorte es el ROI, como hasta ahora (tiene que ser
cuadrado, y conviene centrado).

Medido en la malla de las pruebas (128x160, ROI de 64): con recorte = ROI,
sin fuente −2.0/−2.7 µm y con fuente y amplitud la Tamura se va −53 µm (la
división de la salida amplifica el borde del recorte, que cae en la ventana);
con un recorte de 96, 1.0/1.3 µm con y sin fuente. Con la fuente, el recorte
tiene que ser mayor que el ROI.

### Verificación, lo que cambia

- La ida y vuelta de la Task 3 mide en el centro del disco (|a|² ≥ 0.5):
  con fuente a 0.02 o menos de sin fuente; dividiendo por la media, < 0.5.
- Pruebas nuevas: la salida del Reconstructor con fondo imagen es
  1 + B{c1}/√fondo proyectado, byte a byte con una construcción a mano; el
  √fondo proyectado coincide con |a(s_z)| en la malla por defecto y en otra
  (más fina y desplazada); bajo el piso la salida vale 1 y un fondo con ceros
  no da NaN; un holograma complejo con un fondo que varía aborta.
- Carlos: con recorte, `campo(z, ventana)` es el trozo de `campo(z, recorte)`;
  una ventana fuera del recorte, o un recorte mayor que la malla, abortan.
- A escala real, las cuatro corridas (exacto y Carlos, sin y con fuente) con
  el ROI del USAF y `CARLOS_RECORTE = 1024`, con la tolerancia de siempre en
  las tres métricas.

## Resultado de la verificación a escala real (2026-09-29)

Las cuatro corridas de la Verificación 4 con el código de las Tasks 9 y 10
(amplitud en la reconstrucción, recorte propio de Carlos): el BenchmarkTarget
simulado entero (3000x4000, z = 3 mm, amplitud, intensidad), ROI = el USAF
central (256 px), `CARLOS_RECORTE = 1024`, `PISO = 0.05`, grueso de 2.5 a
3.5 mm con paso 0.01 y fino de 0.001, GPU de desarrollo (RTX 3050, 4 GB).
Error = foco − 3.000 mm; FWHM del barrido grueso.

| motor | fuente | métrica | foco | error | FWHM | alt/riz |
|---|---|---|---|---|---|---|
| exacto | ninguna | tamura | 3.0000 | +0.000 | 0.0163 | 49.2 |
| exacto | ninguna | pearson | 3.0000 | +0.000 | 0.0274 | 72.9 |
| exacto | ninguna | ssim | 3.0000 | +0.000 | 0.0116 | 18.5 |
| exacto | pinhole | tamura | 3.0010 | +0.001 | 0.0296 | 62.1 |
| exacto | pinhole | pearson | 3.0010 | +0.001 | 0.0453 | 61.5 |
| exacto | pinhole | ssim | 3.0000 | +0.000 | 0.1352 | 54.9 |
| carlos | ninguna | tamura | 3.0010 | +0.001 | 0.0377 | 37.5 |
| carlos | ninguna | pearson | 3.0000 | +0.000 | 0.0500 | 69.0 |
| carlos | ninguna | ssim | 2.5070 | **−0.493** | nan | 9.6 |
| carlos | pinhole | tamura | 3.0010 | +0.001 | 0.0355 | 62.9 |
| carlos | pinhole | pearson | 3.0000 | +0.000 | 0.0507 | 66.9 |
| carlos | pinhole | ssim | 3.0000 | +0.000 | 0.1294 | 60.9 |

Tiempos: simular el holograma 41–44 s; motor exacto 12 s de construcción y
122 planos en 13 s (96 ms/z); Carlos, malla remuestreada de hasta 2358x2358,
23–28 s por barrido (180 ms/z), con `reconstruir()` contra la cadena intacta de
`dlhm.py` a 5.4e-07. Ninguna corrida imprimió AVISO.

**Criterio.** Exacto a medio paso grueso (0.005 mm) y Carlos a un paso grueso
(0.01 mm), en las tres métricas. Las doce celdas cumplen salvo una, **Carlos
sin fuente, SSIM**, que el usuario acepta como un **límite medido, no un
fallo**: para esa celda el criterio es Tamura + Pearson a 1 µm (+0.001 y
+0.000), y las otras once mantienen las tres métricas.

**El límite.** Con el recorte de 1024 y sin fuente, SSIM en la ventana de 256
px del USAF no tiene contraste de foco (alt/riz 9.6): 0.3688 en 3.000, solo
0.006 sobre la mediana de la curva lejos del foco (0.3626), y una joroba ancha
de 0.4074 en 2.51 gana el máximo grueso (error −0.493 mm; la FWHM no se
define). Con la fuente (alt/riz 60.9) o con el motor exacto (18.5), SSIM tiene
un pico limpio en 3.000. La primera vez, con una ventana de 1024 px igual al
recorte de Carlos (Task 8, diagnóstico), SSIM enfocaba en 3.000 sin fuente.
**Hipótesis, no medida:** con el recorte de 1024 Carlos no resuelve las barras
más finas del USAF, y sin fuente las franjas de los targets vecinos llenan la
ventana chica, mientras la fuente las oscurece fuera del disco. SSIM es la vara
de medir de la simulación; en el laboratorio manda la Tamura.

### Punto por punto

Verificación:

1. **Suite entera: cumple.** 602 passed, 4 skipped (77 s) en este commit;
   `tests/test_dlhm.py` 51 passed, `tests/test_mi_prueba_dlhm_seleccion.py` 71
   passed. Con `fuente=None` y `piso=0` nada cambia.
2. **`tests/test_dlhm.py`: cumple.** La ida con fuente contra
   Rayleigh-Sommerfeld: 1.1e-5 en el eje y 4.6e-6 en (0.7, 1.9) mm, con
   tolerancia 1e-4. Ida y vuelta con el Airy en 256 px con un pinhole de 20 µm,
   en el centro del disco (|a|² ≥ 0.5): 0.888 con fuente contra 0.893 sin ella
   (diferencia 0.005, tolerancia 0.02). Las guardas de D3, el piso y la GPU
   igual que la CPU pasan.
3. **`tests/test_mi_prueba_dlhm_seleccion.py`: cumple.** 71 passed: ROI por
   defecto con el USAF central y 48 px de margen, combinaciones de D5 que
   abortan, sufijo `_pinhole5um` y `_fondo.npy`, aviso de D6, c = holo/fondo.
4. **Escala real: cumple con un límite medido.** Once de las doce celdas
   dentro de tolerancia (el exacto a 1 µm como mucho, Carlos a 1 µm); la
   doceava, Carlos sin fuente con SSIM (−0.493 mm), documentada arriba. El
   holograma con `FUENTE = "pinhole"`, en el motor exacto y en el de Carlos,
   se ve como un disco brillante; los anillos de Airy (el primero, 1.75 % del
   pico, primer cero a 710 px) no se distinguen en escala lineal.
   El descentrado del ROI (D8) ya no cuenta: Carlos reconstruye su recorte
   centrado (D9) y no hubo aviso.

Revisión («Lo que falló»):

- **Falló 1, el exacto con fuente y SSIM: resuelto.** De −0.035 mm (alt/riz
  20.7) con la división por |a_L|² a +0.000 (54.9) con la amplitud (D4bis);
  Tamura y Pearson, +0.001 y +0.001.
- **Falló 2, Carlos con el ROI del USAF como recorte: resuelto.** Con el ROI de
  256 px como recorte se iba +0.10 mm (Tamura) incluso sin fuente; con el
  recorte propio de 1024 (D9), Tamura +0.001 y Pearson +0.000 con y sin
  fuente.

Revisión («Verificación, lo que cambia»):

- **Ida y vuelta en el centro del disco: cumple.** 0.888 con fuente contra
  0.893 sin ella, a 0.005 (≤ 0.02).
- **Pruebas nuevas del Reconstructor: cumplen.** Incluidas en los 51 de
  `tests/test_dlhm.py`.
- **Pruebas de Carlos con recorte: cumplen.** Incluidas en los 71 de
  `tests/test_mi_prueba_dlhm_seleccion.py`.
- **Las cuatro corridas con `CARLOS_RECORTE = 1024`: cumplen con el límite
  medido de arriba.**

### Abierto

- **La NA real del banco no está medida**, por eso `FUENTE` va apagada por
  defecto.
- **La imagen gemela**, que la división por amplitud amplifica hacia el borde
  de la luz (en 256 px con un pinhole de 20 µm: 0.656 contra 0.817 en todo el
  disco, |a|² ≥ 0.05).
- **El límite de SSIM en Carlos sin fuente**, con su causa solo como hipótesis.
