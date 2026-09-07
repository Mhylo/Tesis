# Barrido con referencia en mi_prueba_angular — diseño

Fecha: 2026-09-03
Rama: main

## Problema

`scripts/mi_prueba_angular.py` carga un holograma, lo propaga a una distancia
escrita a mano y enseña tres imágenes. Eso es todo. No hay barrido, no hay
métrica, y la distancia se ajusta a ojo.

Con la extracción de hologramas ya en el repo, ese script tiene por primera vez
un dato con **verdad de terreno**: `resultados/hologramas/BenchmarkTarget/fft/z0010.000.png`
viene acompañado de un `.txt` que dice de qué objeto salió
(`BenchmarkTarget.png`) y a qué distancia (`Z = 10.0 mm`). O sea que se puede
comprobar si el barrido acierta, en vez de creerle.

Y hay un defecto de fondo que hoy no se puede ni ver. El script hace:

    I = np.asarray(img) / 255.0
    holograma = angularSpectrum(I, -12, LAMB, DELTA, DELTA)

Pasa **la intensidad como si fuera el campo**. El `.txt` dice que ese PNG es
intensidad medida, así que el campo es `sqrt(I)`. Tomarlo como amplitud eleva el
campo al cuadrado, y eso no es sólo contraste feo: **mueve la distancia a la que
enfoca la reconstrucción**. Sin una métrica, ese desplazamiento es invisible.

El propio docstring del archivo se queja además de que su bloque de parámetros
«describe un script más grande que nunca se escribió»: `Z`, `PASOS`, `PAD`,
`METODOS`, `DISPOSITIVO` y `SALIDA` no los usaba nadie. El usuario ya borró los
seis. Este cambio devuelve dos de ellos —`Z` y `PASOS`— haciendo algo de verdad.

## Alcance

Dentro:

- Tres constantes nuevas: `REFERENCIA`, `ENTRADA`, y el par `Z` / `PASOS`.
- Un barrido de distancias que **puntúa cada z contra la referencia** y dice
  dónde enfoca.
- Una figura de cuatro paneles que sustituye a los tres `plt.show()` sueltos.

Fuera, a propósito:

- **`angularSpectrum()` no se toca.** Es la copia verbatim de la implementación
  de referencia y su valor entero es que nadie la ha tocado. Los ejes cruzados
  que lleva dentro son parte de lo que exhibe, no un fallo que arreglar aquí
  (ver D5).
- **No se importa el PROPAGADOR de `CamposT`.** El script lleva
  `angularSpectrum` copiada tal cual de la implementación de referencia, y si
  usara la del paquete, coincidir con el paquete no probaría nada. Esa
  independencia es sobre **propagadores**, no sobre cualquier import: `roi.py`
  no propaga, recorta, así que sí se importa (ver D8). La primera versión de
  este spec decía «no se importa CamposT» a secas, que era más ancho de lo que
  la razón justifica.
- **No vuelven `PAD`, `METODOS` ni `DISPOSITIVO`.** Se borraron porque no hacían
  nada, y este cambio no les da nada que hacer.
- **No hay GPU.** El script es de una sola pasada en NumPy y así se queda.
- **Sin pruebas automáticas.** Es un script de exploración; el repo ya fijó esa
  política en `2026-08-27-retro-intensidad-design.md`.

## Decisiones

### D1 — La referencia es el objeto, y sirve para puntuar

`REFERENCIA` apunta a la imagen del objeto que produjo el holograma. Para el
holograma que hay hoy en `RUTA`, es
`referencia/carlos/DLHM-model-main/DLHM-model-main/data/BenchmarkTarget.png`,
que es lo que dice su `.txt`.

A cada distancia del barrido se correlaciona la reconstrucción con esa imagen.
El barrido deja de ser un ojímetro: devuelve un número por distancia y una `z`
ganadora.

**Se lee con `.convert("L")`.** Ese archivo viene en RGBA, y el holograma en L.
Sin la conversión, las formas no casan.

### D2 — La métrica es correlación de Pearson, no rms

    corr(a, b) = <a - <a>, b - <b>> / (||a - <a>|| · ||b - <b>||)

sobre `|U|²` contra la referencia.

**Por qué no rms:** el PNG del holograma se guardó normalizado por su máximo, así
que la escala absoluta ya no significa nada. La correlación es invariante a
escala y a desplazamiento; el rms exigiría recalibrar el brillo primero, y esa
calibración sería un parámetro más que ajustar a ojo — justo lo que este cambio
existe para quitar.

### D3 — `ENTRADA` convierte un defecto invisible en un número

    "intensidad"   campo = sqrt(I).  Lo correcto para un holograma: un sensor
                   registra |U|², así que el campo es su raíz.
    "amplitud"     campo = I tal cual. Es lo que el script hace hoy.

El defecto es `"intensidad"`, que es lo correcto.

Lo que hace útil el interruptor no es poder equivocarse, es que **ahora la
equivocación se mide**: corriendo las dos opciones, la `z` ganadora se mueve, y
ese desplazamiento es exactamente lo que cuesta interpretar mal la imagen. Sin
la métrica de D1 no habría forma de verlo.

El nombre y los dos valores son los mismos que ya usan los tres
`scripts/retro_*.py`. No se inventa vocabulario.

### D4 — Las distancias van positivas; el menos lo pone el barrido

`Z = (5.0, 20.0)` y `PASOS = 30` definen un `linspace` de distancias
**positivas**, la separación sensor→objeto tal como se mide. El barrido llama a
`angularSpectrum(campo, -z, ...)`.

Es la convención de todo el repo, fijada por contrato en
`tests/test_retropropagacion.py`, y era lo que decía el propio bloque de
parámetros que el usuario borró.

### D5 — El barrido se comprueba a sí mismo

El holograma de `RUTA` se hizo a `Z = 10.0 mm` según su `.txt`. Con `ENTRADA =
"intensidad"`, el pico de correlación tiene que caer cerca de 10 mm. Si no cae
ahí, algo de la cadena está mal — y ésa es toda la gracia de tener una
referencia.

Que funcione depende de un detalle que conviene dejar escrito en el docstring:
ese holograma se hizo con **ejes cruzados** (`EJES_CRUZADOS = True` en su `.txt`)
sobre una malla rectangular de 3000×4000. Este script usa **la misma**
`angularSpectrum` con los mismos ejes cruzados, así que la vuelta deshace la ida
exactamente. Con un holograma hecho con los ejes en su sitio dejaría de cuadrar.

### D6 — Si la referencia no casa con el holograma: aborta, o recorta sin interpolar

Por defecto (`AJUSTAR_FORMA = False`) **aborta** diciendo las dos formas, y
sugiere la constante. Con `AJUSTAR_FORMA = True` **recorta la mayor, centrada**,
hasta la forma común.

**Recortar y no reescalar** es lo que cumple «que queden a la misma escala sin
que la imagen sufra»: un `resize` pasa cada píxel por un filtro de
interpolación; un recorte no toca ninguno de los que sobreviven. Tampoco se
rellena con ceros: meterían negro en la correlación y falsearían el número que
la métrica existe para dar.

Si el recortado acaba siendo el **holograma**, se le exige la proporción de D8
—ahí el recorte cambia la física—. Sobre la referencia no se exige nada: es sólo
el blanco contra el que se puntúa.

**Lo que esto NO arregla, y por eso avisa siempre.** Reconcilia el campo de
visión y nada más. Dos imágenes de geometrías distintas seguirán dando una
correlación sin sentido con las formas ya casadas. Medido sobre el par que lo
motivó —`Simulated_hologram.png` de `referencia/carlos/` contra
`BenchmarkTarget.png`—: las formas casan tras el recorte y la correlación sale
**−0.0349, plana en toda la z**. No hay foco porque no puede haberlo: ese
holograma es DLHM de fuente puntual (λ = 532 nm, δ = 1.85 µm, M = L/z = 4) sobre
un objeto de **fase pura**, y el script propaga onda plana a 633 nm y 3.45 µm
correlacionando `|U|²`.

### D9 — El `.txt` del holograma manda sobre las constantes

Si junto a `RUTA` hay un `.txt` —los escribe `scripts/retro_fft_angular.py`— el
script lee su `lambda [mm]` y su `delta [mm]` y **aborta si no cuadran** con
`LAMB` y `DELTA`.

Sin esta guarda, reconstruir un holograma de 633 nm con `LAMB = 532e-6` devuelve
una `z` perfectamente creíble y equivocada, porque λ entra en la fase del
propagador. Cuando el archivo dice con qué se hizo, no hay razón para adivinarlo.

No puede ser la única guarda: los hologramas de terceros no traen `.txt`, y ése
es justo el caso donde más falta hacía. De ahí el aviso de D6.

Al leer el nombre del `.txt` se corta la extensión **a mano**, no con
`Path.with_suffix()`: para pathlib `z0010.000.npy` deja `z0010.000`, cuyo sufijo
aparente es `.000`, y `with_suffix` se comería los tres decimales. Es el mismo
fallo que ya mordió al escribir estos archivos.

### D7 — Una figura, no tres ventanas

Cuatro paneles en una fila: la referencia, el holograma de entrada, la
reconstrucción en la `z` ganadora, y la curva de correlación contra distancia con
el pico marcado.

Sustituye a los tres `plt.show()` sueltos de hoy. Tres ventanas que hay que
cerrar una a una no dejan comparar nada, que es justo lo que un barrido con
referencia pide hacer.

### D8 — Una ventana por imagen, con la proporción exigida sólo a la del holograma

Dos constantes simétricas, `ROI_HOLOGRAMA` y `ROI_REFERENCIA`, cada una con
cuatro valores: `None`, `True` (ratón sobre **esa** imagen), una tupla
`(x0, y0, ancho, alto)`, o `"misma"` (copia el rectángulo que resolvió la otra).
Sólo una de las dos puede llevar `"misma"`.

Ventanas independientes permiten **misma forma, distinta posición**, que es lo
que sirve cuando los dos planos están descentrados entre sí. El tamaño final
tiene que coincidir o aborta: no se reescala ninguna.

**La proporción se le exige sólo a la ventana que toca el holograma.**
`angularSpectrum` lleva `dfx = 1/(dx*M)` y `dfy = 1/(dy*N)` **cruzados**, así que
el cruce escala las frecuencias por `M/N` y deshacerlo exige la misma razón en la
vuelta. Medido, con recortes de 1000 px de ancho:

    alto   desvío de la razón   corr en z = 10.0   ¿hay pico?
     750          0.0 %              0.9585           sí
     758          1.1 %              0.7935           NO
    1000         25.0 %              0.5388           NO

**Un 1.1 % ya borra el pico**: la curva se vuelve monótona y no hay foco que
encontrar, mientras la correlación sigue dando un 0.54 creíble. Por eso aborta
en vez de avisar, y sin tolerancia — `roi.alto * N == roi.ancho * M`, entero y
exacto. Sobre la **referencia** no se exige nada: es sólo el blanco contra el que
se puntúa.

Se exige distinto según de dónde venga la ventana: de una tupla, **exacta**
—esos números los escribió el usuario—; del ratón, se **agranda** hasta la razón
válida, porque arrastrando no se puede acertar. Es lo que `elegir()` ya
documenta: la ventana devuelta contiene lo que arrastraste.

**Lo que gana:** el barrido fino de 21 pasos baja de **73.8 s a 3.6 s** con una
ventana de 1000×750, y el foco sigue cayendo en 10.000 mm con correlación 0.9585
en vez de 1.0000. Esa caída de 0.04 es el precio del recorte seco, medido.

**Lo sensible que es la posición:** con las dos ventanas en el mismo sitio sale
0.9585; **desplazando la referencia 40 px, 0.3223**. Si se usan ventanas
independientes para corregir un descentrado, hay que acertar a pocos píxeles.

### D9 — El `.txt` del holograma manda sobre las constantes

Si junto a `RUTA` hay un `.txt` —los escribe `scripts/retro_fft_angular.py`— el
script lee su `lambda [mm]` y su `delta [mm]` y **aborta si no cuadran** con
`LAMB` y `DELTA`.

Sin esta guarda, reconstruir un holograma de 633 nm con `LAMB = 532e-6` devuelve
una `z` perfectamente creíble y equivocada, porque λ entra en la fase del
propagador. Cuando el archivo dice con qué se hizo, no hay razón para adivinarlo.

No puede ser la única guarda: los hologramas de terceros no traen `.txt`, y ése
es justo el caso donde más falta hacía. De ahí el aviso de D6.

Al leer el nombre del `.txt` se corta la extensión **a mano**, no con
`Path.with_suffix()`: para pathlib `z0010.000.npy` deja `z0010.000`, cuyo sufijo
aparente es `.000`, y `with_suffix` se comería los tres decimales. Es el mismo
fallo que ya mordió al escribir estos archivos.

### D7 — Una figura, no tres ventanas

Cuatro paneles en una fila: la referencia, el holograma de entrada, la
reconstrucción en la `z` ganadora, y la curva de correlación contra distancia con
el pico marcado.

Sustituye a los tres `plt.show()` sueltos de hoy. Tres ventanas que hay que
cerrar una a una no dejan comparar nada, que es justo lo que un barrido con
referencia pide hacer.

### D8 — La ROI debe conservar EXACTAMENTE la proporción del marco

La ROI se importa de `CamposT.roi` y se recorta sobre las dos imágenes —el
holograma y la referencia— con el mismo rectángulo, elegido sobre la
**referencia**, que es la legible: un holograma a 10 mm es un patrón de franjas.

Y hay una condición que no se puede negociar. `angularSpectrum` lleva
`dfx = 1/(dx*M)` y `dfy = 1/(dy*N)` **cruzados**, así que el cruce escala las
frecuencias por `M/N` y deshacerlo exige la misma razón en la vuelta. Medido
sobre el holograma por defecto, recortes de 1000 px de ancho:

    alto   desvío de la razón   corr en z = 10.0   ¿hay pico?
     750          0.0 %              0.9585           sí
     758          1.1 %              0.7935           NO
    1000         25.0 %              0.5388           NO

**Un 1.1 % ya borra el pico.** No es que degrade: la curva se vuelve monótona y
no hay foco que encontrar, mientras la correlación sigue dando un 0.54
perfectamente creíble. Eso es un error mudo, y por eso **aborta** en vez de
avisar — a diferencia del recorte seco (D2 del spec del ROI), que degrada de
forma graduada y sí avisa.

Sin tolerancia: se compara `roi.alto * N == roi.ancho * M`, entero y exacto.
Con marco 3000×4000 la razón reducida es 3:4, así que las ventanas válidas son
`3k × 4k`.

**Excepción para el ratón.** Arrastrando no se puede acertar una proporción
exacta, así que `elegir()` se sigue de un ajuste que **agranda** la ventana
hasta la razón válida y dice cuánto la cambió. Es lo mismo que `elegir()` ya
documenta —«la ventana devuelta CONTIENE lo que arrastraste»—. La constante
`ROI` no pasa por ahí: esos números los escribió el usuario, y cambiárselos en
silencio sería devolverle una ventana que no pidió.

**Lo que gana:** medido, el barrido fino de 21 pasos baja de **73.8 s a 3.6 s**
con una ROI de 1000×750, y el foco sigue cayendo en 10.000 mm con correlación
0.9585 en vez de 1.0000. Esa caída de 0.04 es el precio del recorte seco, ahora
medido.

## Interfaz

Las constantes, en el bloque que ya existe:

```python
RUTA       = r"C:\Users\User\Desktop\Tesis\resultados\hologramas\BenchmarkTarget\fft\z0010.000.png"
REFERENCIA = r"C:\Users\User\Desktop\Tesis\referencia\carlos\DLHM-model-main\DLHM-model-main\data\BenchmarkTarget.png"
LAMB       = 633e-6
DELTA      = 3.45e-3
ENTRADA    = "intensidad"      # o "amplitud"
Z          = (5.0, 20.0)       # POSITIVAS
PASOS      = 30
```

Salida por consola: una línea por distancia con su correlación, y al final la
`z` ganadora. Salida gráfica: la figura de D7.

## Coste

Medido en esta máquina, malla 3000×4000 (12.0 Mpx):

- una FFT2: 0.81 s → una propagación ≈ 1.6 s
- barrido de 30 pasos: **≈ 50 s**
- pico de memoria de `angularSpectrum`: ≈ 1.07 GB (seis arrays `complex128`)

Viable tal cual. No hace falta reducir la malla ni añadir GPU.

## Verificación

Sin pruebas automáticas. La comprobación es la corrida, y es la de D5:

1. Con `ENTRADA = "intensidad"`, el pico de correlación cae cerca de **10 mm**,
   que es lo que dice `resultados/hologramas/BenchmarkTarget/fft/z0010.000.txt`.
2. Con `ENTRADA = "amplitud"` —lo que el script hacía— el pico se mueve. **Anotar
   cuánto**: ése es el número que justifica el interruptor.
3. La correlación en el pico con `"intensidad"` tiene que ser alta (el objeto se
   reconoce), no un valor mediocre que gane por poco.

Si el punto 1 falla, no se arregla el umbral: se mira por qué, porque significa
que la cadena holograma → barrido no cierra.

## Estado

Diseño aprobado el 2026-09-03. Pendiente el plan de implementación.
