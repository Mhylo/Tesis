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
- **No se importa `CamposT`.** Sigue siendo un script suelto, sin dependencias
  del paquete.
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

### D6 — Si la referencia no casa con el holograma, revienta

Si `REFERENCIA` y `RUTA` no tienen la misma forma, el script **aborta con un
mensaje que dice las dos formas**. No se reescala ninguna de las dos, y no se
recorta a la intersección.

Reescalar en silencio daría una correlación perfectamente calculable y sin
sentido: la métrica compararía dos muestreos distintos del mismo objeto y el
pico caería donde le diera la gana. Es el mismo criterio que `Roi.recortar()` y
que `SinMedir`: fallar fuerte antes que devolver un número que parece bueno.

Hoy casan las dos (3000×4000), pero eso es una coincidencia del holograma que
hay en `RUTA`, no una garantía.

### D7 — Una figura, no tres ventanas

Cuatro paneles en una fila: la referencia, el holograma de entrada, la
reconstrucción en la `z` ganadora, y la curva de correlación contra distancia con
el pico marcado.

Sustituye a los tres `plt.show()` sueltos de hoy. Tres ventanas que hay que
cerrar una a una no dejan comparar nada, que es justo lo que un barrido con
referencia pide hacer.

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
