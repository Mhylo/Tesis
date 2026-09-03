# Extraer el holograma de los tres scripts retro_* — diseño

Fecha: 2026-09-03
Rama: main

## Problema

Los tres `scripts/retro_*.py` autocontenidos (`retro_fft_angular.py`,
`retro_blas.py`, `retro_mpasm.py`) hacen la misma ida y vuelta con distinto
propagador:

    objeto(RUTA) --> U0_obj --+Z--> campo_sensor --−Z--> retropropagado

`campo_sensor` **es el holograma**: el campo en el plano del sensor a `+Z`. Los
tres lo calculan, los tres lo pintan en la figura —el panel se titula
literalmente `holograma |U|^2 a +Z mm`— y los tres **lo tiran** cuando el
proceso termina. No queda en disco.

Eso deja un hueco raro en el repo. `scripts/retro_holograma.py` existe para
retropropagar un holograma **ya grabado**, y hoy no tiene de dónde sacar uno
decente: o tira de `resultados/campos/*.png`, que son campos propagados escritos
por el demo de `pipeline.py` y no hologramas, o del `Simulated_hologram.png` de
`referencia/carlos/`, del que no se controla ni el objeto ni los parámetros.

Los tres scripts fabrican exactamente lo que le falta —un holograma con objeto
conocido, parámetros conocidos y propagador conocido— y lo desperdician.

Y hay un número que el repo cita de memoria en cuatro docstrings distintos: que
retropropagar `sqrt(|U|²)` da correlación **~0.53** con el objeto en vez de
1.00. Nadie puede reproducirlo hoy, porque para eso hacen falta las dos cosas a
la vez: el holograma en intensidad y el campo complejo del mismo instante.

## Alcance

Dentro:

- Los tres scripts escriben, además de lo que ya hacen, **tres archivos** por
  corrida: el holograma en intensidad (PNG), el campo complejo (`.npy`) y los
  parámetros que lo produjeron (`.txt`).
- Cada uno lleva su propia función de guardado. Ver D2.

Fuera, a propósito:

- **La ida, la vuelta, la figura y el barrido no se tocan.** Ni una línea. Este
  cambio sólo añade una escritura al final del camino que ya existe.
- **No se importa `CamposT`.** Es la razón de ser de estos tres scripts (D2).
- **No se guarda el objeto de partida.** Ya existe como archivo, en `RUTA`, y el
  `.txt` anota cuál fue.
- **No se guarda ninguna z del barrido.** El barrido son reconstrucciones, no
  hologramas. El único holograma de la corrida es el de `+Z`.
- **No se toca `scripts/retro_holograma.py`** ni ninguna de las dos CLIs. Son
  los consumidores de esto y ya aceptan los dos formatos sin cambios.
- **No se unifica nada entre los tres scripts.** Ver D2.
- **No se añaden pruebas automáticas.** Ver «Verificación».

## Decisiones

### D1 — Se escriben los dos formatos, y no son redundantes

    resultados/hologramas/<objeto>/<metodo>/z0050.000.png   |campo_sensor|²
                                            z0050.000.npy   campo_sensor
                                            z0050.000.txt   los parámetros

El PNG es **lo que un sensor te habría dado**: intensidad, sin fase. Al
retropropagarlo sale el objeto con su imagen gemela encima.

El `.npy` es **lo que había antes de medirlo**: el campo complejo. Al
retropropagarlo la vuelta deshace la ida exactamente.

`retro_holograma.py` ya distingue los dos por la extensión y dice por consola
cuál tomó, «porque cambia como se lee la figura entera». Tener los dos de la
misma corrida es lo que convierte la comparación en una medida en vez de una
cita.

Dónde se pierde el dato, dicho sin adornos: el PNG normaliza por el máximo —una
escala **lineal**, que no cambia la relación entre valores, pero pierde el
factor absoluto— y después lo cuantiza a 8 bits. El `.npy` conserva todo. Y no
tiene sentido guardar el PNG a 16 bits: `campos.load_field()` hace
`.convert("L")`, así que cualquier imagen entra al paquete a 8 bits de todos
modos.

### D2 — Cada script lleva su copia de la función de guardado

**No se importa `CamposT.pipeline.guardar()`**, aunque haga casi exactamente
esto.

Estos tres scripts son autocontenidos a propósito, y eso no es una casualidad
histórica: es su función. Son el contraste independiente del paquete —si el
paquete y el script coinciden, es que los dos aciertan; si el script importara
del paquete, coincidirían siempre y la comprobación no valdría nada—. Meterles
un `import CamposT` los convierte en otra cosa.

Así que cada uno gana su `guardar_holograma()`, unas 15 líneas. Es duplicación
deliberada y coherente con lo que ya son: los tres ya duplican `nitidez()`,
`pico_de_foco()`, `a_cpu()`, `objeto()` y los tres `pinta_*()`. Va escrito en el
docstring de la función, para que se lea como decisión y no como descuido.

### D3 — Gamma 1.0, lineal, sin excepción

El PNG se escribe **sin gamma**. `retro_holograma.py` dedica un párrafo entero
del docstring a avisar de que los PNG de `resultados/campos/` llevan gamma 0.6,
que tomarles la raíz da `I^0.3` en vez de `I^0.5`, y que eso **mueve la
distancia a la que enfoca la reconstrucción** —justo lo que un barrido intenta
medir—.

Un holograma que salga de aquí no puede llevar esa trampa dentro. Quien lo
consuma pone `GAMMA_GUARDADO = 1.0`, que además es el defecto.

### D4 — El `.npy` se baja a CPU antes de escribirlo

`campo_sensor` vive en la GPU cuando `DISPOSITIVO` es `"auto"` y hay CuPy. Los
tres scripts ya tienen `a_cpu()` para esto; hay que usarla antes de `np.save`.

El dtype se conserva tal cual sale: `complex64` en GPU, `complex128` en CPU. No
se promociona a `complex128` al guardar, porque el archivo debe decir qué se
calculó de verdad, no una versión maquillada. El `.txt` anota cuál fue.

### D5 — El `.txt` anota también lo que el propagador MIDE, no sólo lo que se le pide

Los parámetros comunes a los tres: `objeto` (la ruta de `RUTA`), `lambda`,
`delta`, `Z`, `ENTRADA`, `PHI_MAX`, `dispositivo`, `dtype`, y la malla `MxN`.

Y los propios de cada uno, que son los que hacen que el mismo objeto a la misma
z dé hologramas distintos:

    retro_fft_angular   EJES_CRUZADOS
    retro_blas          frac_ida — la fracción de banda que la máscara conservó
    retro_mpasm         S, R, MAG, KF, y kf_ida — el Kf que de verdad se usó

Los dos últimos no son ajustes: son **resultados** de la ida. `frac_ida` dice
cuánta banda descartó BL-ASM, y `kf_ida` el coeficiente de compresión que MPASM
calculó. Sin ellos, dos hologramas idénticos en parámetros de entrada pueden
diferir y no habría forma de saber por qué.

Formato: texto plano, una clave por línea, `clave = valor`. No JSON: se lee de
un vistazo en cualquier editor, y nadie va a parsearlo con código.

### D6 — Nombres limpios, y por eso hace falta el `.txt`

El nombre es `z{z:08.3f}`, sin `lambda` ni `delta` metidos dentro. Es el mismo
ancho fijo de tres decimales de `nombre_png()`
(`CamposT/retropropagacion.py:140`), que protege
`tests/test_retropropagacion.py::test_los_nombres_del_barrido_no_colisionan`:
tres decimales porque un `linspace` da distancias no enteras y redondearlas
haría que dos cosas distintas escribieran en el mismo archivo, y ancho fijo para
que el orden alfabético siga al orden de la z.

Los scripts **no lo importan** — no importan `CamposT`, por D2— así que el
formato queda escrito a mano en los tres. Es la misma duplicación deliberada y
por la misma razón; lo que no puede pasar es que diverja del original, así que
va con un comentario que nombre de dónde sale.

La consecuencia se acepta con los ojos abiertos: **dos corridas con la misma z y
distinta lambda escriben el mismo nombre y la segunda pisa a la primera**. El
`.txt` es lo que impide que eso se vuelva un error mudo: si lo abres, dice con
qué parámetros se hizo el archivo que hay ahí. Es la alternativa a nombres como
`z0050.000_l633_d3.450.png`, que son ilegibles y que además sólo empujan el
problema un parámetro más allá.

`<objeto>` es el stem de `RUTA`, igual que `retropropagacion.py` usa el stem del
holograma: sin él, dos objetos distintos a la misma z se pisan también.

### D7 — Las salidas van bajo la raíz del repo

Como en `retropropagacion.py` y `propagacion.py`:
`pathlib.Path(__file__).resolve().parent.parent / "resultados" / "hologramas"`.
Una ruta relativa las dejaría en el directorio desde el que se invoque.

No se usa `SALIDA`: esa constante controla las **figuras**, vale `None` por
defecto en los tres, y con `None` no se guardaría ningún holograma —que es justo
lo contrario de lo que se pide—.

## Interfaz

En cada uno de los tres scripts, una función nueva:

```python
def guardar_holograma(campo, z, parametros):
    """campo_sensor -> tres archivos: .png, .npy y .txt."""
```

Y una llamada en `main()`, **justo encima del comentario
`# ---- barrido de foco ---`**, no inmediatamente después de la línea de la ida.
Ese punto existe en los tres scripts y cae en el mismo sitio: después de todos
los `print` de parámetros y de los diagnósticos propios de cada propagador,
antes de que empiece el barrido.

Funcionalmente da igual dónde se ponga —`campo_sensor` ya existe y nadie lo toca
en medio— pero así la consola se lee en orden: primero con qué se hizo, después
qué se escribió. En `retro_mpasm.py` eso además deja el `Kf` impreso justo
encima del `.txt` que lo anota.

Imprime lo que escribió, en el estilo que ya usan las figuras:

```
holograma guardado:
  -> resultados/hologramas/entrada/mpasm/z0050.000.png
  -> resultados/hologramas/entrada/mpasm/z0050.000.npy
  -> resultados/hologramas/entrada/mpasm/z0050.000.txt
```

(`entrada` porque el `RUTA` de `retro_mpasm.py` apunta hoy a
`resultados/campos/entrada.png`. Los otros dos apuntan a `BenchmarkTarget.png`
y escribirían bajo `BenchmarkTarget/`. Los tres tienen además `Z` distinta —10,
200 y 50—, así que tal como están configurados **no colisionan entre sí**.)

## Salidas

`resultados/` está en `.gitignore` salvo `resultados/campos/`, así que estos
hologramas **no entran al repo**. Es lo correcto: son regenerables corriendo el
script, y el `.txt` dice exactamente cómo.

## Verificación

Sin pruebas automáticas nuevas. Estos tres son scripts de exploración y el repo
ya fijó esa política en `2026-08-27-retro-intensidad-design.md`: los tests que
los tocan comprueban funciones compartidas, no los scripts.

La comprobación es manual, y por primera vez es **medible**. Ésta es la
definición de terminado:

1. Correr `retro_mpasm.py` con `Z = 50` y `S = 1`. **Ojo:** eso NO son los
   valores por defecto del archivo committeado -que son `Z = 360.0` y
   `S = 12`- sino los que produjeron las cifras de esta sección, y quedaron
   fuera del commit a propósito (son ediciones de trabajo del usuario). Quien
   quiera reproducir el `~0.89` de aquí tiene que poner esos dos valores a
   mano antes de correr el script. Tienen que aparecer los tres archivos, y el
   `.txt` tiene que decir `S = 1` y el `kf_ida` que la consola imprimió.

2. Coger el `.npy` y ponerlo en `RUTA` de `retro_holograma.py`, con la misma
   `LAMB`, `DELTA` y `Z` que dice el `.txt`. La consola tiene que anunciar el
   camino de campo complejo, y la reconstrucción tiene que devolver el objeto
   **sin gemela**.

3. Repetir con el `.png`, con `GAMMA_GUARDADO = 1.0`. La consola tiene que
   anunciar el camino de intensidad, y ahora sí tiene que aparecer la gemela
   desenfocada encima del objeto.

4. Que el pico de foco del barrido caiga cerca de la `Z` del `.txt` en los dos
   casos. Si el PNG enfoca a una z distinta del `.npy`, hay una gamma metida
   donde no debía (D3).

Los pasos 2 y 3 son el círculo cerrado: el mismo instante físico, medido de las
dos maneras, reconstruido con el mismo código. Es lo que hace reproducible el
~0.53 que hoy se cita de memoria.

## Estado

Diseño aprobado el 2026-09-03. Implementado en los tres scripts: los tres
escriben ya su holograma en los tres formatos, y los archivos están verificados.

## Lo que midió el círculo, y la sorpresa

Reconstruyendo cada holograma con `CamposT.pipeline.propagar()` y correlacionando
`|U|²` contra el objeto de partida:

    holograma          desde .npy   desde .png    objeto           Z
    ---------------------------------------------------------------------
    mpasm/z0050.000      0.8903       0.6297      entrada          50 mm
    fft/z0010.000        0.8018       0.7885      BenchmarkTarget  10 mm
    blas/z0200.000       0.9297         —         BenchmarkTarget 200 mm

**El resultado esperado era que la columna del `.npy` diera ~1.00.** No lo da en
ninguno de los tres, y la razón es distinta en cada uno. Eso NO es un fallo de
esta implementación —los scripts escriben exactamente lo que calculan— sino algo
que la medida sacó a la luz: **la vuelta con el paquete no deshace la ida de
estos tres scripts, porque ninguno de los tres propaga con la identidad.**

    retro_fft_angular   EJES_CRUZADOS = True sobre una malla 3000x4000, o sea NO
                        cuadrada. Los ejes cruzados son el defecto de la
                        implementacion de referencia que este script existe para
                        exhibir; en malla rectangular cambian el resultado, y el
                        paquete -que los tiene en su sitio- no puede deshacerlo.

    retro_blas          la mascara de banda limitada conservo el 10.6 % del
                        espectro (frac_ida = 0.105836). BL-ASM no es la
                        identidad por diseño: devuelve el objeto FILTRADO, y eso
                        es el precio de no aliar. Su propio docstring ya lo dice.

    retro_mpasm         NO es Kf. Rehaciendo la ida a varios S:

                            S= 1  Kf=2.2799  energia en el sensor=74.9%  corr=0.8903
                            S= 6  Kf=1.0000  energia en el sensor=74.9%  corr=0.8903
                            S=12  Kf=1.0000  energia en el sensor=74.9%  corr=0.8903
                            FFT-ASM ida y vuelta          energia=100.0%  corr=1.0000

                        Kf cae a 1.0000 y ni la energia ni la correlacion se
                        mueven un digito: el mecanismo real es otro, y es
                        independiente de S, R y Kf para la ventana central.
                        MPASM evalua la transformada inversa sobre una
                        ventana de salida explicita y TIRA lo que cae afuera.
                        A Z = 50 mm el ensanche de difraccion es
                        sqrt(lambda*z) = 0.178 mm sobre una ventana de
                        1.766 mm, asi que como un cuarto de la energia se sale
                        del borde y nunca llega al plano del sensor. FFT-ASM en
                        cambio envuelve esa energia en vez de tirarla -por eso
                        le da invertible exacto, corr = 1.0000, energia
                        100 %-. Agrandar la ventana de salida lo confirma: con
                        R = 2 se recupera 92.9 % de la energia y con R = 4,
                        98.2 %. Son condiciones de frontera, no compresion.

Lo que **sí** quedó demostrado, y era el objetivo:

- El par `.png` / `.npy` del mismo instante permite comparar los dos caminos, y
  la caída es real y medible. En el caso de MPASM, **0.8903 contra 0.6297**: eso
  es lo que cuesta que el sensor tire la fase.
- Las figuras lo confirman cualitativamente: desde el `.npy` la carta USAF sale
  con las barras nítidas; desde el `.png` el objeto queda casi tapado por la
  gemela y la autocorrelación.

Lo que **no** quedó demostrado: el `~0.53` que citan cuatro docstrings del repo.
Sigue sin ser reproducible, y ahora se sabe por qué no bastaba con guardar los
hologramas: haría falta un caso sin defecto deliberado —malla cuadrada, ejes en
su sitio, sin banda limitada y sin compresión— que ninguno de estos tres scripts
ofrece tal como están configurados. Queda pendiente y **no se ha inventado
ninguna cifra para taparlo**.
