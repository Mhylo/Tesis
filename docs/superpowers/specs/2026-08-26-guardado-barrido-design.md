# Guardado de las reconstrucciones del barrido — diseño

Fecha: 2026-08-26
Rama: referencias-y-versionado

## Problema

Los tres scripts de ida y vuelta (`scripts/retro_blas.py`,
`scripts/retro_fft_angular.py`, `scripts/retro_mpasm.py`) hacen un barrido de
foco: reconstruyen a 25 distancias, por las dos vueltas —A desde el campo
complejo, B desde `sqrt(I)`— y de cada reconstrucción se quedan con un
escalar, `nitidez()`. El campo se descarta acto seguido (`del Ua`, `del Ub`).

Del barrido sólo sobrevive, entonces, la curva de nitidez y una línea que
dice dónde está el pico. Cuando esa línea y la imagen no coinciden no hay
forma de mirar qué pasó a las demás distancias sin volver a correr el barrido
entero cambiando el código. Eso es exactamente lo que ocurrió al revisar el
pico de la vuelta B: la curva decía 50 mm y la reconstrucción mostrada no
estaba enfocada, y para verlo hubo que reconstruir el barrido a mano.

La pila de foco es el dato; hoy se tira.

## Alcance

Dentro:

- Escribir a disco la intensidad reconstruida de cada distancia del barrido,
  por las dos vueltas, en los tres scripts.
- Una constante en el bloque editable para apagarlo.

Fuera, a propósito:

- **El guardarraíl de prominencia del pico.** Que `vuelta B enfoca en z = X`
  se imprima sin comprobar que el pico existe es un defecto real y medido
  (1.5 sigma, y el valor se mueve con la resolución del barrido), pero es un
  cambio de otra naturaleza —qué se afirma— y va aparte. Este diseño sólo
  añade evidencia para poder juzgarlo.
- **Guardar el campo complejo** (`.npy`). Sería lo que hace falta para
  recalcular métricas sin repropagar, pero son 8 MB por distancia en
  complex128 a 512² y ahora mismo nadie los consume. YAGNI.

## Arquitectura

Dos ayudantes nuevos, **una copia en cada script**:

```
nombre_png(z)         -> f"z{z:08.3f}.png"
guardar_png(I, ruta)  -> normaliza por el máximo, gamma 0.5, uint8
```

### Por qué copiados y no importados

`CamposT.pipeline.guardar` y `CamposT.retropropagacion.nombre_png` ya hacen
esto. Importarlos sería menos código, y es justo lo que no se puede hacer: los
tres scripts son autónomos a propósito —"sin importar CamposT: sirve de
contraste independiente del paquete"—, y esa independencia es lo que les da
valor como verificación externa de los propagadores. Un script que importa el
paquete que verifica no verifica nada.

El precio conocido de esa decisión es que las copias diverjan. Se paga como ya
se paga con `sin_piston` y `rms_fase`: con una prueba que las mira a la vez
(`tests/test_fases_retro.py::test_las_tres_copias_dan_el_mismo_numero`).

### Detalles que no son libres

- **`z{z:08.3f}`, tres decimales.** Un `linspace` da distancias no enteras.
  Redondear a entero haría que dos reconstrucciones distintas escribieran en
  el mismo archivo. El ancho fijo mantiene el orden alfabético igual al orden
  del barrido. Es el mismo formato que `CamposT.retropropagacion.nombre_png`,
  y tiene que seguir siéndolo: son cuatro copias, no tres.
- **Gamma 0.5.** Es lo que pintan los paneles de las figuras
  (`(im / im.max()) ** 0.5`). Con gamma 1.0 los PNG del barrido no se
  parecerían a la figura que acompaña al mismo barrido.
- **Guarda del campo nulo.** `I / I.max()` con `I` idénticamente nula da 0/0:
  NaN por todo el array y un PNG de basura, sin error y sin aviso. Es el fallo
  que `pipeline.guardar` ya documenta y arregla; la copia lo lleva también.
  Un negro es un resultado legítimo y hay que poder verlo como tal.

## Ruta de salida

```
resultados/reconstruccion/<objeto>/<metodo>/<A|B>/z00050.000.png
```

- `<objeto>`: stem de `RUTA`. Sin él, correr con `entrada.png` y después con
  `BenchmarkTarget.png` a la misma Z pisa los mismos nombres en silencio. Es
  la misma razón por la que `CamposT.retropropagacion` mete `<holograma>`.
- `<metodo>`: `blas` | `fft` | `mpasm`, uno por script.
- `<A|B>`: las dos vueltas separadas. Son la comparación que motiva el
  barrido; mezcladas en una carpeta no se pueden mirar en secuencia.

## Activación

`GUARDAR_BARRIDO = True`, constante nueva en el bloque editable, junto a
`BARRIDO`. Encendida por defecto.

Apagable porque el coste depende del objeto: a 512² son 82 PNG de ~100 KB por
corrida, pero con `BenchmarkTarget` a 4000×3000 y `REDUCIR_A = None` son 82
archivos de varios MB.

La escritura va dentro del bucle, antes de cada `del`, así que el pico de
memoria no cambia: se escribe el campo que ya estaba vivo y se libera igual.

## Pruebas

`tests/test_guardado_barrido.py`, con el patrón de `test_fases_retro.py`
(fixture parametrizado por módulo, más una prueba que mira las copias juntas):

1. `nombre_png` da lo mismo en los tres scripts **y** en
   `CamposT.retropropagacion.nombre_png`. Es donde de verdad puede divergir:
   son cuatro copias.
2. `nombre_png` no colisiona sobre las distancias de un `linspace` real, que
   es el motivo de los tres decimales.
3. Un campo idénticamente nulo escribe un PNG negro, no NaN.
4. `guardar_png` normaliza por el máximo: escalar la intensidad no cambia los
   bytes del PNG.
5. Los tres `guardar_png` producen bytes idénticos sobre la misma entrada.
