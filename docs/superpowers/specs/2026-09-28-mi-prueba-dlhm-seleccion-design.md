# Selección de foco de barrido_foco en mi_prueba_dlhm — diseño

Fecha: 2026-09-28
Rama: main

## Problema

`scripts/mi_prueba_dlhm.py` reconstruye hologramas DLHM con la cadena de Carlos
y elige el foco con **una sola** métrica: la correlación de Pearson contra una
imagen de referencia reescalada a cada z. Su propia cabecera lo da por roto:
«la reconstrucción funciona, el barrido puntuado no», con correlaciones de
~0.03 a cualquier escala, y concluye que hace falta puntuar **sin referencia**.

`scripts/barrido_foco.py` ya resolvió esa parte en onda plana. Mide cada z con
tres métricas sobre la misma propagación —Tamura, sin referencia y la única
que sirve en el laboratorio, y Pearson y SSIM contra el objeto, que hacen de
vara de medir—. Además barre en dos pasadas (grueso y fino), mide solo dentro
de una región elegida sin recortar el holograma y reporta foco, FWHM, altura,
rizado y alt/riz. El usuario quiere esa misma selección en `mi_prueba_dlhm`.

Hay dos obstáculos que no existen en onda plana:

1. **La región medida de `barrido_foco` no cabe en la cadena de Carlos.**
   Allí el holograma entra entero y `mpasm(ventana=...)` evalúa solo el ROI.
   La cadena de Carlos es por FFT sobre una malla remuestreada que, con el
   sensor entero, pide 15 GB por paso: hoy el ROI tiene que RECORTAR el
   holograma. `CamposT.dlhm.Reconstructor` sí lo permite: guarda una vez el
   espectro del sensor entero (306 MB en 3000x4000) y evalúa en cada z solo
   la ventana que se le pida.
2. **En DLHM la malla de la muestra cambia con z** (paso delta·z/L), y un
   objeto de fase enfoca en el MÍNIMO de contraste de amplitud, no en el
   máximo.

## Alcance

Dentro:

- El bloque de parámetros de selección de `barrido_foco`, con los mismos
  nombres y significado: `MOSTRAR`, `GUARDAR_FIGURA`, `GUARDAR_BARRIDO`,
  `FORMATO_BARRIDO`, `METRICAS`, `EN_CPU`, `SOLAPAR`, `ZOOM_PX`, `Z_MIN`,
  `Z_MAX`, `PASOS`, `PASOS_FINO`, `ROI`, `BORDE_PX`.
- Dos motores de reconstrucción detrás de la misma selección: `MOTOR =
  "exacto"` (CamposT.dlhm, por defecto) o `"carlos"` (la cadena actual).
- `OBJETO = "amplitud" | "fase"`, que sustituye a `COMPARAR`.
- `RUTA = None` simula el holograma con la ida exacta de CamposT.dlhm, como
  `barrido_foco` simula el suyo.
- Las métricas, el barrido, la tabla, el volcado a `.npz` y las dos figuras de
  `barrido_foco`.
- Una suite nueva, `tests/test_mi_prueba_dlhm_seleccion.py`.

Fuera, a propósito:

- **`barrido_foco.py` no se toca.** Tiene trabajo sin commitear (tarea 77,
  ROI y bloques de memoria) y mezclarlo con esto lo enredaría. Por eso las
  métricas se COPIAN y una prueba fija que no diverjan (D5), en vez de sacarlas
  a un módulo común. Ese módulo común puede venir después, cuando la tarea 77
  esté cerrada.
- **Las funciones que usan las pruebas actuales no cambian:** `reconstruir`,
  `malla_remuestreada`, `comprobar_memoria`, `comprobar_equivalencia`,
  `correlacion`, `_reconstruir_referencia` y las copias del backend
  (`elegir_dispositivo`, `a_cpu`, `liberar`, `sincronizar`). Tampoco las cinco
  funciones copiadas de `dlhm.py` de Carlos.
- **No se suprime la imagen gemela** ni se lee un `.txt` junto al holograma.
- **La figura de Kf de `barrido_foco` no se porta** (D10).

## Decisiones

### D1 — Dos motores detrás de la misma selección, el exacto por defecto

La selección (métricas, barrido, región, salida) no sabe qué motor la
alimenta: cada motor es un objeto con un único método,
`campo(z, ventana) -> t` complejo en la ventana pedida. `MOTOR = "exacto"`
es el valor por defecto porque es el único en el que el ROI tiene el sentido
de `barrido_foco` y en el que cabe el sensor entero. `"carlos"` se conserva
porque es la cadena validada contra `reconstruction_dlhm.py` y porque
compararlas sobre el mismo holograma es de lo que va la tesis.

### D2 — La reconstrucción se entrega en la malla del sensor proyectada

Los dos motores devuelven t en la malla que el sensor proyecta desde la
fuente: paso delta·z/L, un píxel por píxel del sensor. En esa malla un detalle
cae en el MISMO píxel a cualquier z: el rayo principal que llega al píxel X
del sensor cruza el plano z en xi = X·z/L, y xi/(delta·z/L) = X/delta. Así que
el ROI es una ventana fija de píxeles, válida en todo el barrido y alineada con
el holograma y con la referencia, igual que en `barrido_foco`, donde la
reconstrucción y |U0| comparten malla.

Con `MOTOR = "exacto"`, una ventana (fila f0, columna c0, alto, ancho) de una
malla M x N se pide como `campo(z, forma=(alto, ancho), centro=(cy, cx))` con
`cy = (f0 + alto/2 - M/2)·delta·z/L` y `cx` igual por columnas. El
`Reconstructor` se construye una vez para el rango `[Z_MIN - paso, Z_MAX +
paso]`, porque el barrido fino puede salirse un paso grueso por cada lado; si
ese rango no cabe en `0 < z < L`, aborta antes de empezar.

### D3 — Motor de Carlos: recorte obligado, centrado, y la huella remuestreada

La cadena de Carlos necesita el holograma recortado al ROI (memoria), y trata
el recorte como centrado en el eje óptico: su onda de referencia sale de
(0, 0) respecto del centro del recorte, como hoy. Si el ROI no está centrado
en la malla, el script lo avisa: la física de ese motor supone el eje en el
centro del recorte.

Su salida cubre todo el ancho del recorte en el plano de la muestra, pero el
objeto ocupa solo el 1/M central: la huella de rayos principales, de
`N_r·z/L` x `M_r·z/L` píxeles sobre una malla de salida `N_r x M_r`. El motor
extrae esa huella y la remuestrea (`cv.resize`, `INTER_AREA`, parte real e
imaginaria por separado, como `resize()`) al tamaño del ROI. Así entrega la
misma malla que el exacto y las métricas no dependen del motor.

Hipótesis a comprobar, no a dar por hecha (ver Verificación): la correlación
de ~0.03 de la cabecera sale de `referencia_a_escala()`, que estira el 1/M
central de la referencia sobre TODA la salida cuando el objeto ocupa solo su
1/M central.

### D4 — OBJETO sustituye a COMPARAR: qué se mide y en qué sentido

Todas las métricas se MAXIMIZAN en el foco, como en `barrido_foco`, para que
argmax, fino, FWHM y alt/riz no cambien.

- `"amplitud"`: Tamura sobre |t|; Pearson y SSIM comparan |t| con |t_ref|.
  Es `barrido_foco` tal cual.
- `"fase"`: Tamura sobre |t| con el signo cambiado, porque un objeto de fase
  tiene en el foco el MÍNIMO de contraste de amplitud (la nota de la tarea 27
  ya lo decía). Pearson y SSIM comparan `angle(t)` con la fase de la
  referencia. Pearson va en valor absoluto, porque el signo depende del
  convenio del modelo (Carlos usa e^{-i phi}); SSIM ya ajusta a·A + b por
  mínimos cuadrados y el signo le da igual.

La referencia se interpreta con `INVERTIR`, igual que al simular: con `True`,
g = 1 - img, y si no, g = img. En amplitud t_ref = g; en fase,
t_ref = exp(i·FASE_MAX·g).

### D5 — Las métricas son copias de las de barrido_foco, y una prueba lo fija

`tamura`, `pearson`, `ssim`, `anchura_media` y `margen_del_pico` se copian de
`barrido_foco.py` con el mismo cuerpo, la misma firma `(A, A_cpu)` y las
mismas globales de módulo (`xp`, `REF`, `REF_CPU`, `REF_RANGO`), que el script
fija antes del barrido. Una prueba saca esas funciones de la fuente de
`barrido_foco.py` con `ast` —importarlo correría su barrido entero— y
comprueba que dan los mismos números que las copias sobre los mismos datos.
El sentido de `"fase"` (D4) se aplica fuera de ellas, envolviéndolas, para
que las copias sigan siendo literales.

### D6 — Sin referencia, solo Tamura; pedir otra cosa aborta

Con un holograma de laboratorio no hay objeto. `REFERENCIA = None` es válido
si `METRICAS` solo contiene `"tamura"`. Si pide `"pearson"` o `"ssim"` sin
referencia, el script aborta antes de reconstruir y dice qué quitar, en vez
de inventar un patrón. Con `RUTA = None` la referencia es obligatoria: es el
objeto que se simula.

### D7 — RUTA = None simula con la ida exacta, y el holograma se guarda

El objeto es la referencia a su tamaño nativo, leída como transmitancia en la
malla delta·Z_REAL/L: con `OBJETO = "amplitud"`, t = g; con `"fase"`,
t = exp(i·FASE_MAX·g). `CamposT.dlhm.holograma` da el holograma de contraste
(`CAMPO = "intensidad"`) o 1 + V (`CAMPO = "complejo"`, sin imagen gemela). Se
guarda en `SALIDA_DIR/holograma.npy`, así que la corrida siguiente puede
leerlo con `RUTA` sin volver a simular (en 3000x4000 la ida tarda del orden
de un minuto).

### D8 — La normalización del holograma grabado: FONDO o la media

El motor exacto necesita el holograma de contraste. Con `RUTA` apuntando a
un archivo (`.png` de 8 bits o `.npy`), `FONDO` es la ruta de la imagen sin
muestra, por la que se divide; con `FONDO = None` se divide por la media del
propio holograma (`fondo="media"` de CamposT.dlhm), que deja dentro la
envolvente del cono del pinhole. El motor de Carlos recibe la intensidad
normalizada a [0, 1], como hoy.

### D9 — BORDE_PX = None mide el cuarto central

Con `ROI = None` la región medida es la malla menos `BORDE_PX` por lado. En
DLHM la periferia trae la imagen gemela y la pérdida de NA del borde del
sensor, y a z alejadas del foco produce gradientes fuertes. Medido el
2026-09-27 en el BenchmarkTarget simulado a z = 3 mm, N = 2048, motor exacto,
con la nitidez de `scripts/verif_dlhm_esferico.py` (energía del gradiente
sobre la media), que no es Tamura: sobre la ventana entera elegía el extremo
del barrido (3.30), y sobre el cuarto central, 3.00. Con Tamura se vuelve a
medir en la Verificación. Por eso `BORDE_PX = None` significa
`3·min(M, N)//8` por lado, que deja el cuarto central del lado corto. Un
número fijo sustituye a esa regla. Con un ROI, `BORDE_PX` no se usa, igual
que en `barrido_foco`.

### D10 — Sin figura de Kf

La figura 3 de `barrido_foco` dibuja el Kf de la Ec. (14) de Zhao, que ninguno
de los dos motores usa. El motor exacto imprime en su lugar su rejilla de
frecuencias (`n_frecuencias` y el periodo P).

### D11 — Las curvas se dibujan reescaladas a [0, 1]

`barrido_foco` divide cada curva por su máximo. Con `OBJETO = "fase"` la
Tamura cambiada de signo es negativa, y dividir por un máximo negativo da la
vuelta a la curva. Aquí cada curva se reescala entre su mínimo y su máximo,
que conserva dónde está el pico y su anchura, que es lo que se compara.

### D12 — La selección vive en funciones, no en main()

Simular, construir el motor, barrer y elegir el foco son funciones con sus
argumentos explícitos, sin leer las constantes del módulo salvo las globales
de las métricas (D5). `main()` solo conecta las constantes con ellas. Es lo
que deja probarlas con mallas de 128 px sin tocar los valores del archivo.

## Interfaz

Los valores por defecto describen el caso que se verifica: el BenchmarkTarget
simulado con los números del montaje.

```python
# ── QUÉ SE MUESTRA Y QUÉ SE GUARDA (idéntico a barrido_foco) ──────────────
MOSTRAR = True
GUARDAR_FIGURA = True
GUARDAR_BARRIDO = False
FORMATO_BARRIDO = "png"            # "png" | "npy" | "ambos"
METRICAS = ("tamura", "pearson", "ssim")
EN_CPU = {"ssim"}
SOLAPAR = True
ZOOM_PX = 200

# ── EL HOLOGRAMA Y EL OBJETO ──────────────────────────────────────────────
RUTA = None                        # holograma grabado (.png/.npy), o None: se simula
FONDO = None                       # solo con RUTA: imagen sin muestra, o None = la media
REFERENCIA = RAIZ / "referencia/carlos/DLHM-model-main/DLHM-model-main/data/BenchmarkTarget.png"
                                   # el objeto; None solo con METRICAS = ("tamura",)
OBJETO = "amplitud"                # "amplitud" | "fase"
INVERTIR = True                    # g = 1 - img (fondo transparente), o img
FASE_MAX = np.pi / 2               # rad, solo con OBJETO = "fase"
Z_REAL = 3.0                       # mm; con RUTA = None, la z que se simula; si no, None o la verdad
CAMPO = "intensidad"               # solo al simular: "intensidad" | "complejo"
LAMB, DELTA, L = MONTAJE.lamb, MONTAJE.delta_x, MONTAJE.L

# ── EL BARRIDO (z = fuente -> muestra) ────────────────────────────────────
Z_MIN, Z_MAX, PASOS = 2.5, 3.5, 101  # grueso: paso 0.01 mm
PASOS_FINO = 21

# ── LA REGIÓN MEDIDA ──────────────────────────────────────────────────────
ROI = None                         # None | True | (X0, Y0, ANCHO, ALTO)
BORDE_PX = None                    # None = 3·min(M, N)//8 por lado (D9)

# ── EL MOTOR ──────────────────────────────────────────────────────────────
MOTOR = "exacto"                   # "exacto" | "carlos"
DISPOSITIVO = "auto"
DTYPE = None
SOBREMUESTREO = None               # solo MOTOR = "carlos"
MEMORIA_MAX_GB = 4.0               # solo MOTOR = "carlos"
FILAS_POR_BLOQUE = 512             # solo MOTOR = "carlos"
```

`SALIDA_DIR = resultados/mi_prueba_dlhm/<nombre>_<motor>_<objeto>[_roi...]`,
con `<nombre>` el del holograma o `sim_<referencia>_z<Z_REAL>mm_<CAMPO>`.

Se van `COMPARAR`, `ROI_HOLOGRAMA`, `ROI_REFERENCIA`, `Z` y la forma actual de
`PASOS`. También `referencia_a_escala()` y `observable()`: quedan sin uso y
ninguna prueba las llama. La hipótesis de D3 se mide ANTES de borrar
`referencia_a_escala()`, con la versión actual.

La salida por consola y en disco es la de `barrido_foco`: la tabla (grueso,
foco, error si hay `Z_REAL`, FWHM, altura, rizado, alt/riz), el reparto de
tiempos por etapa, `curvas.npz`, la figura del barrido con su zoom y la de
campos (holograma y reconstrucción en cada foco elegido, con zoom y el
recuadro del ROI).

## Coste

- Motor exacto, 3000x3000: construir el `Reconstructor` ~5 s una vez, luego
  0.1-0.4 s por z según el tamaño de la ventana. 101 gruesos y un fino de 21
  por pico: del orden de un minuto.
- Simular el BenchmarkTarget entero (3000x4000): del orden de un minuto, una
  vez; después se lee de `holograma.npy`.
- Motor de Carlos con un recorte de 1024: ~1.1 s por paso según la tabla de
  su cabecera, o sea unos 2-3 minutos por barrido.

## Verificación

1. La suite completa sigue verde: las funciones que usan las pruebas no
   cambian.
2. `tests/test_mi_prueba_dlhm_seleccion.py`, en CPU y mallas de 128-256 px:
   - las métricas copiadas dan los mismos números que las de
     `barrido_foco.py` (D5);
   - los dos motores entregan la ventana del ROI alineada con la referencia:
     en `Z_REAL`, la correlación de la ventana con la referencia es alta, y
     con un ROI descentrado la ventana del motor exacto es el trozo
     correspondiente de la reconstrucción entera (D2, D3);
   - con `OBJETO = "fase"`, la Tamura invertida tiene su máximo en `Z_REAL`
     sobre un objeto de fase simulado (D4);
   - un barrido pequeño de punta a punta encuentra `Z_REAL` con los dos
     motores, a menos de un paso fino;
   - pedir Pearson o SSIM sin referencia aborta (D6), y un rango de z que se
     sale de `0 < z < L` con la guarda de un paso también (D2).
3. A escala real, fuera de la suite, con los valores por defecto:
   - motor exacto, amplitud: las tres métricas dan `Z_REAL = 3.000 mm` a
     menos de un paso fino, y se anotan FWHM y alt/riz;
   - motor de Carlos con un ROI centrado de 1024: lo mismo, anotando el
     tiempo;
   - la hipótesis de D3: la correlación en `Z_REAL` con la geometría de
     `referencia_a_escala()` frente a la de la huella;
   - `BORDE_PX`: ventana entera contra el cuarto central (D9);
   - `OBJETO = "fase"` simulado: Tamura invertida en `Z_REAL`.

Si el punto 3 falla en el motor exacto, no se toca el umbral de nada: se mira
por qué, porque significa que la selección no está bien montada sobre el
motor.

## Ajustes del plan (2026-09-28)

Al preparar el plan (`docs/superpowers/plans/2026-09-28-mi-prueba-dlhm-seleccion.md`)
se midió en mallas de 128 px con L = 0.8 mm y z = 0.24 mm (la M del montaje y
la NA de un sensor grande), y eso cambia la letra de esta especificación:

1. **D3, el motor de Carlos.** DIVIDE por |U0|² —`reconstruir()` devuelve
   U·conj(U0), que quita la fase de la referencia pero deja su amplitud, con la
   difracción del borde del recorte—: correlación con el objeto 0.84 dividiendo
   y 0.21 sin dividir. Interpola la huella en las coordenadas exactas de la
   malla del ROI en vez de `cv.resize`, que alinea centros de píxel y corre la
   imagen hasta medio píxel. Y exige un ROI CUADRADO: `reconstruir()` usa el
   mismo paso, delta·P/N, en los dos ejes. Con `ROI = None` sobre un sensor
   rectangular aborta y propone el cuadrado centrado.
2. **D7.** El simulado se guarda como `<nombre>.npy` y no `holograma.npy`: así,
   leerlo después con `RUTA` da el mismo nombre y escribe en la misma carpeta.
3. **Verificación.** La tolerancia del barrido no es «un paso fino»: el error
   lo pone la gemela, no el paso. Con pasos de 5 µm (grueso) y 1 µm (fino) el
   exacto queda a 1.3 µm como mucho y el de Carlos a 2.7 µm. Se exige medio
   paso grueso al exacto y un paso grueso al de Carlos.
4. **Hallazgo.** Con el motor de Carlos y un objeto de fase, la Tamura
   invertida NO enfoca en recortes chicos (se va 19 a 55 µm): el borde del
   recorte domina el contraste de amplitud. Pearson y SSIM sí enfocan. Medido
   después a escala real con un recorte de 1024: ahí sí enfoca, a 5 µm (medio
   paso grueso) sobre un pico ancho. Es un efecto del recorte chico.
5. **Menores.** `margen_del_pico` recibe `ancho` = una décima del barrido (el
   1 mm de `barrido_foco` se comería el barrido DLHM); `GUARDAR_BARRIDO` guarda
   la ventana medida; constante `RESULTADOS` para la raíz de la salida; aviso si
   en fase el fondo no queda en g = 0; en `comprobar_memoria` el texto
   `ROI_HOLOGRAMA` pasa a `ROI`.
6. **La hipótesis de D3, medida** con la versión anterior del script, sobre el
   BenchmarkTarget simulado entero (3000x4000, z = 3 mm, holograma de
   intensidad) y el recorte central de 1024, correlación de |t|² con la
   referencia sin invertir en `Z_REAL`:

   | geometría | correlación |
   |---|---|
   | la vieja, `referencia_a_escala()` | +0.051 |
   | la vieja, dividiendo por \|U0\|² | −0.010 |
   | la vieja con la mejor escala, M de 0.8 a 6.0 | +0.071 (M = 2.20) |
   | la huella, sin dividir | −0.459 |
   | la huella dividiendo: el motor nuevo | −0.731 |
   | el motor exacto sobre la misma ventana | −0.836 |

   El ~0.03 de la cabecera era la geometría: comparaba el 1/M central de la
   referencia estirado sobre TODA la salida, cuando el objeto ocupa solo su 1/M
   central. Ninguna escala lo arregla. El signo negativo es el de la referencia
   sin invertir.
7. **Fuera del plan: un fallo en `CamposT/dlhm.py`.** La ida no podía simular el
   BenchmarkTarget entero: el soporte recortaba la frecuencia del OTRO eje a
   ±1/λ y no a su banda, y con el objeto muestreado a 0.55 µm (por debajo de
   λ/2) la esquina salía evanescente y la rejilla pedía 86.7 GB. Arreglado
   recortando a la banda y ensanchando el soporte guarda·√(λd) por lado (las
   colas de Fresnel, que sin ese ensanche subían el error contra RS a 1.1e-4
   en el trozo de 24x24 px de las pruebas). Contra RS: 5.0e-6 en el eje (antes
   1.8e-5) y 4.3e-6 fuera (antes 4.1e-6); la rejilla de 3000x3000 baja de 30235
   a 20209 frecuencias por eje, y la de 3000x4000 queda en 33503x22295: la ida
   entera tarda 39.5 s en la GPU. Prueba de regresión en `tests/test_dlhm.py`.

## Resultado de la verificación a escala real (2026-09-28)

La tabla completa, con FWHM, alt/riz y tiempos, está en la cabecera de
`scripts/mi_prueba_dlhm.py` («LO QUE SALIÓ»). En resumen, sobre el
BenchmarkTarget simulado entero a z = 3 mm con holograma de intensidad:

- **Motor exacto**, las tres métricas: 3.0000 mm con los valores por defecto,
  con un ROI de 1024 centrado y en el caso peor —Z_REAL a medio paso de la
  rejilla gruesa—, donde el grueso elige un vecino y el fino lo corrige. Con
  el objeto de fase, la Tamura invertida a 1 µm y las otras dos exactas.
- **Motor de Carlos**, ROI de 1024 centrado: a 2 µm en amplitud; en fase, la
  Tamura invertida a 5 µm. Picos unas tres veces más anchos que los del
  exacto: solo ve la NA de su recorte.
- **D9 con Tamura:** la malla entera también da 3.0000, así que el cuarto
  central no es lo que decide el foco, a diferencia de lo medido con la
  nitidez de `verif_dlhm_esferico.py`. Se queda por defecto porque da el pico
  la mitad de ancho (FWHM 0.019 frente a 0.041 mm) y cuesta 3.5 veces menos
  por z (220 frente a 775 ms en la GPU).
- **D3:** la hipótesis se confirmó (tabla del punto 6 de los ajustes).
- **Integración:** la primera corrida real falló porque la ida dejaba ~1.6 GB
  en el pool de CuPy y el motor exacto no cabía; `main()` libera el pool
  después de simular.

## Ampliación: el modelo de Lopera como segundo simulador (2026-09-28)

Pedida por el usuario después de comparar la implementación con Lopera et al.,
*Opt. Express* 32, 48509 (2024): la ida exacta de `CamposT.dlhm` reproduce la
física de su Ec. (1), pero no su modelo de simulación (secciones 4-5), que
sustituye la onda esférica por magnificación + onda plana + Brown-Conrady +
envolvente + 8 bits.

- `SIMULADOR = "exacto" | "lopera"`, solo con `RUTA = None`. `"lopera"` usa su
  `dlhm()`, copiada literal en la sección 2 junto a las otras cinco funciones
  suyas; el alias del módulo pasa a `dlhm_exacto` para no chocar con el nombre.
  Una prueba compara las seis con `referencia/carlos/.../dlhm.py` cuando existe.
- Se llama con `dx_in = 0`: su remuestreo (el paso i) es la identidad con el
  objeto en la malla proyectada, y así no pasa por la rama rota del recorte.
  Verificado bit a bit contra su camino entero a escala real.
- `NA_FUENTE = 0.1`, la de su `main_dlhm.py`: con 0 su envolvente es `exp(-r)`,
  que depende de las unidades.
- Su salida de 8 bits se pasa a [0, 1]. Lleva la envolvente: el motor exacto
  la divide por su media, como un holograma de laboratorio sin `FONDO`.
- Nombre del simulado: `sim_<referencia>_z<Z>mm_lopera`.

**Resultado:** reconstruido con la física exacta, su holograma enfoca 12-20 µm
más allá de `Z_REAL` (2-11 µm con la cadena de Carlos). Quitando uno a uno sus
pasos, el corrimiento sigue (+27 y +16 µm con solo su propagación), así que es
su núcleo paraxial. Su propagación coincide con la ida exacta con un objeto
liso (correlación 1.00000) y se separa con detalle fino (0.895). Tabla completa
en la cabecera del script.

## Estado

Diseño aprobado por partes el 2026-09-28. Implementado y verificado el
2026-09-28 en la rama `mi-prueba-dlhm-seleccion`, sin commit, con el simulador
de Lopera como ampliación.
