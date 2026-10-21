# Manuscrito: hologramas experimentales (objetivo 3) y la comparación ASM–MPASM reunida

Fecha: 2026-10-20. Plazo: borrador completo para Carlos el 21/10 a medianoche;
entrega el 30/10.

## Intención

El borrador que recibe Carlos tiene que decir dos cosas que hoy no dice:

1. **Los hologramas experimentales se reconstruyeron**, y para hacerlo hubo que
   corregir la geometría del banco: el píxel y L que da `CamposT/montaje.py`
   (1.83 µm, 10 mm) no cuadran con la escala del USAF. Además, sobre esos
   mismos hologramas, el MPASM esférico se compara con la cadena de Carlos
   (ASM con onda esférica). Se explican sus parecidos y diferencias, y se aclara
   que el MPASM rinde más donde la propagación es "larga" en el sentido del
   criterio de ventana, no en milímetros.
2. **La comparación ASM–MPASM se cuenta una sola vez.** Hoy está repartida entre
   la Sección 3.2.4 (`sec:metodo-fresnel`), la 4.2.3 (`sec:res-fresnel`) y la
   discusión, que repite los mismos números (0.07 dB, 0.01 dB, 0.34 → 3.05 dB,
   32 MB frente a 32 GB, H(z)H(−z) = 1 dos veces). Pasa a ser un solo ensayo de
   ida y vuelta con las cuatro combinaciones (ASM→ASM, ASM→MPASM, MPASM→ASM y
   MPASM→MPASM), con fase, en tres escalas de distancia, y una explicación de
   cada régimen.

## Objetivos

- Que el borrador del 21/10 no deje `\pendiente` en las piezas de abajo.
- Que cada número del texto salga de un script del repo (regla del
  manuscrito).
- Que "distancia larga" se escriba con el criterio z_eq frente a N(δ/M)²/λ
  (equivalentemente, D = L(L−z)/z frente a Nδ²/λ en la malla del sensor). Nunca
  en milímetros a secas: en el DLHM las distancias físicas son de milímetros, y
  el cociente crece al acercar la muestra a la fuente.

## No objetivos

- No se cambia la conclusión de la investigación cerrada el 08/10: en el banco
  nominal s y K_f no cambian nada medible. Lo nuevo es que los hologramas de
  Carlos, con su geometría real, caen en cocientes de 1.2 a 6.2. Si el
  experimento lo confirma, se dice; si no, también.
- No se editan los objetivos (van literales del anteproyecto).
- No se toca `scripts/mi_prueba_dlhm.py`: su `MotorCarlos` y `reconstruir()` se
  importan.
- Los demás `\pendiente` (resumen, conclusiones, 4.2.2 con los mapas de
  `barrido_obj2`, el número MPASM–CZT, la cita de pyLHM, el repositorio y su
  licencia) quedan fuera de este trabajo. Decidido por el usuario el 20/10: se
  miran después.

## Anti-objetivos (lo que lo haría fracasar aunque compile)

- Presentar el píxel de ~3.8 µm y L ≈ 18.2 mm como medidos. Son inferidos de
  los datos y van marcados como "por confirmar con el laboratorio".
- Llamar "distancia de propagación" a D (148–783 mm). D es la misma propagación
  expresada en la malla del sensor; las distancias físicas son z = 0.41–2.0 mm y
  L − z = 16–18 mm.
- Mezclar las dos geometrías: los números de Carlos (528 nm, 1.83 µm, L = 10,
  z = 3) son para el ensayo simulado del banco; la geometría corregida es solo
  para los hologramas experimentales.
- Validar un propagador con su propia ida y vuelta (ASM→ASM a 66 dB a cualquier
  z no mide nada).

## Restricciones

- GPU de 4 GB: el `Reconstructor` con el sensor entero (3506×4640) pide 6.6 GB;
  se usa un recorte central (2048×2560 cabe). La cadena de Carlos necesita un
  recorte cuadrado (1024, como `CARLOS_RECORTE`).
- Los .tex y `scripts/figuras_manuscrito.py` tienen cambios sin commitear del
  usuario: se edita encima, sin revertir nada. Sin commits salvo que el usuario
  los pida.
- Los datos (dos zip, 3.2 GB) quedan fuera del repo; los scripts reciben la
  carpeta descomprimida como constante `DATOS`.

## Lo decidido con el usuario

- La Sección 4.3 dice que se reconstruyeron los experimentales, con la geometría
  corregida, y los compara con el método de Carlos, explicando sus parecidos y
  que el MPASM, por K_f y s, rinde con más distancia de propagación (en el
  sentido del criterio).
- La comparación ASM–MPASM se reorganiza para que no esté repartida ni repetida,
  con las cuatro combinaciones de ida y vuelta, explicadas.
- Distancias cortas = las del banco DLHM con onda esférica y los números de
  Carlos, **y además** el rango de 5 a 15 mm (onda plana, ya medido). Se suma la
  zona de Fresnel, ya medida, y la transición entre las dos.
- Todo para el borrador del 21/10.

## Diseño

### A. El ensayo ASM–MPASM reunido (método: 3.2.4; resultado: 4.2.3)

Tres escalas, mismas cuatro combinaciones, misma medida (SAM de la Ec. eq:sam y
error máximo del complejo frente al objeto):

1. **Banco DLHM, onda esférica** (nuevo). Geometría de Carlos: λ = 528 nm,
   δ = 1.83 µm, L = 10 mm, z = 3 mm. "ASM" = la cadena de Carlos
   (`reconstruction_dlhm.py`, copiada en `scripts/mi_prueba_dlhm.py`:
   `reconstruir()` y `MotorCarlos`), y "MPASM" = `CamposT.dlhm` (`holograma()`
   para la ida y `Reconstructor` para la vuelta). **Decidido por el usuario
   (20/10):** la ida "ASM" es el espejo de su cadena, no el simulador `dlhm()`
   de Lopera. Se recorren sus pasos al revés: t·e^{ikr}/r en el plano de la
   muestra, sobre la malla remuestreada que cubre el recorte; espectro angular
   por L − z hacia el sensor, con su `angular_spectrum`; remuestreo a la malla
   del sensor. Va con fase
   (holograma complejo U/S_L, sin gemela). Objeto: un USAF binario en el
   recorte; sensor simulado de 1024×1024. *Decisión mía:* además de z = 3 mm,
   z = 1, 2, 5 y 7 mm, para que el cociente D/(Nδ²/λ) cruce 1 dentro del banco
   (con N = 1024 vale 0.7 a z = 7, 3.6 a z = 3 y 14 a z = 1). Script nuevo:
   `scripts/cruzada_dlhm.py`.
2. **5 a 15 mm, onda plana, bajo z_lim = 15 mm.** Ya medido
   (`resultados/comparacion_cruzada/entrada_r4_cercano/`): las cuatro
   combinaciones dan entre 59.4 y 66.2 dB.
3. **Transición y zona de Fresnel.** Lo de 100–1926 mm ya está medido. Se
   rellena el hueco de 15 a 100 mm corriendo `comparacion_cruzada.py` con
   `ZS = (20, 30, 50, 75)` y restaurando la constante al terminar.

La explicación por régimen:
- Cuando el campo cabe en la ventana, las cuatro combinaciones coinciden.
- Cuando se sale, ASM→ASM se deshace a sí mismo y las otras tres coinciden entre
  sí: la información la limita la ventana.
- La prueba de s (sin efecto) y la ventana ampliada (el aporte real del MPASM)
  cierran el régimen de Fresnel.

Una tabla (combinaciones × distancia, marcando z/z_lim) y una figura (SAM contra
z/z_lim, con los tres regímenes). La figura de reconstrucciones a N_F = 10 y 1
se conserva. La discusión deja de repetir números y solo cita.

### B. Montaje y datos (3.3)

- Qué entregó el laboratorio: dos muestras (USAF de amplitud y de fase), 17
  posiciones de la platina, 16 capturas laterales por posición y una imagen sin
  muestra. JPEG de 8 bits, 4640×3506.
- Tabla de parámetros con su origen:
  - medidos o leídos: λ = 528 nm, N_x × N_y y 8 bits;
  - inferidos: píxel 3.94 ± 0.29 µm (se adopta 3.80 µm) y L = 18.2 ± 1.4 mm,
    por confirmar.
- El esquema del montaje (`\pendiente` del capítulo 2) se resuelve con una figura
  esquemática (fuente, muestra, sensor, z, L, M).

### C. Cadena de reconstrucción y foco (3.4)

1. **Normalización por amplitud** con la imagen sin muestra, como en
   `CamposT.dlhm`.
2. **Búsqueda del foco:**
   - barrido paraxial en D sobre la malla del sensor, grueso (0.5 mm) y fino
     (0.1 mm), con la Tamura de |t|: máximo en amplitud, mínimo en fase;
   - las 16 capturas dan la dispersión;
   - refinado con el motor exacto en z para una L dada.
3. **Por qué hace falta la escala:** el enfoque solo fija D, y z depende de L
   (motor exacto: z = 0.39 mm con L = 5 y 9.9 mm con L = 30, a nitidez igual).
4. **Calibración con el USAF:**
   - ajuste de tres barras en los grupos 4, 6 y 7, en z1, z5, z9 y z17;
   - ajuste conjunto de los 17 focos y las 4 magnificaciones, con platina a paso
     fijo, para el píxel, L, z1 y el paso.
5. **Métricas de la comparación con Carlos:**
   - resolución: el elemento más fino del USAF con tres barras distinguibles,
     por contraste de Michelson;
   - contraste de un elemento de referencia;
   - SNR en una zona de fondo;
   - tiempo por plano.

### D. Resultados experimentales (4.3)

1. **Foco:** la tabla de las 17 posiciones (D, z y M) y las curvas.
2. **Geometría:**
   - L implícito por posición, que con 1.83 µm deriva y con 3.80 µm es
     constante;
   - χ²/dof 115 frente a 8;
   - pasos de 0.099 y 0.050 mm;
   - z de 0.415 a 2.003 mm (amplitud) y de 0.459 a 1.274 mm (fase).
3. **Reconstrucciones:** amplitud y fase en varias posiciones.
4. **Comparación MPASM esférico frente a Carlos:**
   - en las 17 posiciones de amplitud: el foco que encuentra cada uno, la
     Tamura y el tiempo;
   - en z1, z9 y z17 (cocientes 6.2, 2.0 y 1.2): resolución, contraste y SNR
     sobre el mismo recorte de 1024, más el exacto con el recorte grande para
     mostrar lo que gana con el campo de visión;
   - fase: foco de cada motor en las 17 posiciones.
5. **Parecidos y diferencias** de las dos cadenas:
   - lo común: la misma onda de referencia esférica y el mismo espectro
     angular;
   - lo distinto: el remuestreo y el recorte frente a la esférica dentro de las
     matrices con K_f < 1, la memoria y el campo de visión;
   - dónde cae cada posición frente al criterio.

### E. Lo demás que se toca

- **Capítulo 1:**
  - el `\pendiente` del alcance: hologramas reconstruidos;
  - la lista de entregables suma las reconstrucciones experimentales y la
    calibración de la geometría.
- **Capítulo 3:** se quita el `\pendiente` de la apertura ("si se reconstruyen los
  experimentales") y el segundo nivel de validación incluye los experimentales.
- **Discusión:**
  - el párrafo de "cuándo pesa el control" cita la 4.2.3 y suma dónde caen los
    hologramas experimentales;
  - el `\pendiente` final sobre el objetivo 3 y sus límites queda escrito: un
    banco, dos muestras y el píxel inferido.

### Figuras nuevas

Van a `manuscrito/figuras/` desde funciones nuevas en
`scripts/figuras_manuscrito.py`:
- la ida y vuelta reunida;
- el foco experimental (curvas);
- la geometría (L implícito y D contra k);
- las reconstrucciones experimentales;
- la comparación con Carlos;
- el esquema del montaje.

### Código

- `scripts/hologramas_carlos/` reúne, limpios, los pasos de
  `resultados/hologramas_carlos/codigo/`:
  - foco paraxial grueso y fino;
  - motor exacto en z para una L;
  - ajuste de tres barras del USAF;
  - ajuste conjunto;
  - comparación con `MotorCarlos`.
- Escriben en `resultados/hologramas_carlos/`.

## Qué queda al criterio de quien lo construya

- La redacción exacta de cada párrafo (el usuario reescribe en su voz).
- Los umbrales de la métrica de resolución y las zonas de fondo del SNR, si se
  dejan escritos.
- El orden interno de las subsecciones de la 4.3.
