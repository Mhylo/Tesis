# Plan de escritura del manuscrito

Armado el 2026-10-07 (semana 12, día 3). Meta: **borrador completo al asesor el
miércoles 21 de octubre**. Así quedan 9 días para correcciones antes de la
entrega del 30 de octubre (semana 15).

Fechas que no se mueven:

| Fecha | Qué |
|---|---|
| Lun 19 oct | Avance 2 (5 %). Se presenta con los capítulos ya escritos. |
| Mié 21 oct | Borrador completo a Carlos (tarea 46). |
| Vie 30 oct | Informe final con carta de aval (tarea 56). |

---

## Tamaño objetivo

Unas 16 000 palabras, entre 45 y 60 páginas con figuras. Hoy hay unas 1 250
palabras de prosa definitiva (introducción y antecedentes) y 26 `\pendiente`.

| Capítulo | Palabras | Hoy |
|---|---|---|
| 1 Introducción | 1 500 | ~900 (falta alcance y estructura, y pasarlo a pasado) |
| 2 Marco teórico | 4 500 | ~350 (antecedentes) |
| 3 Métodos | 4 500 | 0 |
| 4 Resultados y discusión | 5 000 | 0 |
| 5 Conclusiones | 800 | 0 |
| Resumen | 250 | 0 |

Hacen falta unas 1 100 palabras por día durante 13 días. Es viable solo si yo
redacto el primer borrador desde el código y las notas del cronograma, y tú
reescribes en tu voz.

---

## Cómo trabajamos cada sección

1. **Tú, 5 minutos:** el mensaje de la sección en una frase y qué número o
   figura la sostiene.
2. **Yo:** leo el código, los docstrings y las notas del cronograma (ahí están
   los números medidos) y redacto la sección directamente en el `.tex`, con
   `\cite`, comentarios `% Repo:` y la figura final en `manuscrito/figuras/`
   (PDF vectorial, generado por script, nunca a mano).
3. **Tú:** relees y reescribes lo que no suene a ti. Dejas tus dudas como
   `% ?` en el `.tex`.
4. **Yo:** verifico cada número del texto contra el repo antes del commit.
5. Un commit por sección. Subes a Overleaf al final del día.

Regla: **ningún número entra al manuscrito sin un script o un commit que lo
reproduzca.**

---

## Dos frentes en paralelo

El texto no es el único cuello de botella: el objetivo 2 (barrido, tareas
29-30) y el objetivo 3 (hologramas experimentales, tareas 39-42) aún no tienen
resultados. El plan intercala ~4 días técnicos, y mientras corren los cálculos
en GPU se escribe lo que ya está cerrado.

### Recortes de alcance que propongo (decidir con Carlos esta semana)

- **Barrido del objetivo 2 con malla reducida:** λ = 528 nm, δ = 1.83 µm, el
  sensor del banco, z en el recorrido de la platina, s ∈ {1, 2, 3} y Kf
  automático. La NA sale del tamaño del sensor (px_x, px_y), que se lee en los
  propios archivos de los hologramas de Carlos.
- **CZT:** si el 14/10 no está el contraste término a término, se queda en la
  relación teórica (sección 2.4.1) más el contraste contra el `bluestein` de
  pyDHM que ya está medido (tarea 32).
- **Muestra biológica (tarea 36): fuera.** El objetivo 3 pide "muestras de
  resolución conocida", y el USAF basta.
- **Métricas del objetivo 3:** resolución sobre el USAF, contraste y tiempo.
  La SNR solo si sale gratis.

---

## Calendario

### Semana 12 (7-11 oct): lo que ya está cerrado

| Día | Escritura | Técnico |
|---|---|---|
| Mié 7 | Revisión, cronograma y este plan | Commit del árbol del 29/09. Pedir a Carlos: los archivos, **el fondo sin muestra de cada captura**, z y L de cada una y el modelo del sensor |
| Jue 8 | **Cap. 3 §3.1** marco computacional y verificación (tarea 15) | Subir los hologramas de Carlos a `datos/` con metadatos (tarea 38) |
| Vie 9 | **Cap. 4 §4.1** resultados del objetivo 1: Fig. 4 de Zhao, SAM vs z, error vs gaussiano, CPU/GPU | Lanzar de noche el barrido del objetivo 2 (tareas 29-30) |
| Sáb 10 | **Cap. 2 §2.1-2.3:** difracción, ASM-FFT y sus límites, BLAS/WWASM/NUFFT | — |
| Dom 11 | **Cap. 2 §2.4-2.5:** MPASM y CZT, DLHM | (holgura) |

### Semana 13 (12-18 oct): lo nuevo y lo experimental

| Día | Escritura | Técnico |
|---|---|---|
| Lun 12 | **Cap. 3 §3.2:** calibración e iluminación esférica (tareas 28 y 33) | Revisar el barrido |
| Mar 13 | **Cap. 4 §4.2:** mapas de error, criterio de s y Kf, límite del escalado de Fresnel, CZT | Criterio de s y Kf (tarea 31) |
| Mié 14 | (técnico) | **Reconstruir los hologramas experimentales** con los propagadores y la línea base de Carlos, y medir (tareas 39-41). Decidir el recorte de la CZT |
| Jue 15 | **Cap. 3 §3.3-3.4:** datos experimentales, normalización y selección de foco. **Cap. 4 §4.3:** tabla y figuras (tareas 42-43) | — |
| Vie 16 | **Cap. 4 §4.4:** discusión (tarea 45). Diapositivas del avance 2 (tarea 47), sacadas de los capítulos | — |
| Sáb 17 | **Cap. 5** (tarea 50). Introducción: alcance, estructura y tiempos verbales | — |
| Dom 18 | Ensayo del avance 2 | (holgura) |

### Semana 14 (19-21 oct): cierre del borrador

| Día | Qué |
|---|---|
| Lun 19 | **Avance 2.** Anotar la retroalimentación |
| Mar 20 | Resumen. Retroalimentación del avance. Pasada global: cero `\pendiente`, numeración, bibliografía (tareas 53-54). Solicitud formal de la carta de aval (tarea 51) |
| Mié 21 | **Borrador completo a Carlos** (tarea 46) |

Después, del 22 al 30 de octubre: correcciones del asesor (tarea 49), repo
limpio y publicado (tareas 52 y 55) y entrega (tarea 56).

---

## Cambios de estructura del manuscrito

El esqueleto refleja el plan de agosto. Lo más propio del trabajo no tiene un
lugar claro todavía:

- **§3.2 "Propagación con iluminación esférica"**, con tres subsecciones:
  (a) el escalado de Fresnel (paso δ/M y z_eq), que es paraxial;
  (b) el MPASM con la onda esférica exacta (`CamposT/dlhm.py`);
  (c) la emisión de la fuente y la normalización por amplitud.
  Hoy el texto dice "validados contra RS", y eso es solo media verdad: el
  escalado coincide con RS para objetos lisos y falla con detalle fino.
- **§3.4 "Selección del plano de foco"** (Tamura, Pearson, SSIM; grueso y
  fino), porque sin verdad de terreno el foco *es* el método de medida del
  objetivo 3.
- **Resultado del objetivo 2:** el límite del escalado de Fresnel (0.89 de
  correlación en el eje, 0.63 fuera, con detalle fino) y el corrimiento de
  +14 a +20 µm del modelo paraxial de Lopera. Hay que escribirlo como "régimen
  paraxial", no como "error del paper".
- **Nombrar los propagadores una sola vez y usar siempre esos nombres.** El
  objetivo 1 habla de tres (FFT-ASM, BLAS, MPASM), el cronograma de cuatro, y
  la línea base del laboratorio es otra cosa. Propuesta: FFT-ASM, BLAS, MPASM y
  MPASM esférico, frente a la línea base (ASM con onda esférica de Carlos).

## Bibliografía que falta (verificar cada una antes de citar)

Doce entradas es poco para una tesis. Faltan los clásicos que el texto ya
necesita:

- Gabor (1948), el origen de la holografía en línea.
- Garcia-Sucerquia, Xu, Jericho, Klages, Jericho y Kreuzer (2006), *Appl. Opt.*,
  la referencia de DLHM.
- Rabiner, Schafer y Rader (1969), la CZT.
- Wang, Bovik, Sheikh y Simoncelli (2004), el SSIM.
- La referencia de la métrica de Tamura que usa el script.
- El paper o manual del sensor del banco.

## Figuras (carpeta `manuscrito/figuras/`, hoy vacía)

1. Esquema del montaje DLHM (fuente, muestra, sensor, z, L)
2. Muestreo de la función de transferencia: FFT-ASM frente a BLAS y MPASM
3. Reproducción de la Fig. 4 de Zhao 2020
4. SAM vs z de los tres propagadores frente a RS
5. Escalado CPU/GPU (`resultados/tiempos.png`, `aceleracion.png`)
6. Mapas de error del barrido (objetivo 2)
7. Escalado de Fresnel frente al esférico exacto, objeto liso y con detalle fino
8. Emisión de la fuente: pinhole, Lopera y holograma de referencia
9. Curvas de foco (Tamura, Pearson, SSIM) en el BenchmarkTarget
10. Hologramas experimentales de Carlos
11. Reconstrucciones experimentales, una por propagador, en el mismo ROI del USAF
12. Tabla de métricas del objetivo 3

## Antes de entregar

- [ ] Buscar `pendiente` en el proyecto: cero resultados.
- [ ] Mes de la entrega en `\fechaentrega` (`main.tex`).
- [ ] Revisar si EAFIT pide declarar el uso de herramientas de IA. Si lo pide,
      va una nota en la metodología o en los agradecimientos.
