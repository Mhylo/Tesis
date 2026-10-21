# Revisión del manuscrito

Empezó el 21 de octubre de 2026, sobre el borrador que va a Carlos. Se revisa una
sección por iteración: resumen, cap. 1, cap. 2, cap. 3, cap. 4 y cap. 5. Lo que
era seguro se corrigió en el `.tex`, con un commit por sección. Aquí va solo lo
que necesita una decisión tuya o de Carlos. El texto no se tocó en estos puntos.

## Generales

- **Declarar el uso de herramientas de IA.** Está en la lista "Antes de
  entregar" del plan. Hay que revisar si EAFIT lo pide. Si lo pide, va una nota
  en la metodología o en los agradecimientos.
- **pyDHM y la CZT (4.2.1).** El número de la tarea 32 existe: en el escenario
  de la Tabla 1, el Fresnel–Bluestein de pyDHM se aparta de la analítica entre
  0.011 y 0.23, y el MPASM con s = 4, entre 1.2e-4 y 5.5e-4
  (`scripts/contraste_referencias.py`). Antes de escribirlo hay dos cuidados:
  - la causa no está establecida. Los ángulos son de unos 2e-4 rad, así que
    "es paraxial" no la explica;
  - el script encontró que `pyDHM.fresnel` y `pyDHM.bluestein` usan `np.exp2`
    donde va `np.exp`. Es la librería de Carlos, así que hay que hablarlo con
    él antes de que entre al documento.
- **Cita de pyLHM** (`03_metodos.tex`, pendiente). La entrada de la
  bibliografía no tiene URL ni autores. Hay que confirmar con Carlos cómo se
  cita.
- **Repositorio y licencia** (`01_introduccion.tex`, pendiente; tareas 52 y 55).

## Resumen

- Tiene unas 290 palabras, y el plan pedía unas 250. No se recortó porque
  cada frase lleva un número de las conclusiones.
- El abstract en inglés es opcional: las directrices solo piden el resumen.
  Está comentado en `resumen.tex`.

## Capítulo 1

- **Justificación, última frase.** Tu `% ?` sigue abierto: ¿se mantiene "la
  formación experimental e instrumental"? Los hologramas los registró el
  laboratorio. Lo experimental del trabajo fue inferir la geometría del
  registro y reconstruir. Una salida es cambiarla por "el tratamiento de datos
  experimentales".
- **Contribuciones.** El planteamiento dice que no existían criterios
  publicados para elegir los parámetros del MPASM, y la tesis entrega uno: el
  cociente de la ventana. El concilio del 11/10 sugería un párrafo de
  contribuciones al final de la introducción:
  - el MPASM esférico;
  - el criterio de ventana;
  - la referencia RS radial;
  - las dos erratas de Zhao;
  - la geometría inferida.

  Es una adición, no una corrección, así que no se escribió.

## Capítulo 2

- **Lista de símbolos.** Se aclararon en el texto M (filas o magnificación) y
  z_k (punto de la transformada Z). Quedan letras con más de un uso, cada una
  definida donde aparece:
  - W: ancho del sensor (Ec. na), parámetro de la CZT y ventana de Fresnel
    (Ec. ventana-chirp);
  - A: espectro A_0, parámetro de la CZT y matriz A_n(f);
  - s: sobremuestreo y variable de integración de la Ec. ventana-chirp;
  - z_k: punto de la CZT y posición de la platina (3.4 y 4.3).

  Una lista de símbolos en los preliminares lo resolvería sin tocar las
  ecuaciones. Es una adición, así que no se hizo.
- **Antecedentes, penúltima frase.** "La reconstrucción de rutina en el
  montaje DLHM del laboratorio se realiza con un propagador de espectro
  angular..." repite la 2.5.1. El concilio del 11/10 sugería quitarla. No se
  tocó porque es una decisión de estilo.

## Capítulo 3

- **La L adoptada.** La 3.3 dice "se adoptaron δ = 3.80 µm y L = 18.23 mm".
  La Tabla 3.1 da, con 3.80 µm fijo, L = 18.19 ± 0.29 mm, igual que
  `calibracion.json` (`pixel_380.L = 18.194`). Pero `correr.py` usa
  `L_EXP = 18.229`, con el comentario "ajuste conjunto", y ningún archivo de
  resultados lo reproduce. La diferencia, 0.035 mm, cae dentro de la
  incertidumbre. Hay dos salidas:
  - decir de dónde sale el 18.23;
  - adoptar 18.19 y volver a correr `correr.py exacto` y `comparar.py`, lo que
    toma horas en GPU.

  Lo más barato es la primera, si sabes de dónde salió.
- **Más letras con dos usos** (para la lista de símbolos): k (número de onda
  y posición de la platina, 3.4) y r (distancia en las integrales RS y factor
  de la malla de salida del MPASM). El diámetro del pinhole pasó a d_p, porque
  d ya era L − z.
