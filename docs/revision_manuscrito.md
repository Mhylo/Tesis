# Revisión del manuscrito

Empezó el 21 de octubre de 2026, sobre el borrador que va a Carlos. Se revisa una
sección por iteración: resumen, cap. 1, cap. 2, cap. 3, cap. 4 y cap. 5. Lo que
era seguro se corrigió en el `.tex`, con un commit por sección. Aquí va solo lo
que necesita una decisión tuya o de Carlos. El texto no se tocó en estos puntos.

## Generales

- **Declarar el uso de herramientas de IA.** Está en la lista "Antes de
  entregar" del plan. Hay que revisar si EAFIT lo pide. Si lo pide, va una nota
  en la metodología o en los agradecimientos.
- **La CZT: resuelta el 21/10, sin pyDHM.** La 4.2.2 nueva hace el contraste
  término a término que la tarea 32 dejó pendiente (`scripts/contraste_czt.py`,
  `tests/test_contraste_czt.py`):
  - el MPASM calculado con la CZT de SciPy y con sus matrices da lo mismo
    hasta 4e-10;
  - en CPU, los dos caminos cuestan lo mismo en orden de magnitud;
  - la formulación esférica no es una CZT.

  El número de pyDHM no se usó. Queda algo para decirle a Carlos, fuera de la
  tesis: `pyDHM.fresnel` y `pyDHM.bluestein` usan `np.exp2` donde va `np.exp`.
  Corregido, el error del Fresnel–Bluestein a 6 m baja de 0.23 a 0.059
  (`scripts/contraste_referencias.py`).
- **La GPU no estaba visible el 21/10 por la tarde.** CuPy daba
  `cudaErrorNoDevice` dentro y fuera del sandbox, y nvidia-smi pedía permisos
  de administrador. Por eso el contraste con la CZT cita los tiempos en GPU de
  la Tabla 4.1. Si hay que volver a correr algo en la GPU (por ejemplo, con
  otra L), conviene revisar antes el modo de energía o el controlador.
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

## Capítulo 4

- **Los 5 dB a 0.5 m (4.1).** "40.5 dB frente a 35.4 dB a 0.5 m" queda sin
  causa. Ahí el MPASM usa s = 10 y K_f = 1, así que no es el caso que
  coincide con el FFT-ASM (s = K_f = 1). El concilio del 11/10 ya lo marcó:
  un jurado con el artículo de Zhao lo va a preguntar. Hay dos salidas:
  - una frase con la causa, si la conoces;
  - declararla abierta: "la causa de esa diferencia no se investigó".

  No se tocó porque decir cualquiera de las dos es decisión tuya.
- **La ida esférica contra RS (4.2.4, límites).** Ahora los límites remiten a
  las Secciones 4.1 y 4.2.3, pero las dos validan el MPASM de onda plana. La
  ida del banco usa el MPASM esférico. Según el concilio del 11/10, esa ida
  "se contrastó con RS y da 3.6e-6". Si se ubica el script que lo produce,
  ese número cerraría el límite en la 4.2.1. Si no, el límite queda como
  está.

## Capítulo 5

- Se cotejó contra el capítulo 4 ya revisado y no cambió nada.
- **"Un orden de magnitud, no una frontera."** El párrafo de la pregunta (el
  del concilio) dice que el criterio "es un orden de magnitud, no una
  frontera, porque sale del escalado paraxial". La 4.2.3 muestra que en los
  mapas del banco sí traza una frontera nítida, pero en 3 y no en 1, porque
  depende del relleno. Las dos cosas son compatibles, y la 4.2.3 lo dice
  ("que caiga en 3 y no en 1 es lo que quiere decir que el criterio sea un
  orden de magnitud"). Si quieres que la conclusión lo recoja, basta con
  agregar al final de esa frase: "y su valor exacto depende del relleno: en
  el banco, con relleno al doble, la frontera cae en 3". No se tocó porque
  es el texto que aprobaste.
- **Extensión.** El capítulo tiene unas 1500 palabras, frente a las 800 del
  plan. Los párrafos del objetivo 3 y de la pregunta repiten varios números
  (el recorte 20 veces mayor, el contraste que casi se duplica).
