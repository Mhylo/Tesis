# manuscrito/

El documento del trabajo de grado en LaTeX, en la versión entregada. No se
modifica.

## Compilar

Con latexmk (pdflatex y biber), desde esta carpeta:

```bash
latexmk -pdf -outdir=build -interaction=nonstopmode main.tex
```

El PDF y los auxiliares quedan en `build/`, fuera del repositorio.

## Qué hay

| Archivo | Contenido |
|---|---|
| [`main.tex`](main.tex), [`preambulo.tex`](preambulo.tex) | El documento y su formato. |
| [`capitulos/`](capitulos/) | Portada, resumen, agradecimientos, los capítulos 1 a 5 y el Anexo A. |
| [`referencias.bib`](referencias.bib) | La bibliografía. |
| [`figuras/`](figuras/) | Las figuras, en PDF. Las dibuja [`scripts/figuras_manuscrito.py`](../scripts/figuras_manuscrito.py), salvo `obj1_sam_vs_z.pdf` (Fig. 4.1), que dibuja [`scripts/exactitud.py`](../scripts/exactitud.py). |

| Capítulo | Archivo |
|---|---|
| 1 Introducción | [`01_introduccion.tex`](capitulos/01_introduccion.tex) |
| 2 Marco teórico y estado del arte | [`02_marco_teorico.tex`](capitulos/02_marco_teorico.tex) |
| 3 Materiales y métodos | [`03_metodos.tex`](capitulos/03_metodos.tex) |
| 4 Resultados y análisis | [`04_resultados.tex`](capitulos/04_resultados.tex) |
| 5 Conclusiones y trabajos futuros | [`05_conclusiones.tex`](capitulos/05_conclusiones.tex) |
| A Interferometría holográfica simulada | [`anexo_a.tex`](capitulos/anexo_a.tex) |

## De dónde sale cada número

Cada figura y tabla, con su script y sus datos: [INDICE.md](../INDICE.md#las-figuras-del-manuscrito).
Dentro de los `.tex`, un comentario `% Repo: ...` sobre cada sección y junto a
cada figura da las mismas rutas, y la fecha de la medida cuando importa.

El manuscrito dice *FFT-ASM* y *cadena del laboratorio*; la presentación dice
*ASM* y *método del laboratorio*. Son lo mismo.
