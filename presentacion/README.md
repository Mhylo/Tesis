# presentacion/

Las figuras que se proyectaron en la defensa. Qué diapositiva usa cada una, con
su script y sus datos: [INDICE.md](../INDICE.md#la-presentación-de-la-defensa).

## Qué hay

| Archivo | Contenido |
|---|---|
| [`figuras_charla.py`](figuras_charla.py) | Dibuja las figuras de la charla. Llama a las funciones de [`scripts/figuras_manuscrito.py`](../scripts/figuras_manuscrito.py) con letra para proyectar, o trae su propia versión con los mismos CSV. En la charla el espectro angular con FFT se llama *ASM*: la función `solo_asm()` cambia el rótulo «FFT-ASM» al dibujar. |
| [`figuras_charla/`](figuras_charla/) | Las figuras, en PDF y en PNG (el PNG es el que va en la diapositiva). Cuatro, las `g_*`, son copia de las de [`docs/guia_manuscrito/figuras/`](../docs/guia_manuscrito/figuras/). |

Las diapositivas que no tienen figura aquí usan la del manuscrito tal cual, de
[`manuscrito/figuras/`](../manuscrito/figuras/).

## Rehacer las figuras

Desde la raíz:

```bash
Tesis_env/Scripts/python.exe -m presentacion.figuras_charla
```

Necesita `pdftoppm` (viene con MiKTeX o poppler) para los PNG, los CSV de
`resultados/` y, para algunas figuras, los `.npy` que no están en el
repositorio; [INDICE.md](../INDICE.md#lo-que-no-está-en-el-repositorio) dice con
qué orden salen.

## Lo que no está aquí

El `.pptx` de la defensa y el script que lo arma (`generar_defensa.py`, que
maneja PowerPoint por COM y lleva las notas del orador) quedan fuera del
repositorio, igual que `ecuaciones/` y `figuras/`: las ecuaciones compuestas
con LaTeX, las copias rasterizadas de las figuras del manuscrito y los recortes
de los hologramas del laboratorio, que no se publican.
