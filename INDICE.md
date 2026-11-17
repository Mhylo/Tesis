# Índice: de cada diapositiva y cada figura a su script y sus datos

Para encontrar rápido de dónde sale lo que se mostró en la defensa o lo que
dice el manuscrito. Cada fila lleva a la figura, al script que la dibuja (y la
función), a los datos que lee y al lugar del manuscrito donde está el número.

- [La presentación de la defensa](#la-presentación-de-la-defensa) (53 diapositivas)
- [Las figuras del manuscrito](#las-figuras-del-manuscrito) (16) y
  [sus tablas](#las-tablas-del-manuscrito) (8)
- [Lo que no está en el repositorio](#lo-que-no-está-en-el-repositorio) y cómo
  se regenera

**Dos nombres para lo mismo.** En la charla se dice *ASM* y *método del
laboratorio*; el manuscrito dice *FFT-ASM* y *cadena del laboratorio*. En el
código son `CamposT.propagadores.fft_asm` y `MotorCarlos`
([`scripts/mi_prueba_dlhm.py`](scripts/mi_prueba_dlhm.py)).

**Cómo correr.** Desde la raíz, con el entorno del [README](README.md#instalación)
(`Tesis_env/Scripts/python.exe` en Windows): `python -m scripts.exactitud`, etc.
Los datos se leen de [`resultados/`](resultados/README.md); los que están en el
repositorio son los CSV y JSON chicos. Los `.npy` grandes no: abajo dice con qué
orden salen.

## La presentación de la defensa

Las figuras que se proyectan están en
[`presentacion/figuras_charla/`](presentacion/figuras_charla/) (PDF y PNG) y las
dibuja [`presentacion/figuras_charla.py`](presentacion/figuras_charla.py), que
reutiliza las funciones de
[`scripts/figuras_manuscrito.py`](scripts/figuras_manuscrito.py) con letra para
proyectar y los mismos CSV. Cuando una diapositiva usa la figura del manuscrito
tal cual, la fila apunta a [`manuscrito/figuras/`](manuscrito/figuras/). Las
ecuaciones de las diapositivas son las del manuscrito; la columna de la derecha
da su número.

`fc` = `presentacion/figuras_charla.py`; `fm` = `scripts/figuras_manuscrito.py`;
`fg` = `docs/guia_manuscrito/figuras_guia.py`; `hc` = `scripts/hologramas_carlos/`.

### Introducción y marco teórico

| # | Título | Figura · script | Datos | Manuscrito |
|---|---|---|---|---|
| 1 | Implementación y validación de propagadores numéricos con muestreo controlado para la reconstrucción de hologramas en microscopía holográfica digital sin lentes | Recorte del holograma de amplitud, posición 9, captura 1 | Holograma del laboratorio, [no publicado](#lo-que-no-está-en-el-repositorio) | Portada |
| 2 | Índice | — | — | Capítulos 1 a 5 |
| 3 | El DLHM registra sin lentes el holograma de la muestra, y la imagen se recupera propagándolo de vuelta hasta ella | [`esquema_montaje.pdf`](manuscrito/figuras/esquema_montaje.pdf) · `fm.figura_esquema_montaje()`; el holograma de la posición 9 y su reconstrucción | Holograma: no publicado. Reconstrucción: `resultados/hologramas_carlos/imagen_amp_z9.npy` (`python -m scripts.hologramas_carlos.correr imagenes`) | Fig. 2.2 · §2.5 |
| 4 | Los tres propagadores del trabajo calculan el mismo espectro angular con muestreos distintos | Ecuaciones · [`CamposT/propagadores.py`](CamposT/propagadores.py) (`fft_asm`, `blas`, `mpasm`) | — | Ec. 2.2 · §2.2 a §2.4 |
| 5 | El MPASM libera el muestreo, pero solo se había probado en simulación | Texto | — | §1.1 · §2.6 |
| 6 | Pregunta de investigación | Texto | — | §1.1 · §1.2 |
| 7 | El espectro angular con FFT impone a todo el cálculo el muestreo de la entrada | Ecuaciones · `CamposT.propagadores.fft_asm` | — | Ec. 2.4 y 2.5 · §2.2 |
| 8 | Las soluciones conocidas corrigen la FFT desde fuera, y pierden detalle o gastan más memoria | Ecuaciones · `CamposT.propagadores.blas` | — | Ec. 2.7 · §2.3 |
| 9 | El MPASM flexibiliza el muestreo: elige las frecuencias del espectro y la malla de salida | [`kf_vs_z.png`](presentacion/figuras_charla/kf_vs_z.png) · `fc.kf_vs_z()`, la Fig. 2.1 con z en cm | Ninguno: evalúa `CamposT.propagadores.kf_auto` con `CamposT.montaje.TABLA1` | Fig. 2.1 · Ec. 2.8, 2.9 y 2.12 · §2.4 |
| 10 | El sensor registra un holograma en línea, y reconstruir es retropropagarlo | Ecuaciones · [`CamposT/dlhm.py`](CamposT/dlhm.py) | — | Ec. 2.14 y 2.15 · §2.5 |
| 11 | El ASM falla cuando el campo propagado ya no cabe en la ventana | Ecuaciones · el criterio en [`hc/geometria.py`](scripts/hologramas_carlos/geometria.py) | — | Ec. 2.6 y 3.4 · §3.2.1 · §3.2.4 |

### Metodología

| # | Título | Figura · script | Datos | Manuscrito |
|---|---|---|---|---|
| 12 | El trabajo siguió cuatro fases, cada una con un producto verificado | Texto | — | Capítulo 3 |
| 13 | Fase 1: los tres propagadores en un mismo paquete, verificados contra referencias independientes | Esquema · [`CamposT/`](CamposT/README.md) y [`tests/`](tests/README.md) | — | §3.1 · §3.1.1 |
| 14 | Fase 2: el MPASM permite muestrear de manera adecuada la iluminación esférica | Ecuaciones · `CamposT.dlhm` (`ventana_chirp`, `Reconstructor`); los 9.8 s y 0.56 s, en su docstring | — | Ec. 3.5 · §3.2.2 |
| 15 | Fase 2: un barrido de validez contra Rayleigh–Sommerfeld y una ida y vuelta con objeto conocido | Esquema · [`CamposT/barrido.py`](CamposT/barrido.py) (la malla), [`scripts/barrido_obj2.py`](scripts/barrido_obj2.py), [`scripts/comparacion_cruzada.py`](scripts/comparacion_cruzada.py), [`scripts/cruzada_dlhm.py`](scripts/cruzada_dlhm.py) | — | Ec. 3.6 · §3.2.3 · §3.2.4 |
| 16 | Fases 3 y 4: hologramas del laboratorio sin geometría conocida, y dos métodos sobre el mismo recorte | Tabla y pasos · [`hc/`](scripts/hologramas_carlos/) (`correr.py`, `comparar.py`); el método del laboratorio, `MotorCarlos` en `scripts/mi_prueba_dlhm.py` | [`calibracion.json`](resultados/hologramas_carlos/calibracion.json) (`hc.correr calibracion`) | Tabla 3.1 · §3.3 · §3.4 |

### Resultados: objetivo 1

| # | Título | Figura · script | Datos | Manuscrito |
|---|---|---|---|---|
| 17 | 4.1 Implementación y verificación | Separador | — | §4.1 |
| 18 | Los tres propagadores coinciden con códigos independientes y con una referencia exacta | Tabla de tres números: 1.9 × 10⁻⁸ de [`scripts/contraste_referencias.py`](scripts/contraste_referencias.py) (necesita pyDHM, ver README); 4 × 10⁻¹⁶, el MPASM frente al `MatrixDftCPU` de [`referencia/zhao2020/`](referencia/zhao2020/), copiado en [`scripts/retro_mpasm.py`](scripts/retro_mpasm.py) (`comprobar_equivalencia()`); 2.0 × 10⁻⁶ de [`tests/test_rs1.py`](tests/test_rs1.py) | — | §4.1.1 |
| 19 | Cuando el haz desborda la ventana, el MPASM sigue exacto y el ASM cae | [`sam_vs_z.png`](presentacion/figuras_charla/sam_vs_z.png) · `fc.sam_vs_z()` | [`resultados/exactitud/sam_vs_z.csv`](resultados/exactitud/sam_vs_z.csv) (`python -m scripts.exactitud`) | Fig. 4.1 · §4.1.2 |
| 20 | La verificación encontró un error en el código publicado y una errata en el artículo | [`errata_kf.png`](presentacion/figuras_charla/errata_kf.png) · `fc.errata_kf()`, la Fig. 4.2 con z en cm | Ninguno: `kf_auto(..., formula="codigo")`; la errata de la Ec. (16), en [`CamposT/metricas.py`](CamposT/metricas.py) y [`tests/test_metricas.py`](tests/test_metricas.py) | Fig. 4.2 · Ec. 2.12 y 3.3 · §4.1.3 |
| 21 | El costo computacional del MPASM es mayor que el del ASM, y la GPU lo vuelve manejable | [`tiempos_N.png`](presentacion/figuras_charla/tiempos_N.png) · `fc.tiempos_N()`, el panel (a) de la Fig. 4.3 | [`resultados/escalado_N.csv`](resultados/escalado_N.csv) (`python -m scripts.comparacion`) | Tabla 4.1 · Fig. 4.3 · §4.1.4 |

### Resultados: objetivo 2

| # | Título | Figura · script | Datos | Manuscrito |
|---|---|---|---|---|
| 22 | 4.2 Validez en el DLHM | Separador | — | §4.2 |
| 23 | Con detalle del micrómetro, el escalado paraxial se degrada lejos del eje y el MPASM esférico se mantiene | [`mpasm_esferico.png`](presentacion/figuras_charla/mpasm_esferico.png) · `fc.mpasm_esferico()` | [`resultados/dlhm_esferico/campo_N3000.csv`](resultados/dlhm_esferico/campo_N3000.csv) (`python -m scripts.verif_dlhm_esferico campo --n 3000`) | Fig. 4.4 · §4.2.1 |
| 24 | Con onda plana, la matriz del MPASM es una CZT; con la esférica, la ventana W le quita esa forma | Ecuaciones · [`scripts/contraste_czt.py`](scripts/contraste_czt.py), [`tests/test_contraste_czt.py`](tests/test_contraste_czt.py) | [`acuerdo.csv`](resultados/contraste_czt/acuerdo.csv) y [`tiempos.csv`](resultados/contraste_czt/tiempos.csv) de `resultados/contraste_czt/` (`python -m scripts.contraste_czt`) | Tabla 4.2 · §2.4.1 · §4.2.2 |
| 25 | En el banco, el ASM con relleno falla solo donde el cociente pasa de 3 | [`mapas_dos.png`](presentacion/figuras_charla/mapas_dos.png) · `fc.mapas_dos()`, los paneles (a) y (e) de la Fig. 4.5 | [`resultados/barrido_obj2/barrido.csv`](resultados/barrido_obj2/barrido.csv) (`python -m scripts.barrido_obj2`; la frontera, `--frontera`) | Fig. 4.5 · Tabla 4.3 · §4.2.3 |
| 26 | Fuera de la ventana decide el sobremuestreo del MPASM, y su Kf automático es el adecuado | [`celda_ejemplo.png`](presentacion/figuras_charla/celda_ejemplo.png) · `fc.celda_ejemplo()`: la celda z = 3 mm, NA ≈ 0.06 | `barrido.csv` | §4.2.3 |
| 27 | Con la misma ventana, el ASM y el MPASM recuperan la misma información | [`ida_vuelta.png`](presentacion/figuras_charla/ida_vuelta.png) · `fc.ida_vuelta()`, la Fig. 4.7 con la leyenda corta | (a) `resultados/comparacion_cruzada/{entrada_r4_cercano, entrada_r4_transicion, entrada, entrada_r4}/resumen.csv` (`scripts/comparacion_cruzada.py`); (b) [`resultados/cruzada_dlhm/resumen.csv`](resultados/cruzada_dlhm/resumen.csv) (`scripts/cruzada_dlhm.py`) | Fig. 4.7 · Tabla 4.4 · §4.2.4 |
| 28 | Una ventana mayor mejora la reconstrucción, y el MPASM la permite | [`fresnel_reconstrucciones.png`](presentacion/figuras_charla/fresnel_reconstrucciones.png) · `fc.fresnel_reconstrucciones()` | [`entrada_r4/resumen.csv`](resultados/comparacion_cruzada/entrada_r4/resumen.csv) y los campos `entrada_r4/z*/*.npy`, [no versionados](#lo-que-no-está-en-el-repositorio); s de 1 a 4: `entrada_s4/` frente a `entrada/` | Fig. 4.8 · Tabla 4.5 · §4.2.4 |

### Resultados: objetivo 3

Los hologramas son del laboratorio y no están en el repositorio; los scripts
los leen como dice [la última sección](#lo-que-no-está-en-el-repositorio).

| # | Título | Figura · script | Datos | Manuscrito |
|---|---|---|---|---|
| 29 | 4.3 Hologramas experimentales | Separador | — | §4.3 |
| 30 | Los hologramas son de 17 posiciones de la platina: al alejar la muestra de la fuente, la magnificación baja de 44 a 9 | Los hologramas crudos de las posiciones 1, 9 y 17, captura 1 | M = L/z de [`focos_exactos.csv`](resultados/hologramas_carlos/focos_exactos.csv) (`hc.correr exacto`) | §3.3 · §4.3 |
| 31 | El foco no fija la geometría | [`exp_z_frente_a_L.png`](presentacion/figuras_charla/exp_z_frente_a_L.png) · `fc.exp_z_frente_a_L()` | [`z_frente_a_L.csv`](resultados/hologramas_carlos/z_frente_a_L.csv) (`hc.correr zL`) y [`focos_D.csv`](resultados/hologramas_carlos/focos_D.csv) (`hc.correr foco`) | Fig. 4.11 · Ec. 3.8 · §4.3.2 |
| 32 | Los hologramas se reconstruyen con detalle hasta el grupo 7, en amplitud y en fase | [`exp_reconstrucciones.png`](presentacion/figuras_charla/exp_reconstrucciones.png) · `fc.exp_reconstrucciones()`, la Fig. 4.13 con barra de escala | `resultados/hologramas_carlos/imagen_{amp,phase}_z*.npy` (`hc.correr imagenes`, no versionados) y `focos_exactos.csv` | Fig. 4.13 · §4.3.3 |
| 33 | Con la misma ventana, los dos métodos dan casi la misma imagen | [`exp_misma_ventana.png`](presentacion/figuras_charla/exp_misma_ventana.png) · `fc.exp_misma_ventana()`, la Fig. 4.14 en dos filas | `t_exacto_z*.npy` y `t_carlos_z*.npy` (`hc.comparar metricas`, no versionados); [`metricas_carlos.csv`](resultados/hologramas_carlos/metricas_carlos.csv); las cajas, [`hc/cajas_metricas.json`](scripts/hologramas_carlos/cajas_metricas.json) | Fig. 4.14 · Tabla 4.6 · §4.3.4 |
| 34 | La ventaja del MPASM esférico es de memoria: en la misma GPU reconstruye un recorte veinte veces mayor | [`memoria_cadena.png`](presentacion/figuras_charla/memoria_cadena.png) · `fc.memoria_cadena()` | Los GB van escritos en la función; los calcula `memoria_carlos()` de [`hc/comparar.py`](scripts/hologramas_carlos/comparar.py) y quedan en [`tiempos_carlos.csv`](resultados/hologramas_carlos/tiempos_carlos.csv) (`hc.comparar tiempos`) | §3.4 · §4.3.4 |
| 35 | Con una ventana de 2048 × 2560 píxeles, el contraste de los detalles de 5 a 10 µm casi se duplica | [`exp_mas_ventana.png`](presentacion/figuras_charla/exp_mas_ventana.png) · `fc.exp_mas_ventana()` | `t_exacto_z*.npy` y la zona central de `imagen_amp_z*.npy` (`hc.comparar.zona_grande`); `metricas_carlos.csv` (columnas `contraste_grande`) | Fig. 4.14 · Tabla 4.6 · §4.3.4 |
| 36 | En simulación a 193 cm, la ventana ampliada del MPASM recupera las barras que la ventana del sensor pierde | [`asm_vs_mpasm_lejos.png`](presentacion/figuras_charla/asm_vs_mpasm_lejos.png) · `fc.asm_vs_mpasm_lejos()`, z = 1926 mm (N_F = 1) | `entrada_r4/resumen.csv` y `entrada_r4/z1926.000/*.npy` (no versionados); el objeto, [`resultados/campos/entrada.png`](resultados/campos/entrada.png) | Fig. 4.8 y 4.9 · §4.2.4 |

### Conclusiones, referencias y respaldo

| # | Título | Figura · script | Datos | Manuscrito |
|---|---|---|---|---|
| 37 | Los objetivos 1 y 2 quedaron cumplidos, y el 3 sin una de sus métricas | Tabla · cada número sale del capítulo 4 | — | §1.2 · §5.1 |
| 38 | El control del muestreo conviene cuando el campo no cabe; lo que mejora la reconstrucción es una ventana mayor | [`resumen_q.png`](presentacion/figuras_charla/resumen_q.png) · `fc.resumen_q()` | `barrido.csv`; los extremos de los demás ensayos, escritos en la función, son los del manuscrito | §4.4 · §5.1 |
| 39 | Quedan cuatro caminos, empezando por medir la fidelidad | Texto | — | §5.2 |
| 40 | Gracias | — | — | — |
| 41–43 | Referencias | Lista | — | [`manuscrito/referencias.bib`](manuscrito/referencias.bib) |
| 44 | Respaldo | Separador | — | — |
| 45 | Con s = 3, el MPASM es confiable en 95 de las 98 celdas del mapa | [`mapas_banco.png`](presentacion/figuras_charla/mapas_banco.png) · `fm.figura_mapas_banco()`, dibujada por `fc` con «ASM» en los rótulos | `barrido.csv` (`python -m scripts.barrido_obj2 --resumen 1e-3`) | Fig. 4.5 · Tabla 4.3 · §4.2.3 |
| 46 | Ni la mitad ni el doble mejoran el Kf automático, con ningún s | [`mapas_kf.pdf`](manuscrito/figuras/mapas_kf.pdf) · `fm.figura_mapas_kf()` | `barrido.csv` | Fig. 4.6 · §4.2.3 |
| 47 | Con Kf > 1, el error lo fija cuánta banda se recorta, no el periodo | [`g_s1.png`](presentacion/figuras_charla/g_s1.png) · `fg.figura_s1()` | `barrido.csv` | §4.2.3 |
| 48 | El 3 de la frontera sale de la geometría del relleno al doble | [`g_rayos.png`](presentacion/figuras_charla/g_rayos.png) · `fg.figura_rayos()` | Ninguno: es un esquema | §4.2.3 |
| 49 | El escalado de Fresnel lleva el DLHM a una onda plana, en dos mallas equivalentes | [`g_tres_lecturas.png`](presentacion/figuras_charla/g_tres_lecturas.png) · `fg.figura_tres_lecturas()` | Ninguno: es un esquema | Ec. 3.4 · §3.2.1 |
| 50 | En el MPASM esférico, cada píxel aporta un bloque de frecuencias en la dirección de su rayo | [`g_esferica.png`](presentacion/figuras_charla/g_esferica.png) · `fg.figura_esferica()` | Ninguno: λ, δ y L del banco | Ec. 3.5 · §3.2.2 |
| 51 | La escala del USAF fija la geometría, y la platina la confirma con pasos redondos | [`recta_MD.png`](presentacion/figuras_charla/recta_MD.png) · `fc.recta_MD()` | [`L_implicito.csv`](resultados/hologramas_carlos/L_implicito.csv) (`hc.correr calibracion`); el grupo 5, [`grupo5.csv`](resultados/hologramas_carlos/grupo5.csv) (`hc.correr grupo5`) | Ec. 3.9 · §3.3 · §4.3.2 |
| 52 | Con la geometría adoptada, la platina avanza a pasos redondos y L casi no varía | [`exp_geometria.pdf`](manuscrito/figuras/exp_geometria.pdf) · `fm.figura_exp_geometria()` | `focos_exactos.csv` y `L_implicito.csv` | Fig. 4.12 · §4.3.2 |
| 53 | En un montaje largo, el propagador decide al simular y no al medir | [`interferometria.png`](presentacion/figuras_charla/interferometria.png) · `fm.figura_interferometria()`, dibujada por `fc` con «ASM» en los rótulos | [`resultados/interferometria_larga/resumen.csv`](resultados/interferometria_larga/resumen.csv) y `dphi_q20_*.npy` (no versionados), de `python -m scripts.interferometria_larga` | Fig. A.1 · Tabla A.1 · Anexo A |

## Las figuras del manuscrito

Todas están en [`manuscrito/figuras/`](manuscrito/figuras/). Las dibuja
[`scripts/figuras_manuscrito.py`](scripts/figuras_manuscrito.py)
(`python -m scripts.figuras_manuscrito`) salvo la 4.1, que dibuja
`scripts/exactitud.py`. Ninguna vuelve a calcular: leen lo que dejó el script de
la columna de datos. En cada `.tex`, un comentario `% Repo:` junto a la figura
dice lo mismo.

| Fig. | Qué muestra | Archivo · función | Datos (y el script que los produce) |
|---|---|---|---|
| 2.1 | K_f automático en función de la distancia | [`kf_vs_z.pdf`](manuscrito/figuras/kf_vs_z.pdf) · `figura_kf_vs_z()` | Ninguno: `kf_auto` con el escenario de Zhao et al. (`CamposT.montaje.TABLA1`) |
| 2.2 | Esquema del montaje DLHM | [`esquema_montaje.pdf`](manuscrito/figuras/esquema_montaje.pdf) · `figura_esquema_montaje()` | Ninguno: es un esquema |
| 4.1 | Exactitud de los tres propagadores en el escenario de Zhao et al. | [`obj1_sam_vs_z.pdf`](manuscrito/figuras/obj1_sam_vs_z.pdf) · `scripts/exactitud.py` (`figura_sam`) | [`resultados/exactitud/sam_vs_z.csv`](resultados/exactitud/sam_vs_z.csv) (`python -m scripts.exactitud`) |
| 4.2 | La Ec. (14) de Zhao et al.: artículo frente a código publicado | [`errata_kf.pdf`](manuscrito/figuras/errata_kf.pdf) · `figura_errata_kf()` | Ninguno: `kf_auto` con las dos fórmulas |
| 4.3 | Tiempo de propagación en función de N y de s | [`tiempos.pdf`](manuscrito/figuras/tiempos.pdf) · `figura_tiempos()` | [`escalado_N.csv`](resultados/escalado_N.csv) y [`escalado_s.csv`](resultados/escalado_s.csv) (`python -m scripts.comparacion`) |
| 4.4 | MPASM con iluminación esférica exacta frente al escalado paraxial | [`mpasm_esferico.pdf`](manuscrito/figuras/mpasm_esferico.pdf) · `figura_mpasm_esferico()` | [`dlhm_esferico/campo_N3000.csv`](resultados/dlhm_esferico/campo_N3000.csv) (`python -m scripts.verif_dlhm_esferico campo --n 3000`) |
| 4.5 | Mapas de validez en la geometría del banco | [`mapas_banco.pdf`](manuscrito/figuras/mapas_banco.pdf) · `figura_mapas_banco()` | [`barrido_obj2/barrido.csv`](resultados/barrido_obj2/barrido.csv) (`python -m scripts.barrido_obj2`) |
| 4.6 | El MPASM con K_f distinto del automático | [`mapas_kf.pdf`](manuscrito/figuras/mapas_kf.pdf) · `figura_mapas_kf()` | `barrido.csv` |
| 4.7 | Ida y vuelta: las cuatro combinaciones de FFT-ASM y MPASM | [`ida_vuelta_cuatro.pdf`](manuscrito/figuras/ida_vuelta_cuatro.pdf) · `figura_ida_vuelta_cuatro()` | `comparacion_cruzada/{entrada_r4_cercano, entrada_r4_transicion, entrada, entrada_r4}/resumen.csv` (`scripts/comparacion_cruzada.py`) y [`cruzada_dlhm/resumen.csv`](resultados/cruzada_dlhm/resumen.csv) (`scripts/cruzada_dlhm.py`) |
| 4.8 | Ida y vuelta con fase a N_F = 10 y N_F = 1 | [`fresnel_reconstrucciones.pdf`](manuscrito/figuras/fresnel_reconstrucciones.pdf) · `figura_fresnel_reconstrucciones()` | [`entrada_r4/resumen.csv`](resultados/comparacion_cruzada/entrada_r4/resumen.csv) y `entrada_r4/z*/*.npy` (no versionados; `scripts/comparacion_cruzada.py` con sus valores por defecto) |
| 4.9 | Exactitud y resolución de la reconstrucción en la zona de Fresnel | [`fresnel_metricas.pdf`](manuscrito/figuras/fresnel_metricas.pdf) · `figura_fresnel_metricas()` | `entrada_r4/resumen.csv` y [`parametros.txt`](resultados/comparacion_cruzada/entrada_r4/parametros.txt) |
| 4.10 | Curvas de enfoque de los hologramas experimentales | [`exp_foco.pdf`](manuscrito/figuras/exp_foco.pdf) · `figura_exp_foco()` | [`curvas_amp.npz`](resultados/hologramas_carlos/curvas_amp.npz), [`curvas_phase.npz`](resultados/hologramas_carlos/curvas_phase.npz) y [`focos_D.csv`](resultados/hologramas_carlos/focos_D.csv) (`hc.correr foco`) |
| 4.11 | El foco no fija L | [`exp_z_frente_a_L.pdf`](manuscrito/figuras/exp_z_frente_a_L.pdf) · `figura_exp_z_frente_a_L()` | [`z_frente_a_L.csv`](resultados/hologramas_carlos/z_frente_a_L.csv) (`hc.correr zL`) y `focos_D.csv` |
| 4.12 | Geometría inferida de los hologramas experimentales | [`exp_geometria.pdf`](manuscrito/figuras/exp_geometria.pdf) · `figura_exp_geometria()` | [`focos_exactos.csv`](resultados/hologramas_carlos/focos_exactos.csv) (`hc.correr exacto`) y [`L_implicito.csv`](resultados/hologramas_carlos/L_implicito.csv) (`hc.correr calibracion`) |
| 4.13 | Reconstrucciones de los hologramas experimentales | [`exp_reconstrucciones.pdf`](manuscrito/figuras/exp_reconstrucciones.pdf) · `figura_exp_reconstrucciones()` | `imagen_{amp,phase}_z*.npy` (no versionados; `hc.correr imagenes`) y `focos_exactos.csv` |
| 4.14 | MPASM esférico frente a la cadena del laboratorio | [`exp_carlos.pdf`](manuscrito/figuras/exp_carlos.pdf) · `figura_exp_carlos()` | `t_exacto_z*.npy`, `t_carlos_z*.npy` e `imagen_amp_z*.npy` (no versionados; `hc.comparar metricas`, `hc.correr imagenes`), [`metricas_carlos.csv`](resultados/hologramas_carlos/metricas_carlos.csv) y [`cajas_metricas.json`](scripts/hologramas_carlos/cajas_metricas.json) |
| A.1 | Interferometría holográfica simulada a distancias largas | [`interferometria.pdf`](manuscrito/figuras/interferometria.pdf) · `figura_interferometria()` | [`interferometria_larga/resumen.csv`](resultados/interferometria_larga/resumen.csv) y `dphi_q20_*.npy` (no versionados; `python -m scripts.interferometria_larga`) |

`hc` = `scripts/hologramas_carlos/`: `hc.correr foco` es
`python -m scripts.hologramas_carlos.correr foco`, y así las demás.

## Las tablas del manuscrito

Las tablas están escritas en el `.tex`; sus números salen de estos archivos.

| Tabla | Qué muestra | Datos (y el script que los produce) |
|---|---|---|
| 3.1 | Parámetros del montaje y de los hologramas experimentales | Nominales: [`CamposT/montaje.py`](CamposT/montaje.py). Inferidos: [`calibracion.json`](resultados/hologramas_carlos/calibracion.json) (`hc.correr calibracion`) |
| 4.1 | Tiempo de una propagación en CPU y en GPU | [`escalado_N.csv`](resultados/escalado_N.csv) y [`escalado_s.csv`](resultados/escalado_s.csv) (`python -m scripts.comparacion`) |
| 4.2 | El MPASM calculado con sus matrices y con la CZT | [`acuerdo.csv`](resultados/contraste_czt/acuerdo.csv) y [`tiempos.csv`](resultados/contraste_czt/tiempos.csv) (`python -m scripts.contraste_czt`) |
| 4.3 | Validez de los propagadores en la geometría del banco | `barrido.csv` (`python -m scripts.barrido_obj2 --resumen 1e-3` y `--resumen 1e-2`) |
| 4.4 | Ida y vuelta: SAM de las cuatro combinaciones | `comparacion_cruzada/*/resumen.csv` y `cruzada_dlhm/resumen.csv` |
| 4.5 | Ida y vuelta en la zona de Fresnel: ventana fija frente a ampliada | [`entrada_r4/resumen.csv`](resultados/comparacion_cruzada/entrada_r4/resumen.csv) |
| 4.6 | Contraste del USAF: MPASM esférico frente a la cadena del laboratorio | [`metricas_carlos.csv`](resultados/hologramas_carlos/metricas_carlos.csv) (`hc.comparar metricas`); los tiempos de la §4.3.4, [`tiempos_carlos.csv`](resultados/hologramas_carlos/tiempos_carlos.csv) (`hc.comparar tiempos`) |
| A.1 | Interferometría simulada: error de la ida | [`interferometria_larga/resumen.csv`](resultados/interferometria_larga/resumen.csv) |

Los números sueltos del texto también tienen su `% Repo:` en el `.tex`, sobre
cada sección: por ejemplo, `04_resultados.tex` dice de dónde sale cada cifra de
la §4.1 (`scripts/exactitud.py`, `contraste_referencias.py`, `comparacion.py` y
`tests/test_rs1.py`).

## Lo que no está en el repositorio

- **Los hologramas experimentales.** Los registró el laboratorio de Óptica
  Aplicada de la Universidad EAFIT (USAF de amplitud y de fase, 17 posiciones
  × 16 capturas, más la referencia sin muestra; 4640 × 3506 píxeles, JPEG) y no
  se publican: se le piden al grupo. Los scripts de
  [`scripts/hologramas_carlos/`](scripts/hologramas_carlos/) los buscan en
  `~/Downloads/hologramas_carlos/` (`datos_amp/` y `datos_phase/`, una carpeta
  `z1` a `z17` por posición), o donde diga la variable de entorno
  `HOLOGRAMAS_CARLOS` (`datos.py`). Del repositorio sí son las medidas que
  salen de ellos: los CSV, el JSON y las curvas de enfoque de
  `resultados/hologramas_carlos/`.
- **Los campos `.npy`** de `resultados/`, de 2 a 64 MB cada uno. Se regeneran
  con el script de la tabla:

  | Archivos | Orden |
  |---|---|
  | `comparacion_cruzada/entrada_r4/z*/*.npy` | `python -m scripts.comparacion_cruzada` (valores por defecto: R = 4) |
  | `cruzada_dlhm/*.npy` | `python -m scripts.cruzada_dlhm` |
  | `hologramas_carlos/imagen_*.npy` (40 MB) | `python -m scripts.hologramas_carlos.correr imagenes` (necesita los hologramas) |
  | `hologramas_carlos/t_exacto_*.npy`, `t_carlos_*.npy` | `python -m scripts.hologramas_carlos.comparar metricas` (ídem) |
  | `interferometria_larga/dphi_q20_*.npy` | `python -m scripts.interferometria_larga` |

- **El `.pptx` de la defensa y el script que lo arma**
  (`presentacion/generar_defensa.py`), con las notas del orador. Del
  repositorio son las figuras que se proyectan; ver
  [`presentacion/README.md`](presentacion/README.md).
- **Los repos de terceros** de `referencia/carlos/` (pyDHM y pyLHM): se clonan de
  GitHub, como dice el [README](README.md#referenciacarlos).
