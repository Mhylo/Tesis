# Manuscrito: objetivo 3 experimental y comparación ASM–MPASM reunida. Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** que el borrador del 21/10 cuente la reconstrucción de los hologramas experimentales de Carlos, con la geometría corregida y la comparación con su cadena, y la comparación ASM–MPASM en un solo lugar, con las cuatro combinaciones de ida y vuelta en tres escalas de distancia.

**Arquitectura:** el código de análisis de los hologramas pasa de `resultados/hologramas_carlos/codigo/` (borrador) a un paquete limpio y probado, `scripts/hologramas_carlos/`. El caso esférico del banco va en un script propio, `scripts/cruzada_dlhm.py`. Los dos escriben CSV y `.npy` en `resultados/`. `scripts/figuras_manuscrito.py` los lee y deja los PDF en `manuscrito/figuras/`. El texto se edita en los `.tex` existentes.

**Stack:** Python del entorno `Tesis_env` (NumPy, CuPy, SciPy, OpenCV, Pillow, Matplotlib, pytest), `CamposT` (`dlhm`, `propagadores`, `metricas`), `scripts/mi_prueba_dlhm.py` (importado, no editado), LaTeX con latexmk.

**Spec:** `docs/superpowers/specs/2026-10-20-manuscrito-obj3-asm-mpasm-design.md`

## Restricciones globales

- **Geometría del banco simulado (de Carlos):** λ = 528e-6 mm, δ = 1.83e-3 mm, L = 10 mm. Las z son 1, 2, 3, 5 y 7 mm, en un sensor simulado de 1024×1024.
- **Geometría experimental (inferida):** δ = 3.80e-3 mm, L = 18.229 mm, λ = 528e-6 mm. En el texto siempre va marcada como "inferida, por confirmar con el laboratorio".
- **Las dos geometrías nunca se mezclan.** Ninguna función del paquete experimental toma δ ni L de `CamposT.montaje.MONTAJE`: se pasan explícitos.
- **D = L(L−z)/z no es una distancia física.** En el texto se usa solo junto con su significado: la misma propagación expresada en la malla del sensor. "Distancia larga" se escribe como el cociente z_eq/(N(δ/M)²/λ), que es igual a D/(Nδ²/λ).
- **Archivos del usuario:**
  - `scripts/mi_prueba_dlhm.py` no se edita;
  - en `scripts/figuras_manuscrito.py` y en los `.tex` solo se agrega o reescribe lo que dice el plan, sin revertir sus cambios sin commitear;
  - `scripts/comparacion_cruzada.py` se edita solo para correr y se restaura igual.
- **Sin `git commit`** (decisión del usuario). Los pasos de commit de la plantilla se sustituyen por "verificar `git diff --stat`".
- **GPU de 4 GB.** El `Reconstructor` del sensor entero no cabe: se usan recortes centrales de 2048×2560 (exacto) y de 1024 (comparación con Carlos).
- **Datos fuera del repo.** Constante `DATOS` = carpeta con `datos_amp/` y `datos_phase/` descomprimidos de `Downloads/4_Amplitude_USAF.zip` y `Downloads/10_Phase_USAF.zip`. Los scripts leen con Python `-I` cuando abren esos archivos.
- **Correr:** `Tesis_env/Scripts/python.exe -m pytest ...`; scripts con `Tesis_env/Scripts/python.exe -m scripts.<modulo>`, y `CUDA_PATH` apuntando a CUDA v13.3.

## Foco de revisión

1. **El signo del foco en fase:** un objeto de fase enfoca en el mínimo de contraste. Si un barrido de fase toma el máximo, todo el conjunto de fase sale corrido. Prueba en la Tarea 1.
2. **La convención de signo de la ida espejo:** el `angular_spectrum` de Carlos usa e^{−ikz}. Una ida con el signo cambiado retropropaga en vez de propagar, y la ida y vuelta igual "cierra". Prueba en la Tarea 3: la ida espejo concuerda con `holograma(complejo=True)` en el centro.
3. **El recorte de Carlos centrado y cuadrado:** si la ventana no está centrada en el eje, su esférica sale descentrada y el foco se corre. `MotorCarlos` aborta con ROI fuera del recorte; la Tarea 4 usa `recorte=1024` y ROI dentro.
4. **La división por un fondo casi cero en el JPEG:** sin piso, las esquinas oscuras de `ref.jpg` explotan. Todas las rutas usan piso 0.05; prueba en la Tarea 1.
5. **El ajuste de tres barras sobre perfiles con anillos de gemela:** el método de mínimos falló en esta sesión. Prueba en la Tarea 1 con un perfil sintético con oscilaciones dentro de las barras.

---

### Tarea 1: paquete `scripts/hologramas_carlos/` (foco, geometría, USAF) con pruebas

**Archivos:**
- Crear `scripts/hologramas_carlos/__init__.py`, con un docstring que describe los datos, la constante `DATOS` y el orden de corrida.
- Crear `scripts/hologramas_carlos/datos.py`, `foco.py`, `geometria.py` y `usaf.py`, desde los borradores de `resultados/hologramas_carlos/codigo/`: `paraxial.py`, `lote_D.py`, `lote_fino.py`, `usaf_ajuste.py` y `ajuste_conjunto.py`.
- Prueba: `tests/test_hologramas_carlos.py`.

**Interfaces que produce:**
- `datos.leer(conjunto: str, k: int, captura: str) -> tuple[np.ndarray, np.ndarray]`: el holograma y `ref`, en float64. `conjunto` es `"amp"` o `"phase"`.
- `foco.contraste(holo, ref, piso=0.05) -> np.ndarray`: I/ref − 1, y 0 bajo piso·max(ref).
- `foco.Paraxial(c, delta, lamb, pad=2, xp=None)`, con el método `.campo(D, ventana=None)`, que devuelve 1 + U retropropagado −D en la malla del sensor.
- `foco.tamura(A, xp) -> float`.
- `foco.barrer(P, Ds, ventana, signo: str) -> np.ndarray`, la curva. `signo` es `"max"` (amplitud) o `"min"` (fase), y la curva se devuelve ya orientada para maximizar.
- `foco.pico(Ds, curva, tendencia_mm=20.0) -> float`: quita la mediana móvil y hace interpolación parabólica.
- `geometria.D_de(z, L)`, `geometria.z_de(D, L)`, `geometria.M_de(z, L)`, `geometria.zeq_de(z, L)`.
- `geometria.cociente_ventana(z, L, delta, N, lamb) -> float`: z_eq/(N(δ/M)²/λ).
- `geometria.ajuste_conjunto(ks, D_medido, eD, kM, M_medido, eM, D_de_M, delta_supuesto) -> dict`, con las claves p, L, z1, dz, sus errores y chi2_dof. El modelo es el de `ajuste_conjunto.py`: D_medido = L(L−z)/z/p², M_medido = (L/z)/p y z_k = z1 + (k−1)dz.
- `usaf.lp_mm(g, e) -> float`.
- `usaf.ajustar_barras(perfil) -> tuple[float, float]`: w en px y el sigma del borde. Es el modelo de tres barras con erf de `usaf_ajuste.py`, con varios arranques.
- `usaf.magnificacion(w_px, g, e, delta_um) -> float`: 2w·δ / (1000/lp_mm).

- [ ] **Paso 1: escribir las pruebas**, en `tests/test_hologramas_carlos.py`:
  - `test_D_z_ida_y_vuelta`: `z_de(D_de(z, L), L) == approx(z)` para z de 0.4 a 2.0 y L de 10 a 18.229.
  - `test_cociente_ventana`: con z = 0.4146, L = 18.229, δ = 3.8e-3, N = 4640 y λ = 528e-6, da approx 6.17 con rel = 0.01.
  - `test_ajuste_conjunto_recupera`:
    - datos sintéticos con p = 2.0765, L = 18.229, z1 = 0.4167, dz = 0.09906, 17 posiciones y 4 M;
    - ruido gaussiano de 0.1 % con semilla 0;
    - el ajuste devuelve p con rel = 0.03, L con rel = 0.05 y dz con rel = 0.01;
    - con p fijo en 1, chi2_dof > 20 × el chi2_dof libre.
  - `test_barras_con_anillos`:
    - perfil de tres barras, w = 67.0 px y borde sigma 1.5;
    - sumarle cos(2π x/9) × 0.3 dentro de las barras y ruido de 0.05;
    - `ajustar_barras` devuelve w approx 67.0, con abs = 1.0.
  - `test_foco_gabor_sintetico`:
    - objeto de amplitud: tres barras opacas sobre 1, en 256×256;
    - holograma de Gabor de onda plana por BLAS a D = 40 mm, con δ = 3.8e-3 y λ = 528e-6;
    - `pico(Ds, barrer(...,"max"))` con Ds de 20 a 60, paso 0.5, da 40 con abs = 1.0.
  - `test_foco_fase_es_minimo`: el mismo con t = exp(1j·0.8·barras). Con `"min"` da 40 ± 1; con `"max"` no da 40 ± 1.
  - `test_contraste_piso`: con ref de esquina 0.01·max, `contraste` vale 0 ahí y es finito en todo el array.

- [ ] **Paso 2: correrlas y ver que fallan.** `Tesis_env/Scripts/python.exe -m pytest tests/test_hologramas_carlos.py -v` debe dar ModuleNotFoundError en `scripts.hologramas_carlos`.

- [ ] **Paso 3: implementar los cuatro módulos** con las firmas de arriba, pasando el código del borrador a funciones:
  - sin rutas del scratchpad;
  - δ, L y λ siempre como argumento;
  - CuPy si lo hay, NumPy si no, con `CamposT.backend.get_xp("auto")`.

- [ ] **Paso 4: correrlas y ver que pasan.** El mismo comando, todas en PASS.

- [ ] **Paso 5: verificar.** `git diff --stat` solo muestra archivos nuevos, sin commit.

### Tarea 2: corridas experimentales (foco, calibración, motor exacto) a `resultados/hologramas_carlos/`

**Archivos:**
- Crear `scripts/hologramas_carlos/correr.py`. Cada subcomando escribe en `resultados/hologramas_carlos/`:
  - `foco`: grueso de 5 a 200 con paso 0.5, en la captura 1, más fino de ±7 (amplitud, 16 capturas) o de ±9 (fase, capturas impares), con paso 0.1. Escribe `focos_D.csv` (conjunto, k, captura, D) y `curvas_<conjunto>.npz`.
  - `calibracion`: ajuste de barras con las cajas de `scripts/hologramas_carlos/cajas_usaf.json` más el ajuste conjunto. Escribe `calibracion.json` (M por posición y el resultado del ajuste) y `L_implicito.csv`.
  - `exacto`: `Reconstructor` en recorte de 2048×2560 con δ = 3.8e-3 y L = 18.229, en las 17 posiciones × los 2 conjuntos, captura 1. Escribe `focos_exactos.csv` (z_foco, M, Tamura).
  - `zL`: z del foco para L en {12, 15, 18.229, 22, 26} con δ = 3.8e-3, y en {5, 6, 8, 10, 12, 15, 20, 30} con δ = 1.83e-3, sobre amp z9/1. Escribe `z_frente_a_L.csv`.
- Crear `scripts/hologramas_carlos/cajas_usaf.json`, con las cajas ya medidas en esta sesión:
  - z17, captura 5: g4e2 v, g4e3 v y g4e2 h;
  - z9, captura 1: grupos 4 y 6;
  - z5, captura 1: grupo 6;
  - z1, captura 1: grupos 7 y 6.
  
  Las coordenadas están en el historial de esta sesión y en las llamadas a `usaf_ajuste.py`.

**Interfaces:** consume la Tarea 1. Produce los CSV y JSON nombrados, con las columnas de arriba.

- [ ] **Paso 1: correr `foco`** y comparar contra los números de la sesión. Amplitud z1 mediana 185.32 y z17 34.78, con ±0.3. Fase z9 mediana 86.1, con ±1.
- [ ] **Paso 2: correr `calibracion`.** Esperado:
  - chi2_dof libre ≈ 8;
  - p = 2.15 ± 0.16;
  - L = 18.9 ± 1.4;
  - dz = 0.0978;
  - L implícito con δ = 1.83: de 9.09 a 10.27; con δ = 3.80: de 17.94 a 18.53.
- [ ] **Paso 3: correr `exacto` y `zL`.** Esperado:
  - amplitud: z1 = 0.4146 y z17 = 2.0026, con ±0.002;
  - el ajuste lineal da paso 0.0991 ± 0.001;
  - fase: paso 0.050 ± 0.002;
  - con δ = 3.8 y L = 18.229, `zL` da 1.2023 en z9/1.
- [ ] **Paso 4: verificar** que existen los cinco archivos de salida y que `git diff --stat` no toca archivos del usuario.

### Tarea 3: `scripts/cruzada_dlhm.py`, las cuatro combinaciones en el banco con onda esférica

**Archivos:**
- Crear `scripts/cruzada_dlhm.py`.
- Prueba: `tests/test_cruzada_dlhm.py`.

**Interfaces:**
- Consume:
  - de `scripts.mi_prueba_dlhm`: `point_src`, `angular_spectrum`, `resize`, `malla_remuestreada`, `reconstruir`, `onda_de_referencia` e `interpolar_centrado`;
  - de `CamposT.dlhm`: `holograma`, `Reconstructor` y `coordenadas`;
  - `CamposT.metricas.sam`.
- Produce:
  - `ida_carlos(t, z, L, lamb, delta, N) -> np.ndarray` (complejo N×N): el holograma complejo 1 + V en la malla del sensor. Es el espejo de `reconstruir()`:
    1. t, en la malla δ·z/L de N×N, se interpola a la malla remuestreada de `malla_remuestreada(N, N, z, lamb, delta)`, que cubre la ventana Nδ, valiendo 1 fuera de su huella;
    2. se multiplica por `point_src(n_r, n_r, z, 0, 0, lamb, N·delta/n_r)`;
    3. se propaga con `angular_spectrum(·, Nδ, Nδ, k, −(L − z))`, con el signo negativo porque el suyo es e^{−ikz};
    4. se divide por `point_src(n_r, n_r, L, ...)`;
    5. se pasa a N×N con `resize`.
  - `vuelta_carlos(h, z, L, lamb, delta) -> np.ndarray`: t en la malla δ·z/L, N×N. Es `MotorCarlos` con `recorte = N` y la ventana entera.
  - `ida_mpasm(t, z, L, lamb, delta, N) -> np.ndarray`: `1 + holograma(..., complejo=True, banda=500.0)`.
  - `vuelta_mpasm(h, z, L, lamb, delta) -> np.ndarray`: `Reconstructor(h, delta, lamb, L, z).campo(z)`.
  - `main()`:
    - objeto USAF binario de 1024 a partir de `resultados/campos/entrada.png` (el mismo de `comparacion_cruzada.py`), escalado a 1024 y suavizado con σ = 1 px para no pedir banda infinita;
    - z en (1, 2, 3, 5, 7) con L = 10, δ = 1.83e-3, λ = 528e-6 y N = 1024;
    - las cuatro combinaciones;
    - escribe `resultados/cruzada_dlhm/resumen.csv` con z, D, cociente, ida, vuelta, sam_db y error_max, y los `.npy` de las reconstrucciones a z = 3 y z = 1.

- [ ] **Paso 1: escribir las pruebas**, en `tests/test_cruzada_dlhm.py`, con N = 256, z = 3, L = 10 y objeto de barras suaves:
  - `test_ida_vuelta_carlos_cierra`: `vuelta_carlos(ida_carlos(t))` frente a t da error máximo < 1e-3 en el 50 % central.
  - `test_ida_espejo_concuerda_con_exacta`: la correlación de `ida_carlos` con `ida_mpasm` en el 25 % central es > 0.95 (la parte paraxial coincide). Si el signo de la propagación estuviera cambiado, daría < 0.5.
  - `test_vuelta_mpasm_de_ida_mpasm`: SAM > 30 dB frente a t, en el 50 % central.

- [ ] **Paso 2: correrlas y ver que fallan** (ImportError): `Tesis_env/Scripts/python.exe -m pytest tests/test_cruzada_dlhm.py -v`.

- [ ] **Paso 3: implementar** con las firmas de arriba. El objeto en la huella se interpola con `interpolar_centrado`, sin cv.resize: mismo criterio que la vuelta.

- [ ] **Paso 4: correrlas y ver que pasan.**

- [ ] **Paso 5: correr `main()`.** Comprobar que el CSV tiene 5 × 4 filas, que la columna del cociente vale ≈ 14.2, 6.2, 3.6, 1.5 y 0.66, y que ASM→ASM y MPASM→MPASM dan SAM alto donde el cociente es < 1.

### Tarea 4: comparación experimental, MPASM esférico frente a la cadena de Carlos

**Archivos:**
- Crear `scripts/hologramas_carlos/comparar.py`.
- Crear las cajas de métricas en `scripts/hologramas_carlos/cajas_metricas.json`, dentro del recorte central de 1024: para z1, z9 y z17 de amplitud, captura 1, los elementos del USAF visibles más una caja de fondo. Se eligen viendo la reconstrucción exacta del recorte.
- Prueba: agregar a `tests/test_hologramas_carlos.py`.

**Interfaces:**
- Consume: las Tareas 1 y 2 (focos exactos) y `scripts.mi_prueba_dlhm.MotorCarlos`, con `recorte=1024`, `raiz=sqrt(ref)` y `raiz_min=sqrt(0.05)·max`.
- Produce:
  - `michelson(perfil) -> float`: (max − min)/(max + min) del perfil suavizado sobre las tres barras;
  - `snr(A, caja_fondo) -> float`: media/desviación;
  - `comparar(conjunto, k, captura) -> dict`, con el z del foco de cada motor (barrido ±3 % alrededor del exacto, 41 planos, Tamura con el signo del conjunto), la Tamura en el foco y los ms por plano;
  - `resultados/hologramas_carlos/carlos_vs_exacto.csv`, en las 17 posiciones × los 2 conjuntos;
  - `metricas_carlos.csv`, en z1, z9 y z17 de amplitud: contraste por elemento y lp/mm en la muestra, el elemento más fino con contraste ≥ 0.2 y el SNR;
  - los `.npy` de t en el foco de los dos motores, y del exacto en 2048×2560, para las figuras.

- [ ] **Paso 1: escribir las pruebas:**
  - `test_michelson_barras`: barras de 0.2 sobre 1 dan approx 0.667, con abs = 0.01.
  - `test_snr_plano`: un fondo de media 1 y desviación 0.1 da approx 10, con rel = 0.05.
- [ ] **Paso 2: correrlas y ver que fallan.**
- [ ] **Paso 3: implementar `comparar.py`.** Definir las cajas mirando las reconstrucciones del recorte; las del USAF ya identificadas sirven de guía.
- [ ] **Paso 4: correrlas y ver que pasan.** Correr el subcomando completo y comprobar que los focos del exacto en el recorte de 1024 caen a ≤ 1 % de `focos_exactos.csv`.
- [ ] **Paso 5: verificar `git diff --stat`.**

### Tarea 5: rellenar el hueco de transición con `comparacion_cruzada.py`

**Archivos:** `scripts/comparacion_cruzada.py`, solo para correr y restaurando después.

- [ ] **Paso 1:** cambiar `ZS = (192.6, 385.2, 963.0, 1926.0)` por `ZS = (20.0, 30.0, 50.0, 75.0)`. En `SALIDA`, agregar el sufijo `+ "_transicion"` al nombre de la carpeta.
- [ ] **Paso 2:** correr `Tesis_env/Scripts/python.exe -m scripts.comparacion_cruzada`. Esperado: `resultados/comparacion_cruzada/entrada_r4_transicion/resumen.csv` con 4 × 5 filas.
- [ ] **Paso 3:** restaurar las dos líneas al texto original y comprobar que `git diff scripts/comparacion_cruzada.py` sale vacío.

### Tarea 6: figuras en `scripts/figuras_manuscrito.py`

**Archivos:** modificar `scripts/figuras_manuscrito.py`, solo agregando funciones al final y su llamada en el bloque principal.

**Interfaces:** consume las salidas de las Tareas 2 a 5. Produce estos PDF en `manuscrito/figuras/`:
- `ida_vuelta_cuatro.pdf`: SAM contra el cociente z/z_lim (o D/(Nδ²/λ) en el caso esférico), con las cuatro combinaciones. Tres paneles o tres tramos con fondo distinto: el banco esférico; de 5 a 15 mm; transición y Fresnel. La línea vertical en 1.
- `exp_foco.pdf`: las curvas, a partir de `fig1` de `resultados/hologramas_carlos/codigo/figuras.py`.
- `exp_geometria.pdf`: D contra k con el ajuste, y L implícito frente a su media (`fig2`).
- `exp_z_frente_a_L.pdf` (`fig4`).
- `exp_reconstrucciones.pdf`: amplitud en z1, z5, z9 y z17, y fase en z1, z9 y z17.
- `exp_carlos.pdf`: el mismo recorte con el exacto y con Carlos en z1, z9 y z17, más el contraste contra lp/mm.
- `esquema_montaje.pdf`: fuente, muestra, sensor, z, L − z, L, el cono y M.

Se mantiene la paleta del archivo (`COLOR`, `TINTA`, `SUAVE`) y su `estilo()`.

- [ ] **Paso 1:** implementar las siete funciones `figura_*`. Cada una lee sus datos con rutas relativas a `RESULTADOS`.
- [ ] **Paso 2:** correr `Tesis_env/Scripts/python.exe -m scripts.figuras_manuscrito` y comprobar que los siete PDF existen. Abrir cada uno, convertido a PNG, y revisar a ojo los textos superpuestos, los ejes y que no haya escalas en "pixel de 1.83 um" en las figuras experimentales finales: ahí D se rotula como "malla del sensor".
- [ ] **Paso 3:** comprobar con `git diff scripts/figuras_manuscrito.py` que las líneas sin commitear previas del usuario siguen intactas.

### Tarea 7: redacción del manuscrito

**Archivos:**
- `manuscrito/capitulos/01_introduccion.tex`:
  - el `\pendiente` de las líneas ~161-163 solo pierde su parte de hologramas; el de repositorio y licencia sigue;
  - la lista de entregables suma las reconstrucciones experimentales y la calibración de la geometría.
- `manuscrito/capitulos/02_marco_teorico.tex`: la Figura `esquema_montaje.pdf` resuelve el `\pendiente` de la línea ~343.
- `manuscrito/capitulos/03_metodos.tex`:
  - **apertura:** se quita el `\pendiente` de las líneas ~39-41, y el segundo nivel de validación incluye los hologramas experimentales;
  - **3.2.4** (`sec:metodo-fresnel`) pasa a llamarse "Ida y vuelta: FFT-ASM frente a MPASM". El ensayo se cuenta una vez: las cuatro combinaciones, por qué solo las cruzadas informan, las tres escalas (banco esférico con la ida espejo, de 5 a 15 mm, transición y Fresnel), la prueba de s y la ventana ampliada. Se conservan las ecuaciones existentes (`eq:paso-ampliado`) y la etiqueta, para no romper referencias;
  - **3.3** (`sec:montaje`): los datos de Carlos y una tabla de parámetros con su origen;
  - **3.4** (`sec:metricas`): normalización, foco (D grueso y fino, la Tamura con su signo), por qué hace falta la escala (la cresta de z contra L), la calibración con el USAF y el ajuste conjunto, y las métricas de la comparación.
- `manuscrito/capitulos/04_resultados.tex`:
  - **4.2.3** (`sec:res-fresnel`): se reescribe como el resultado único del ensayo de ida y vuelta, con la Tabla de las cuatro combinaciones por distancia y la Figura `ida_vuelta_cuatro.pdf`. Se conservan `fig:fresnel-rec`, `fig:fresnel-met` y `tab:fresnel` donde sigan aportando;
  - **4.3** (`sec:res-obj3`): pasa a llamarse "Reconstrucción de hologramas experimentales (objetivo 3)" y cuenta foco, geometría, reconstrucciones, comparación con Carlos (parecidos y diferencias) y dónde cae cada posición frente al criterio;
  - **discusión:** se quitan los números repetidos de la 4.2.3 y se cita. Se resuelven los `\pendiente` de las líneas ~442 (solo la parte del criterio con los experimentales; los mapas del banco siguen pendientes) y ~478.

- [ ] **Paso 1:** escribir cada sección con los números de los CSV de las Tareas 2 a 5, y nada que no salga de ellos. Un comentario `% Repo:` por sección, con el script y el archivo de datos.
- [ ] **Paso 2:** compilar con `latexmk -pdf -outdir=build main.tex` en `manuscrito/`. Esperado: sin errores, y `grep -c "undefined" build/main.log` = 0 referencias indefinidas nuevas.
- [ ] **Paso 3:** comprobar con `grep -n "pendiente" manuscrito/capitulos/*.tex` que solo quedan los fuera de alcance de la spec: resumen, conclusiones, 4.2.2, CZT, pyLHM y repositorio/licencia.
- [ ] **Paso 4:** pasar `grep -n "mm" manuscrito/capitulos/04_resultados.tex` sobre los párrafos nuevos y revisar que ningún D (148–783 mm) aparezca como distancia física.

### Tarea 8: revisión final

- [ ] **Paso 1:** suite completa: `Tesis_env/Scripts/python.exe -m pytest -q`. Esperado: lo de antes más las pruebas nuevas, todo en PASS (las de GPU se saltan solas si no hay).
- [ ] **Paso 2:** revisión del conjunto contra la spec, con la skill superpowers:requesting-code-review o leyéndolo uno mismo.
- [ ] **Paso 3:** al usuario, un resumen de qué cambió en cada capítulo y de la tabla de números, para que reescriba en su voz.
