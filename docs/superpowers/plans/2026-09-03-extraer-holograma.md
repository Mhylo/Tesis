# Extraer el holograma de los tres retro_* — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que los tres `scripts/retro_*.py` autocontenidos escriban a disco el holograma que ya calculan y hoy tiran, en tres archivos: intensidad (PNG), campo complejo (`.npy`) y parámetros (`.txt`).

**Architecture:** Cada script gana una función `guardar_holograma()` y una llamada en `main()`. Nada más se toca: la ida, la vuelta, la figura y el barrido quedan idénticos. La función está **duplicada a propósito** en los tres, porque estos scripts no importan `CamposT` y ésa es su razón de existir.

**Tech Stack:** Python 3, NumPy, PIL, CuPy opcional. Sin dependencias nuevas: los tres ya importan `pathlib`, `numpy` y `PIL.Image`.

**Spec:** `docs/superpowers/specs/2026-09-03-extraer-holograma-design.md`

## Global Constraints

- **Unidades: milímetros para todo.** `633 nm -> 633e-6`, `3.45 um -> 3.45e-3`.
- **El PNG se escribe SIN GAMMA (lineal).** Un PNG de intensidad guardado como `I^0.6` se lee luego como `I^0.3` al tomarle la raíz, y eso **mueve la distancia a la que enfoca la reconstrucción** — justo lo que un barrido intenta medir. Ver el docstring de `scripts/retro_holograma.py`.
- **NO se importa `CamposT` en ninguno de los tres scripts.** Es la razón de ser de estos archivos: son el contraste independiente del paquete, y si dependieran de él, coincidir con él dejaría de significar algo. Por eso `guardar_holograma()` se duplica en los tres en vez de importarse de `CamposT.pipeline`, y por eso el formato de nombre `z{z:08.3f}` se escribe a mano en vez de importar `nombre_png()`. **Es duplicación deliberada, mandada por la decisión D2 del spec**, y va documentada como tal en el docstring de la función.
- **El `.npy` se baja a CPU con `a_cpu()` antes de escribirlo**: `campo_sensor` vive en la GPU cuando hay CuPy. El dtype se conserva tal cual (`complex64` en GPU, `complex128` en CPU); no se promociona al guardar, porque el archivo debe decir qué se calculó de verdad.
- **La ida, la vuelta, la figura y el barrido no se tocan.** Ni una línea.
- **Las salidas van bajo la raíz del repo**, no relativas al directorio de invocación: `pathlib.Path(__file__).resolve().parent.parent / "resultados" / "hologramas"`.
- **No se usa la constante `SALIDA`**: ésa controla las figuras y vale `None` por defecto en los tres, con lo que no se guardaría ningún holograma.
- **Sin pruebas automáticas nuevas.** Estos tres son scripts de exploración y el repo ya fijó esa política en `docs/superpowers/specs/2026-08-27-retro-intensidad-design.md`: los tests que los tocan comprueban funciones compartidas, no los scripts. La verificación es la corrida manual de cada tarea, más la Task 4.
- **Comentarios y docstrings en español**, diciendo *por qué*, no *qué*. **Los tres `retro_*.py` están escritos en ASCII sin tildes**: respeta ese estilo al editarlos.
- **Mensajes de commit en ASCII sin tildes.**
- Correr con el intérprete del entorno: `./Tesis_env/Scripts/python.exe`.

---

### Task 1: `retro_fft_angular.py` escribe su holograma

**Files:**
- Modify: `scripts/retro_fft_angular.py`

**Interfaces:**
- Consumes: nada de tareas anteriores.
- Produces: el patrón que las Tasks 2 y 3 repiten. `guardar_holograma(campo, z, parametros)` — escribe tres archivos y no devuelve nada, como `pipeline.guardar()` — y una constante de módulo `METODO = "fft"`.

- [ ] **Step 1: Añadir la constante `METODO`**

En `scripts/retro_fft_angular.py`, junto a las demás constantes del bloque `PARAMETROS`, después de `SALIDA` y antes de la sección siguiente:

```python
#: Nombre corto del propagador, para la ruta de salida del holograma. Es el
#: mismo vocabulario que usa METODOS en CamposT: fft, blas, mpasm.
METODO = "fft"
```

- [ ] **Step 2: Añadir la función `guardar_holograma`**

En `scripts/retro_fft_angular.py`, justo antes de la sección `MAIN` (la cabecera `# ════...` que precede a `def main():`), añade:

```python
def guardar_holograma(campo, z, parametros):
    """campo_sensor -> tres archivos, y una linea por archivo en consola.

        .png   |campo|^2 normalizado por su maximo, SIN GAMMA. Es lo que mide
               un sensor: al retropropagarlo sale el objeto CON su imagen
               gemela encima.
        .npy   el campo complejo tal cual, en el dtype en que se calculo. Es lo
               que habia ANTES de medirlo: al retropropagarlo la vuelta deshace
               la ida y devuelve el objeto exacto.
        .txt   con que se hizo.

    Los dos primeros no son redundantes: el PNG es lo que un sensor te habria
    dado, el .npy es lo que habia antes de que lo midiera.
    scripts/retro_holograma.py distingue los dos por la extension.

    SIN GAMMA A PROPOSITO. Un PNG de intensidad guardado como I^0.6 -lo que
    hace CamposT/pipeline.py por defecto- se lee luego como I^0.3 al tomarle la
    raiz, y eso no es solo contraste feo: MUEVE LA DISTANCIA a la que enfoca la
    reconstruccion, que es justo lo que un barrido intenta medir.

    EL .txt NO ES ADORNO. El nombre del archivo lleva la z y nada mas, asi que
    dos corridas con la misma z y distinta lambda escriben el mismo nombre y la
    segunda pisa a la primera. El .txt es lo que impide que eso sea un error
    mudo: dice con que parametros se hizo el archivo que hay ahi.

    ESTA FUNCION ESTA DUPLICADA en los tres scripts retro_* A PROPOSITO. Podria
    importarse de CamposT.pipeline.guardar(), que hace casi esto mismo, pero
    estos tres no importan el paquete: son el contraste INDEPENDIENTE contra el.
    Si dependieran de el, coincidir con el dejaria de significar algo. Es la
    misma duplicacion deliberada que ya tienen nitidez(), pico_de_foco(),
    a_cpu(), objeto() y los tres pinta_*().
    """
    A = a_cpu(campo)

    # Bajo la raiz del repo, se lance el script desde donde se lance: una ruta
    # relativa lo dejaria en el directorio de invocacion.
    destino = (pathlib.Path(__file__).resolve().parent.parent / "resultados"
               / "hologramas" / pathlib.Path(RUTA).stem / METODO)
    destino.mkdir(parents=True, exist_ok=True)

    # z{:08.3f}: el mismo formato que nombre_png() en
    # CamposT/retropropagacion.py, escrito a mano porque aqui no se importa el
    # paquete. Tres decimales para que dos z distintas no escriban el mismo
    # archivo, y ancho fijo para que el orden alfabetico siga al de la z.
    #
    # OJO: las extensiones se PEGAN con f-string, NO con Path.with_suffix().
    # Para pathlib "z0010.000" ya tiene sufijo -".000"- y with_suffix(".png")
    # lo SUSTITUIRIA en vez de anadirlo, dejando z0010.png. Eso se cargaria los
    # tres decimales enteros: z = 10.0 y z = 10.5 escribirian en el MISMO
    # archivo, en silencio, que es exactamente lo que este formato existe para
    # impedir.
    base = destino / f"z{z:08.3f}"

    I = np.abs(A) ** 2
    m = I.max()
    # Un campo identicamente nulo daria 0/0: NaN por todo el array y un PNG de
    # basura, sin error y sin aviso. Un negro es un resultado legitimo.
    I = I / m if m > 0 else np.zeros_like(I)
    Image.fromarray((I * 255).astype(np.uint8)).save(f"{base}.png")

    np.save(f"{base}.npy", A)

    with open(f"{base}.txt", "w", encoding="utf-8") as f:
        for clave, valor in parametros.items():
            f.write(f"{clave} = {valor}\n")

    print("holograma guardado:")
    for ext in (".png", ".npy", ".txt"):
        print(f"  -> {base}{ext}")
```

- [ ] **Step 3: Llamarla desde `main()`**

En `scripts/retro_fft_angular.py`, dentro de `main()`, localiza el comentario que abre el barrido:

```python
    # ---- barrido de foco ----------------------------------------------------
```

Inserta **justo encima** de esa linea (es el mismo punto en los tres scripts: despues de todos los `print` de parametros y diagnosticos, antes de que empiece el barrido):

```python
    # Se guarda aqui y no justo tras la ida para que la consola se lea en
    # orden: primero con que se hizo, despues que se escribio. campo_sensor ya
    # existe y nadie lo toca en medio.
    guardar_holograma(campo_sensor, Z, {
        "objeto": RUTA,
        "propagador": "FFT-ASM (espectro angular por FFT)",
        "lambda [mm]": LAMB,
        "delta [mm]": DELTA,
        "Z [mm]": Z,
        "malla": f"{M}x{N}",
        "ENTRADA": ENTRADA,
        "PHI_MAX": PHI_MAX,
        "dispositivo": dev,
        "dtype": np.dtype(dtype).name,
        "EJES_CRUZADOS": EJES_CRUZADOS,
    })
```

- [ ] **Step 4: Correr el script y comprobar los tres archivos**

```bash
MPLBACKEND=Agg ./Tesis_env/Scripts/python.exe scripts/retro_fft_angular.py
```


Expected: en la consola, un bloque `holograma guardado:` con tres rutas bajo `resultados/hologramas/BenchmarkTarget/fft/`. El nombre debe ser `z0010.000` (el `Z` del script es `10.0`).

Comprueba los tres:

```bash
./Tesis_env/Scripts/python.exe -c "
import numpy as np, pathlib
from PIL import Image
b = pathlib.Path('resultados/hologramas/BenchmarkTarget/fft/z0010.000')
im = Image.open(f'{b}.png'); print('png :', im.size, im.mode)
A = np.load(f'{b}.npy'); print('npy :', A.shape, A.dtype, 'complejo:', np.iscomplexobj(A))
print('txt :'); print(pathlib.Path(f'{b}.txt').read_text(encoding='utf-8'))"
```

Expected: el PNG en modo `L`, el `.npy` complejo y con la misma forma, y el `.txt` con las once claves, entre ellas `EJES_CRUZADOS`.

- [ ] **Step 5: Comprobar que la figura y el barrido no cambiaron**

```bash
git diff --stat scripts/retro_fft_angular.py
```

Expected: sólo inserciones (más las líneas de contexto), **cero borrados** salvo los estrictamente necesarios para insertar. Ninguna línea de `espectro_angular(...)`, de `pinta_*`, del barrido ni de la figura debe aparecer como modificada.

- [ ] **Step 6: Commit**

```bash
git add scripts/retro_fft_angular.py
git commit -m "retro_fft_angular: escribir el holograma en vez de tirarlo

campo_sensor es el campo en el plano del sensor a +Z, o sea el holograma.
El script lo calculaba, lo pintaba en la figura y lo perdia al terminar.
Ahora escribe tres archivos: el PNG de intensidad -lo que mide un sensor,
con gemela-, el .npy del campo complejo -la vuelta exacta- y un .txt con
los parametros.

El PNG va SIN GAMMA. Guardado como I^0.6 se leeria como I^0.3 al tomarle
la raiz, y eso mueve la distancia a la que enfoca la reconstruccion.

guardar_holograma() se duplica en los tres retro_* a proposito: estos
scripts no importan CamposT porque son el contraste independiente contra
el, y si dependieran de el, coincidir dejaria de significar algo.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `retro_blas.py` escribe su holograma

**Files:**
- Modify: `scripts/retro_blas.py`

**Interfaces:**
- Consumes: el patrón de la Task 1, repetido aquí íntegro (la duplicación es la decisión D2, no un descuido).
- Produces: `guardar_holograma(campo, z, parametros)` y `METODO = "blas"` en este archivo.

- [ ] **Step 1: Añadir la constante `METODO`**

En `scripts/retro_blas.py`, junto a las demás constantes del bloque `PARAMETROS`, después de `SALIDA`:

```python
#: Nombre corto del propagador, para la ruta de salida del holograma. Es el
#: mismo vocabulario que usa METODOS en CamposT: fft, blas, mpasm.
METODO = "blas"
```

- [ ] **Step 2: Añadir la función `guardar_holograma`**

En `scripts/retro_blas.py`, justo antes de la sección `MAIN`:

```python
def guardar_holograma(campo, z, parametros):
    """campo_sensor -> tres archivos, y una linea por archivo en consola.

        .png   |campo|^2 normalizado por su maximo, SIN GAMMA. Es lo que mide
               un sensor: al retropropagarlo sale el objeto CON su imagen
               gemela encima.
        .npy   el campo complejo tal cual, en el dtype en que se calculo. Es lo
               que habia ANTES de medirlo: al retropropagarlo la vuelta deshace
               la ida y devuelve el objeto exacto.
        .txt   con que se hizo.

    Los dos primeros no son redundantes: el PNG es lo que un sensor te habria
    dado, el .npy es lo que habia antes de que lo midiera.
    scripts/retro_holograma.py distingue los dos por la extension.

    SIN GAMMA A PROPOSITO. Un PNG de intensidad guardado como I^0.6 -lo que
    hace CamposT/pipeline.py por defecto- se lee luego como I^0.3 al tomarle la
    raiz, y eso no es solo contraste feo: MUEVE LA DISTANCIA a la que enfoca la
    reconstruccion, que es justo lo que un barrido intenta medir.

    EL .txt NO ES ADORNO. El nombre del archivo lleva la z y nada mas, asi que
    dos corridas con la misma z y distinta lambda escriben el mismo nombre y la
    segunda pisa a la primera. El .txt es lo que impide que eso sea un error
    mudo: dice con que parametros se hizo el archivo que hay ahi.

    ESTA FUNCION ESTA DUPLICADA en los tres scripts retro_* A PROPOSITO. Podria
    importarse de CamposT.pipeline.guardar(), que hace casi esto mismo, pero
    estos tres no importan el paquete: son el contraste INDEPENDIENTE contra el.
    Si dependieran de el, coincidir con el dejaria de significar algo. Es la
    misma duplicacion deliberada que ya tienen nitidez(), pico_de_foco(),
    a_cpu(), objeto() y los tres pinta_*().
    """
    A = a_cpu(campo)

    # Bajo la raiz del repo, se lance el script desde donde se lance: una ruta
    # relativa lo dejaria en el directorio de invocacion.
    destino = (pathlib.Path(__file__).resolve().parent.parent / "resultados"
               / "hologramas" / pathlib.Path(RUTA).stem / METODO)
    destino.mkdir(parents=True, exist_ok=True)

    # z{:08.3f}: el mismo formato que nombre_png() en
    # CamposT/retropropagacion.py, escrito a mano porque aqui no se importa el
    # paquete. Tres decimales para que dos z distintas no escriban el mismo
    # archivo, y ancho fijo para que el orden alfabetico siga al de la z.
    #
    # OJO: las extensiones se PEGAN con f-string, NO con Path.with_suffix().
    # Para pathlib "z0010.000" ya tiene sufijo -".000"- y with_suffix(".png")
    # lo SUSTITUIRIA en vez de anadirlo, dejando z0010.png. Eso se cargaria los
    # tres decimales enteros: z = 10.0 y z = 10.5 escribirian en el MISMO
    # archivo, en silencio, que es exactamente lo que este formato existe para
    # impedir.
    base = destino / f"z{z:08.3f}"

    I = np.abs(A) ** 2
    m = I.max()
    # Un campo identicamente nulo daria 0/0: NaN por todo el array y un PNG de
    # basura, sin error y sin aviso. Un negro es un resultado legitimo.
    I = I / m if m > 0 else np.zeros_like(I)
    Image.fromarray((I * 255).astype(np.uint8)).save(f"{base}.png")

    np.save(f"{base}.npy", A)

    with open(f"{base}.txt", "w", encoding="utf-8") as f:
        for clave, valor in parametros.items():
            f.write(f"{clave} = {valor}\n")

    print("holograma guardado:")
    for ext in (".png", ".npy", ".txt"):
        print(f"  -> {base}{ext}")
```

- [ ] **Step 3: Llamarla desde `main()`**

En `scripts/retro_blas.py`, dentro de `main()`, localiza el comentario que abre el barrido:

```python
    # ---- barrido de foco ----------------------------------------------------
```

Inserta **justo encima** de esa linea (es el mismo punto en los tres scripts: despues de todos los `print` de parametros y diagnosticos, antes de que empiece el barrido):

```python
    # Se guarda aqui y no justo tras la ida para que la consola se lea en
    # orden: primero con que se hizo, despues que se escribio. campo_sensor ya
    # existe y nadie lo toca en medio.
    #
    # frac_ida no es un ajuste, es un RESULTADO de la ida: la fraccion del
    # espectro que la mascara de banda limitada conservo. Va al .txt porque sin
    # el, dos hologramas con los mismos parametros de entrada pueden diferir y
    # no habria forma de saber por que.
    guardar_holograma(campo_sensor, Z, {
        "objeto": RUTA,
        "propagador": "BL-ASM (espectro angular de banda limitada)",
        "lambda [mm]": LAMB,
        "delta [mm]": DELTA,
        "Z [mm]": Z,
        "malla": f"{M}x{N}",
        "ENTRADA": ENTRADA,
        "PHI_MAX": PHI_MAX,
        "dispositivo": dev,
        "dtype": np.dtype(dtype).name,
        "frac_ida": f"{frac_ida:.6f}",
    })
```

- [ ] **Step 4: Correr el script y comprobar los tres archivos**

```bash
MPLBACKEND=Agg ./Tesis_env/Scripts/python.exe scripts/retro_blas.py
```


Expected: bloque `holograma guardado:` con tres rutas bajo `resultados/hologramas/BenchmarkTarget/blas/`, con nombre `z0200.000` (el `Z` del script es `200.0`).

```bash
./Tesis_env/Scripts/python.exe -c "
import numpy as np, pathlib
from PIL import Image
b = pathlib.Path('resultados/hologramas/BenchmarkTarget/blas/z0200.000')
im = Image.open(f'{b}.png'); print('png :', im.size, im.mode)
A = np.load(f'{b}.npy'); print('npy :', A.shape, A.dtype, 'complejo:', np.iscomplexobj(A))
print('txt :'); print(pathlib.Path(f'{b}.txt').read_text(encoding='utf-8'))"
```

Expected: el `.txt` incluye `frac_ida`, y su valor coincide con la fracción de banda que el script imprimió por consola.

- [ ] **Step 5: Comprobar que nada más cambió**

```bash
git diff --stat scripts/retro_blas.py
```

Expected: sólo inserciones. Ninguna línea de `espectro_angular_bl(...)`, del barrido ni de la figura modificada.

- [ ] **Step 6: Commit**

```bash
git add scripts/retro_blas.py
git commit -m "retro_blas: escribir el holograma en vez de tirarlo

Mismo cambio que en retro_fft_angular: campo_sensor pasa a disco como PNG
de intensidad, .npy del campo complejo y .txt de parametros.

El .txt de este lleva ademas frac_ida, que no es un ajuste sino un
RESULTADO de la ida: la fraccion del espectro que la mascara de banda
limitada conservo. Sin el, dos hologramas con los mismos parametros de
entrada pueden diferir y no hay forma de saber por que.

guardar_holograma() es una copia de la de retro_fft_angular, a proposito:
estos scripts no importan CamposT porque son el contraste independiente
contra el.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `retro_mpasm.py` escribe su holograma

**Files:**
- Modify: `scripts/retro_mpasm.py`

**Interfaces:**
- Consumes: el patrón de las Tasks 1 y 2, repetido aquí íntegro (decisión D2).
- Produces: `guardar_holograma(campo, z, parametros)` y `METODO = "mpasm"` en este archivo.

- [ ] **Step 1: Añadir la constante `METODO`**

En `scripts/retro_mpasm.py`, junto a las demás constantes del bloque `PARAMETROS`, después de `SALIDA`:

```python
#: Nombre corto del propagador, para la ruta de salida del holograma. Es el
#: mismo vocabulario que usa METODOS en CamposT: fft, blas, mpasm.
METODO = "mpasm"
```

- [ ] **Step 2: Añadir la función `guardar_holograma`**

En `scripts/retro_mpasm.py`, justo antes de la sección `MAIN`:

```python
def guardar_holograma(campo, z, parametros):
    """campo_sensor -> tres archivos, y una linea por archivo en consola.

        .png   |campo|^2 normalizado por su maximo, SIN GAMMA. Es lo que mide
               un sensor: al retropropagarlo sale el objeto CON su imagen
               gemela encima.
        .npy   el campo complejo tal cual, en el dtype en que se calculo. Es lo
               que habia ANTES de medirlo: al retropropagarlo la vuelta deshace
               la ida y devuelve el objeto exacto.
        .txt   con que se hizo.

    Los dos primeros no son redundantes: el PNG es lo que un sensor te habria
    dado, el .npy es lo que habia antes de que lo midiera.
    scripts/retro_holograma.py distingue los dos por la extension.

    SIN GAMMA A PROPOSITO. Un PNG de intensidad guardado como I^0.6 -lo que
    hace CamposT/pipeline.py por defecto- se lee luego como I^0.3 al tomarle la
    raiz, y eso no es solo contraste feo: MUEVE LA DISTANCIA a la que enfoca la
    reconstruccion, que es justo lo que un barrido intenta medir.

    EL .txt NO ES ADORNO. El nombre del archivo lleva la z y nada mas, asi que
    dos corridas con la misma z y distinta lambda escriben el mismo nombre y la
    segunda pisa a la primera. El .txt es lo que impide que eso sea un error
    mudo: dice con que parametros se hizo el archivo que hay ahi.

    ESTA FUNCION ESTA DUPLICADA en los tres scripts retro_* A PROPOSITO. Podria
    importarse de CamposT.pipeline.guardar(), que hace casi esto mismo, pero
    estos tres no importan el paquete: son el contraste INDEPENDIENTE contra el.
    Si dependieran de el, coincidir con el dejaria de significar algo. Es la
    misma duplicacion deliberada que ya tienen nitidez(), pico_de_foco(),
    a_cpu(), objeto() y los tres pinta_*().
    """
    A = a_cpu(campo)

    # Bajo la raiz del repo, se lance el script desde donde se lance: una ruta
    # relativa lo dejaria en el directorio de invocacion.
    destino = (pathlib.Path(__file__).resolve().parent.parent / "resultados"
               / "hologramas" / pathlib.Path(RUTA).stem / METODO)
    destino.mkdir(parents=True, exist_ok=True)

    # z{:08.3f}: el mismo formato que nombre_png() en
    # CamposT/retropropagacion.py, escrito a mano porque aqui no se importa el
    # paquete. Tres decimales para que dos z distintas no escriban el mismo
    # archivo, y ancho fijo para que el orden alfabetico siga al de la z.
    #
    # OJO: las extensiones se PEGAN con f-string, NO con Path.with_suffix().
    # Para pathlib "z0010.000" ya tiene sufijo -".000"- y with_suffix(".png")
    # lo SUSTITUIRIA en vez de anadirlo, dejando z0010.png. Eso se cargaria los
    # tres decimales enteros: z = 10.0 y z = 10.5 escribirian en el MISMO
    # archivo, en silencio, que es exactamente lo que este formato existe para
    # impedir.
    base = destino / f"z{z:08.3f}"

    I = np.abs(A) ** 2
    m = I.max()
    # Un campo identicamente nulo daria 0/0: NaN por todo el array y un PNG de
    # basura, sin error y sin aviso. Un negro es un resultado legitimo.
    I = I / m if m > 0 else np.zeros_like(I)
    Image.fromarray((I * 255).astype(np.uint8)).save(f"{base}.png")

    np.save(f"{base}.npy", A)

    with open(f"{base}.txt", "w", encoding="utf-8") as f:
        for clave, valor in parametros.items():
            f.write(f"{clave} = {valor}\n")

    print("holograma guardado:")
    for ext in (".png", ".npy", ".txt"):
        print(f"  -> {base}{ext}")
```

- [ ] **Step 3: Llamarla desde `main()`**

En `scripts/retro_mpasm.py`, dentro de `main()`, localiza el comentario que abre el barrido:

```python
    # ---- barrido de foco ----------------------------------------------------
```

Inserta **justo encima** de esa linea (es el mismo punto en los tres scripts: despues de todos los `print` de parametros y diagnosticos, antes de que empiece el barrido):

```python
    # Se guarda aqui y no justo tras la ida para que la consola se lea en
    # orden: primero con que se hizo, despues que se escribio. Ademas deja el
    # Kf impreso justo encima del .txt que lo anota.
    #
    # kf_ida no es un ajuste, es un RESULTADO de la ida: el coeficiente de
    # compresion frecuencial que MPASM calculo. KF es lo que se le PIDIO -None
    # significa "calculalo tu"-. Van los dos porque no son lo mismo, y sin
    # kf_ida dos hologramas con los mismos parametros de entrada pueden diferir
    # sin que se pueda saber por que.
    guardar_holograma(campo_sensor, Z, {
        "objeto": RUTA,
        "propagador": "MPASM (espectro angular por producto matricial)",
        "lambda [mm]": LAMB,
        "delta [mm]": DELTA,
        "Z [mm]": Z,
        "malla": f"{M}x{N}",
        "ENTRADA": ENTRADA,
        "PHI_MAX": PHI_MAX,
        "dispositivo": dev,
        "dtype": np.dtype(dtype).name,
        "S": S,
        "R": R,
        "MAG": MAG,
        "KF": KF,
        "kf_ida": f"{kf_ida:.6f}",
    })
```

- [ ] **Step 4: Correr el script y comprobar los tres archivos**

```bash
MPLBACKEND=Agg ./Tesis_env/Scripts/python.exe scripts/retro_mpasm.py
```


Expected: bloque `holograma guardado:` con tres rutas bajo `resultados/hologramas/entrada/mpasm/` — **`entrada`, no `BenchmarkTarget`**: el `RUTA` de este script apunta a `resultados/campos/entrada.png`. El nombre es `z0050.000` (su `Z` es `50.0`).

```bash
./Tesis_env/Scripts/python.exe -c "
import numpy as np, pathlib
from PIL import Image
b = pathlib.Path('resultados/hologramas/entrada/mpasm/z0050.000')
im = Image.open(f'{b}.png'); print('png :', im.size, im.mode)
A = np.load(f'{b}.npy'); print('npy :', A.shape, A.dtype, 'complejo:', np.iscomplexobj(A))
print('txt :'); print(pathlib.Path(f'{b}.txt').read_text(encoding='utf-8'))"
```

Expected: el `.txt` incluye `S`, `R`, `MAG`, `KF` y `kf_ida`, y `kf_ida` coincide con el valor que el script imprimió en la línea `Kf: ... en la ida`.

- [ ] **Step 5: Comprobar que nada más cambió**

```bash
git diff --stat scripts/retro_mpasm.py
```

Expected: sólo inserciones. Ninguna línea de `mpasm_bloques(...)`, del barrido ni de la figura modificada.

- [ ] **Step 6: Commit**

```bash
git add scripts/retro_mpasm.py
git commit -m "retro_mpasm: escribir el holograma en vez de tirarlo

Mismo cambio que en los otros dos: campo_sensor pasa a disco como PNG de
intensidad, .npy del campo complejo y .txt de parametros.

El .txt de este lleva S, R, MAG y KF -lo que se le pide- y ademas kf_ida
-lo que MPASM calculo de verdad-. No son lo mismo: KF = None significa
'calculalo tu', y sin kf_ida dos hologramas con los mismos parametros de
entrada pueden diferir sin que se pueda saber por que.

guardar_holograma() es una copia de la de los otros dos, a proposito:
estos scripts no importan CamposT porque son el contraste independiente
contra el.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Cerrar el círculo y anotar lo medido

**Files:**
- Modify: `docs/superpowers/specs/2026-09-03-extraer-holograma-design.md` (sección `## Estado`)

**Interfaces:**
- Consumes: los hologramas escritos por las Tasks 1-3, en particular `resultados/hologramas/entrada/mpasm/z0050.000.{png,npy,txt}`.
- Produces: nada de código. El deliverable es la verificación corrida y su resultado anotado.

Ésta es la razón de ser de todo lo anterior: el mismo instante físico, medido de las dos maneras, reconstruido con el mismo código. Es lo que convierte en reproducible el `~0.53` de correlación que cuatro docstrings del repo citan de memoria.

- [ ] **Step 1: Reconstruir desde el `.npy` — la vuelta exacta**

Lee el `.txt` para saber con qué parámetros se hizo. Después edita `scripts/retro_holograma.py` y pon **temporalmente**:

```python
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\hologramas\entrada\mpasm\z0050.000.npy"
LAMB = 633e-6
DELTA = 3.45e-3
Z = 50.0
RECORTAR = False
ROI = None
BARRIDO = None
USAR_ANGULAR = True
USAR_BLAS = False
USAR_MPASM = False
SALIDA = "resultados/circulo/npy"
```

`SALIDA` es lo que hace que la figura se escriba a disco: con `MPLBACKEND=Agg` no se abre ninguna ventana, asi que si no la guardas no hay nada que mirar.

(Los valores de `LAMB`, `DELTA` y `Z` deben ser los que diga el `.txt`; los de arriba son los que tiene `retro_mpasm.py` hoy. Si no coinciden, manda el `.txt`.)

```bash
MPLBACKEND=Agg ./Tesis_env/Scripts/python.exe scripts/retro_holograma.py
```

Expected: la consola anuncia `campo complejo (la vuelta es exacta, sin gemela)`.

Abre `resultados/circulo/npy/retropropagacion.png` y mirala. La columna de la reconstruccion tiene que devolver el objeto **limpio**, sin una copia desenfocada superpuesta.

- [ ] **Step 2: Reconstruir desde el `.png` — con gemela**

Cambia sólo la extensión de `RUTA`:

```python
RUTA = r"C:\Users\User\Desktop\Tesis\resultados\hologramas\entrada\mpasm\z0050.000.png"
GAMMA_GUARDADO = 1.0
SALIDA = "resultados/circulo/png"
```

```bash
MPLBACKEND=Agg ./Tesis_env/Scripts/python.exe scripts/retro_holograma.py
```

Expected: la consola anuncia `intensidad medida, campo = sqrt(I) (con gemela)`.

Abre `resultados/circulo/png/retropropagacion.png`. Ahora la reconstruccion trae la **imagen gemela desenfocada encima** del objeto. Ponla al lado de la del Step 1: la diferencia entre las dos es exactamente lo que cuesta que un sensor tire la fase.

- [ ] **Step 3: Medir la correlación de los dos caminos contra el objeto**

Con `resultados/campos/entrada.png` como verdad de terreno:

```bash
./Tesis_env/Scripts/python.exe -c "
import numpy as np, pathlib
from PIL import Image
from CamposT.pipeline import propagar

LAMB, DELTA, Z = 633e-6, 3.45e-3, 50.0
b = pathlib.Path('resultados/hologramas/entrada/mpasm/z0050.000')

obj = np.asarray(Image.open('resultados/campos/entrada.png').convert('L'), float) / 255.0

def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))

# desde el campo complejo
U = np.load(f'{b}.npy')
rec_npy, _ = propagar(U, DELTA, LAMB, -Z, metodo='fft', pad=1, device='cpu')

# desde la intensidad medida, que es lo unico que da un sensor
I = np.asarray(Image.open(f'{b}.png').convert('L'), float) / 255.0
rec_png, _ = propagar(np.sqrt(I).astype(complex), DELTA, LAMB, -Z, metodo='fft', pad=1, device='cpu')

print(f'desde .npy (campo complejo): corr = {corr(np.abs(rec_npy)**2, obj):.4f}')
print(f'desde .png (intensidad)    : corr = {corr(np.abs(rec_png)**2, obj):.4f}')"
```

Expected: la primera correlación claramente más alta que la segunda. **Anota los dos números** — son el deliverable de esta tarea.

- [ ] **Step 4: Deshacer los cambios temporales de `retro_holograma.py`**

```bash
git checkout scripts/retro_holograma.py
```

**Cuidado:** ese archivo tenía cambios sin commitear antes de empezar (`LAMB`, `S_MPASM`, `RUTA`, `Z`). Antes de correr el `checkout`, guarda el diff:

```bash
git diff scripts/retro_holograma.py > /tmp/retro_holograma_local.diff
```

y vuelve a aplicarlo después con `git apply /tmp/retro_holograma_local.diff`. Si el `git diff` sale vacío, no hay nada que restaurar y el `checkout` es seguro.

```bash
git status --short scripts/
```

Expected: `scripts/retro_holograma.py` en el mismo estado que tenía al empezar la tarea.

- [ ] **Step 5: Anotar lo medido en el spec**

En `docs/superpowers/specs/2026-09-03-extraer-holograma-design.md`, sustituye la sección `## Estado` entera por:

```markdown
## Estado

Diseño aprobado el 2026-09-03. Implementado en los tres scripts.

Círculo cerrado y medido sobre `resultados/hologramas/entrada/mpasm/z0050.000`,
reconstruyendo con FFT-ASM contra `resultados/campos/entrada.png`:

    desde el .npy (campo complejo)  corr = <valor del Step 3>
    desde el .png (intensidad)      corr = <valor del Step 3>

La segunda es la que un sensor de verdad te deja: la caída es el precio de que
la medida tire la fase, y es de donde sale la imagen gemela.
```

Sustituye `<valor del Step 3>` por los dos números que salieron. **Si no salieron, no inventes ninguno**: escribe en su lugar qué falló.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-09-03-extraer-holograma-design.md
git commit -m "Cerrar el circulo: medir lo que cuesta que el sensor tire la fase

El mismo holograma reconstruido por los dos caminos que ahora escriben los
tres retro_*: desde el campo complejo y desde la intensidad. La diferencia
entre las dos correlaciones es el precio de la imagen gemela, y hasta ahora
era un numero que cuatro docstrings citaban de memoria sin que nadie
pudiera reproducirlo.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Notas para quien implemente

**Las Tasks 1-3 son casi idénticas y eso es correcto.** `guardar_holograma()` aparece tres veces con el mismo cuerpo. No la factorices a un módulo común ni la importes de `CamposT`: es la decisión D2 del spec, y la razón está en el docstring de la propia función. Si te parece mal, dilo en el informe en vez de arreglarlo por tu cuenta.

**Lo único que cambia entre las tres** es el valor de `METODO`, el nombre del propagador en el `.txt`, y las claves extra: `EJES_CRUZADOS` en fft, `frac_ida` en blas, y `S`/`R`/`MAG`/`KF`/`kf_ida` en mpasm. Las dos últimas son **resultados** de la ida, no ajustes.

**Los tres scripts terminan en `plt.show()` y se quedan esperando a que cierres la ventana.** Por eso todas las corridas de este plan van con `MPLBACKEND=Agg` delante: con ese backend `plt.show()` es un no-op y el proceso termina solo. No hace falta tocar ni una linea de los scripts, y las figuras que se guardan con `SALIDA` se escriben igual.

**Los tres tienen `RUTA` y `Z` distintos** hoy (`BenchmarkTarget`/10, `BenchmarkTarget`/200, `entrada`/50), así que escriben en sitios distintos y no se pisan entre sí. No los homogeneices.

**`resultados/` está en `.gitignore`** salvo `resultados/campos/`, así que ninguno de los hologramas entra al repo. Es lo correcto: son regenerables, y el `.txt` dice cómo.

**Lo que NO hay que hacer:** tocar la ida, la vuelta, la figura o el barrido; importar `CamposT` en los tres scripts; guardar el objeto de partida; guardar las z del barrido; tocar `scripts/retro_holograma.py` fuera de la Task 4 (y allí, deshaciéndolo).
