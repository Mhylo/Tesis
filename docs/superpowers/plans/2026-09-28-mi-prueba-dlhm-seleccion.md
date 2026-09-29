# Selección de foco de barrido_foco en mi_prueba_dlhm — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que `scripts/mi_prueba_dlhm.py` elija el foco de un holograma DLHM con los mismos parámetros de selección que `scripts/barrido_foco.py` (Tamura, Pearson, SSIM; grueso + fino; ROI; tabla, `curvas.npz` y figuras), con dos motores detrás: el exacto de `CamposT.dlhm` (por defecto) y la cadena de Carlos.

**Architecture:** dos motores con la misma interfaz `campo(z, ventana) -> t` en la malla que el sensor proyecta desde la fuente (paso `DELTA*z/L`, un píxel por píxel del sensor), y la selección de `barrido_foco` escrita como funciones con argumentos explícitos (`barrido`, `enfocar`, `region_medida`, `rango_del_barrido`...) que `main()` conecta con las constantes. Las métricas se copian literales de `barrido_foco.py` y una prueba con `ast` fija que no diverjan.

**Tech Stack:** Python 3, NumPy/CuPy, OpenCV (`cv2`), scikit-image (`structural_similarity`), matplotlib, `CamposT.dlhm`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-mi-prueba-dlhm-seleccion-design.md` (D1–D12). Los ajustes que salieron al preparar este plan están en su sección «Ajustes del plan».

## Global Constraints

- `scripts/barrido_foco.py` y `scripts/mpasm_minimo.py` NO se tocan: tienen trabajo sin commitear del usuario (tarea 77).
- No cambian las funciones que usan las pruebas actuales: `reconstruir`, `malla_remuestreada`, `comprobar_equivalencia`, `correlacion`, `_reconstruir_referencia`, `elegir_dispositivo`, `a_cpu`, `liberar`, `sincronizar`, ni las cinco copiadas de `dlhm.py` de Carlos (`ifts`, `fts`, `resize`, `point_src`, `angular_spectrum`). De `comprobar_memoria` solo cambia el texto `ROI_HOLOGRAMA` → `ROI` de sus dos mensajes (la constante desaparece).
- Unidades: milímetros para todo. Montaje por defecto: `LAMB, DELTA, L = MONTAJE.lamb, MONTAJE.delta_x, MONTAJE.L` (528 nm, 1.83 µm, 10 mm).
- Consola en ASCII (la de Windows es cp1252); comentarios y docstrings del script, en el estilo del archivo (sin tildes).
- Commits: SOLO si el usuario lo pide. Rama `main`; si pide commit, primero rama. Nada de push.
- Python del entorno: `Tesis_env/Scripts/python.exe`. Pruebas: `Tesis_env/Scripts/python.exe -m pytest ...` desde la raíz.

## Mapa de archivos

- Modificar `scripts/mi_prueba_dlhm.py`: cabecera, bloque de parámetros, secciones nuevas 4 (motores), 5 (métricas), 6 (barrido), 7 (configuración y región), 8 (objeto, holograma y salida), `main()` nuevo; se borran `referencia_a_escala`, `observable`, `_roi_de`, `COMPARAR`, `ROI_HOLOGRAMA`, `ROI_REFERENCIA`, `Z`.
- Crear `tests/test_mi_prueba_dlhm_seleccion.py`.
- Modificar `docs/superpowers/specs/2026-09-28-mi-prueba-dlhm-seleccion-design.md`: sección «Ajustes del plan» y «Estado».
- Scratch (no va al repo): `medir_hipotesis_d3.py` y `verificar_escala_real.py` en el scratchpad de la sesión.

---

### Task 1: Medir la hipótesis de D3 con la versión actual

Antes de borrar `referencia_a_escala()` (spec, Interfaz). No toca el repo.

**Files:**
- Create (scratch): `<scratchpad>/medir_hipotesis_d3.py`

- [ ] **Step 1: Escribir el script de medida**

Simula el BenchmarkTarget entero (3000x4000) a z = 3 mm con la ida exacta, lo guarda para reusarlo en la Task 7, toma el recorte central de 1024 y mide la correlación en `Z_REAL` con cuatro geometrías: la vieja (`referencia_a_escala`, sin dividir), la vieja dividiendo por |U0|², la huella sin dividir y la huella dividiendo (el motor nuevo). Añade el mejor ajuste de escala de la vieja, como el barrido de M de 0.8 a 6 de la cabecera, y la exacta sobre la misma ventana.

```python
"""D3: de donde salia el ~0.03 de mi_prueba_dlhm. Con la version ACTUAL."""
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import cv2 as cv
import numpy as np
import cupy as cp

RAIZ = pathlib.Path(r"C:\Users\User\Desktop\Tesis")
AQUI = pathlib.Path(__file__).parent
sys.path.insert(0, str(RAIZ))
from CamposT import dlhm
from CamposT.backend import a_numpy, liberar_memoria
from scripts import mi_prueba_dlhm as mp

LAMB, DELTA, L, Z = 528e-6, 1.83e-3, 10.0, 3.0
REF = RAIZ / "referencia/carlos/DLHM-model-main/DLHM-model-main/data/BenchmarkTarget.png"


def interpolar_centrado(A, paso, forma, paso_nuevo):
    def eje(A, n_nuevo, axis):
        n = A.shape[axis]
        u = (np.arange(n_nuevo) - n_nuevo / 2) * paso_nuevo / paso + n / 2
        i0 = np.clip(np.floor(u).astype(np.int64), 0, n - 2)
        w = (u - i0).reshape([n_nuevo, 1] if axis == 0 else [1, n_nuevo])
        return np.take(A, i0, axis=axis) * (1 - w) + np.take(A, i0 + 1, axis=axis) * w
    return eje(eje(A, forma[0], 0), forma[1], 1)


img = cv.imread(str(REF), cv.IMREAD_GRAYSCALE).astype(np.float64) / 255
archivo = AQUI / "holo_bench_z3_intensidad.npy"
if archivo.exists():
    c = np.load(archivo)
else:
    t0 = time.perf_counter()
    c = a_numpy(dlhm.holograma(1 - img, DELTA * Z / L, LAMB, Z, L, img.shape, DELTA))
    print(f"holograma {c.shape} en {time.perf_counter() - t0:.1f} s")
    np.save(archivo, c.astype(np.float32))
    liberar_memoria()
M, N = c.shape
n = 1024
f0, c0 = (M - n) // 2, (N - n) // 2
h = c[f0:f0 + n, c0:c0 + n]
h = h / h.max()                      # como el png /255 de antes
ref = img[f0:f0 + n, c0:c0 + n]      # la de antes: sin invertir

Rec = a_numpy(mp.reconstruir(h, Z, LAMB, DELTA, L, xp=cp, dtype=np.complex64, sobremuestreo=None))
Nr = Rec.shape[0]
Q, P = h.shape
Nm, Mm = mp.malla_remuestreada(P, Q, Z, LAMB, DELTA)
U0 = a_numpy(mp.espectro_angular(mp.fuente_puntual(Nm, Mm, L, 0, 0, LAMB, DELTA * P / Nm, xp=cp, dtype=np.complex64),
                                 P * DELTA, Q * DELTA, 2 * np.pi / LAMB, L - Z, xp=cp, dtype=np.complex64))
div = Rec / np.abs(U0) ** 2
paso_c = DELTA * n / Nr
fp = lambda A: interpolar_centrado(A, paso_c, (n, n), DELTA * Z / L)

vieja = mp.correlacion(mp.observable(Rec, "intensidad"), mp.referencia_a_escala(ref, L / Z, Rec.shape))
vieja_div = mp.correlacion(np.abs(div) ** 2, mp.referencia_a_escala(ref, L / Z, Rec.shape))
huella = mp.correlacion(np.abs(fp(Rec)) ** 2, ref)
huella_div = mp.correlacion(np.abs(fp(div)) ** 2, ref)
escalas = np.linspace(0.8, 6.0, 27)
mejor = max(escalas, key=lambda m: abs(mp.correlacion(np.abs(Rec) ** 2, mp.referencia_a_escala(ref, m, Rec.shape))))
r_mejor = mp.correlacion(np.abs(Rec) ** 2, mp.referencia_a_escala(ref, mejor, Rec.shape))
R = dlhm.Reconstructor(c, DELTA, LAMB, L, Z)
p = DELTA * Z / L
te = a_numpy(R.campo(Z, forma=(n, n), centro=((f0 + n / 2 - M / 2) * p, (c0 + n / 2 - N / 2) * p)))
exacta = mp.correlacion(np.abs(te) ** 2, ref)
print(f"recorte central {n}x{n} de {M}x{N}, malla de Carlos {Nr}, z = {Z} mm, L = {L} mm")
print(f"  geometria vieja (referencia_a_escala), sin dividir: {vieja:+.4f}")
print(f"  geometria vieja, dividiendo por |U0|^2:            {vieja_div:+.4f}")
print(f"  mejor escala vieja (M de 0.8 a 6.0): M = {mejor:.2f}, {r_mejor:+.4f}")
print(f"  huella, sin dividir:                               {huella:+.4f}")
print(f"  huella, dividiendo por |U0|^2 (el motor nuevo):    {huella_div:+.4f}")
print(f"  motor exacto sobre la misma ventana:               {exacta:+.4f}")
```

- [ ] **Step 2: Correrlo y anotar**

Run: `Tesis_env/Scripts/python.exe <scratchpad>/medir_hipotesis_d3.py`
Expected: la geometría vieja en |corr| < 0.1 a cualquier escala, la huella dividiendo > 0.7 en valor absoluto (signo negativo: la referencia de antes no está invertida). Los seis números van a la cabecera del script (Task 5) y a la especificación.

- [ ] **Step 3: Escribir en la especificación los ajustes del plan**

Añadir a `docs/superpowers/specs/2026-09-28-mi-prueba-dlhm-seleccion-design.md`, antes de «Estado», una sección «Ajustes del plan (2026-09-28)» con lo que se midió en mallas de 128 px (L = 0.8 mm, z = 0.24 mm) al preparar el plan y cambia la letra de la especificación:

1. D3: el motor de Carlos DIVIDE por |U0|² (0.84 contra 0.21 sin dividir), interpola la huella en las coordenadas exactas de la malla del ROI en vez de `cv.resize` (que corre hasta medio píxel), y exige un ROI CUADRADO (`reconstruir()` usa el mismo paso en los dos ejes). Con `ROI = None` sobre un sensor rectangular aborta y propone el cuadrado centrado.
2. D7: el simulado se guarda como `<nombre>.npy` y no `holograma.npy`, para que leerlo con `RUTA` dé el mismo nombre y la misma carpeta.
3. Verificación: la tolerancia del barrido no es «un paso fino». El error lo pone la gemela, no el paso: medido 1.3 µm con el exacto y 2.7 µm con el de Carlos, con pasos de 5 µm (grueso) y 1 µm (fino). Se exige medio paso grueso al exacto y un paso grueso al de Carlos.
4. Hallazgo: con el motor de Carlos y un objeto de fase, la Tamura invertida NO enfoca en recortes chicos (se va 19 a 55 µm): el borde del recorte domina el contraste de amplitud. Pearson y SSIM sí enfocan. Se mide a escala real.
5. `margen_del_pico` recibe `ancho` = una décima del barrido (el 1 mm de `barrido_foco` se comería el barrido DLHM); `GUARDAR_BARRIDO` guarda la ventana medida; constante `RESULTADOS` para la raíz de salida; aviso si en fase el fondo no queda en g = 0; en `comprobar_memoria` el texto `ROI_HOLOGRAMA` pasa a `ROI`.
6. Los números de la hipótesis de D3 (Step 2).

---

### Task 2: Las métricas copiadas y el sentido de OBJETO

**Files:**
- Modify: `scripts/mi_prueba_dlhm.py` (imports; sección 5 nueva tras la 3)
- Create: `tests/test_mi_prueba_dlhm_seleccion.py`

**Interfaces:**
- Produces: globales `xp`, `REF`, `REF_CPU`, `REF_RANGO`; `tamura(A, A_cpu)`, `pearson(A, A_cpu)`, `ssim(A, A_cpu)`, `anchura_media(zs, c)`, `margen_del_pico(zs, c, z_pico, ancho=1.0)` (copias literales); `FUNCION`, `COLOR`; `clase_de(nombre, objeto) -> "amplitud" | "fase"`; `mapa(t, clase, xp) -> float64`; `puntuar(nombre, A, A_cpu, objeto) -> float`; `fijar_referencia(ref, xp_)`; `etiquetas(objeto) -> dict`.

- [ ] **Step 1: Escribir las pruebas de D5 y D4**

`tests/test_mi_prueba_dlhm_seleccion.py`, primera parte:

```python
"""La seleccion de foco de barrido_foco en scripts/mi_prueba_dlhm.py.

Todo en CPU y en mallas de 128x160 px, con la lambda y el delta del montaje.
Para que una malla tan chica tenga la NA del banco, L y z se reducen a 0.8 y
0.24 mm (M = 3.33, la del montaje): el sensor de 128 px abarca asi 0.21 de NA
desde la muestra. Con L = 10 mm esa malla veria 0.017 y el objeto saldria
borrado a cualquier z.

Lo que se fija (D1-D12 de
docs/superpowers/specs/2026-09-28-mi-prueba-dlhm-seleccion-design.md):

- D5  las metricas copiadas dan los mismos numeros que las de barrido_foco,
      y tienen el mismo cuerpo;
- D4  con OBJETO = "fase" la Tamura cambia de signo y Pearson va en valor
      absoluto, y la Tamura invertida enfoca en Z_REAL;
- D2  la ventana del motor exacto es el trozo de la reconstruccion entera, y
      los dos motores entregan la ventana alineada con la referencia;
- D3  el motor de Carlos exige un ROI cuadrado y divide por |U0|^2;
- el barrido grueso + fino encuentra Z_REAL con los dos motores;
- D6  Pearson o SSIM sin referencia abortan; D2 un rango que se sale de
      0 < z < L con la guarda del fino tambien;
- y main() de punta a punta: simula, guarda el holograma y lo vuelve a leer.
"""

import ast
import functools
import pathlib

import matplotlib
matplotlib.use("Agg")           # antes de importar el script, que trae pyplot

import cv2 as cv
import numpy as np
import pytest
from skimage.metrics import structural_similarity

from CamposT import dlhm
from scripts import mi_prueba_dlhm as mp

RAIZ = pathlib.Path(__file__).resolve().parents[1]
BARRIDO_FOCO = RAIZ / "scripts" / "barrido_foco.py"

#: Las que viajan copiadas desde barrido_foco.py.
COPIADAS = ("tamura", "pearson", "ssim", "anchura_media", "margen_del_pico")

LAMB, DELTA = 528e-6, 1.83e-3
L, Z = 0.8, 0.24
FORMA = (128, 160)
#: (fila, columna, alto, ancho): centrada y cuadrada, vale para los dos motores
VENTANA = (32, 48, 64, 64)
#: paso grueso de los barridos de prueba, 5 um
PASO = 0.005


def hay_gpu():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


sin_gpu = pytest.mark.skipif(not hay_gpu(), reason="no hay CUDA en esta maquina")


@pytest.fixture
def globales():
    """Deja como estaban las globales de las metricas: fijar_referencia las pisa."""
    antes = (mp.xp, mp.REF, mp.REF_CPU, mp.REF_RANGO)
    yield
    mp.xp, mp.REF, mp.REF_CPU, mp.REF_RANGO = antes


# ── D5: las metricas son las de barrido_foco ────────────────────────────────

def de_barrido_foco(ref):
    """Las COPIADAS sacadas de la FUENTE de barrido_foco.py, con sus globales.

    Con ast y no importando: barrido_foco.py corre su barrido entero al
    importarse. Se compilan solo esas cinco definiciones, en un espacio con las
    globales que leen.
    """
    arbol = ast.parse(BARRIDO_FOCO.read_text(encoding="utf-8"))
    defs = [n for n in arbol.body
            if isinstance(n, ast.FunctionDef) and n.name in COPIADAS]
    assert sorted(d.name for d in defs) == sorted(COPIADAS)
    espacio = {"np": np, "xp": np, "structural_similarity": structural_similarity,
               "REF": ref, "REF_CPU": ref, "REF_RANGO": float(ref.max() - ref.min())}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(BARRIDO_FOCO), "exec"),
         espacio)
    return espacio


def test_las_metricas_copiadas_dan_los_mismos_numeros(globales):
    """Mismos datos, mismos numeros, bit a bit: son la misma cuenta.

    Entran tambien los casos degenerados -un mapa constante, uno nulo- porque
    son los que recorren las ramas que devuelven 0.0 en vez de un NaN.
    """
    rng = np.random.default_rng(0)
    ref = rng.random((48, 64))
    bf = de_barrido_foco(ref)
    mp.fijar_referencia(ref, np)
    for A in (rng.random((48, 64)), 3 * ref + 1, -ref, np.ones((48, 64)),
              np.zeros((48, 64))):
        for n in ("tamura", "pearson", "ssim"):
            assert getattr(mp, n)(A, A) == bf[n](A, A), n
    zs = np.linspace(2.5, 3.5, 101)
    for c in (np.exp(-((zs - 3.0) / 0.05) ** 2) + 0.01 * rng.random(101),
              np.linspace(0.0, 1.0, 101), np.ones(101)):
        np.testing.assert_array_equal(mp.anchura_media(zs, c),
                                      bf["anchura_media"](zs, c))
        for ancho in (0.1, 1.0):
            z_pico = zs[int(np.argmax(c))]
            np.testing.assert_array_equal(mp.margen_del_pico(zs, c, z_pico, ancho),
                                          bf["margen_del_pico"](zs, c, z_pico, ancho))


def _cuerpos(ruta):
    """{nombre: firma y cuerpo sin el docstring} de las COPIADAS de un archivo."""
    arbol = ast.parse(pathlib.Path(ruta).read_text(encoding="utf-8"))
    cuerpos = {}
    for n in arbol.body:
        if isinstance(n, ast.FunctionDef) and n.name in COPIADAS:
            cuerpo = n.body
            if (isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant)
                    and isinstance(cuerpo[0].value.value, str)):
                cuerpo = cuerpo[1:]
            cuerpos[n.name] = ast.dump(n.args) + ast.dump(ast.Module(body=cuerpo, type_ignores=[]))
    return cuerpos


def test_las_copias_tienen_el_mismo_cuerpo():
    """Los docstrings pueden hablar de DLHM; el codigo no puede cambiar.

    Cubre las ramas que los datos de la prueba de arriba no pisen. Si falla,
    alguien toco una de las dos: se cambian las dos o ninguna.
    """
    assert _cuerpos(mp.__file__) == _cuerpos(BARRIDO_FOCO)


# ── D4: el sentido de OBJETO ────────────────────────────────────────────────

def test_en_fase_la_tamura_cambia_de_signo_y_pearson_va_en_valor_absoluto(globales):
    rng = np.random.default_rng(1)
    ref = rng.random((32, 32))
    mp.fijar_referencia(ref, np)
    A = 2 - ref                                    # anticorrelada con la referencia
    assert mp.puntuar("tamura", A, A, "fase") == -mp.tamura(A, A)
    assert mp.puntuar("tamura", A, A, "amplitud") == mp.tamura(A, A)
    assert mp.puntuar("pearson", A, A, "amplitud") == pytest.approx(-1.0)
    assert mp.puntuar("pearson", A, A, "fase") == pytest.approx(1.0)
    assert mp.puntuar("ssim", A, A, "fase") == mp.ssim(A, A)


def test_en_fase_tamura_mide_la_amplitud_y_las_de_referencia_la_fase():
    assert mp.clase_de("tamura", "fase") == "amplitud"
    assert mp.clase_de("pearson", "fase") == mp.clase_de("ssim", "fase") == "fase"
    assert {mp.clase_de(n, "amplitud") for n in mp.FUNCION} == {"amplitud"}
    t = np.exp(1j * np.array([[0.5, -1.0]])) * 2
    np.testing.assert_allclose(mp.mapa(t, "amplitud", np), [[2.0, 2.0]])
    np.testing.assert_allclose(mp.mapa(t, "fase", np), [[0.5, -1.0]])
    assert mp.mapa(t.astype(np.complex64), "fase", np).dtype == np.float64
```

- [ ] **Step 2: Correrlas y ver que fallan**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q`
Expected: FAIL con `AttributeError: module 'scripts.mi_prueba_dlhm' has no attribute 'fijar_referencia'` (o `tamura`).

- [ ] **Step 3: Implementar la sección 5**

En `scripts/mi_prueba_dlhm.py`, añadir a los imports `from skimage.metrics import structural_similarity`, y tras la sección 3 (después de `correlacion`):

```python
# ════════════════════════════════════════════════════════════════════════════
#  5. LAS METRICAS  --  COPIADAS DE scripts/barrido_foco.py
# ════════════════════════════════════════════════════════════════════════════
#
# tamura, pearson, ssim, anchura_media y margen_del_pico son las de
# barrido_foco con el MISMO cuerpo -solo cambian los docstrings- y leen las
# mismas globales de modulo: xp, REF, REF_CPU y REF_RANGO, que
# fijar_referencia() pone antes del barrido. tests/test_mi_prueba_dlhm_seleccion.py
# saca las de barrido_foco de su fuente con ast (importarlo correria su barrido
# entero) y comprueba que den los mismos numeros y tengan el mismo cuerpo. Si
# hay que cambiar una, se cambian las dos.
#
# Todas reciben el mapa YA en la ventana medida -A en el dispositivo, A_cpu el
# mismo en NumPy- y se MAXIMIZAN en el foco. El sentido de OBJETO = "fase" va
# fuera, en puntuar(), para que las copias sigan siendo literales.

#: Las globales de las metricas. De arranque, la CPU y sin referencia.
xp = np
REF = REF_CPU = REF_RANGO = None


def tamura(A, A_cpu):
    """sqrt(std(A) / media(A)). SIN REFERENCIA: la unica que sirve en el banco.

    No depende de la escala del campo, que en DLHM no significa nada: la fija
    el fondo por el que se dividio el holograma.
    """
    m = A.mean()
    if m <= 0:
        return 0.0
    return float(xp.sqrt(A.std() / m))


def pearson(A, A_cpu):
    """Correlacion de Pearson entre A y REF. CON REFERENCIA.

    Invariante a escala y a offset: la reconstruccion no sale con la escala del
    objeto y, con un holograma de intensidad, trae la gemela encima.
    """
    a = A - A.mean()
    r = REF - REF.mean()
    d = xp.sqrt((a * a).sum() * (r * r).sum())
    if d <= 0:
        return 0.0
    return float((a * r).sum() / d)


def ssim(A, A_cpu):
    """SSIM entre A -llevada a la escala de REF por minimos cuadrados- y REF.

    SSIM no es invariante a escala; por eso el ajuste a*A + b. Va en la CPU:
    skimage no tiene version GPU.
    """
    va = A_cpu.var()
    if va <= 0:
        return 0.0
    a = float(((A_cpu - A_cpu.mean()) * (REF_CPU - REF_CPU.mean())).mean() / va)
    b = float(REF_CPU.mean() - a * A_cpu.mean())
    return float(structural_similarity(a * A_cpu + b, REF_CPU, data_range=REF_RANGO))


FUNCION = {"tamura": tamura, "pearson": pearson, "ssim": ssim}
COLOR = {"tamura": "C0", "pearson": "C1", "ssim": "C2"}


def anchura_media(zs, c):
    """Anchura a media altura del pico de c(zs), en mm, o nan si no se puede.

    La curva se reescala a [0, 1] y se interpola donde cruza 0.5 a cada lado
    del maximo. nan si el pico llega al extremo sin bajar de la mitad. Sale del
    barrido GRUESO: no resuelve picos de menos de un par de pasos.
    """
    lo, hi = c.min(), c.max()
    if hi <= lo:
        return float("nan")
    y = (c - lo) / (hi - lo)
    i = int(np.argmax(y))
    cortes = []
    for sentido in (-1, +1):
        j = i
        while 0 <= j + sentido < len(y) and y[j + sentido] > 0.5:
            j += sentido
        k = j + sentido
        if not (0 <= k < len(y)):
            return float("nan")   # el pico no baja de media altura dentro del barrido
        t = (y[j] - 0.5) / (y[j] - y[k])
        cortes.append(zs[j] + t * (zs[k] - zs[j]))
    return abs(cortes[1] - cortes[0])


def margen_del_pico(zs, c, z_pico, ancho=1.0):
    """(altura del pico, rizado del fondo, y su cociente) de la curva c(zs).

    El fondo es lo que queda a mas de `ancho` mm del pico. El rizado es la std
    de las diferencias entre vecinos / sqrt(2): mata las tendencias suaves y
    deja el salto punto a punto. La altura, contra la MEDIANA del fondo. En
    DLHM el pico es estrecho: main() pasa ancho = una decima del barrido.
    """
    fuera = np.abs(zs - z_pico) > ancho
    if fuera.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    o = c[fuera]
    base = float(np.median(o))
    rizado = float(np.std(np.diff(o)) / np.sqrt(2))
    altura = float(c.max() - base)
    return altura, rizado, (altura / rizado if rizado > 0 else float("inf"))


def clase_de(nombre, objeto):
    """Que mapa mide una metrica: "amplitud" (|t|) o "fase" (angle(t)).

    Tamura mira siempre la amplitud, porque es contraste. Las de referencia
    miran lo que el objeto modula: con OBJETO = "fase", la fase.
    """
    return "fase" if objeto == "fase" and nombre != "tamura" else "amplitud"


def mapa(t, clase, xp):
    """El mapa real que mide una metrica, en float64.

    En doble aunque t venga en complex64: las metricas suman millones de
    pixeles, y en simple el error acumulado es del orden de lo que separa dos
    pasos vecinos del barrido.
    """
    if clase == "fase":
        return xp.angle(t).astype(np.float64)
    return xp.abs(t).astype(np.float64)


def puntuar(nombre, A, A_cpu, objeto):
    """FUNCION[nombre] con el sentido de OBJETO: todas se maximizan en el foco.

    "fase": la Tamura cambia de signo, porque un objeto de fase enfoca en el
    MINIMO de contraste de amplitud (en el foco |t| = 1), y Pearson va en valor
    absoluto, porque el signo de la fase depende del convenio del modelo
    (Carlos usa e^{-i phi}). SSIM no cambia: ya ajusta a*A + b, y el signo de
    a le da igual.
    """
    v = FUNCION[nombre](A, A_cpu)
    if objeto == "fase":
        if nombre == "tamura":
            return -v
        if nombre == "pearson":
            return abs(v)
    return v


def fijar_referencia(ref, xp_):
    """Las globales que leen las metricas copiadas.

    ref es el mapa de la referencia en la ventana medida -el de la clase de las
    metricas con referencia-, o None sin referencia: entonces solo vale Tamura.
    xp_ es el modulo del motor: REF vive donde vive A.
    """
    global xp, REF, REF_CPU, REF_RANGO
    xp = xp_
    if ref is None:
        REF = REF_CPU = REF_RANGO = None
        return
    REF_CPU = np.asarray(ref, dtype=np.float64)
    REF = xp_.asarray(REF_CPU)
    REF_RANGO = float(REF_CPU.max() - REF_CPU.min())


def etiquetas(objeto):
    """La leyenda de cada curva, segun lo que mide con ese OBJETO."""
    if objeto == "fase":
        return {"tamura": "-Tamura de |t|  (SIN referencia)",
                "pearson": "|Pearson| de angle(t) con la fase  (con referencia)",
                "ssim": "SSIM de angle(t) con la fase  (con referencia)"}
    return {"tamura": "Tamura de |t|  (SIN referencia)",
            "pearson": "Pearson de |t| con la referencia  (con referencia)",
            "ssim": "SSIM de |t| con la referencia  (con referencia)"}
```

- [ ] **Step 4: Correr las pruebas**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py tests/test_gpu_mi_prueba.py tests/test_malla_remuestreada.py -q`
Expected: todo PASS.

- [ ] **Step 5: Checkpoint** — sin commit (se piden al final).

---

### Task 3: Los dos motores

**Files:**
- Modify: `scripts/mi_prueba_dlhm.py` (sección 4 nueva, antes de la 5; import `from CamposT import dlhm`)
- Test: `tests/test_mi_prueba_dlhm_seleccion.py`

**Interfaces:**
- Consumes: `reconstruir`, `malla_remuestreada`, `fuente_puntual`, `espectro_angular` (sección 2b/3, sin cambios).
- Produces: `MotorExacto(holo, lamb, delta, L_fuente, z_rango, fondo=None, device="auto", dtype=None)` con `.campo(z, ventana)`, `.xp`, `.forma`, `.ventana_maxima`, `.describir()`; `MotorCarlos(holo, ventana, lamb, delta, L_fuente, xp=np, dtype=np.complex128, sobremuestreo=None)` con `.campo(z, ventana)`, `.xp`, `.ventana`, `.ventana_maxima`, `.describir(z_rango)`; `interpolar_centrado(A, paso, forma, paso_nuevo, xp=np)`; `onda_de_referencia(forma, z, lamb, delta, L_fuente, xp=np, dtype=np.complex128, sobremuestreo=None)`. `ventana` es siempre `(fila, columna, alto, ancho)`.

- [ ] **Step 1: Escribir las pruebas de D2 y D3**

Añadir a `tests/test_mi_prueba_dlhm_seleccion.py`:

```python
# ── el objeto y sus hologramas ──────────────────────────────────────────────

def objeto_de_prueba(forma=FORMA):
    """a en [0, 1]: barras de 3, 4 y 6 px, tres puntos y una L, alrededor del
    centro y lejos del borde, con un paso bajo suave para que sea de banda
    limitada. Lo que se reconstruye bien con la NA de estas mallas."""
    M, N = forma
    a = np.zeros(forma)
    cy, cx = M // 2, N // 2
    for k, w in enumerate((3, 4, 6)):
        x0 = cx - 30 + k * 20
        for j in range(3):
            a[cy - 25:cy - 5, x0 + 2 * j * w:x0 + 2 * j * w + w] = 1
    yy, xx = np.ogrid[:M, :N]
    for dy, dx in ((10, -20), (12, -8), (15, 5)):
        a[(yy - cy - dy) ** 2 + (xx - cx - dx) ** 2 <= 9] = 1
    a[cy + 5:cy + 25, cx + 15:cx + 19] = 1
    a[cy + 21:cy + 25, cx + 15:cx + 30] = 1
    fy, fx = np.fft.fftfreq(M)[:, None], np.fft.fftfreq(N)[None, :]
    a = np.fft.ifft2(np.fft.fft2(a) * np.exp(-(fy ** 2 + fx ** 2) / (2 * 0.25 ** 2))).real
    return np.clip(a, 0, 1)


@functools.lru_cache(maxsize=None)
def holograma(tipo, campo):
    """(t, holograma) del objeto de prueba a Z. tipo "amplitud": t = 1 - 0.8a;
    "fase": t = exp(i*pi/2*a). campo "complejo" (1 + V) o "intensidad"."""
    a = objeto_de_prueba()
    t = 1 - 0.8 * a if tipo == "amplitud" else np.exp(1j * np.pi / 2 * a)
    V = dlhm.a_numpy(dlhm.holograma(t, DELTA * Z / L, LAMB, Z, L, FORMA, DELTA,
                                    complejo=True, device="cpu"))
    return t, (1 + V if campo == "complejo" else np.abs(1 + V) ** 2)


def motor_de(nombre, c, rango=(Z - 0.06, Z + 0.06)):
    if nombre == "exacto":
        return mp.MotorExacto(c, LAMB, DELTA, L, rango, device="cpu")
    return mp.MotorCarlos(c, VENTANA, LAMB, DELTA, L)


# ── D2 y D3: la ventana de cada motor ───────────────────────────────────────

def test_la_ventana_descentrada_del_exacto_es_el_trozo_de_la_entera():
    """D2: una ventana (fila, columna, alto, ancho) de la malla M x N se pide
    con centro = (f0 + alto/2 - M/2, c0 + ancho/2 - N/2) * delta*z/L, y es el
    trozo correspondiente de la reconstruccion entera. Descentrada, rectangular
    y en una malla rectangular: un eje cruzado no pasa. Medido: 2.6e-14."""
    _, c = holograma("amplitud", "intensidad")
    m = motor_de("exacto", c)
    entera = m.campo(Z, m.ventana_maxima)
    assert entera.shape == FORMA
    f0, c0, alto, ancho = 10, 70, 50, 70
    trozo = m.campo(Z, (f0, c0, alto, ancho))
    assert trozo.shape == (alto, ancho)
    assert np.abs(trozo - entera[f0:f0 + alto, c0:c0 + ancho]).max() < 1e-10


@pytest.mark.parametrize("nombre, minimo", [("exacto", 0.98), ("carlos", 0.75)])
def test_los_dos_motores_entregan_la_ventana_alineada_con_la_referencia(nombre, minimo):
    """D2 y D3: en Z_REAL la ventana de cada motor se parece a la referencia en
    su sitio, y bastante menos corrida 2 px. Holograma complejo, sin gemela.

    Medido: exacto 0.991 (0.496 corrida); Carlos 0.842 (0.598). El de Carlos
    queda por debajo porque solo ve su recorte -menos NA- y porque su malla
    remuestreada es mas gruesa que la del ROI. SIN dividir por |U0|^2 daba 0.21:
    ver MotorCarlos.
    """
    t, c = holograma("amplitud", "complejo")
    f0, c0, alto, ancho = VENTANA
    g = np.abs(t)[f0:f0 + alto, c0:c0 + ancho]
    A = np.abs(motor_de(nombre, c).campo(Z, VENTANA))
    assert A.shape == (alto, ancho)
    en_su_sitio = mp.correlacion(A, g)
    corrida = mp.correlacion(A[2:, 2:], g[:-2, :-2])
    assert en_su_sitio > minimo
    assert en_su_sitio > corrida + 0.15


@pytest.mark.parametrize("sobremuestreo", [None, 1])
def test_la_onda_de_referencia_es_la_de_dentro_de_reconstruir(sobremuestreo):
    """reconstruir(unos) = U0*conj(U0) = |U0|^2: onda_de_referencia() repite la
    malla, el paso y las funciones de reconstruir(), en las dos ramas."""
    unos = np.ones((64, 64))
    Rec = mp.reconstruir(unos, Z, LAMB, DELTA, L, sobremuestreo=sobremuestreo)
    U0 = mp.onda_de_referencia(unos.shape, Z, LAMB, DELTA, L, sobremuestreo=sobremuestreo)
    assert Rec.shape == U0.shape
    assert np.abs(Rec - np.abs(U0) ** 2).max() < 1e-12 * np.abs(Rec).max()


def test_el_motor_de_carlos_exige_un_roi_cuadrado():
    """reconstruir() usa el mismo paso, delta*P/N, en los dos ejes: con un
    recorte rectangular la esferica sale con otra curvatura en x que en y."""
    _, c = holograma("amplitud", "intensidad")
    with pytest.raises(SystemExit) as fallo:
        mp.MotorCarlos(c, (32, 48, 64, 80), LAMB, DELTA, L)
    assert "CUADRADO" in str(fallo.value)


def test_el_motor_de_carlos_solo_reconstruye_su_recorte():
    _, c = holograma("amplitud", "intensidad")
    m = motor_de("carlos", c)
    assert m.ventana_maxima == VENTANA
    with pytest.raises(ValueError):
        m.campo(Z, (0, 0, 64, 64))


def test_interpolar_centrado_es_exacto_en_lo_lineal():
    """Una rampa lineal por eje se interpola sin error, y la malla nueva que se
    sale de la vieja aborta en vez de extrapolar."""
    n, p = 40, 1.0
    x = (np.arange(n) - n / 2) * p
    A = (2 * x[:, None] + 3 * x[None, :]) * (1 + 1j)
    B = mp.interpolar_centrado(A, p, (25, 30), 0.7)
    y2 = (np.arange(25) - 25 / 2) * 0.7
    x2 = (np.arange(30) - 30 / 2) * 0.7
    np.testing.assert_allclose(B, (2 * y2[:, None] + 3 * x2[None, :]) * (1 + 1j), atol=1e-12)
    with pytest.raises(ValueError):
        mp.interpolar_centrado(A, p, (60, 60), 1.0)
```

- [ ] **Step 2: Correrlas y ver que fallan**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q -k "ventana or motor or onda or interpolar"`
Expected: FAIL con `AttributeError: ... 'MotorExacto'`.

- [ ] **Step 3: Implementar la sección 4**

Añadir el import `from CamposT import dlhm` y, entre la sección 3 y la 5:

```python
# ════════════════════════════════════════════════════════════════════════════
#  4. LOS DOS MOTORES
# ════════════════════════════════════════════════════════════════════════════
#
# La seleccion (metricas, barrido, region) no sabe que motor la alimenta: los
# dos tienen campo(z, ventana) -> t complejo en el dispositivo, sobre la
# ventana (fila, columna, alto, ancho) de la malla que el sensor PROYECTA
# desde la fuente: paso DELTA*z/L, un pixel por pixel del sensor.
#
# En esa malla un detalle cae en el MISMO pixel a toda z: el rayo principal
# que llega al pixel X del sensor cruza el plano z en xi = X*z/L, y
# xi/(DELTA*z/L) = X/DELTA. Asi que el ROI es una ventana fija de pixeles,
# valida en todo el barrido y alineada con el holograma y con la referencia,
# como en barrido_foco, donde la reconstruccion y |U0| comparten malla.


class MotorExacto:
    """CamposT.dlhm.Reconstructor: el holograma ENTERO, y en cada z la ventana.

    El espectro del sensor con la esferica dentro se calcula una vez, al
    construirlo; cada campo() cuesta la funcion de transferencia y los dos
    productos de salida, en proporcion a la ventana. Por eso aqui el ROI tiene
    el sentido de barrido_foco: no recorta el holograma, solo evalua menos.

    Se construye para el rango de z de TODO el barrido, fino incluido (ver
    rango_del_barrido): la rejilla de frecuencias se dimensiona para el peor z
    del rango, y pedir una z fuera aborta.
    """

    def __init__(self, holo, lamb, delta, L_fuente, z_rango, fondo=None,
                 device="auto", dtype=None):
        self.R = dlhm.Reconstructor(holo, delta, lamb, L_fuente, z_rango,
                                    fondo=fondo, device=device, dtype=dtype)
        self.xp = self.R.xp
        self.forma = self.R.forma
        self.delta, self.L = float(delta), float(L_fuente)
        #: la ventana mas grande que sabe dar: la malla entera
        self.ventana_maxima = (0, 0, *self.forma)

    def campo(self, z, ventana):
        f0, c0, alto, ancho = ventana
        M, N = self.forma
        p = self.delta * z / self.L
        centro = ((f0 + alto / 2 - M / 2) * p, (c0 + ancho / 2 - N / 2) * p)
        return self.R.campo(z, forma=(alto, ancho), centro=centro)

    def describir(self):
        (nfy, nfx), (Py, Px) = self.R.n_frecuencias, self.R.P
        mb = nfy * nfx * np.dtype(self.R.dtype).itemsize / 2 ** 20
        return (f"rejilla de frecuencias {nfy}x{nfx}, periodo {Py:.3f} x "
                f"{Px:.3f} mm, espectro de {mb:.0f} MB")


def interpolar_centrado(A, paso, forma, paso_nuevo, xp=np):
    """A, en una malla centrada de paso `paso`, en la centrada forma x paso_nuevo.

    Lineal por eje, real e imaginaria juntas. "Centrada" es el convenio de todo
    el archivo: el indice i esta en (i - n/2)*paso. Se interpola en las
    coordenadas EXACTAS y no con cv.resize, que alinea los centros de pixel de
    las dos mallas y no sus coordenadas: corre la imagen hasta medio pixel (es
    el residuo de (delta - p)/(2M) que cita malla_remuestreada).

    Aborta si la malla nueva se sale de la vieja: extrapolar seria inventar.
    """
    def eje(A, n_nuevo, axis):
        n = A.shape[axis]
        u = (np.arange(n_nuevo) - n_nuevo / 2) * paso_nuevo / paso + n / 2
        if u[0] < -1e-9 or u[-1] > n - 1 + 1e-9:
            raise ValueError(
                f"la malla nueva ({n_nuevo} puntos de {paso_nuevo:.4g} mm) se sale "
                f"de la vieja ({n} de {paso:.4g} mm)")
        i0 = np.clip(np.floor(u).astype(np.int64), 0, n - 2)
        forma_w = [1, 1]
        forma_w[axis] = n_nuevo
        w = xp.asarray((u - i0).reshape(forma_w), dtype=A.real.dtype)
        return (xp.take(A, xp.asarray(i0), axis=axis) * (1 - w)
                + xp.take(A, xp.asarray(i0 + 1), axis=axis) * w)
    return eje(eje(A, forma[0], 0), forma[1], 1)


def onda_de_referencia(forma, z, lamb, delta, L_fuente, xp=np,
                       dtype=np.complex128, sobremuestreo=None):
    """U0: la onda de la fuente retropropagada a la muestra, la de reconstruir().

    reconstruir() la calcula por dentro y no la devuelve: la esferica en el
    sensor, recortada al holograma, propagada L - z. Esta es la misma cuenta
    -misma malla, remuestreada o no, mismo paso, mismas funciones-, asi que
    reconstruir(unos) es |U0|^2. Lo fija una prueba.
    """
    Q, P = forma
    if sobremuestreo is None:
        N, M = malla_remuestreada(P, Q, z, lamb, delta)
    else:
        N, M = Q, P
    kw = dict(xp=xp, dtype=dtype)
    referencia = fuente_puntual(N, M, L_fuente, 0, 0, lamb, (delta * P) / N, **kw)
    return espectro_angular(referencia, P * delta, Q * delta, 2 * np.pi / lamb,
                            L_fuente - z, **kw)


class MotorCarlos:
    """reconstruir() -la cadena de Carlos- sobre el recorte del ROI.

    TRES COSAS LA HACEN COMPARABLE CON LA EXACTA:

    1. EL RECORTE ES OBLIGADO, CENTRADO Y CUADRADO. Obligado por memoria: con
       el sensor entero la malla remuestreada pide 15 GB por paso (ver
       comprobar_memoria). Centrado porque reconstruir() pone la fuente en el
       centro del recorte: con un ROI descentrado la fisica de este motor ya
       no es la del banco, y main() lo avisa. Cuadrado porque reconstruir()
       usa el mismo paso, delta*P/N, en los dos ejes: con un recorte
       rectangular la esferica sale con otra curvatura en x que en y.

    2. SE DIVIDE POR |U0|^2. reconstruir() devuelve Rec = U*conj(U0), que quita
       la FASE de la referencia pero deja su AMPLITUD, |U0|^2, con la
       difraccion del borde del recorte encima: es la causa 1 de la cabecera,
       "la reconstruccion queda DOMINADA por la difraccion del borde".
       Rec/|U0|^2 = U/U0 la quita. Medido en una malla de 128 px con el objeto
       de las pruebas: correlacion con el objeto 0.84 dividiendo, 0.21 sin
       dividir. Cuesta una propagacion mas por z (tres en vez de dos).

    3. SE DEVUELVE LA HUELLA, EN LA MALLA DEL ROI. La salida de reconstruir()
       abarca en la muestra el mismo ancho que el recorte en el sensor, pero el
       objeto que el recorte ve ocupa solo su 1/M central: los rayos
       principales que llegan al recorte cruzan el plano z en su ancho por z/L.
       Esa huella se interpola en la malla del ROI, paso delta*z/L, que es la
       del motor exacto: asi las metricas no dependen del motor.

    El holograma entra normalizado a [0, 1], como antes: una escala global no
    cambia ninguna metrica.
    """

    def __init__(self, holo, ventana, lamb, delta, L_fuente, xp=np,
                 dtype=np.complex128, sobremuestreo=None):
        holo = np.asarray(holo)
        f0, c0, alto, ancho = (int(v) for v in ventana)
        if alto != ancho:
            M, N = holo.shape
            lado = min(alto, ancho)
            raise SystemExit(
                f"El motor de Carlos necesita un ROI CUADRADO y la region medida "
                f"es de {alto}x{ancho} px (alto x ancho): reconstruir() usa el "
                f"mismo paso en los dos ejes, delta*P/N, y con un recorte "
                f"rectangular la esferica sale con otra curvatura en x que en y."
                f"\nUsa un ROI cuadrado y centrado, por ejemplo\n\n"
                f"    ROI = ({(N - lado) // 2}, {(M - lado) // 2}, {lado}, {lado})\n")
        h = holo[f0:f0 + alto, c0:c0 + ancho]
        self.h = h / np.abs(h).max()
        self.ventana = self.ventana_maxima = (f0, c0, alto, ancho)
        self.lamb, self.delta, self.L = float(lamb), float(delta), float(L_fuente)
        self.xp, self.dtype, self.sobremuestreo = xp, dtype, sobremuestreo

    def campo(self, z, ventana):
        if tuple(int(v) for v in ventana) != self.ventana:
            raise ValueError(f"el motor de Carlos solo reconstruye su recorte, "
                             f"{self.ventana}; se pidio {tuple(ventana)}")
        kw = dict(xp=self.xp, dtype=self.dtype, sobremuestreo=self.sobremuestreo)
        Rec = reconstruir(self.h, z, self.lamb, self.delta, self.L, **kw)
        U0 = onda_de_referencia(self.h.shape, z, self.lamb, self.delta, self.L, **kw)
        Rec /= self.xp.abs(U0) ** 2
        del U0
        n = self.h.shape[0]
        return interpolar_centrado(Rec, self.delta * n / Rec.shape[0], (n, n),
                                   self.delta * z / self.L, self.xp)

    def describir(self, z_rango):
        n = self.h.shape[0]
        if self.sobremuestreo is not None:
            return (f"recorte {n}x{n} sin remuestrear (SOBREMUESTREO = "
                    f"{self.sobremuestreo}): alia, ver la cabecera")
        z_lo, z_hi = min(z_rango), max(z_rango)
        lo = malla_remuestreada(n, n, z_lo, self.lamb, self.delta)[0]
        hi = malla_remuestreada(n, n, z_hi, self.lamb, self.delta)[0]
        return (f"recorte {n}x{n}, malla remuestreada {lo}x{lo} a z = {z_lo:.3f} "
                f"mm y {hi}x{hi} a z = {z_hi:.3f} mm")
```

- [ ] **Step 4: Correr las pruebas**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py tests/test_gpu_mi_prueba.py tests/test_malla_remuestreada.py -q`
Expected: todo PASS.

- [ ] **Step 5: Checkpoint** — sin commit.

---

### Task 4: La selección: rango, región, configuración, barrido y foco

**Files:**
- Modify: `scripts/mi_prueba_dlhm.py` (secciones 6 y 7 nuevas, tras la 5)
- Test: `tests/test_mi_prueba_dlhm_seleccion.py`

**Interfaces:**
- Consumes: `MotorExacto`, `MotorCarlos` (Task 3); `clase_de`, `mapa`, `puntuar`, `fijar_referencia`, `FUNCION` (Task 2); `a_cpu` (2b).
- Produces: `barrido(motor, ventana, zs, metricas, objeto, en_cpu=(), solapar=False, reloj=None, guardar=None) -> {nombre: np.ndarray}`; `enfocar(motor, ventana, zs, pasos_fino, metricas, objeto, en_cpu=(), solapar=False, reloj=None, guardar_en=None) -> (curvas, picos, finos, z_foco)`; `rango_del_barrido(z_min, z_max, pasos, L_fuente) -> (zs, paso, (z_lo, z_hi))`; `region_medida(forma, roi, borde_px, imagen=None, titulo="") -> ((fila, columna, alto, ancho), Roi | None)`; `comprobar_configuracion(metricas, formato, objeto, motor, campo, ruta, referencia, z_real)`.

- [ ] **Step 1: Escribir las pruebas**

Añadir a `tests/test_mi_prueba_dlhm_seleccion.py`:

```python
# ── D4: la Tamura invertida enfoca un objeto de fase ────────────────────────

def test_en_fase_la_tamura_invertida_enfoca_en_z_real(globales):
    """Un objeto de fase pura tiene |t| = 1 en el foco: contraste de amplitud
    MINIMO. Con el signo cambiado, el maximo cae en Z. Holograma complejo y
    motor exacto: medido -0.241 en Z frente a -0.40 en los vecinos. (Con el de
    Carlos y un recorte de 64 px no pasa: el borde del recorte domina el
    contraste de amplitud. Ver la cabecera del script.)"""
    _, c = holograma("fase", "complejo")
    mp.fijar_referencia(None, np)
    zs = np.linspace(Z - 0.05, Z + 0.05, 21)
    curvas = mp.barrido(motor_de("exacto", c), VENTANA, zs, ("tamura",), "fase")
    assert zs[int(np.argmax(curvas["tamura"]))] == pytest.approx(Z, abs=1e-9)


# ── el barrido grueso + fino ────────────────────────────────────────────────

@pytest.mark.parametrize("desfase", [0.0, 0.0023])
@pytest.mark.parametrize("nombre, tolerancia", [("exacto", PASO / 2), ("carlos", PASO)])
def test_el_barrido_grueso_y_fino_encuentra_z_real(nombre, tolerancia, desfase, globales):
    """Holograma de INTENSIDAD, las tres metricas, grueso de 5 um y fino de 1 um.

    desfase 0.0023 deja Z fuera de la rejilla gruesa: es el fino el que tiene
    que encontrarlo. La tolerancia no es un paso fino: el error lo pone la
    gemela, no el paso. Medido: el exacto a +1.3 um como mucho, el de Carlos a
    -2.7 um (su pico es mas ancho y plano: ve menos NA). Medio paso grueso y
    un paso grueso son la cota de que el fino sirve.
    """
    t, c = holograma("amplitud", "intensidad")
    f0, c0, alto, ancho = VENTANA
    zs = np.linspace(Z + desfase - 0.05, Z + desfase + 0.05, 21)
    m = motor_de(nombre, c, (zs[0] - PASO, zs[-1] + PASO))
    mp.fijar_referencia(mp.mapa(t[f0:f0 + alto, c0:c0 + ancho], "amplitud", np), np)
    metricas = ("tamura", "pearson", "ssim")
    curvas, picos, finos, z_foco = mp.enfocar(m, VENTANA, zs, 11, metricas, "amplitud",
                                              en_cpu={"ssim"})
    assert set(finos) == set(picos.values())
    for n in metricas:
        assert len(curvas[n]) == len(zs)
        assert abs(z_foco[n] - Z) < tolerancia, (n, z_foco[n])


def test_repartir_las_metricas_entre_cpu_y_dispositivo_no_cambia_ninguna_curva(globales):
    t, c = holograma("amplitud", "intensidad")
    f0, c0, alto, ancho = VENTANA
    m = motor_de("exacto", c)
    mp.fijar_referencia(mp.mapa(t[f0:f0 + alto, c0:c0 + ancho], "amplitud", np), np)
    zs = np.linspace(Z - 0.01, Z + 0.01, 5)
    metricas = ("tamura", "pearson", "ssim")
    reloj = {}
    a = mp.barrido(m, VENTANA, zs, metricas, "amplitud", en_cpu=set(), reloj=reloj)
    b = mp.barrido(m, VENTANA, zs, metricas, "amplitud", en_cpu={"ssim"}, reloj=reloj)
    for n in metricas:
        np.testing.assert_array_equal(a[n], b[n])
    assert set(reloj) == {"reconstruir", *metricas}


@sin_gpu
def test_solapar_no_cambia_ningun_numero(globales):
    """Encadenar la GPU con la CPU mueve el reloj, no las curvas: bit a bit."""
    import cupy as cp
    t, c = holograma("amplitud", "intensidad")
    f0, c0, alto, ancho = VENTANA
    m = mp.MotorExacto(c, LAMB, DELTA, L, (Z - 0.06, Z + 0.06), device="gpu",
                       dtype=np.complex64)
    mp.fijar_referencia(mp.mapa(t[f0:f0 + alto, c0:c0 + ancho], "amplitud", np), cp)
    zs = np.linspace(Z - 0.01, Z + 0.01, 5)
    metricas = ("tamura", "pearson", "ssim")
    reloj = {}
    a = mp.barrido(m, VENTANA, zs, metricas, "amplitud", en_cpu={"ssim"}, solapar=False)
    b = mp.barrido(m, VENTANA, zs, metricas, "amplitud", en_cpu={"ssim"}, solapar=True,
                   reloj=reloj)
    for n in metricas:
        np.testing.assert_array_equal(a[n], b[n])
    assert "espera GPU" in reloj


# ── D6 y D2: lo que aborta antes de reconstruir ─────────────────────────────

@pytest.mark.parametrize("metricas", [("pearson",), ("tamura", "ssim")])
def test_pedir_pearson_o_ssim_sin_referencia_aborta(metricas):
    with pytest.raises(SystemExit) as fallo:
        mp.comprobar_configuracion(metricas, "png", "amplitud", "exacto",
                                   "intensidad", "holo.png", None, None)
    assert '("tamura",)' in str(fallo.value)


def test_solo_tamura_sin_referencia_vale():
    mp.comprobar_configuracion(("tamura",), "png", "amplitud", "exacto",
                               "intensidad", "holo.png", None, None)


@pytest.mark.parametrize("cambio", [
    dict(metricas=()), dict(metricas=("nitidez",)), dict(formato="tif"),
    dict(objeto="intensidad"), dict(motor="rapido"), dict(campo="fase"),
    dict(referencia=None), dict(z_real=None)])
def test_una_configuracion_invalida_aborta(cambio):
    """Con RUTA = None: simular necesita el objeto y la z."""
    kw = dict(metricas=("tamura",), formato="png", objeto="amplitud", motor="exacto",
              campo="intensidad", ruta=None, referencia="objeto.png", z_real=3.0)
    kw.update(cambio)
    with pytest.raises(SystemExit):
        mp.comprobar_configuracion(**kw)


def test_el_rango_lleva_la_guarda_del_fino():
    zs, paso, (lo, hi) = mp.rango_del_barrido(0.2, 0.3, 11, L)
    assert paso == pytest.approx(0.01)
    assert (lo, hi) == (zs[0] - paso, zs[-1] + paso)


@pytest.mark.parametrize("z_min, z_max, pasos", [
    (0.005, 0.1, 11),       # la guarda baja de z = 0
    (0.5, 0.795, 11),       # la guarda pasa de L = 0.8
    (0.3, 0.2, 11),         # al reves
    (0.2, 0.3, 1)])         # sin paso
def test_un_rango_que_se_sale_de_la_muestra_aborta(z_min, z_max, pasos):
    with pytest.raises(SystemExit):
        mp.rango_del_barrido(z_min, z_max, pasos, L)


# ── la region medida ────────────────────────────────────────────────────────

def test_sin_roi_se_mide_el_cuarto_central_del_lado_corto():
    """D9: BORDE_PX = None es 3*min(M, N)//8 por lado."""
    assert mp.region_medida((3000, 4000), None, None) == ((1125, 1125, 750, 1750), None)
    assert mp.region_medida((3000, 4000), None, 1000)[0] == (1000, 1000, 1000, 2000)


def test_un_roi_de_tupla_se_da_en_x_y_y_se_devuelve_en_filas():
    ventana, roi = mp.region_medida((3000, 4000), (100, 200, 1000, 750), None)
    assert ventana == (200, 100, 750, 1000)      # (fila, columna, alto, ancho)
    assert (roi.x0, roi.y0, roi.ancho, roi.alto) == (100, 200, 1000, 750)


@pytest.mark.parametrize("roi, borde", [
    (None, 480), (None, -1), ((3900, 0, 200, 200), None), ((0, 0, 50, 50), None),
    ((1, 2, 3), None)])
def test_una_region_que_no_sirve_aborta(roi, borde):
    with pytest.raises(SystemExit):
        mp.region_medida((1000, 4000), roi, borde)
```

- [ ] **Step 2: Correrlas y ver que fallan**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q`
Expected: FAIL con `AttributeError: ... 'barrido'` (y las de configuración/región con `comprobar_configuracion` / `region_medida`).

- [ ] **Step 3: Implementar las secciones 6 y 7**

```python
# ════════════════════════════════════════════════════════════════════════════
#  6. EL BARRIDO  --  el de barrido_foco, con el motor en vez de mpasm()
# ════════════════════════════════════════════════════════════════════════════

def barrido(motor, ventana, zs, metricas, objeto, en_cpu=(), solapar=False,
            reloj=None, guardar=None):
    """{nombre: curva}: todas las metricas medidas sobre la MISMA reconstruccion.

    Es el barrido() de barrido_foco con el motor en lugar de mpasm() y hasta
    dos mapas por z en lugar de uno: con OBJETO = "fase" la Tamura mide |t| y
    las de referencia angle(t) (clase_de). Cada mapa baja a la CPU una vez.

    en_cpu   las metricas que corren en la CPU: reciben A = None y solo pueden
             usar A_cpu. Meter una que use A la rompe, y no en silencio.
    solapar  con GPU y alguna metrica en_cpu, encadena: lanza la
             reconstruccion del paso i y, mientras la GPU la calcula, cierra en
             la CPU las metricas del paso i-1. No cambia ningun numero -las
             curvas se escriben por INDICE, porque el paso i-1 se cierra una
             vuelta mas tarde-, solo el reloj.
    reloj    dict de segundos por etapa que se ACUMULA: el grueso y los finos
             comparten uno. "reconstruir" es lanzar el motor y esperarlo; con
             solapar, "espera GPU": lo que la CPU no consiguio esconder.
    guardar  f(t, z), llamada en cada z, o None.
    """
    xp_ = motor.xp
    en_gpu_lista = [n for n in metricas if n not in en_cpu]
    en_cpu_lista = [n for n in metricas if n in en_cpu]
    solapando = solapar and xp_ is not np and bool(en_cpu_lista)
    clave = "espera GPU" if solapando else "reconstruir"
    reloj = {} if reloj is None else reloj
    for k in (clave, *metricas):
        reloj.setdefault(k, 0.0)
    clases = {clase_de(n, objeto) for n in metricas}
    curvas = {n: np.empty(len(zs)) for n in metricas}
    pendiente = None                       # (indice, mapas en la CPU) del paso anterior

    def cerrar_en_cpu(i, mapas_cpu):
        for n in en_cpu_lista:
            t0 = time.perf_counter()
            curvas[n][i] = puntuar(n, None, mapas_cpu[clase_de(n, objeto)], objeto)
            reloj[n] += time.perf_counter() - t0

    for i, z in enumerate(zs):
        t0 = time.perf_counter()
        t = motor.campo(z, ventana)        # en la GPU lanza y vuelve enseguida
        mapas = {c: mapa(t, c, xp_) for c in clases}
        lanzar = time.perf_counter() - t0

        # la GPU esta ocupada con lo de arriba: cerrar el paso anterior
        if pendiente is not None:
            cerrar_en_cpu(*pendiente)
            pendiente = None

        # AQUI se espera al motor: bajar a la CPU sincroniza
        t0 = time.perf_counter()
        mapas_cpu = {c: a_cpu(A) for c, A in mapas.items()}
        reloj[clave] += lanzar + time.perf_counter() - t0

        for n in en_gpu_lista:
            t0 = time.perf_counter()
            c = clase_de(n, objeto)
            curvas[n][i] = puntuar(n, mapas[c], mapas_cpu[c], objeto)
            reloj[n] += time.perf_counter() - t0

        if guardar is not None:
            guardar(t, z)
        del t, mapas

        if solapando:
            pendiente = (i, mapas_cpu)     # se cierra en la vuelta siguiente
        else:
            cerrar_en_cpu(i, mapas_cpu)

    if pendiente is not None:              # el ultimo paso no tiene siguiente
        cerrar_en_cpu(*pendiente)
    return curvas


def enfocar(motor, ventana, zs, pasos_fino, metricas, objeto, en_cpu=(),
            solapar=False, reloj=None, guardar_en=None):
    """El barrido grueso y un fino por cada pico distinto, como barrido_foco.

    -> (curvas, picos, finos, z_foco): las curvas gruesas, el indice grueso del
    maximo de cada metrica, {indice: (zs_fino, curvas_finas)} y la z elegida.

    El fino va de zs[i] - paso a zs[i] + paso, un paso grueso por cada lado: por
    eso rango_del_barrido() da ese rango al motor exacto. Se lanza UNO por
    indice grueso distinto, compartido por las metricas que lo eligieron:
    reconstruir dos veces a la misma z no da dos respuestas.

    guardar_en(subcarpeta) -> f(t, z) o None: "grueso" y "fino_z<zs[i]>".
    """
    paso = zs[1] - zs[0]
    kw = dict(en_cpu=en_cpu, solapar=solapar, reloj=reloj)
    guardar = guardar_en or (lambda _: None)
    curvas = barrido(motor, ventana, zs, metricas, objeto,
                     guardar=guardar("grueso"), **kw)
    picos = {n: int(np.argmax(c)) for n, c in curvas.items()}
    finos, z_foco = {}, {}
    for i in sorted(set(picos.values())):
        zs_f = np.linspace(zs[i] - paso, zs[i] + paso, pasos_fino)
        finos[i] = (zs_f, barrido(motor, ventana, zs_f, metricas, objeto,
                                  guardar=guardar(f"fino_z{zs[i]:.4f}"), **kw))
    for n, i in picos.items():
        zs_f, cf = finos[i]
        z_foco[n] = float(zs_f[int(np.argmax(cf[n]))])
    return curvas, picos, finos, z_foco


# ════════════════════════════════════════════════════════════════════════════
#  7. LA CONFIGURACION Y LA REGION MEDIDA
# ════════════════════════════════════════════════════════════════════════════

def comprobar_configuracion(metricas, formato, objeto, motor, campo, ruta,
                            referencia, z_real):
    """Aborta ANTES de leer ni reconstruir nada si la combinacion no tiene sentido."""
    if formato not in ("png", "npy", "ambos"):
        raise SystemExit(f'FORMATO_BARRIDO = {formato!r}: usa "png", "npy" o "ambos".')
    if not metricas:
        raise SystemExit("METRICAS esta vacio: no hay nada con que elegir el foco.")
    for n in metricas:
        if n not in FUNCION:
            raise SystemExit(f"METRICAS trae {n!r}, y solo existen {tuple(FUNCION)}.")
    if objeto not in ("amplitud", "fase"):
        raise SystemExit(f'OBJETO = {objeto!r}: usa "amplitud" o "fase".')
    if motor not in ("exacto", "carlos"):
        raise SystemExit(f'MOTOR = {motor!r}: usa "exacto" o "carlos".')
    if ruta is None:
        if referencia is None:
            raise SystemExit("RUTA = None simula el holograma de REFERENCIA, y "
                             "REFERENCIA tambien es None: no hay objeto que simular.")
        if z_real is None:
            raise SystemExit("RUTA = None simula el holograma a Z_REAL, y Z_REAL es "
                             "None: di a que z esta la muestra.")
        if campo not in ("intensidad", "complejo"):
            raise SystemExit(f'CAMPO = {campo!r}: usa "intensidad" o "complejo".')
    con_referencia = [n for n in metricas if n != "tamura"]
    if referencia is None and con_referencia:
        raise SystemExit(
            f"METRICAS pide {', '.join(con_referencia)} y REFERENCIA = None: sin el "
            f"objeto no hay contra que compararlas.\nCon un holograma de laboratorio "
            f'pon METRICAS = ("tamura",), la unica que no necesita referencia.')


def rango_del_barrido(z_min, z_max, pasos, L_fuente):
    """-> (zs, paso, (z_lo, z_hi)): el grueso, y el rango que cubre tambien el fino.

    El fino se sale del grueso un paso por cada lado, asi que el motor exacto
    se construye para [z_min - paso, z_max + paso] y ese rango tiene que caber
    en 0 < z < L: la muestra esta ENTRE la fuente y el sensor.
    """
    if pasos < 2:
        raise SystemExit(f"PASOS = {pasos}: hacen falta dos o mas para tener paso.")
    zs = np.linspace(z_min, z_max, pasos)
    paso = zs[1] - zs[0]
    z_lo, z_hi = zs[0] - paso, zs[-1] + paso
    if not 0 < z_lo < z_hi < L_fuente:
        raise SystemExit(
            f"El barrido de {z_min:g} a {z_max:g} mm con paso {paso:g}, y el fino que "
            f"se sale un paso por cada lado, va de {z_lo:g} a {z_hi:g} mm: tiene que "
            f"quedar en 0 < z < L = {L_fuente:g} mm, con z_min < z_max. La muestra "
            f"esta ENTRE la fuente y el sensor.")
    return zs, paso, (z_lo, z_hi)


def region_medida(forma, roi, borde_px, imagen=None, titulo=""):
    """-> ((fila, columna, alto, ancho), Roi o None): donde miden las metricas.

    roi = None                   la malla menos borde_px por lado; con
                                 borde_px = None, 3*min(M, N)//8, que deja el
                                 cuarto central del lado corto (D9: la
                                 periferia trae la gemela y la perdida de NA
                                 del borde del sensor).
    roi = True                   se arrastra con el raton sobre `imagen`.
    roi = (X0, Y0, ANCHO, ALTO)  ventana fija, X0 la columna e Y0 la fila.

    Con un ROI borde_px no se usa, como en barrido_foco.
    """
    M, N = forma
    if roi is None:
        borde = 3 * min(M, N) // 8 if borde_px is None else int(borde_px)
        if borde < 0 or min(M, N) - 2 * borde < 64:
            raise SystemExit(
                f"BORDE_PX = {borde} por lado no deja una region de 64 px de lado en "
                f"una malla {M}x{N}. Bajalo, o fija un ROI.")
        return (borde, borde, M - 2 * borde, N - 2 * borde), None
    if roi is True:
        if imagen is None:
            raise ValueError("ROI = True necesita la imagen sobre la que arrastrar")
        r = elegir(imagen, titulo)
    else:
        if not (isinstance(roi, (tuple, list)) and len(roi) == 4):
            raise SystemExit(
                f"ROI = {roi!r} no es ninguno de los valores validos:\n\n"
                f"    None                    el centro menos BORDE_PX\n"
                f"    True                    la arrastras con el raton\n"
                f"    (X0, Y0, ANCHO, ALTO)   ventana fija\n")
        r = Roi(*roi)
    if r.x0 + r.ancho > N or r.y0 + r.alto > M:
        raise SystemExit(f"ROI {r} se sale de la malla {M}x{N}.")
    if min(r.ancho, r.alto) < 64:
        raise SystemExit(f"ROI {r}: menos de 64 px de lado no da para medir contraste.")
    return (r.y0, r.x0, r.alto, r.ancho), r
```

- [ ] **Step 4: Correr las pruebas**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py tests/test_gpu_mi_prueba.py tests/test_malla_remuestreada.py -q`
Expected: todo PASS (la de `solapar` solo con GPU).

- [ ] **Step 5: Checkpoint** — sin commit.

---

### Task 5: El objeto, el holograma, la salida y main()

**Files:**
- Modify: `scripts/mi_prueba_dlhm.py`: cabecera, bloques de parámetros (secciones 1-5 de constantes), sección 8 nueva, `main()` nuevo; borrar `referencia_a_escala`, `observable`, `_roi_de`; en `comprobar_memoria`, `ROI_HOLOGRAMA` → `ROI` en los dos mensajes.
- Test: `tests/test_mi_prueba_dlhm_seleccion.py`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: `leer_imagen(ruta)`, `objeto_de(img, objeto, invertir, fase_max)`, `fondo_de(img)`, `simular(t, lamb, delta, L_fuente, z, campo, device="auto", dtype=None)`, `vista(a, objeto="amplitud")`, `carpeta_de_salida(raiz, ruta, referencia, z_real, campo, motor, objeto, roi) -> (Path, nombre)`, `guardar_instante(t, z, carpeta, formato, objeto)`, `informe_de_focos(zs, curvas, picos, z_foco, metricas, z_real=None, ancho=None) -> str`, `guardar_curvas(ruta, zs, curvas, finos, metricas)`, `figura_barrido(...) -> Figure`, `figura_campos(...) -> Figure`, `main() -> z_foco`; constante nueva `RESULTADOS`.

- [ ] **Step 1: Escribir la prueba de punta a punta**

```python
# ── main() de punta a punta ─────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path, monkeypatch, globales):
    """main() sobre el objeto de prueba, en la CPU y escribiendo en tmp_path.

    La referencia es un PNG de 8 bits con el fondo negro y el objeto blanco,
    como el BenchmarkTarget: con INVERTIR = True, t = 1 - img."""
    ruta = tmp_path / "objeto.png"
    cv.imwrite(str(ruta), np.round(255 * objeto_de_prueba()).astype(np.uint8))
    valores = dict(
        MOSTRAR=False, GUARDAR_FIGURA=True, GUARDAR_BARRIDO=False,
        METRICAS=("tamura", "pearson", "ssim"), EN_CPU={"ssim"}, SOLAPAR=True,
        RUTA=None, FONDO=None, REFERENCIA=ruta, OBJETO="amplitud", INVERTIR=True,
        Z_REAL=Z, CAMPO="intensidad", LAMB=LAMB, DELTA=DELTA, L=L,
        Z_MIN=Z - 0.05, Z_MAX=Z + 0.05, PASOS=21, PASOS_FINO=11,
        ROI=(48, 32, 64, 64), BORDE_PX=None, MOTOR="exacto", DISPOSITIVO="cpu",
        DTYPE=None, SOBREMUESTREO=None, RESULTADOS=tmp_path / "salida")
    for k, v in valores.items():
        monkeypatch.setattr(mp, k, v)
    return tmp_path / "salida" / "sim_objeto_z0.24mm_intensidad_exacto_amplitud_roi48-32-64x64"


def test_main_simula_guarda_y_encuentra_el_foco(config, monkeypatch):
    z_foco = mp.main()
    for f in ("sim_objeto_z0.24mm_intensidad.npy", "curvas.npz",
              "barrido_metricas.png", "campos.png"):
        assert (config / f).exists(), f
    assert all(abs(z - Z) < PASO / 2 for z in z_foco.values()), z_foco
    # lo guardado se vuelve a leer sin simular, y cae en la misma carpeta
    monkeypatch.setattr(mp, "RUTA", config / "sim_objeto_z0.24mm_intensidad.npy")
    z_foco_2 = mp.main()
    assert all(abs(z_foco_2[n] - z_foco[n]) <= 2 * PASO / 10 for n in z_foco), (z_foco, z_foco_2)
    curvas = np.load(config / "curvas.npz")
    assert {"zs", "grueso_tamura", "grueso_pearson", "grueso_ssim"} <= set(curvas.files)


def test_main_con_el_motor_de_carlos(config, monkeypatch):
    monkeypatch.setattr(mp, "MOTOR", "carlos")
    z_foco = mp.main()
    carpeta = config.parent / config.name.replace("_exacto_", "_carlos_")
    assert (carpeta / "curvas.npz").exists()
    assert all(abs(z - Z) < PASO for z in z_foco.values()), z_foco


def test_main_en_fase_y_solo_con_tamura(config, monkeypatch):
    """El caso del laboratorio: sin referencia, solo Tamura. Aqui con un objeto
    de fase simulado con fondo en g = 0 (INVERTIR = False) y holograma
    complejo, que es donde la Tamura invertida tiene su pico limpio."""
    for k, v in dict(OBJETO="fase", INVERTIR=False, CAMPO="complejo",
                     METRICAS=("tamura",), EN_CPU=set()).items():
        monkeypatch.setattr(mp, k, v)
    z_foco = mp.main()
    assert abs(z_foco["tamura"] - Z) < PASO / 2, z_foco
```

- [ ] **Step 2: Correr y ver que falla**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q -k main`
Expected: FAIL (`main()` todavía es el viejo: `AttributeError: ... 'RESULTADOS'` al parchear, o `cv.imread` de `RUTA = None`).

- [ ] **Step 3: Reescribir el bloque de parámetros**

Sustituir la sección 1 entera (desde el separador `1. PARAMETROS` hasta `FILAS_POR_BLOQUE = 512`) y los imports por:

```python
import pathlib
import sys
import time

# ════════════════════════════════════════════════════════════════════════════
#  1. QUE SE MUESTRA Y QUE SE GUARDA  --  los mismos de scripts/barrido_foco.py
# ════════════════════════════════════════════════════════════════════════════

#: True abre las figuras al terminar (bloquea hasta cerrarlas).
MOSTRAR = True

#: True guarda ademas esas figuras como PNG en la carpeta de salida.
GUARDAR_FIGURA = True

#: True guarda la reconstruccion de CADA z del barrido (grueso y fino). A
#: diferencia de barrido_foco guarda la VENTANA MEDIDA, no la malla entera: el
#: motor de Carlos no tiene otra.
GUARDAR_BARRIDO = False

#: Que se guarda por cada z: "png" (|t|^2, o angle(t) con OBJETO = "fase", a
#: [0, 1]), "npy" (t complejo, tal cual) o "ambos".
FORMATO_BARRIDO = "png"

#: Que metricas se evaluan en cada z, en orden de dibujo. "tamura" es la unica
#: sin referencia: la unica que sirve con un holograma de laboratorio.
METRICAS = ("tamura", "pearson", "ssim")

#: Cuales corren en la CPU: reciben A = None y solo usan A_cpu. SSIM, porque
#: skimage no tiene version GPU. Meter una que use A la rompe a la primera z.
EN_CPU = {"ssim"}

#: True encadena: la GPU reconstruye el paso i+1 mientras la CPU mide el i. No
#: cambia ningun numero, solo el reloj. Sin GPU no hace nada.
SOLAPAR = True

#: Lado en pixeles del recuadro que se amplia en la figura de campos.
ZOOM_PX = 200

import cv2 as cv                                                     # noqa: E402
import matplotlib                                                    # noqa: E402
if not MOSTRAR:
    matplotlib.use("Agg")   # sin ventana: no hace falta backend grafico
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
from skimage.metrics import structural_similarity                    # noqa: E402

try:
    import cupy as cp
except Exception:                      # sin CuPy, sin CUDA, o CuPy roto
    cp = None

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from CamposT import dlhm                                             # noqa: E402
from CamposT.montaje import MONTAJE                                  # noqa: E402
from CamposT.roi import Roi, elegir                                  # noqa: E402

# ════════════════════════════════════════════════════════════════════════════
#  2. EL HOLOGRAMA Y EL OBJETO
# ════════════════════════════════════════════════════════════════════════════

#: El holograma grabado (.png de 8 bits o .npy), o None para SIMULARLO con la
#: ida exacta de CamposT.dlhm a partir de REFERENCIA. El simulado se guarda en
#: la carpeta de salida como <nombre>.npy: pon aqui esa ruta y la corrida
#: siguiente no vuelve a simular (en 3000x4000 la ida tarda del orden de un
#: minuto) y escribe en la misma carpeta.
RUTA = None

#: Solo con RUTA y MOTOR = "exacto": la imagen SIN muestra, por la que se
#: divide el holograma. None divide por su propia media, que deja dentro la
#: envolvente del cono del pinhole. El motor de Carlos no la usa: recibe la
#: intensidad normalizada a [0, 1].
FONDO = None

#: EL OBJETO, en la malla que el sensor proyecta desde la fuente: paso
#: DELTA*z/L y la MISMA forma que el holograma, un pixel por pixel del sensor.
#: Con RUTA = None es lo que se simula; si no, contra lo que se comparan
#: Pearson y SSIM. None solo con METRICAS = ("tamura",).
REFERENCIA = RAIZ / "referencia/carlos/DLHM-model-main/DLHM-model-main/data/BenchmarkTarget.png"

#: "amplitud": t = g, y se mide |t|, como en barrido_foco.
#: "fase":     t = exp(i*FASE_MAX*g). La Tamura va con el signo cambiado -un
#:             objeto de fase enfoca en el MINIMO de contraste de amplitud- y
#:             Pearson (en valor absoluto) y SSIM comparan angle(t) con la
#:             fase de la referencia.
OBJETO = "amplitud"

#: g = 1 - img (fondo transparente, barras opacas: la holografia en linea de
#: Gabor necesita el fondo) o g = img. OJO en "fase": el fondo tiene que quedar
#: en g = 0 o el borde de la ventana es un escalon de fase; el BenchmarkTarget
#: tiene el fondo negro, asi que en fase va con INVERTIR = False.
INVERTIR = True

#: rad: la fase de g = 1. Solo con OBJETO = "fase".
FASE_MAX = np.pi / 2

#: mm. Con RUTA = None, la z a la que se simula. Con RUTA, la verdad si se sabe
#: -la tabla da el error del foco contra ella- o None.
Z_REAL = 3.0

#: Solo al simular: "intensidad" (|1 + V|^2 = I/I_ref, lo que mide el sensor,
#: con imagen gemela) o "complejo" (1 + V, sin gemela).
CAMPO = "intensidad"

#: El montaje del laboratorio, CamposT/montaje.py: 528 nm, 1.83 um, L = 10 mm.
#: L -fuente a sensor- se mide una vez en el banco y NO se barre.
LAMB, DELTA, L = MONTAJE.lamb, MONTAJE.delta_x, MONTAJE.L

# ════════════════════════════════════════════════════════════════════════════
#  3. EL BARRIDO  (z = fuente -> muestra, con L fijo)
# ════════════════════════════════════════════════════════════════════════════

#: Grueso, en mm: paso 0.01 mm. Mover z mueve el foco Y la escala a la vez
#: (M = L/z), pero en la malla proyectada un detalle no cambia de pixel.
Z_MIN, Z_MAX, PASOS = 2.5, 3.5, 101
#: Barrido fino: +-1 paso grueso alrededor de cada pico distinto.
PASOS_FINO = 21

# ════════════════════════════════════════════════════════════════════════════
#  4. LA REGION MEDIDA
# ════════════════════════════════════════════════════════════════════════════

#: Donde miden las metricas, en pixeles de la malla del holograma:
#:     None                    la malla menos BORDE_PX por lado.
#:     True                    la arrastras con el raton sobre la referencia
#:                             (o sobre el holograma si no hay).
#:     (X0, Y0, ANCHO, ALTO)   ventana fija: X0 la columna e Y0 la fila de la
#:                             esquina superior izquierda.
#: Con el motor exacto el holograma entra ENTERO y solo se evalua la ventana.
#: Con el de Carlos la ventana es ademas el RECORTE del holograma, y tiene que
#: ser cuadrada y conviene centrada (ver MotorCarlos).
ROI = None

#: Pixeles por lado que no se miden con ROI = None. None = 3*min(M, N)//8, el
#: cuarto central del lado corto: medido el 2026-09-27 con la nitidez de
#: scripts/verif_dlhm_esferico.py, la ventana entera elegia el extremo del
#: barrido (3.30) y el cuarto central, 3.00. Ver la cabecera.
BORDE_PX = None

# ════════════════════════════════════════════════════════════════════════════
#  5. EL MOTOR
# ════════════════════════════════════════════════════════════════════════════

#: "exacto": CamposT.dlhm, la esferica exacta dentro del MPASM, el holograma
#:           entero. "carlos": reconstruir(), la cadena de reconstruction_dlhm.py,
#:           sobre el recorte del ROI. Ver la cabecera.
MOTOR = "exacto"
```

y a continuación, sin cambios de texto, las constantes que ya existían y que la cadena de Carlos usa: `DISPOSITIVO`, `DTYPE`, `MEMORIA_MAX_GB`, `SOBREMUESTREO`, `DE_LA_CONSTANTE`, `FILAS_POR_BLOQUE` (con sus comentarios `#:`; en los de `MEMORIA_MAX_GB` y `SOBREMUESTREO` añadir «Solo con MOTOR = "carlos".»). Al final del bloque:

```python
#: Carpeta raiz de la salida: <RESULTADOS>/<nombre>_<motor>_<objeto>[_roi...]
RESULTADOS = RAIZ / "resultados" / "mi_prueba_dlhm"
```

Se borran `COMPARAR`, `ROI_HOLOGRAMA`, `ROI_REFERENCIA`, `Z` y los `RUTA`/`REFERENCIA`/`LAMB`/`DELTA`/`L`/`PASOS` viejos.

- [ ] **Step 4: Borrar lo que queda sin uso y arreglar el texto de comprobar_memoria**

Borrar `referencia_a_escala()`, `observable()` y `_roi_de()`. En `comprobar_memoria`, cambiar `f"ROI_HOLOGRAMA, o usa DISPOSITIVO = 'cpu'."` por `f"ROI (cuadrado y centrado), o usa DISPOSITIVO = 'cpu'."` y `f"submuestrear no ayuda,\nhay que RECORTAR. Con ROI_HOLOGRAMA:\n\n"` por `f"submuestrear no ayuda,\nhay que RECORTAR. Con ROI:\n\n"`.

- [ ] **Step 5: Implementar la sección 8 y main()**

```python
# ════════════════════════════════════════════════════════════════════════════
#  8. EL OBJETO, EL HOLOGRAMA Y LA SALIDA
# ════════════════════════════════════════════════════════════════════════════

def leer_imagen(ruta):
    """Un .npy tal cual (real o complejo), o una imagen en gris a [0, 1] (/255)."""
    ruta = pathlib.Path(ruta)
    if not ruta.exists():
        raise SystemExit(f"No existe:\n    {ruta}")
    if ruta.suffix.lower() == ".npy":
        a = np.load(ruta)
    else:
        a = cv.imread(str(ruta), cv.IMREAD_GRAYSCALE)
        if a is None:
            raise SystemExit(f"No pude leer la imagen:\n    {ruta}")
        a = a.astype(np.float64) / 255.0
    if a.ndim != 2:
        raise SystemExit(f"{ruta} no es una imagen 2-D: forma {a.shape}.")
    return a


def objeto_de(img, objeto, invertir, fase_max):
    """La transmitancia t del objeto: g = 1 - img o img, y t = g o exp(i*fase_max*g)."""
    g = 1 - img if invertir else img
    if objeto == "fase":
        return np.exp(1j * fase_max * g)
    return g.astype(np.complex128)


def fondo_de(img):
    """La mediana del marco de la imagen: lo que hay alrededor del objeto."""
    return float(np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])))


def simular(t, lamb, delta, L_fuente, z, campo, device="auto", dtype=None):
    """El holograma de t con la ida exacta, en un sensor de la forma de t. En NumPy.

    t va en la malla que el sensor proyecta, paso delta*z/L: un pixel del objeto
    por pixel del sensor. "intensidad" es el de contraste |1 + V|^2 = I/I_ref;
    "complejo", 1 + V, sin imagen gemela.
    """
    h = a_cpu(dlhm.holograma(t, delta * z / L_fuente, lamb, z, L_fuente, t.shape,
                             delta, complejo=campo == "complejo", device=device,
                             dtype=dtype))
    return 1 + h if campo == "complejo" else h


def vista(a, objeto="amplitud"):
    """Lo que se dibuja o se guarda como PNG, a [0, 1].

    Un campo complejo: |a|^2, o angle(a) con OBJETO = "fase". Un holograma
    real, tal cual. Cada uno a su propio maximo (o rango).
    """
    a = np.asarray(a)
    if objeto == "fase" and np.iscomplexobj(a):
        v = np.angle(a)
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    v = np.abs(a) ** 2 if np.iscomplexobj(a) else a.astype(np.float64)
    m = v.max()
    return v / m if m > 0 else v


def carpeta_de_salida(raiz, ruta, referencia, z_real, campo, motor, objeto, roi):
    """-> (carpeta, nombre). carpeta = raiz/<nombre>_<motor>_<objeto>[_roi...].

    nombre es el del holograma, o sim_<referencia>_z<Z_REAL>mm_<CAMPO> si se
    simula: el simulado se guarda como <nombre>.npy, asi que leerlo despues con
    RUTA da el mismo nombre y escribe en la misma carpeta.
    """
    if ruta is None:
        nombre = f"sim_{pathlib.Path(referencia).stem}_z{z_real:g}mm_{campo}"
    else:
        nombre = pathlib.Path(ruta).stem
    carpeta = f"{nombre}_{motor}_{objeto}"
    if roi is not None:
        carpeta += f"_roi{roi.x0}-{roi.y0}-{roi.ancho}x{roi.alto}"
    return pathlib.Path(raiz) / carpeta, nombre


def guardar_instante(t, z, carpeta, formato, objeto):
    """La ventana reconstruida a z, en png, npy o los dos."""
    carpeta.mkdir(parents=True, exist_ok=True)
    t = a_cpu(t)
    nombre = f"z{z:08.4f}mm"
    if formato in ("npy", "ambos"):
        np.save(carpeta / f"{nombre}.npy", t)
    if formato in ("png", "ambos"):
        plt.imsave(carpeta / f"{nombre}.png", vista(t, objeto), cmap="gray",
                   vmin=0, vmax=1)


def informe_de_focos(zs, curvas, picos, z_foco, metricas, z_real=None, ancho=None):
    """La tabla de barrido_foco, como texto: grueso, foco, error, FWHM y margen.

    ancho es el que va a margen_del_pico(): el fondo es lo que queda a mas de
    ancho mm del pico. None = una decima del barrido, porque en DLHM el pico
    mide centesimas de mm y el 1 mm de barrido_foco se comeria el barrido.
    """
    paso = zs[1] - zs[0]
    ancho = (zs[-1] - zs[0]) / 10 if ancho is None else ancho
    lineas = [f"{'metrica':>9} {'grueso':>9} {'foco':>9} {'error':>9} {'FWHM':>9} "
              f"{'altura':>9} {'rizado':>9} {'alt/riz':>8}"]
    for n in metricas:
        c = curvas[n]
        w = anchura_media(zs, c)
        alt, riz, razon = margen_del_pico(zs, c, zs[picos[n]], ancho)
        error = f"{z_foco[n] - z_real:+9.4f}" if z_real is not None else f"{'-':>9}"
        lineas.append(f"{n:>9} {zs[picos[n]]:9.4f} {z_foco[n]:9.4f} {error} {w:9.4f} "
                      f"{alt:9.4f} {riz:9.4f} {razon:8.1f}")
    lineas.append(
        f"(error = foco - Z_REAL; FWHM del barrido grueso, no resuelve por debajo de "
        f"~{2 * paso:g} mm; altura sobre la mediana del fondo -a mas de {ancho:g} mm "
        f"del pico- y rizado punto a punto de ese fondo, en unidades de la propia "
        f"metrica: solo alt/riz se compara entre metricas)")
    return "\n".join(lineas)


def guardar_curvas(ruta, zs, curvas, finos, metricas):
    """Las curvas en crudo, siempre: volver a dibujarlas no debe exigir barrer."""
    volcado = {"zs": zs, **{f"grueso_{n}": curvas[n] for n in metricas}}
    for i, (zs_f, cf) in finos.items():
        volcado[f"fino_{i}_zs"] = zs_f
        volcado.update({f"fino_{i}_{n}": cf[n] for n in metricas})
    np.savez(ruta, **volcado)


def figura_barrido(zs, curvas, finos, z_foco, metricas, objeto, z_real=None, titulo=""):
    """Las curvas, cada una reescalada a [0, 1] entre su minimo y su maximo.

    Y no dividida por su maximo como en barrido_foco: con OBJETO = "fase" la
    Tamura cambiada de signo es negativa, y dividir por un maximo negativo le
    da la vuelta. Reescalar conserva donde esta el pico y su anchura, que es lo
    que se compara.
    """
    paso = zs[1] - zs[0]
    etiqueta = etiquetas(objeto)
    escala = {}
    for n in metricas:
        todas = np.concatenate([curvas[n]] + [cf[n] for _, cf in finos.values()])
        lo, hi = float(todas.min()), float(todas.max())
        escala[n] = (lo, hi - lo if hi > lo else 1.0)
    centro = z_real if z_real is not None else z_foco[metricas[0]]

    fig, (b1, b2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                 gridspec_kw={"width_ratios": [2, 1]})
    for n in metricas:
        lo, r = escala[n]
        b1.plot(zs, (curvas[n] - lo) / r, "o-", ms=3, lw=1.2, color=COLOR[n],
                label=etiqueta[n])
        for zs_f, cf in finos.values():
            b1.plot(zs_f, (cf[n] - lo) / r, ".-", ms=4, lw=1, color=COLOR[n], alpha=0.55)
        b1.axvline(z_foco[n], color=COLOR[n], lw=1, ls=":")
    if z_real is not None:
        b1.axvline(z_real, color="k", ls="--", lw=1.2, label=f"Z_REAL = {z_real:g} mm")
    b1.set_xlabel("z = fuente -> muestra [mm]")
    b1.set_ylabel("metrica reescalada a [0, 1]")
    b1.set_title("barrido: sin referencia vs con referencia\n"
                 "(punteado claro: barrido fino; vertical de color: foco elegido)",
                 fontsize=10)
    b1.legend(fontsize=8, loc="upper right")
    b1.grid(alpha=0.25)

    m = (zs > centro - 6 * paso) & (zs < centro + 6 * paso)
    for n in metricas:
        lo, r = escala[n]
        b2.plot(zs[m], (curvas[n][m] - lo) / r, "o-", ms=4, lw=1.2, color=COLOR[n],
                label=n)
        for zs_f, cf in finos.values():
            b2.plot(zs_f, (cf[n] - lo) / r, ".-", ms=4, lw=1, color=COLOR[n], alpha=0.55)
    if z_real is not None:
        b2.axvline(z_real, color="k", ls="--", lw=1.2)
    b2.set_xlim(centro - 6 * paso, centro + 6 * paso)
    b2.set_xlabel("z [mm]")
    donde = "Z_REAL" if z_real is not None else f"el foco de {metricas[0]}"
    b2.set_title(f"zoom +-{6 * paso:g} mm alrededor de {donde}", fontsize=10)
    b2.legend(fontsize=8)
    b2.grid(alpha=0.25)
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def figura_campos(paneles, forma, ventana, zoom_px, con_roi, titulo=""):
    """Cada panel entero y, debajo, su zoom sobre la ventana medida. En pixeles.

    paneles: [(imagen a [0, 1], (fila, columna) de su esquina en la malla,
    titulo)]. Un panel mas chico que la malla -la reconstruccion del motor de
    Carlos, que solo da su recorte- se dibuja en su sitio. Todos comparten la
    malla del holograma: en pixeles un detalle no cambia de sitio con z.
    """
    M, N = forma
    f0, c0, alto, ancho = ventana
    zoom = min(zoom_px, M, N)
    zf = min(max(f0 + alto // 2 - zoom // 2, 0), M - zoom)
    zc = min(max(c0 + ancho // 2 - zoom // 2, 0), N - zoom)
    fig, ejes = plt.subplots(2, len(paneles), figsize=(4.1 * len(paneles), 8.6),
                             squeeze=False)
    for k, (img, (pf, pc), tit) in enumerate(paneles):
        h, w = img.shape
        ext = [pc - 0.5, pc + w - 0.5, pf + h - 0.5, pf - 0.5]
        arriba, abajo = ejes[0][k], ejes[1][k]
        for ax in (arriba, abajo):
            ax.imshow(img, extent=ext, cmap="gray", vmin=0, vmax=1)
            ax.set_xlabel("x [px]")
        arriba.set_xlim(-0.5, N - 0.5)
        arriba.set_ylim(M - 0.5, -0.5)
        arriba.set_title(tit, fontsize=10)
        if con_roi:
            arriba.add_patch(plt.Rectangle((c0 - 0.5, f0 - 0.5), ancho, alto,
                                           fill=False, ec="tab:red", lw=1))
        abajo.set_xlim(zc - 0.5, zc + zoom - 0.5)
        abajo.set_ylim(zf + zoom - 0.5, zf - 0.5)
        abajo.set_title(f"zoom {'en el ROI' if con_roi else 'central'} {zoom}x{zoom} px",
                        fontsize=9)
    ejes[0][0].set_ylabel("y [px]")
    ejes[1][0].set_ylabel("y [px]")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  9. MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Conecta las constantes con las funciones. Devuelve {metrica: z de foco}."""
    comprobar_configuracion(METRICAS, FORMATO_BARRIDO, OBJETO, MOTOR, CAMPO, RUTA,
                            REFERENCIA, Z_REAL)
    xp_, dev = elegir_dispositivo(DISPOSITIVO)
    dtype = DTYPE or (np.complex64 if dev == "gpu" else np.complex128)
    zs, paso, z_rango = rango_del_barrido(Z_MIN, Z_MAX, PASOS, L)

    # --- el objeto y el holograma -------------------------------------------
    t_ref = None
    if REFERENCIA is not None:
        img = leer_imagen(REFERENCIA)
        t_ref = objeto_de(img, OBJETO, INVERTIR, FASE_MAX)
        g_fondo = 1 - fondo_de(img) if INVERTIR else fondo_de(img)
        if RUTA is None and OBJETO == "fase" and abs(g_fondo) > 0.1:
            print(f"AVISO: el fondo del objeto vale g = {g_fondo:.2f}, o sea fase "
                  f"{FASE_MAX * g_fondo:.2f} rad, y fuera de la ventana el objeto es "
                  f"transparente:\n  el borde de la ventana es un escalon de fase. "
                  f"Cambia INVERTIR para dejar el fondo en g = 0.")
    t0 = time.perf_counter()
    if RUTA is None:
        holo = simular(t_ref, LAMB, DELTA, L, Z_REAL, CAMPO, dev, dtype)
        fondo = None
        print(f"holograma simulado ({CAMPO}) de {pathlib.Path(REFERENCIA).name} a "
              f"z = {Z_REAL:g} mm: {holo.shape[0]}x{holo.shape[1]} en "
              f"{time.perf_counter() - t0:.1f} s")
    else:
        holo = leer_imagen(RUTA)
        fondo = "media" if FONDO is None else leer_imagen(FONDO)
        if FONDO is not None and fondo.shape != holo.shape:
            raise SystemExit(f"FONDO es de {fondo.shape} y el holograma de {holo.shape}.")
        print(f"holograma {RUTA}: {holo.shape[0]}x{holo.shape[1]}")
    if t_ref is not None and t_ref.shape != holo.shape:
        raise SystemExit(
            f"La referencia es de {t_ref.shape[0]}x{t_ref.shape[1]} y el holograma de "
            f"{holo.shape[0]}x{holo.shape[1]}. Tiene que ser el objeto en la malla que "
            f"el sensor proyecta,\nun pixel por pixel del sensor: si no lo es, pon "
            f'REFERENCIA = None y METRICAS = ("tamura",).')
    M, N = holo.shape

    # --- la region medida y la salida ---------------------------------------
    imagen = vista(t_ref if t_ref is not None else holo, OBJETO) if ROI is True else None
    ventana, roi = region_medida((M, N), ROI, BORDE_PX, imagen,
                                 "arrastra la region donde enfocar")
    f0, c0, alto, ancho = ventana
    if roi is not None:
        print(f"ROI = ({roi.x0}, {roi.y0}, {roi.ancho}, {roi.alto})   # repite esta ventana")
    salida, nombre = carpeta_de_salida(RESULTADOS, RUTA, REFERENCIA, Z_REAL, CAMPO,
                                       MOTOR, OBJETO, roi)
    salida.mkdir(parents=True, exist_ok=True)
    if RUTA is None:
        np.save(salida / f"{nombre}.npy", holo)
        print(f"  -> {salida / f'{nombre}.npy'}  (ponlo en RUTA para no volver a simular)")

    print(f"  lambda {LAMB * 1e6:.0f} nm | delta {DELTA * 1e3:.2f} um | L = {L:g} mm | "
          f"objeto {OBJETO} | motor {MOTOR}")
    print(f"  grueso: {PASOS} pasos de {Z_MIN:g} a {Z_MAX:g} mm (paso {paso:g} mm, "
          f"M = L/z de {L / Z_MAX:.3f} a {L / Z_MIN:.3f}); fino: {PASOS_FINO} pasos "
          f"(paso {2 * paso / (PASOS_FINO - 1):.3g} mm)")
    print(f"  dispositivo {dev.upper()} | dtype {np.dtype(dtype).name} | fase en float64")

    # --- el motor -----------------------------------------------------------
    t0 = time.perf_counter()
    if MOTOR == "exacto":
        motor = MotorExacto(holo, LAMB, DELTA, L, z_rango, fondo=fondo, device=dev,
                            dtype=dtype)
        sincronizar(motor.xp)
        print(f"  motor exacto: {motor.describir()}; construido en "
              f"{time.perf_counter() - t0:.1f} s")
    else:
        if abs(f0 + alto / 2 - M / 2) > 1 or abs(c0 + ancho / 2 - N / 2) > 1:
            print(f"AVISO: el ROI no esta centrado en la malla {M}x{N}. El motor de "
                  f"Carlos pone la fuente en el centro\n  del recorte: su fisica ya no "
                  f"es la del banco.")
        motor = MotorCarlos(holo, ventana, LAMB, DELTA, L, xp_, dtype, SOBREMUESTREO)
        comprobar_memoria(motor.h, z_rango, LAMB, DELTA, MEMORIA_MAX_GB, xp_, dtype)
        print(f"  motor de Carlos: {motor.describir(z_rango)}")
        print(f"  reconstruir() vs la cadena intacta de dlhm.py: "
              f"{comprobar_equivalencia(xp_, dtype):.2e}")

    # --- la referencia de las metricas --------------------------------------
    clase_ref = clase_de("pearson", OBJETO)
    ref = (None if t_ref is None
           else mapa(t_ref[f0:f0 + alto, c0:c0 + ancho], clase_ref, np))
    fijar_referencia(ref, motor.xp)
    print(f"metricas: {', '.join(METRICAS)} sobre {alto}x{ancho} px de {M}x{N} "
          f"(x {c0}-{c0 + ancho}, y {f0}-{f0 + alto} px)")

    # --- el barrido ---------------------------------------------------------
    guardar_en = None
    if GUARDAR_BARRIDO:
        def guardar_en(sub):
            return lambda t, z: guardar_instante(t, z, salida / sub, FORMATO_BARRIDO,
                                                 OBJETO)
    reloj = {}
    sincronizar(motor.xp)
    t0 = time.perf_counter()
    curvas, picos, finos, z_foco = enfocar(motor, ventana, zs, PASOS_FINO, METRICAS,
                                           OBJETO, en_cpu=EN_CPU, solapar=SOLAPAR,
                                           reloj=reloj, guardar_en=guardar_en)
    sincronizar(motor.xp)
    t_barrido = time.perf_counter() - t0
    for n, i in picos.items():
        if i in (0, len(zs) - 1):
            print(f"AVISO: el maximo de {n} cae en el extremo z = {zs[i]:g} mm; el "
                  f"barrido no acota ese foco.")

    # --- la reconstruccion en cada foco elegido -----------------------------
    z_unicos = sorted({round(z, 6) for z in z_foco.values()})
    quien = {z: [n for n in METRICAS if round(z_foco[n], 6) == z] for z in z_unicos}
    recon = {z: a_cpu(motor.campo(z, motor.ventana_maxima)) for z in z_unicos}
    liberar(motor.xp)

    # --- lo que se sabe -----------------------------------------------------
    print()
    print(informe_de_focos(zs, curvas, picos, z_foco, METRICAS, Z_REAL))
    print()
    n_rec = len(zs) + PASOS_FINO * len(finos)
    print(f"{n_rec} reconstrucciones en {t_barrido:.1f} s"
          + (" (incluye guardar cada z)" if GUARDAR_BARRIDO else ""))
    print("  reparto: " + ", ".join(f"{k} {v:.1f} s ({v / n_rec * 1e3:.0f} ms/z)"
                                     for k, v in reloj.items()))
    guardar_curvas(salida / "curvas.npz", zs, curvas, finos, METRICAS)
    print("->", salida / "curvas.npz")

    # --- figuras ------------------------------------------------------------
    cabeza = (f"{nombre}  ·  motor {MOTOR}  ·  objeto {OBJETO}  ·  lambda "
              f"{LAMB * 1e6:.0f} nm  ·  delta {DELTA * 1e3:.2f} um  ·  L = {L:g} mm  ·  {M}x{N}")
    fig1 = figura_barrido(zs, curvas, finos, z_foco, METRICAS, OBJETO, Z_REAL, cabeza)
    paneles = []
    if t_ref is not None:
        paneles.append((vista(t_ref, OBJETO), (0, 0),
                        f"referencia ({'fase' if OBJETO == 'fase' else '|t|^2'})"))
    paneles.append((vista(holo), (0, 0), "holograma (entrada)"))
    fv, cv_, _, _ = motor.ventana_maxima
    for z in z_unicos:
        paneles.append((vista(recon[z], OBJETO), (fv, cv_),
                        f"reconstruido a {z:.4f} mm\n" + " + ".join(quien[z])))
    fig2 = figura_campos(paneles, (M, N), ventana, ZOOM_PX, roi is not None,
                         f"Campos  ·  {cabeza}  ·  cada panel a su propio rango")
    if GUARDAR_FIGURA:
        for nombre_fig, f in (("barrido_metricas", fig1), ("campos", fig2)):
            f.savefig(salida / f"{nombre_fig}.png", dpi=110)
            print("->", salida / f"{nombre_fig}.png")
    if MOSTRAR:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)
    return z_foco


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Reescribir la cabecera**

Sustituir el docstring del módulo por uno que diga, en este orden: QUÉ HACE (la selección de `barrido_foco` sobre DLHM, con referencia a `barrido_foco.py`); LOS DOS MOTORES (qué es cada uno, por qué el exacto por defecto, qué exige el de Carlos); LA MALLA ES LA DEL SENSOR PROYECTADA (la invariancia en píxeles); OBJETO = "fase"; EL 0.03 DE ANTES, EXPLICADO (los números de la Task 1 y las cinco hipótesis descartadas, resumidas); VALIDADO CONTRA reconstruction_dlhm.py (el bloque actual, con el OJO de la malla par y el del 1e-6); EL ROI DEL MOTOR DE CARLOS (la tabla de memoria actual); EL MODELO DE REFERENCIA SOLO GENERA EN UNA CONFIGURACIÓN (el bloque actual sobre `dlhm()` de Carlos, íntegro); y UNIDADES. La sección «LO QUE SALIÓ» se añade en la Task 6 con los números a escala real.

- [ ] **Step 7: Correr la suite nueva y las dos viejas del script**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py tests/test_gpu_mi_prueba.py tests/test_malla_remuestreada.py -q`
Expected: todo PASS.

- [ ] **Step 8: Checkpoint** — sin commit.

---

### Task 6: Verificación a escala real

Con los valores por defecto (BenchmarkTarget 3000x4000, z = 3 mm, montaje, GPU). No toca el repo salvo la cabecera y la especificación al final.

**Files:**
- Create (scratch): `<scratchpad>/verificar_escala_real.py`
- Modify: `scripts/mi_prueba_dlhm.py` (cabecera: «LO QUE SALIÓ»), `docs/superpowers/specs/2026-09-28-mi-prueba-dlhm-seleccion-design.md` («Estado»)

- [ ] **Step 1: Escribir el conductor**

```python
"""Corre main() de mi_prueba_dlhm con variantes, sin editar el archivo."""
import sys

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, r"C:\Users\User\Desktop\Tesis")
from scripts import mi_prueba_dlhm as mp

SIM = (mp.RESULTADOS / "sim_BenchmarkTarget_z3mm_intensidad_exacto_amplitud"
       / "sim_BenchmarkTarget_z3mm_intensidad.npy")
CENTRADO = (1488, 988, 1024, 1024)        # 1024 centrado en 3000x4000
mp.MOSTRAR = False
variante = sys.argv[1]
if variante == "defecto":
    pass
elif variante == "exacto_roi":
    mp.RUTA, mp.ROI = SIM, CENTRADO
elif variante == "carlos":
    mp.RUTA, mp.ROI, mp.MOTOR = SIM, CENTRADO, "carlos"
elif variante == "borde0":
    mp.RUTA, mp.BORDE_PX, mp.METRICAS = SIM, 0, ("tamura",)
elif variante == "fase":
    mp.OBJETO, mp.INVERTIR = "fase", False
else:
    raise SystemExit(variante)
print(mp.main())
```

- [ ] **Step 2: Correr las cinco variantes y anotar**

Run, una a una (la primera simula y guarda el holograma que usan las demás):
`Tesis_env/Scripts/python.exe <scratchpad>/verificar_escala_real.py defecto`, luego `exacto_roi`, `carlos`, `borde0`, `fase`.

Expected, según la especificación (Verificación, punto 3):
- `defecto`: las tres métricas a menos de medio paso grueso de 3.000 mm; se anotan FWHM y alt/riz y el tiempo.
- `exacto_roi` y `carlos`: el foco de las dos sobre la misma ventana de 1024 y el tiempo de cada una.
- `borde0`: la Tamura sobre la malla entera frente al cuarto central de `defecto` (D9).
- `fase`: la Tamura invertida, Pearson y SSIM en 3.000 mm.

Si el motor exacto falla, no se toca ningún umbral: se busca por qué (la especificación lo pide así).

- [ ] **Step 3: Escribir «LO QUE SALIÓ» en la cabecera y cerrar la especificación**

Añadir a la cabecera de `scripts/mi_prueba_dlhm.py` la tabla de cada variante con sus números medidos (foco, error, FWHM, alt/riz, tiempo), y en la especificación cambiar «Estado» a «Implementado el 2026-09-28; verificación a escala real en la cabecera del script».

- [ ] **Step 4: Suite completa**

Run: `Tesis_env/Scripts/python.exe -m pytest -q`
Expected: todo PASS (la suite entera, no solo las tres del script).

- [ ] **Step 5: Checkpoint** — sin commit. Proponer al usuario los commits (en rama nueva): spec + plan, script + pruebas; y la fila del cronograma si la quiere.
