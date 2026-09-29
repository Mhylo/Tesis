# La emisión de la fuente en la ida DLHM — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que la ida exacta de `CamposT/dlhm.py` pueda simular la emisión real de la fuente (el Airy del pinhole de 5 µm o cualquier a(sin θ)), que la reconstrucción la deshaga con un fondo y un piso, y que `scripts/mi_prueba_dlhm.py` lo use con `FUENTE`/`PISO` y mida por defecto sobre el grupo USAF central.

**Architecture:** La fuente es una amplitud a(s), s = sin θ, que apodiza la ida por dentro: multiplica el objeto por a(s_z) antes de propagar y la referencia del sensor pasa a ser a(s_L), así que la intensidad es |a_L + V|². `iluminacion()` da |a_L|², el fondo simulado. El `Reconstructor` gana `piso`: donde |fondo| < piso·max|fondo| pone c − 1 = 0 en vez de dividir. `mi_prueba_dlhm.py` solo cablea eso y cambia su ROI por defecto.

**Tech Stack:** Python 3.12 (`Tesis_env`), NumPy, SciPy (`scipy.special.j1`), CuPy opcional, pytest, OpenCV (solo en el script).

**Spec:** `docs/superpowers/specs/2026-09-29-fuente-dlhm-design.md` (D1-D8).

## Global Constraints

- Unidades: milímetros para todo. λ = 528e-6, δ = 1.83e-3, L = 10, z = 3; el pinhole es `MONTAJE.pinhole = 5e-3` (5 µm de diámetro).
- `fuente=None` (y `piso=0.0`) deja todo idéntico a hoy: la suite actual tiene que seguir verde sin tocar ninguna de sus pruebas.
- a es una AMPLITUD con signo, con a(0) = 1. No se confunde con intensidad.
- `FUENTE` va apagada por defecto (`None`): la NA real no está medida.
- No se toca `scripts/barrido_foco.py` ni `scripts/mpasm_minimo.py`: tienen trabajo del usuario sin commitear (tarea 77). No se hace `git add` de nada que no sea de este plan.
- No se tocan las seis funciones copiadas de Carlos en la sección 2 de `mi_prueba_dlhm.py` (`ifts`, `fts`, `resize`, `point_src`, `angular_spectrum`, `dlhm`) ni las métricas copiadas de `barrido_foco.py`.
- Comentarios y docstrings en el estilo del archivo que se edita: `CamposT/dlhm.py` con tildes; `scripts/mi_prueba_dlhm.py` y `tests/test_mi_prueba_dlhm_seleccion.py` en ASCII sin tildes; el cronograma en ASCII puro.
- Python del proyecto: `Tesis_env/Scripts/python.exe` (desde la raíz del repo, en Git Bash). El `python` del sistema no tiene cv2.
- Rama: `fuente-dlhm`. Mensajes de commit en español, terminando con la línea `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Mapa de archivos

- `CamposT/dlhm.py` — la física: `fuente_pinhole`, `_comprobar_fuente`, `_evaluar_fuente`, `iluminacion` (nuevas), `holograma(..., fuente=)`, `_contraste(..., piso)`, `Reconstructor(..., piso=)`, y la sección «LA FUENTE» del docstring.
- `tests/test_dlhm.py` — `rs_directo(..., emision=)` y las pruebas de la fuente y del piso.
- `scripts/mi_prueba_dlhm.py` — `FUENTE`, `PISO`, `ROI` por defecto, `fuente_de`, `sufijo_fuente`, `contraste_enmascarado`, `zona_iluminada`, y el cableado en `comprobar_configuracion`, `simular`, `carpeta_de_salida`, `MotorExacto` y `main`.
- `tests/test_mi_prueba_dlhm_seleccion.py` — las pruebas de lo anterior.
- `docs/superpowers/specs/2026-09-29-fuente-dlhm-design.md`, `docs/Cronograma_TG_DLHM.xlsx` — el resultado a escala real (Task 8).

---

### Task 1: La fuente como función: `fuente_pinhole`, sus guardas e `iluminacion`

**Files:**
- Modify: `CamposT/dlhm.py` (el `__all__` y una sección nueva antes de `# ---...--- la DFT con el chirp dentro`)
- Test: `tests/test_dlhm.py` (sección nueva antes de `# ---...--- la ida`)

**Interfaces:**
- Produces:
  - `fuente_pinhole(diametro: float, lamb: float) -> Callable[[np.ndarray], np.ndarray]` — a(s) = 2·J1(x)/x con x = π·diametro·s/lamb; a(0) = 1.
  - `_comprobar_fuente(fuente) -> None` — `ValueError` si a(0) ≠ 1 (tolerancia 1e-6), no finita o de otra forma.
  - `_evaluar_fuente(fuente, y, x, R) -> np.ndarray` — a(s) en la malla (y, x) a distancia R, s = ρ/sqrt(ρ² + R²), forma (y.size, x.size); `ValueError` si la forma no casa o hay no finitos.
  - `iluminacion(forma, delta, L, fuente, centro=(0.0, 0.0)) -> np.ndarray` — |a(s_L)|² en el sensor, float64 NumPy; unos con `fuente=None`.

- [ ] **Step 1: Write the failing tests**

En `tests/test_dlhm.py`, justo ANTES de la línea `# ------------------------------------------------------------------ la ida`, añadir:

```python
# ------------------------------------------------------------------ la fuente
#: Primer cero de 2 J1(x)/x: x = 3.8317..., o sea s = 1.2197 lamb/d.
X_CERO = 3.8317059702075125


def test_fuente_pinhole_es_el_airy_de_amplitud():
    """a(0) = 1, el primer cero donde dice la teoría y 2 J1(x)/x en puntos
    sueltos, con el signo: entre el primer cero y el segundo es negativa."""
    from scipy.special import j1
    d = 5e-3
    a = dlhm.fuente_pinhole(d, LAMB)
    assert a(np.zeros(1))[0] == 1.0
    s0 = X_CERO * LAMB / (np.pi * d)
    assert s0 / (LAMB / d) == pytest.approx(1.2197, abs=1e-4)
    assert abs(a(np.array([s0]))[0]) < 1e-12
    s = np.array([0.01, 0.05, 0.1, 0.2, 0.3])
    x = np.pi * d * s / LAMB
    np.testing.assert_allclose(a(s), 2 * j1(x) / x, rtol=1e-12)
    assert a(np.array([0.2]))[0] < 0


def test_fuente_pinhole_rechaza_un_diametro_no_positivo():
    with pytest.raises(ValueError, match="diametro"):
        dlhm.fuente_pinhole(0.0, LAMB)


def test_iluminacion_es_el_modulo_al_cuadrado_de_a_en_el_sensor():
    a = dlhm.fuente_pinhole(5e-3, LAMB)
    X0 = (0.7, 1.9)
    Ys = (np.arange(16) - 8) * DELTA + X0[0]
    Xs = (np.arange(24) - 12) * DELTA + X0[1]
    rho = np.hypot(Xs[None, :], Ys[:, None])
    esperado = a(rho / np.sqrt(rho**2 + L**2)) ** 2
    np.testing.assert_allclose(dlhm.iluminacion((16, 24), DELTA, L, a, centro=X0),
                               esperado, rtol=1e-12)
    np.testing.assert_array_equal(dlhm.iluminacion((16, 24), DELTA, L, None),
                                  np.ones((16, 24)))


@pytest.mark.parametrize("fuente, mensaje", [
    (lambda s: 0.5 * np.ones_like(s), r"a\(0\) = 1"),
    (lambda s: np.ones(3), "forma"),
    (lambda s: np.where(s > 1e-3, np.nan, 1.0), "finit")])
def test_una_fuente_mal_hecha_aborta(fuente, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        dlhm.iluminacion((16, 16), DELTA, L, fuente, centro=(0.7, 1.9))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_dlhm.py -q -p no:cacheprovider -k "fuente or iluminacion"`
Expected: FAIL con `AttributeError: module 'CamposT.dlhm' has no attribute 'fuente_pinhole'` (y `iluminacion`).

- [ ] **Step 3: Implement**

En `CamposT/dlhm.py`, cambiar el `__all__`:

```python
__all__ = ["magnificacion", "distancia_equivalente", "coordenadas", "esferica",
           "fuente_pinhole", "iluminacion", "ventana_chirp", "holograma",
           "Reconstructor", "reconstruir", "reconstruir_escalado", "memoria",
           "memoria_holograma"]
```

y justo ANTES de la línea `# ------------------------------------------------------ la DFT con el chirp dentro` insertar:

```python
# ------------------------------------------------------------------ la fuente
def fuente_pinhole(diametro, lamb):
    """La emisión de un pinhole circular iluminado uniforme, en campo lejano:
    a(s) = 2·J1(x)/x con x = pi·diametro·s/lamb y s = sin(theta).

    a(0) = 1 y es una AMPLITUD con signo: entre el primer cero, s =
    1.2197·lamb/diametro, y el segundo, a es negativa. Con el pinhole del banco
    (5 um, 528 nm) el primer cero cae a sin(theta) = 0.1288. Es campo lejano ya
    en la muestra: el número de Fresnel del pinhole a z = 3 mm es 0.004.
    """
    if not diametro > 0:
        raise ValueError(f"diametro = {diametro!r}: tiene que ser > 0")
    from scipy.special import j1
    c = np.pi * diametro / lamb

    def a(s):
        x = c * np.asarray(s, dtype=np.float64)
        out = np.ones_like(x)
        m = np.abs(x) > 1e-8                  # por debajo, 1 - x²/8 = 1 en float64
        out[m] = 2 * j1(x[m]) / x[m]
        return out
    return a


def _comprobar_fuente(fuente):
    """a(0) = 1, finita y con la forma de su entrada, o ValueError."""
    a0 = np.asarray(fuente(np.zeros(1)))
    if a0.shape != (1,) or not np.all(np.isfinite(a0)) or abs(a0[0] - 1) > 1e-6:
        raise ValueError(f"la fuente tiene que dar a(0) = 1 (finita, con la forma de "
                         f"su entrada), y fuente(zeros(1)) = {a0!r}")


def _evaluar_fuente(fuente, y, x, R):
    """a(s) sobre la malla (y, x) a distancia R de la fuente, s = rho/sqrt(rho² + R²).

    En NumPy y float64: a varía despacio frente a la banda del objeto y se sube
    al dispositivo ya evaluada.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    rho = np.hypot(x[None, :], y[:, None])
    s = rho / np.sqrt(rho * rho + R * R)
    a = np.asarray(fuente(s))
    if a.shape != s.shape:
        raise ValueError(f"la fuente devolvió forma {a.shape} para una entrada {s.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("la fuente devolvió valores no finitos")
    return a


def iluminacion(forma, delta, L, fuente, centro=(0.0, 0.0)):
    """|a(s_L)|² en el sensor: la imagen SIN muestra, en la normalización del
    holograma (dividida por |e^{ikR}/R|²). Es el fondo por el que se divide
    para reconstruir. Unos con fuente=None. En NumPy, float64.

    No lleva lamb: la función fuente ya la trae dentro.
    """
    ny, nx = forma
    if fuente is None:
        return np.ones((ny, nx))
    _comprobar_fuente(fuente)
    (dy, dx), (cy, cx) = _por_eje(delta), _por_eje(centro)
    a = _evaluar_fuente(fuente, coordenadas(ny, dy, cy), coordenadas(nx, dx, cx), L)
    return np.abs(a) ** 2
```

Nota: `iluminacion` llama a `_comprobar_fuente` antes de evaluar, así que la prueba de `a(0) = 0.5` aborta con el mensaje `a(0) = 1`, la de `np.ones(3)` en `_comprobar_fuente` (forma `(3,)` ≠ `(1,)`, mensaje con «forma») y la del NaN en `_evaluar_fuente` («no finitos»).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_dlhm.py -q -p no:cacheprovider -k "fuente or iluminacion"`
Expected: PASS (6 pruebas: 3 + las 3 parametrizadas).

- [ ] **Step 5: Commit**

```bash
git add CamposT/dlhm.py tests/test_dlhm.py
git commit -m "dlhm: la fuente como amplitud a(sin theta), el Airy del pinhole y el fondo que ilumina

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: La ida con la fuente dentro: `holograma(..., fuente=)`

**Files:**
- Modify: `CamposT/dlhm.py` (`holograma`: firma, docstring, evaluación, `J`, salida)
- Test: `tests/test_dlhm.py` (`rs_directo` y pruebas nuevas al final de la sección «la ida»; prueba GPU al final)

**Interfaces:**
- Consumes (Task 1): `fuente_pinhole`, `_comprobar_fuente`, `_evaluar_fuente`, `iluminacion`.
- Produces: `holograma(t, paso, lamb, z, L, forma, delta, *, centro_objeto=(0.0, 0.0), centro=(0.0, 0.0), banda=None, guarda=4.0, margen=1.1, complejo=False, fuente=None, device="auto", dtype=None)`. Con `fuente`: `complejo=True` → V (campo del objeto relativo a la esférica ideal S_L); `complejo=False` → |a_L + V|².

- [ ] **Step 1: Write the failing tests**

En `tests/test_dlhm.py`, cambiar `rs_directo` para que acepte la emisión (el parámetro se llama `emision` porque `fuente` ya es una variable local de la función):

```python
def rs_directo(t, paso, centro_obj, Ys, Xs, zm=Z, Lm=L, emision=None):
    """Rayleigh-Sommerfeld I directo (Lopera, Ec. 1), campo de t - 1 sobre la
    malla (Ys, Xs) del sensor, dividido por la esférica en el sensor. zm y Lm,
    los del montaje salvo que se pidan otros. emision = a(sin theta): pesa cada
    punto del objeto con la amplitud que la fuente le manda."""
    n = t.shape[0]
    g = (np.arange(n) - n / 2) * paso
    xs = (g + centro_obj[1])[None, :].repeat(n, 0).ravel()
    ys = (g + centro_obj[0])[:, None].repeat(n, 1).ravel()
    fuente = (t - 1).ravel()
    m = np.abs(fuente) > 1e-12
    xs, ys, fuente = xs[m], ys[m], fuente[m]
    rho = np.sqrt(xs**2 + ys**2 + zm**2)
    fuente = fuente * np.exp(1j * K * rho) / rho * paso**2
    if emision is not None:
        fuente = fuente * emision(np.hypot(xs, ys) / rho)
    XX, YY = np.meshgrid(Xs, Ys)
    U = np.zeros(XX.size, complex)
    dm = Lm - zm
    for i in range(0, xs.size, 256):
        dx = XX.ravel()[None, :] - xs[i:i + 256, None]
        dy = YY.ravel()[None, :] - ys[i:i + 256, None]
        r = np.sqrt(dx * dx + dy * dy + dm * dm)
        U += fuente[i:i + 256] @ (dm / (2 * np.pi * r * r) * (1 / r - 1j * K) * np.exp(1j * K * r))
    r = np.sqrt(XX**2 + YY**2 + Lm**2)
    return U.reshape(XX.shape) / (np.exp(1j * K * r) / r)
```

(Es la función de hoy con el parámetro nuevo y las dos líneas `if emision is not None:`; el resto no cambia.)

Justo DESPUÉS de `test_holograma_de_contraste_es_el_modulo_de_uno_mas_V` (antes de `# ---...--- la vuelta`), añadir:

```python
@pytest.mark.parametrize("X0", [(0.0, 0.0), (0.7, 1.9)])
def test_la_ida_con_fuente_coincide_con_rayleigh_sommerfeld(X0):
    """El Airy del pinhole de 5 um dentro de la ida, contra RS directo con la
    misma emisión. Fuera del eje el objeto cae a sin(theta) = 0.199, pasado el
    primer cero (0.1288): a vale -0.099 y un signo o una coordenada mal
    puestos no pasan (la ida SIN fuente contra esta RS da 11.1, medido).
    Medido con la implementación: 1.1e-5 en el eje y 4.6e-6 fuera."""
    a = dlhm.fuente_pinhole(5e-3, LAMB)
    centro_obj = (X0[0] / M, X0[1] / M)
    t = manchas(40, DO, [(0.0, -3e-3), (4e-3, 4e-3), (-4e-3, 2e-3)], 1e-3)
    n_s = 24
    V = dlhm.holograma(t, DO, LAMB, Z, L, (n_s, n_s), 4 * DELTA, centro_objeto=centro_obj,
                       centro=X0, complejo=True, fuente=a, device="cpu")
    Ys = (np.arange(n_s) - n_s / 2) * 4 * DELTA + X0[0]
    Xs = (np.arange(n_s) - n_s / 2) * 4 * DELTA + X0[1]
    ref = rs_directo(t, DO, centro_obj, Ys, Xs, emision=a)
    assert error_relativo(a_numpy(V), ref) < 1e-4


def test_con_fuente_la_intensidad_es_el_modulo_de_a_mas_V():
    """La referencia en el sensor ya no es 1 sino a_L: I/|S_L|² = |a_L + V|².
    Y sin objeto (t = 1, V = 0) lo que queda es exactamente iluminacion()."""
    a = dlhm.fuente_pinhole(5e-3, LAMB)
    X0 = (0.7, 1.9)
    t = manchas(32, DO, [(0.0, 0.0)], 3e-3)
    kw = dict(centro_objeto=(X0[0] / M, X0[1] / M), centro=X0, fuente=a, device="cpu")
    V = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (32, 32), DELTA, complejo=True, **kw))
    c = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (32, 32), DELTA, **kw))
    Ys = (np.arange(32) - 16) * DELTA + X0[0]
    Xs = (np.arange(32) - 16) * DELTA + X0[1]
    rho = np.hypot(Xs[None, :], Ys[:, None])
    a_L = a(rho / np.sqrt(rho**2 + L**2))
    np.testing.assert_allclose(c, np.abs(a_L + V) ** 2, rtol=1e-12)
    vacio = a_numpy(dlhm.holograma(np.ones((32, 32)), DO, LAMB, Z, L, (32, 32), DELTA, **kw))
    np.testing.assert_allclose(vacio, dlhm.iluminacion((32, 32), DELTA, L, a, centro=X0),
                               rtol=1e-12)


def test_una_fuente_constante_da_lo_mismo_que_ninguna():
    t = manchas(32, DO, [(0.0, 0.0)], 3e-3)
    uno = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (32, 32), DELTA,
                                 fuente=lambda s: np.ones_like(s), device="cpu"))
    nada = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (32, 32), DELTA, device="cpu"))
    np.testing.assert_allclose(uno, nada, rtol=1e-12)


def test_la_ida_con_una_fuente_mal_hecha_aborta():
    t = manchas(16, DO, [(0.0, 0.0)], 3e-3)
    with pytest.raises(ValueError, match=r"a\(0\) = 1"):
        dlhm.holograma(t, DO, LAMB, Z, L, (16, 16), DELTA,
                       fuente=lambda s: 2 * np.ones_like(s), device="cpu")
```

Al final del archivo, DESPUÉS de `test_gpu_da_lo_mismo_que_cpu` y antes de `if __name__ == "__main__":`, añadir:

```python
@pytest.mark.gpu
@pytest.mark.skipif(not gpu_disponible(), reason="sin GPU CUDA")
def test_gpu_da_lo_mismo_que_cpu_con_fuente():
    """La fuente se evalúa en la CPU en float64 y sube ya evaluada: en la GPU
    solo cambia la mantisa. Medido: 1.6e-6."""
    a = dlhm.fuente_pinhole(5e-3, LAMB)
    X0 = (0.7, 1.9)
    t = manchas(64, DO, [(0.0, 0.0), (0.01, -0.01)], 3e-3)
    kw = dict(centro_objeto=(X0[0] / M, X0[1] / M), centro=X0, fuente=a)
    cpu = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (64, 64), DELTA, device="cpu", **kw))
    gpu = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (64, 64), DELTA, device="gpu", **kw))
    fondo = dlhm.iluminacion((64, 64), DELTA, L, a, centro=X0)
    assert error_relativo(gpu - fondo, cpu - fondo) < 1e-4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_dlhm.py -q -p no:cacheprovider -k "con_fuente or fuente_constante or fuente_mal"`
Expected: FAIL con `TypeError: holograma() got an unexpected keyword argument 'fuente'`.

- [ ] **Step 3: Implement**

En `CamposT/dlhm.py`, en `holograma`:

1. La firma:

```python
def holograma(t, paso, lamb, z, L, forma, delta, *, centro_objeto=(0.0, 0.0),
              centro=(0.0, 0.0), banda=None, guarda=4.0, margen=1.1, complejo=False,
              fuente=None, device="auto", dtype=None):
```

2. En el docstring, reemplazar el párrafo de `complejo` (desde `    complejo False: el holograma de contraste` hasta `el campo del objeto relativo a la referencia, sin imagen gemela.`) por:

```
    complejo False: el holograma de contraste |1 + V|² = I/I_ref, que es lo
             que mide el sensor dividido por el fondo. True: V = U_obj/S_L,
             el campo del objeto relativo a la esférica ideal, sin imagen
             gemela.
    fuente   None: fuente puntual ideal, isótropa (lo de siempre). O una
             función a(s), s = sin(theta), con a(0) = 1: la AMPLITUD que la
             fuente emite en cada dirección (fuente_pinhole() da el Airy de
             un pinhole). Apodiza la ida por dentro: el objeto se ilumina con
             a(s_z)·e^{ikr}/r y la referencia en el sensor es a(s_L)·S_L, así
             que complejo=False da |a_L + V|² (= I/|S_L|²) y complejo=True da V.
             iluminacion() es |a_L|², el fondo. Supone que a varía despacio
             frente a la banda del objeto (el Airy de 5 um, 1.6 mm^-1 en la
             muestra, frente a 911): no se comprueba.
```

3. Justo DESPUÉS de la línea `    Ys, Xs = coordenadas(ny, dy, cy), coordenadas(nx, dx, cx)` (y antes de `    rho_max = ...`), insertar la evaluación, que aborta antes de la ida si la fuente está mal hecha:

```python
    if fuente is not None:
        _comprobar_fuente(fuente)
        a_obj = _evaluar_fuente(fuente, yo, xo, z)
        a_sen = _evaluar_fuente(fuente, Ys, Xs, L)
```

4. Justo DESPUÉS de la línea `    J = a_dispositivo(t - 1, xp, dtype) * esferica(yo, xo, z, lamb, xp, dtype, parte="residuo")`, insertar:

```python
    if fuente is not None:
        J *= a_dispositivo(a_obj, xp, dtype)
```

5. Reemplazar las dos últimas líneas de la función:

```python
    V *= esferica(Ys, Xs, L, lamb, xp, dtype, parte="inversa")
    return V if complejo else xp.abs(1 + V) ** 2
```

por:

```python
    V *= esferica(Ys, Xs, L, lamb, xp, dtype, parte="inversa")
    if complejo:
        return V
    if fuente is None:
        return xp.abs(1 + V) ** 2
    return xp.abs(a_dispositivo(a_sen, xp, dtype) + V) ** 2
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_dlhm.py -q -p no:cacheprovider`
Expected: PASS, todo `tests/test_dlhm.py` (las 23 de antes, las 6 de Task 1 y las 5 nuevas; la GPU se salta si no hay CUDA). La de RS con fuente medida en el prototipo: 1.1e-5 y 4.6e-6.

- [ ] **Step 5: Commit**

```bash
git add CamposT/dlhm.py tests/test_dlhm.py
git commit -m "dlhm: la ida apodiza con la emision de la fuente, contra RS con la misma emision

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: El piso del Reconstructor, y la sección «LA FUENTE»

**Files:**
- Modify: `CamposT/dlhm.py` (`_contraste`, `Reconstructor.__init__` y su docstring, docstring del módulo)
- Test: `tests/test_dlhm.py` (sección «la vuelta», después de `test_un_holograma_crudo_sin_fondo_aborta`)

**Interfaces:**
- Consumes (Tasks 1-2): `fuente_pinhole`, `iluminacion`, `holograma(..., fuente=)`.
- Produces: `_contraste(holo, fondo, piso=0.0) -> np.ndarray` (c − 1); `Reconstructor(holo, delta, lamb, L, z, *, fondo=None, piso=0.0, centro=(0.0, 0.0), margen=1.1, device="auto", dtype=None)`; `reconstruir(..., piso=...)` por `**kw`.

- [ ] **Step 1: Write the failing tests**

Después de `test_un_holograma_crudo_sin_fondo_aborta`, añadir:

```python
def test_el_piso_anula_el_contraste_donde_el_fondo_es_oscuro():
    """Bajo piso·max|fondo| no se divide: c - 1 = 0, 'sin información', como lo
    que cae fuera del sensor. Un fondo con ceros no da NaN."""
    fondo = np.ones((32, 32))
    fondo[:, :8] = 1e-3
    fondo[0, 0] = 0.0
    c1 = dlhm._contraste(1.2 * fondo, fondo, piso=0.05)
    assert np.isfinite(c1).all()
    np.testing.assert_array_equal(c1[:, :8], 0.0)
    np.testing.assert_allclose(c1[:, 8:], 0.2, rtol=1e-12)


def test_sin_piso_el_contraste_es_el_de_siempre():
    rng = np.random.default_rng(3)
    fondo = 0.5 + rng.random((16, 16))
    holo = 0.5 + rng.random((16, 16))
    np.testing.assert_array_equal(dlhm._contraste(holo, fondo, piso=0.0), holo / fondo - 1)


@pytest.mark.parametrize("fondo", [None, "media", 2.0])
def test_el_piso_sin_imagen_de_fondo_aborta(fondo):
    with pytest.raises(ValueError, match="piso"):
        dlhm._contraste(np.ones((8, 8)), fondo, piso=0.05)


@pytest.mark.parametrize("piso", [-0.1, 1.0])
def test_un_piso_fuera_de_rango_aborta(piso):
    with pytest.raises(ValueError, match="piso"):
        dlhm._contraste(np.ones((8, 8)), np.ones((8, 8)), piso=piso)


def test_ida_y_vuelta_con_el_airy_dentro_del_disco():
    """Holograma de INTENSIDAD con la emisión del pinhole, reconstruido
    dividiendo por iluminacion() con piso. Un pinhole de 20 um para que el
    disco (|a|² >= 0.05, 138 px de radio) quepa en 256 px. Dentro del disco
    rinde lo mismo que sin fuente; sin fondo o sin piso, no. Medido: 0.8193 con
    fuente, 0.8165 sin fuente, 0.253 dividiendo por la media y 0.023 con el
    fondo sin piso."""
    n = 256
    a = dlhm.fuente_pinhole(20e-3, LAMB)
    t = manchas(n, DO, [(0.0, 0.0), (0.03, -0.02), (-0.025, 0.03), (0.02, 0.035)], 6e-3)
    fondo = dlhm.iluminacion((n, n), DELTA, L, a)
    luz = fondo >= 0.05
    sin = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, device="cpu"))
    con = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, fuente=a, device="cpu"))

    def corr(holo, **kw):
        tr = a_numpy(dlhm.reconstruir(holo, DELTA, LAMB, Z, L, device="cpu", **kw))
        return correlacion((tr - 1)[luz], (t - 1)[luz])

    c_sin, c_con = corr(sin), corr(con, fondo=fondo, piso=0.05)
    assert c_con >= c_sin - 0.02, (c_con, c_sin)
    assert corr(con, fondo="media") < 0.5
    assert corr(con, fondo=fondo) < 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_dlhm.py -q -p no:cacheprovider -k "piso or airy"`
Expected: FAIL con `TypeError: _contraste() got an unexpected keyword argument 'piso'` y `TypeError: Reconstructor.__init__() got an unexpected keyword argument 'piso'`.

- [ ] **Step 3: Implement**

En `CamposT/dlhm.py`:

1. `_contraste`: la firma pasa a `def _contraste(holo, fondo, piso=0.0):`. En su docstring, añadir antes de `Siempre en coma flotante`:

```
    piso > 0 (con un fondo que sea una imagen): donde |fondo| < piso·max|fondo|
    no se divide y c - 1 = 0, como lo que cae fuera del sensor. Dividir ahí
    explota en los ceros de la iluminación, y subir el fondo a un mínimo daría
    c - 1 = -1: una zona opaca falsa cuyo borde difracta hacia dentro.
```

y, justo DESPUÉS de las dos líneas

```python
    holo = a_numpy(holo)
    holo = holo.astype(np.result_type(holo.dtype, np.float64), copy=False)
```

insertar:

```python
    if not 0 <= piso < 1:
        raise ValueError(f"piso = {piso!r}: va en [0, 1)")
    if piso > 0:
        if fondo is None or isinstance(fondo, str) or np.ndim(fondo) == 0:
            raise ValueError(f"piso = {piso} necesita un fondo que sea una imagen, y "
                             f"fondo = {fondo!r}")
        f = a_numpy(fondo)
        if f.shape != holo.shape:
            raise ValueError(f"el fondo es {f.shape} y el holograma {holo.shape}")
        vale = np.abs(f) >= piso * np.abs(f).max()
        c1 = np.zeros(holo.shape, dtype=np.result_type(holo.dtype, f.dtype))
        np.divide(holo, f, out=c1, where=vale)
        c1[vale] -= 1
        return c1
```

2. `Reconstructor.__init__`: la firma pasa a

```python
    def __init__(self, holo, delta, lamb, L, z, *, fondo=None, piso=0.0,
                 centro=(0.0, 0.0), margen=1.1, device="auto", dtype=None):
```

y la línea `        c1 = _contraste(holo, fondo)` pasa a `        c1 = _contraste(holo, fondo, piso)`. En el docstring de la clase, después del párrafo de `fondo`, añadir:

```
    piso     con un fondo imagen: fracción de max|fondo| bajo la que c - 1 = 0
             en vez de dividir (ver _contraste). Para la iluminación de una
             fuente real (iluminacion()), que tiene ceros. 0: se divide todo.
```

3. En el docstring del módulo, justo ANTES del párrafo que empieza `UNIDADES: milímetros para todo, como en montaje.py.`, insertar:

```
LA FUENTE. Por defecto la ida supone una fuente puntual ideal, isótropa, y
devuelve I/I_ref: por el escalado de Fresnel eso es un holograma de Gabor de
onda plana del objeto magnificado, sin firma global de la esférica (la caída
|e^{ikR}/R|² de la esférica ideal es solo 1.17 del centro a la esquina del
sensor de 3000x4000). El «patrón esférico» de un holograma de laboratorio es
la EMISIÓN de la fuente: el pinhole de 5 um del banco da un Airy con el primer
cero a sin(theta) = 0.1288, 710 px del centro del sensor a L = 10 mm. Con
fuente = a(s) la ida la incluye por dentro, no multiplicando a la salida: el
objeto se ilumina con a(s_z)·e^{ikr}/r y la referencia en el sensor es a(s_L),
así que el holograma es |a_L + V|². Es exacto mientras la emisión sea ya su
campo lejano en la muestra (el pinhole, a z = 3 mm: número de Fresnel 0.004).
Para reconstruir basta dividir por iluminacion() = |a_L|²: c - 1 ≈ V/a_L +
c.c., y como a_z ≈ a_L en la proyección de cada punto (el mismo rayo), el
cociente deshace la envolvente en los dos planos. Falla cerca de los ceros de
a; ahí el Reconstructor pone c - 1 = 0 bajo un piso. Medido en 256 px con un
pinhole de 20 um: correlación con el objeto dentro del disco 0.819 con fuente
y 0.817 sin ella; sin piso, 0.023; dividiendo por la media, 0.253. Dividir por
amplitud -(I - fondo)/sqrt(fondo) en el sensor y la salida entre sqrt(fondo)
proyectado- solo es exacto con campo complejo: con intensidad amplifica la
gemela hacia el borde de la luz, y en el ROI del USAF del BenchmarkTarget
entero empata (0.7906 frente a 0.7900 de esta división; 0.7990 sin fuente).

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_dlhm.py -q -p no:cacheprovider`
Expected: PASS entero.

- [ ] **Step 5: Commit**

```bash
git add CamposT/dlhm.py tests/test_dlhm.py
git commit -m "dlhm: el Reconstructor no divide bajo un piso del fondo, y el docstring explica la fuente

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `mi_prueba_dlhm`: las piezas de la fuente (sin cablear aún `main`)

**Files:**
- Modify: `scripts/mi_prueba_dlhm.py` (constantes `FUENTE` y `PISO` en 1b; `comprobar_configuracion`, `simular`, `carpeta_de_salida`, `MotorExacto`; funciones nuevas `fuente_de`, `sufijo_fuente`, `contraste_enmascarado`, `zona_iluminada`)
- Test: `tests/test_mi_prueba_dlhm_seleccion.py` (sección nueva al final)

**Interfaces:**
- Consumes (Tasks 1-3): `dlhm_exacto.fuente_pinhole`, `dlhm_exacto.iluminacion`, `holograma(..., fuente=)`, `Reconstructor(..., piso=)`.
- Produces:
  - `comprobar_configuracion(metricas, formato, objeto, motor, campo, ruta, referencia, z_real, simulador="exacto", fuente=None, piso=0.0)`.
  - `fuente_de(valor, pinhole, lamb)` → `None` o la función a(s).
  - `sufijo_fuente(valor, pinhole)` → `""`, `"_pinhole5um"` (con `pinhole = 5e-3`) o `"_fuente"`.
  - `carpeta_de_salida(raiz, ruta, referencia, z_real, campo, motor, objeto, roi, sufijo="")`.
  - `simular(t, lamb, delta, L_fuente, z, campo, device="auto", dtype=None, simulador="exacto", na_fuente=0.1, fuente=None)`.
  - `contraste_enmascarado(holo, fondo, piso)` → c = holo/fondo donde |fondo| ≥ piso·max|fondo| y fondo ≠ 0, y 1 donde no.
  - `zona_iluminada(fondo, ventana, piso)` → `(fraccion: float, lado: int)`.
  - `MotorExacto(holo, lamb, delta, L_fuente, z_rango, fondo=None, piso=0.0, device="auto", dtype=None)`.

- [ ] **Step 1: Write the failing tests**

Al final de `tests/test_mi_prueba_dlhm_seleccion.py` añadir:

```python
# ── la fuente (2026-09-29) ──────────────────────────────────────────────────

@pytest.mark.parametrize("cambio", [
    dict(fuente="pinhole", ruta="holo.npy"),       # un holograma grabado ya trae su fuente
    dict(fuente="pinhole", simulador="lopera"),     # el suyo trae su envolvente
    dict(fuente="pinhole", campo="complejo"),
    dict(fuente="led"), dict(fuente=3.0),
    dict(piso=1.0), dict(piso=-0.1)])
def test_una_fuente_o_un_piso_que_no_casan_abortan(cambio):
    kw = dict(metricas=("tamura",), formato="png", objeto="amplitud", motor="exacto",
              campo="intensidad", ruta=None, referencia="objeto.png", z_real=3.0)
    kw.update(cambio)
    with pytest.raises(SystemExit):
        mp.comprobar_configuracion(**kw)


def test_fuente_pinhole_o_una_funcion_pasan():
    kw = dict(metricas=("tamura",), formato="png", objeto="amplitud", motor="exacto",
              campo="intensidad", ruta=None, referencia="objeto.png", z_real=3.0,
              piso=0.05)
    mp.comprobar_configuracion(fuente="pinhole", **kw)
    mp.comprobar_configuracion(fuente=lambda s: np.ones_like(s), **kw)


def test_fuente_de_y_su_sufijo():
    assert mp.fuente_de(None, 5e-3, LAMB) is None
    a = mp.fuente_de("pinhole", 5e-3, LAMB)
    s = np.array([0.0, 0.05, 0.2])
    np.testing.assert_array_equal(a(s), dlhm.fuente_pinhole(5e-3, LAMB)(s))
    f = lambda s: np.ones_like(s)                   # noqa: E731
    assert mp.fuente_de(f, 5e-3, LAMB) is f
    assert mp.sufijo_fuente(None, 5e-3) == ""
    assert mp.sufijo_fuente("pinhole", 5e-3) == "_pinhole5um"
    assert mp.sufijo_fuente(f, 5e-3) == "_fuente"


def test_la_carpeta_de_un_simulado_con_fuente_lleva_su_sufijo():
    carpeta, nombre = mp.carpeta_de_salida("raiz", None, "dir/BenchmarkTarget.png", 3.0,
                                           "intensidad", "exacto", "amplitud", None,
                                           "_pinhole5um")
    assert nombre == "sim_BenchmarkTarget_z3mm_intensidad_pinhole5um"
    assert carpeta.name == "sim_BenchmarkTarget_z3mm_intensidad_pinhole5um_exacto_amplitud"
    # con RUTA el nombre es el del archivo, lleve lo que lleve
    _, nombre = mp.carpeta_de_salida("raiz", "x/holo.npy", None, None, "intensidad",
                                     "exacto", "amplitud", None, "_pinhole5um")
    assert nombre == "holo"


def test_simular_con_fuente_es_la_ida_con_fuente():
    a = dlhm.fuente_pinhole(3.2e-3, LAMB)
    t = 1 - 0.8 * objeto_de_prueba()
    h = mp.simular(t, LAMB, DELTA, L, Z, "intensidad", device="cpu", fuente=a)
    esperado = dlhm.a_numpy(dlhm.holograma(t, DELTA * Z / L, LAMB, Z, L, FORMA, DELTA,
                                           fuente=a, device="cpu"))
    np.testing.assert_array_equal(h, esperado)


def test_contraste_enmascarado_divide_solo_donde_hay_luz():
    fondo = np.ones((8, 8))
    fondo[:, :2] = 1e-3
    fondo[0, 0] = 0.0
    c = mp.contraste_enmascarado(1.5 * fondo, fondo, 0.05)
    np.testing.assert_array_equal(c[:, :2], 1.0)
    np.testing.assert_allclose(c[:, 2:], 1.5)
    # con piso 0 un cero del fondo tampoco divide
    assert np.isfinite(mp.contraste_enmascarado(1.5 * fondo, fondo, 0.0)).all()


def test_zona_iluminada_mide_la_fraccion_oscura_y_el_mayor_cuadrado_centrado():
    """Disco de radio 50 centrado en (100, 150): el mayor cuadrado centrado que
    cabe tiene medio lado m con sqrt(2) m <= 50, o sea m = 35 y lado 70."""
    yy, xx = np.mgrid[:200, :300]
    fondo = ((yy - 100) ** 2 + (xx - 150) ** 2 <= 50 ** 2).astype(float)
    fr, lado = mp.zona_iluminada(fondo, (80, 130, 40, 40), 0.05)
    assert fr == 0.0 and lado == 70
    fr, _ = mp.zona_iluminada(fondo, (0, 0, 64, 64), 0.05)
    assert fr == 1.0
    fr, _ = mp.zona_iluminada(fondo, (100, 100, 64, 64), 0.05)
    assert 0.0 < fr < 1.0


def test_el_motor_exacto_pasa_el_piso(globales):
    """Con fuente y piso el motor exacto reconstruye lo mismo que el
    Reconstructor con esos argumentos."""
    a = dlhm.fuente_pinhole(3.2e-3, LAMB)
    t = 1 - 0.8 * objeto_de_prueba()
    h = mp.simular(t, LAMB, DELTA, L, Z, "intensidad", device="cpu", fuente=a)
    fondo = dlhm.iluminacion(FORMA, DELTA, L, a)
    m = mp.MotorExacto(h, LAMB, DELTA, L, (Z - 0.01, Z + 0.01), fondo=fondo, piso=0.05,
                       device="cpu")
    R = dlhm.Reconstructor(h, DELTA, LAMB, L, (Z - 0.01, Z + 0.01), fondo=fondo, piso=0.05,
                           device="cpu")
    np.testing.assert_allclose(dlhm.a_numpy(m.campo(Z, (0, 0, *FORMA))),
                               dlhm.a_numpy(R.campo(Z)), rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q -p no:cacheprovider -k "fuente or piso or enmascarado or zona_iluminada"`
Expected: FAIL (`comprobar_configuracion() got an unexpected keyword argument 'fuente'`, `module ... has no attribute 'fuente_de'`, etc.).

- [ ] **Step 3: Implement**

En `scripts/mi_prueba_dlhm.py`:

1. Constantes. Justo DESPUÉS de la línea `NA_FUENTE = 0.1` (y antes del comentario `#: El montaje del laboratorio, ...`), insertar:

```python

#: La EMISION de la fuente, solo al simular con SIMULADOR = "exacto" y CAMPO =
#: "intensidad":
#:   None       fuente puntual ideal, isotropa: ilumina el sensor entero por
#:              igual. Lo de siempre.
#:   "pinhole"  el Airy del pinhole de MONTAJE.pinhole (5 um): primer cero a
#:              sin(theta) = 0.1288, 710 px del centro del sensor a L = 10 mm.
#:              Es el disco brillante con anillos de un holograma de
#:              laboratorio, el "patron esferico" de los de Lopera y Carlos.
#:   a(s)       cualquier funcion de s = sin(theta) con a(0) = 1: AMPLITUD, no
#:              intensidad.
#: Apagada por defecto: la NA real no esta medida (MONTAJE.NA), y la puede fijar
#: el objetivo que enfoca el laser en vez del pinhole. Con fuente el holograma
#: se guarda junto a su fondo, <nombre>_fondo.npy, y los dos motores lo dividen
#: por el (ver PISO).
FUENTE = None

#: Fraccion de max|fondo| bajo la que no se divide por el fondo: ahi c - 1 = 0
#: en el motor exacto y c = 1 en el de Carlos, "sin informacion", como lo que
#: cae fuera del sensor. Dividir en los ceros de Airy explota: medido en 256 px,
#: sin piso la correlacion con el objeto cae de 0.82 a 0.02. Se usa con FUENTE
#: y, en el motor exacto, con FONDO; con FONDO = None no. Con el pinhole, 0.05
#: deja un disco de 555 px de radio (0.01, 632; 0.1, 504).
PISO = 0.05
```

2. El comentario de `FONDO` (el bloque `#: Solo con RUTA y MOTOR = "exacto": ...`) pasa a:

```python
#: Solo con RUTA y MOTOR = "exacto": la imagen SIN muestra, por la que se
#: divide el holograma, con PISO. None divide por su propia media, que deja
#: dentro la envolvente del cono del pinhole. El motor de Carlos no la usa:
#: recibe la intensidad normalizada a [0, 1]. Un simulado con FUENTE se relee
#: con RUTA = <nombre>.npy y FONDO = <nombre>_fondo.npy.
```

3. `comprobar_configuracion`: la firma pasa a

```python
def comprobar_configuracion(metricas, formato, objeto, motor, campo, ruta,
                            referencia, z_real, simulador="exacto", fuente=None,
                            piso=0.0):
```

y, justo ANTES del bloque `    con_referencia = [n for n in metricas if n != "tamura"]`, insertar:

```python
    if not 0 <= piso < 1:
        raise SystemExit(f"PISO = {piso!r}: va en [0, 1).")
    if fuente is not None:
        if (isinstance(fuente, str) and fuente != "pinhole") or not (
                isinstance(fuente, str) or callable(fuente)):
            raise SystemExit(f'FUENTE = {fuente!r}: usa None, "pinhole" o una funcion '
                             f'a(s) con a(0) = 1.')
        if ruta is not None:
            raise SystemExit("FUENTE solo vale al simular (RUTA = None): un holograma "
                             "grabado ya trae su fuente. Pasa su imagen sin muestra en "
                             "FONDO y deja FUENTE = None.")
        if simulador != "exacto":
            raise SystemExit('FUENTE va con SIMULADOR = "exacto": el modelo de Lopera '
                             'trae su propia envolvente (NA_FUENTE).')
        if campo != "intensidad":
            raise SystemExit('FUENTE va con CAMPO = "intensidad": con "complejo" el '
                             'holograma es 1 + V, normalizado por la esferica ideal, y la '
                             'fuente no tiene donde ir.')
```

4. Justo DESPUÉS de la función `fondo_de` (antes de `def simular(`), insertar:

```python
def fuente_de(valor, pinhole, lamb):
    """FUENTE -> la funcion a(s) de CamposT.dlhm, o None."""
    if valor is None:
        return None
    if isinstance(valor, str):
        return dlhm_exacto.fuente_pinhole(pinhole, lamb)
    return valor


def sufijo_fuente(valor, pinhole):
    """Lo que FUENTE anade al nombre del simulado: '', _pinhole<d>um o _fuente."""
    if valor is None:
        return ""
    if isinstance(valor, str):
        return f"_pinhole{pinhole * 1e3:g}um"
    return "_fuente"


def contraste_enmascarado(holo, fondo, piso):
    """c = holo/fondo donde |fondo| >= piso*max|fondo| (y fondo != 0), y 1 donde
    no: lo que recibe el motor de Carlos con la fuente simulada. El 1 no es
    opaco: es "sin informacion", como lo que cae fuera del recorte."""
    holo = np.asarray(holo, dtype=np.float64)
    fondo = np.asarray(fondo, dtype=np.float64)
    vale = (np.abs(fondo) >= piso * np.abs(fondo).max()) & (fondo != 0)
    c = np.ones_like(holo)
    np.divide(holo, fondo, out=c, where=vale)
    return c


def zona_iluminada(fondo, ventana, piso):
    """-> (fraccion de la ventana bajo el piso, lado del mayor cuadrado centrado
    en la malla que queda entero sobre el piso). Para el AVISO de main(): lo que
    cae bajo el piso no se mide, se enmascara."""
    luz = np.abs(np.asarray(fondo, dtype=np.float64))
    luz = luz >= piso * luz.max()
    f0, c0, alto, ancho = ventana
    fraccion = float(1 - luz[f0:f0 + alto, c0:c0 + ancho].mean())
    M, N = luz.shape
    lo, hi = 0, min(M, N) // 2                      # medio lado
    while lo < hi:
        m = (lo + hi + 1) // 2
        if luz[M // 2 - m:M // 2 + m, N // 2 - m:N // 2 + m].all():
            lo = m
        else:
            hi = m - 1
    return fraccion, 2 * lo
```

5. `simular`: la firma pasa a

```python
def simular(t, lamb, delta, L_fuente, z, campo, device="auto", dtype=None,
            simulador="exacto", na_fuente=0.1, fuente=None):
```

en su docstring, después del párrafo de `simulador = "exacto"`, añadir `    fuente: la emision a(s) (CamposT.dlhm), solo con "exacto".` y en la llamada a la ida añadir `fuente=fuente`:

```python
    h = a_cpu(dlhm_exacto.holograma(t, delta * z / L_fuente, lamb, z, L_fuente, t.shape,
                                    delta, complejo=campo == "complejo", fuente=fuente,
                                    device=device, dtype=dtype))
```

6. `carpeta_de_salida`: la firma pasa a
`def carpeta_de_salida(raiz, ruta, referencia, z_real, campo, motor, objeto, roi, sufijo=""):`,
su docstring dice `sim_<referencia>_z<Z_REAL>mm_<CAMPO o lopera><sufijo de FUENTE>`, y la línea del nombre simulado pasa a

```python
        nombre = f"sim_{pathlib.Path(referencia).stem}_z{z_real:g}mm_{campo}{sufijo}"
```

7. `MotorExacto.__init__`:

```python
    def __init__(self, holo, lamb, delta, L_fuente, z_rango, fondo=None, piso=0.0,
                 device="auto", dtype=None):
        self.R = dlhm_exacto.Reconstructor(holo, delta, lamb, L_fuente, z_rango,
                                    fondo=fondo, piso=piso, device=device, dtype=dtype)
```

(el resto del método no cambia).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q -p no:cacheprovider`
Expected: PASS entero (todas las de antes y las nuevas).

- [ ] **Step 5: Commit**

```bash
git add scripts/mi_prueba_dlhm.py tests/test_mi_prueba_dlhm_seleccion.py
git commit -m "mi_prueba_dlhm: FUENTE y PISO, y las piezas para simular y dividir por el fondo

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `main()` con la fuente: fondo, aviso, motores y lo que se guarda

**Files:**
- Modify: `scripts/mi_prueba_dlhm.py` (`main`)
- Test: `tests/test_mi_prueba_dlhm_seleccion.py` (fixture `config` y pruebas de punta a punta)

**Interfaces:**
- Consumes (Task 4): `fuente_de`, `sufijo_fuente`, `contraste_enmascarado`, `zona_iluminada`, `comprobar_configuracion(..., fuente, piso)`, `simular(..., fuente=)`, `carpeta_de_salida(..., sufijo)`, `MotorExacto(..., piso=)`; de `CamposT.dlhm`, `iluminacion(forma, delta, L, fuente)`.
- Produces: con `FUENTE`, en la carpeta de salida `<nombre>.npy` y `<nombre>_fondo.npy`; con `RUTA` + `FONDO` la corrida se repite con el mismo fondo y piso.

- [ ] **Step 1: Write the failing tests**

En la fixture `config`, añadir al dict `valores` las dos constantes nuevas (así ninguna prueba depende de sus valores por defecto): después de `SIMULADOR="exacto", NA_FUENTE=0.1,` poner `FUENTE=None, PISO=0.05,`.

Al final del archivo añadir:

```python
#: Pinhole que da, con L = 0.8 mm, un disco (|a|² >= 0.05) de ~69 px de radio:
#: la ventana de config (48, 32, 64, 64), con las esquinas a 45 px, cae dentro.
PINHOLE_PRUEBA = 3.2e-3


def test_main_con_fuente_guarda_el_fondo_y_encuentra_el_foco(config, monkeypatch):
    """Medido con el prototipo: el exacto a 2.0 um (1.0 sin fuente), el de
    Carlos a 2.7 um, las mismas cotas que sin fuente."""
    a = dlhm.fuente_pinhole(PINHOLE_PRUEBA, LAMB)
    monkeypatch.setattr(mp, "FUENTE", a)
    z_foco = mp.main()
    carpeta = config.parent / config.name.replace("_intensidad_", "_intensidad_fuente_")
    nombre = "sim_objeto_z0.24mm_intensidad_fuente"
    holo = np.load(carpeta / f"{nombre}.npy")
    fondo = np.load(carpeta / f"{nombre}_fondo.npy")
    np.testing.assert_allclose(fondo, dlhm.iluminacion(FORMA, DELTA, L, a), rtol=1e-12)
    M, N = FORMA
    # el disco brillante: medido 283 veces el nivel de la esquina
    assert holo[M // 2 - 5:M // 2 + 5, N // 2 - 5:N // 2 + 5].mean() > 100 * holo[:10, :10].mean()
    # y dividido por su fondo, dentro del disco, es un holograma de contraste
    luz = fondo >= 0.05 * fondo.max()
    assert 0.5 < (holo[luz] / fondo[luz]).mean() < 1.5
    assert all(abs(z - Z) < PASO / 2 for z in z_foco.values()), z_foco
    # releido con RUTA y FONDO, sin FUENTE: el mismo foco, en la misma carpeta
    monkeypatch.setattr(mp, "FUENTE", None)
    monkeypatch.setattr(mp, "RUTA", carpeta / f"{nombre}.npy")
    monkeypatch.setattr(mp, "FONDO", carpeta / f"{nombre}_fondo.npy")
    z_foco_2 = mp.main()
    assert all(abs(z_foco_2[n] - z_foco[n]) <= 2 * PASO / 10 for n in z_foco), (z_foco, z_foco_2)


def test_main_con_fuente_y_el_motor_de_carlos(config, monkeypatch):
    monkeypatch.setattr(mp, "FUENTE", dlhm.fuente_pinhole(PINHOLE_PRUEBA, LAMB))
    monkeypatch.setattr(mp, "MOTOR", "carlos")
    z_foco = mp.main()
    assert all(abs(z - Z) < PASO for z in z_foco.values()), z_foco


def test_main_avisa_si_la_region_medida_sale_de_la_luz(config, monkeypatch, capsys):
    """Una ventana casi entera: sus esquinas (a 91 px) salen del disco de 69."""
    for k, v in dict(FUENTE=dlhm.fuente_pinhole(PINHOLE_PRUEBA, LAMB),
                     ROI=(8, 8, 144, 112), METRICAS=("tamura",), EN_CPU=set()).items():
        monkeypatch.setattr(mp, k, v)
    mp.main()
    salida = capsys.readouterr().out
    assert "AVISO" in salida and "PISO" in salida
    lado = mp.zona_iluminada(dlhm.iluminacion(FORMA, DELTA, L, mp.FUENTE),
                             (8, 8, 112, 144), 0.05)[1]
    assert f"{lado}, {lado})" in salida
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q -p no:cacheprovider -k "main_con_fuente or main_avisa"`
Expected: FAIL (el simulado sale sin fuente: no existe `<nombre>_fondo.npy` y la carpeta no lleva `_fuente`; no hay AVISO).

- [ ] **Step 3: Implement**

En `scripts/mi_prueba_dlhm.py`, dentro de `main()`:

1. La llamada inicial:

```python
    comprobar_configuracion(METRICAS, FORMATO_BARRIDO, OBJETO, MOTOR, CAMPO, RUTA,
                            REFERENCIA, Z_REAL, SIMULADOR, FUENTE, PISO)
    fuente = fuente_de(FUENTE, MONTAJE.pinhole, LAMB)
```

2. En la rama `if RUTA is None:`, la simulación y el fondo:

```python
        holo = simular(t_ref, LAMB, DELTA, L, Z_REAL, CAMPO, dev, dtype, SIMULADOR,
                       NA_FUENTE, fuente=fuente)
        # la ida deja en el pool de CuPy lo que uso -en 3000x4000, ~1.6 GB- y el
        # motor no lo encontraria libre
        liberar(xp_)
        if fuente is not None:
            # la imagen sin muestra, en la normalizacion del holograma
            fondo = dlhm_exacto.iluminacion(holo.shape, DELTA, L, fuente)
        else:
            # el de Lopera lleva la envolvente de la fuente y 8 bits: se divide por
            # su media, como uno de laboratorio sin FONDO
            fondo = None if SIMULADOR == "exacto" else "media"
```

(reemplaza desde `holo = simular(` hasta `fondo = None if SIMULADOR == "exacto" else "media"`, conservando el `print(f"holograma simulado ...")` que sigue).

3. Justo DESPUÉS del bloque `if t_ref is not None and t_ref.shape != holo.shape: raise SystemExit(...)`, antes de `    M, N = holo.shape`, insertar:

```python
    # el piso solo cuando se divide por una imagen: la simulada o FONDO
    piso = PISO if isinstance(fondo, np.ndarray) else 0.0
```

4. Justo DESPUÉS de `    f0, c0, alto, ancho = ventana`, insertar el aviso:

```python
    if piso > 0:
        fraccion, lado = zona_iluminada(fondo, ventana, piso)
        if fraccion > 0:
            print(f"AVISO: el {100 * fraccion:.0f} % de la region medida cae donde el "
                  f"fondo esta bajo PISO = {PISO:g} de su maximo: ahi no se divide y no "
                  f"hay informacion.\n  El mayor cuadrado centrado que queda iluminado "
                  f"es de {lado} px:\n\n    ROI = ({(N - lado) // 2}, {(M - lado) // 2}, "
                  f"{lado}, {lado})\n")
```

5. La carpeta de salida y lo que se guarda:

```python
    salida, nombre = carpeta_de_salida(RESULTADOS, RUTA, REFERENCIA, Z_REAL, tipo,
                                       MOTOR, OBJETO, roi,
                                       sufijo_fuente(FUENTE, MONTAJE.pinhole))
    salida.mkdir(parents=True, exist_ok=True)
    if RUTA is None:
        np.save(salida / f"{nombre}.npy", holo)
        if fuente is None:
            print(f"  -> {salida / f'{nombre}.npy'}  (ponlo en RUTA para no volver a simular)")
        else:
            np.save(salida / f"{nombre}_fondo.npy", fondo)
            print(f"  -> {salida / f'{nombre}.npy'} y {nombre}_fondo.npy  (ponlos en RUTA "
                  f"y FONDO, con FUENTE = None, para no volver a simular)")
```

6. Los motores:

```python
    if MOTOR == "exacto":
        motor = MotorExacto(holo, LAMB, DELTA, L, z_rango, fondo=fondo, piso=piso,
                            device=dev, dtype=dtype)
```

y en la rama del de Carlos, la línea `motor = MotorCarlos(holo, ventana, ...)` pasa a

```python
        # con la fuente simulada recibe el contraste; con RUTA y FONDO, como siempre
        h_carlos = holo if fuente is None else contraste_enmascarado(holo, fondo, piso)
        motor = MotorCarlos(h_carlos, ventana, LAMB, DELTA, L, xp_, dtype, SOBREMUESTREO)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q -p no:cacheprovider`
Expected: PASS entero.

- [ ] **Step 5: Commit**

```bash
git add scripts/mi_prueba_dlhm.py tests/test_mi_prueba_dlhm_seleccion.py
git commit -m "mi_prueba_dlhm: main simula con la fuente, divide por su fondo y avisa si se mide en lo oscuro

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: El ROI por defecto sobre el grupo USAF central

**Files:**
- Modify: `scripts/mi_prueba_dlhm.py` (constante `ROI` y su comentario; la etiqueta «el defecto» de la tabla de la cabecera)
- Test: `tests/test_mi_prueba_dlhm_seleccion.py`

**Interfaces:**
- Consumes (Task 4): `zona_iluminada`; de `CamposT.dlhm`, `fuente_pinhole`, `iluminacion`.
- Produces: `ROI = (1884, 1306, 256, 256)`.

- [ ] **Step 1: Write the failing test**

Al final de `tests/test_mi_prueba_dlhm_seleccion.py`:

```python
BENCHMARK = RAIZ / "referencia/carlos/DLHM-model-main/DLHM-model-main/data/BenchmarkTarget.png"


@pytest.mark.skipif(not BENCHMARK.exists(),
                    reason="referencia/carlos no va en el repo (.gitignore)")
def test_el_roi_por_defecto_encierra_el_usaf_central():
    """Spec D8: un cuadrado con el grupo USAF del centro del BenchmarkTarget
    entero, 48 px de margen como minimo, ningun otro grupo dentro y todo el
    cuadrado en la luz del pinhole del banco (PISO 0.05)."""
    from scipy import ndimage as ndi
    from CamposT.montaje import MONTAJE
    img = cv.imread(str(BENCHMARK), cv.IMREAD_GRAYSCALE) > 127
    M, N = img.shape
    x0, y0, ancho, alto = mp.ROI
    assert ancho == alto
    etiquetas, _ = ndi.label(ndi.binary_dilation(img, iterations=12))
    objetos = ndi.find_objects(etiquetas)

    def distancia(sl):
        return np.hypot((sl[0].start + sl[0].stop) / 2 - M / 2,
                        (sl[1].start + sl[1].stop) / 2 - N / 2)
    usaf = 1 + min(range(len(objetos)), key=lambda i: distancia(objetos[i]))
    ys, xs = np.nonzero(img & (etiquetas == usaf))
    assert ys.min() - y0 >= 48 and y0 + alto - 1 - ys.max() >= 48
    assert xs.min() - x0 >= 48 and x0 + ancho - 1 - xs.max() >= 48
    assert set(np.unique(etiquetas[y0:y0 + alto, x0:x0 + ancho])) <= {0, usaf}
    a = dlhm.fuente_pinhole(MONTAJE.pinhole, MONTAJE.lamb)
    fondo = dlhm.iluminacion((M, N), MONTAJE.delta_x, MONTAJE.L, a)
    assert mp.zona_iluminada(fondo, (y0, x0, alto, ancho), 0.05)[0] == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q -p no:cacheprovider -k usaf_central`
Expected: FAIL con `TypeError: cannot unpack non-iterable NoneType object` (hoy `ROI = None`).

- [ ] **Step 3: Implement**

En `scripts/mi_prueba_dlhm.py`:

1. La constante y su comentario (sección 1d): la línea `ROI = None` pasa a

```python
#:
#: El defecto es el grupo USAF del CENTRO del BenchmarkTarget: filas 1357-1512
#: y columnas 1932-2091 de la imagen (pixeles > 0.5), y este cuadrado de 256
#: px le deja 48-51 px por lado para sus franjas sin tocar otro grupo. Su
#: esquina mas lejana queda a 239 px del centro del sensor: dentro de la luz del
#: pinhole con FUENTE = "pinhole" (disco de 555 px con PISO = 0.05). NO esta
#: centrado -65.5 px arriba y 11.5 a la derecha del centro-: el motor exacto no
#: lo nota, y el de Carlos, que pone la fuente en el centro de su recorte, lo
#: avisa. Con otro holograma en RUTA se cambia a mano o se pone None o True.
ROI = (1884, 1306, 256, 256)
```

2. En la cabecera, la etiqueta de la primera variante de la tabla «LO QUE SALIO el 2026-09-28» deja de ser el defecto. Reemplazar la línea

```
      (750x1750 px: el defecto)         pearson     3.0000   0.047    57.7
```

por (misma longitud, no se descuadra la tabla):

```
      (750x1750 px, ROI = None)         pearson     3.0000   0.047    57.7
```

y reemplazar las cuatro líneas que van desde `LO QUE SALIO el 2026-09-28, con los valores por defecto: el BenchmarkTarget` hasta `0.01 mm y fino de 0.001 mm, en la GPU de desarrollo (RTX 3050 de 4 GB):` por estas cinco (el mismo texto con `de ese dia (ROI = None)` y los saltos reajustados para no pasar de 79 columnas):

```
LO QUE SALIO el 2026-09-28, con los valores por defecto de ese dia (ROI =
None): el BenchmarkTarget simulado entero (3000x4000 px, z = 3 mm, holograma
de INTENSIDAD, con gemela) con el montaje (528 nm, 1.83 um, L = 10 mm), grueso
de 2.5 a 3.5 mm con paso 0.01 mm y fino de 0.001 mm, en la GPU de desarrollo
(RTX 3050 de 4 GB):
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q -p no:cacheprovider`
Expected: PASS entero (las pruebas de `main` fijan su propio `ROI` en `config`).

- [ ] **Step 5: Commit**

```bash
git add scripts/mi_prueba_dlhm.py tests/test_mi_prueba_dlhm_seleccion.py
git commit -m "mi_prueba_dlhm: el ROI por defecto es el grupo USAF central del BenchmarkTarget

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: La suite entera

**Files:** ninguno (solo verificación).

- [ ] **Step 1: Run the whole suite**

Run: `Tesis_env/Scripts/python.exe -m pytest -q -p no:cacheprovider`
Expected: PASS entero (con GPU corren también las marcadas `gpu`). Si algo de lo que existía falla, es una regresión de las Tasks 1-6: se arregla ahí, no se toca la prueba.

- [ ] **Step 2: Confirm nothing else was touched**

Run: `git status --short`
Expected: solo `M scripts/barrido_foco.py`, `M scripts/mpasm_minimo.py` (del usuario, sin tocar) y los `??` que ya estaban. Nada de este plan sin commitear.

---

### Task 8: A escala real, y dejarlo escrito

**Files:**
- Modify: `scripts/mi_prueba_dlhm.py` (cabecera: sección «LA FUENTE» con los números)
- Modify: `docs/superpowers/specs/2026-09-29-fuente-dlhm-design.md` (sección final «Resultado de la verificación a escala real»)
- Modify: `docs/Cronograma_TG_DLHM.xlsx` (tarea 80)

**Interfaces:**
- Consumes: todo lo anterior. Las corridas escriben en `resultados/mi_prueba_dlhm/` (no se commitea: es salida).

- [ ] **Step 1: Las cuatro corridas**

Cada una simula el BenchmarkTarget entero (~40 s en la GPU) y barre 2.5-3.5 mm. Desde la raíz, en Git Bash, una por una, copiando de cada salida la tabla de focos (foco, error, FWHM, alt/riz) y el tiempo:

```bash
MPLBACKEND=Agg Tesis_env/Scripts/python.exe -c "import scripts.mi_prueba_dlhm as mp; mp.MOSTRAR = False; mp.main()"
```

```bash
MPLBACKEND=Agg Tesis_env/Scripts/python.exe -c "import scripts.mi_prueba_dlhm as mp; mp.MOSTRAR = False; mp.FUENTE = 'pinhole'; mp.main()"
```

```bash
MPLBACKEND=Agg Tesis_env/Scripts/python.exe -c "import scripts.mi_prueba_dlhm as mp; mp.MOSTRAR = False; mp.MOTOR = 'carlos'; mp.main()"
```

```bash
MPLBACKEND=Agg Tesis_env/Scripts/python.exe -c "import scripts.mi_prueba_dlhm as mp; mp.MOSTRAR = False; mp.MOTOR = 'carlos'; mp.FUENTE = 'pinhole'; mp.main()"
```

Expected (spec, Verificación 4): con el exacto, las tres métricas a menos de medio paso grueso (0.005 mm) de 3.000; con el de Carlos, a menos de un paso grueso (0.01 mm). Mirar `campos.png` de las dos corridas con fuente: el holograma tiene que verse como un disco brillante con anillos de Airy.

Si el de Carlos falla con el ROI descentrado, repetir sus dos corridas con un cuadrado centrado que contiene el USAF, y anotar las dos:

```bash
MPLBACKEND=Agg Tesis_env/Scripts/python.exe -c "import scripts.mi_prueba_dlhm as mp; mp.MOSTRAR = False; mp.MOTOR = 'carlos'; mp.ROI = (1808, 1308, 384, 384); mp.main()"
```

Si el exacto falla, NO se toca ningún umbral: se vuelve a systematic-debugging.

- [ ] **Step 2: La cabecera de mi_prueba_dlhm.py**

Justo ANTES del párrafo que empieza `EL ~0.03 DE ANTES, EXPLICADO.`, insertar una sección en ASCII con este esqueleto, rellenando la tabla con los números del Step 1:

```
LA FUENTE (FUENTE, PISO), 2026-09-29. La ida exacta supone por defecto una
fuente puntual ideal y da I/I_ref: sin la envolvente de los hologramas de
Lopera y Carlos, que es la EMISION de la fuente y no la propagacion (medido
sobre este objeto, media radial centro/borde: exacto 0.98, con la esferica
ideal sin dividir 1.17, con la envolvente de dlhm() 58.6, su modelo 54.7).
FUENTE = "pinhole" mete el Airy del pinhole de 5 um dentro de la ida (primer
cero a 710 px del centro) y los dos motores dividen por su fondo, con PISO.
Con el ROI por defecto (el USAF central, 256 px) y el mismo barrido:

    motor     fuente     metrica     foco     FWHM   alt/riz
    exacto    ninguna    tamura      ...
    ...

(lo que salio, y si el de Carlos necesito el ROI centrado).
```

- [ ] **Step 3: La spec**

Al final de `docs/superpowers/specs/2026-09-29-fuente-dlhm-design.md`, añadir una sección `## Resultado de la verificación a escala real (2026-09-29)` con la misma tabla y una línea por cada punto de la Verificación que se cumplió o no, con su número.

- [ ] **Step 4: El cronograma**

Tarea 80 en la hoja Tareas, fila 83, con la convención de siempre (ASCII puro, semana 11, viernes 02/10/2026). Copiar el estilo de la fila 82 y extender el autofiltro:

```bash
Tesis_env/Scripts/python.exe - <<'EOF'
import copy, openpyxl
ruta = "docs/Cronograma_TG_DLHM.xlsx"
wb = openpyxl.load_workbook(ruta)
ws = wb["Tareas"]
assert ws.cell(82, 1).value in (79, "79"), ws.cell(82, 1).value
ids = [ws.cell(r, 1).value for r in range(4, 83)]
assert [int(v) for v in ids] == list(range(1, 80)), "IDs con huecos"
fila = 83
valores = [
    80 if isinstance(ws.cell(82, 1).value, int) else "80",
    11 if isinstance(ws.cell(82, 2).value, int) else "11",
    "F2",
    "Anadir la emision de la fuente a la ida DLHM exacta y el ROI sobre el USAF central",
    "CamposT/dlhm.py (fuente, iluminacion, piso) + scripts/mi_prueba_dlhm.py (FUENTE, PISO, ROI)",
    "Alta", "Hecho", ws.cell(82, 8).value,
    "NOTAS: <rellenar en ASCII con: el diagnostico (no era regresion: HEAD regenera el "
    "holograma del 28/09 a 1.8e-4; la envolvente de Lopera/Carlos es la emision de la "
    "fuente, 0.98 vs 58.6 centro/borde), los hashes de los commits, RS con fuente 1.1e-5 y "
    "4.6e-6, la ida y vuelta 0.819/0.817 (sin piso 0.023), la tabla del Step 1 y lo que "
    "quede abierto: la NA real sin medir>",
]
for col, v in enumerate(valores, start=1):
    c = ws.cell(fila, col, v)
    arriba = ws.cell(82, col)
    c.font, c.border = copy.copy(arriba.font), copy.copy(arriba.border)
    c.fill, c.alignment = copy.copy(arriba.fill), copy.copy(arriba.alignment)
    c.number_format = arriba.number_format
ws.row_dimensions[fila].height = ws.row_dimensions[82].height
ws.auto_filter.ref = f"A3:I{fila}"
wb.save(ruta)
print("ok", ws.auto_filter.ref)
EOF
```

Antes de correrlo, sustituir el texto `NOTAS: <...>` por las notas reales en ASCII (sin tildes ni ñ), con los números medidos del Step 1 y los hashes de los commits de las Tasks 1-6 (`git log --oneline fuente-dlhm -8`). La fecha límite se copia de la fila 82 (misma semana 11).

Run después: `Tesis_env/Scripts/python.exe -c "import openpyxl; ws = openpyxl.load_workbook('docs/Cronograma_TG_DLHM.xlsx')['Tareas']; print([ws.cell(83, c).value for c in range(1, 8)], ws.auto_filter.ref)"`
Expected: la fila 80 con `Hecho` y el autofiltro `A3:I83`.

- [ ] **Step 5: Commit**

```bash
git add scripts/mi_prueba_dlhm.py docs/superpowers/specs/2026-09-29-fuente-dlhm-design.md docs/Cronograma_TG_DLHM.xlsx
git commit -m "Cronograma y cabecera: la fuente en la ida DLHM, medida a escala real

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Apéndice (2026-09-29): división por amplitud y recorte propio de Carlos

La primera corrida de la Task 8 se bloqueó (ver la spec, «Revisión tras la
verificación a escala real»): con la D4 SSIM se va −0.035 mm, y Carlos con el
ROI del USAF (256 px) no tiene NA. El usuario decidió: dividir por amplitud
(D4bis, D5bis) y darle a Carlos su propio recorte centrado de 1024 px (D9).
Las Tasks 1-7 siguen hechas; estas tres sustituyen a la Task 8.

### Task 9: La reconstrucción divide por amplitud

**Files:**
- Modify: `CamposT/dlhm.py` (`__all__`, `contraste_amplitud` y `_bilineal` nuevas, `_contraste` sin piso, `Reconstructor.__init__`, `_raiz_proyectada` nueva, `campo`, docstrings)
- Test: `tests/test_dlhm.py` (las pruebas del piso de la Task 3 se migran; nuevas)

**Interfaces:**
- Produces:
  - `contraste_amplitud(holo, fondo, piso=0.0) -> (c1: ndarray, raiz: ndarray float64, raiz_min: float)`, pública.
  - `_bilineal(A, Y, X, yq, xq) -> ndarray` (A en la malla uniforme Y, X; fuera, 0).
  - `_contraste(holo, fondo)` — vuelve a no tener `piso`.
  - `Reconstructor(holo, delta, lamb, L, z, *, fondo=None, piso=0.0, ...)`: con `fondo` imagen, normaliza por amplitud; `Reconstructor._raiz_proyectada(y, x, z) -> ndarray`.

- [ ] **Step 1: Las pruebas**

En `tests/test_dlhm.py`, reemplazar las cinco funciones de la Task 3 —desde `def test_el_piso_anula_el_contraste_donde_el_fondo_es_oscuro():` hasta el final de `test_ida_y_vuelta_con_el_airy_dentro_del_disco` (la línea `    assert corr(con, fondo=fondo) < 0.5`), con sus dos decoradores `@pytest.mark.parametrize`— por:

```python
def test_el_piso_anula_el_contraste_donde_el_fondo_es_oscuro():
    """Bajo sqrt(piso·max|fondo|) no se divide: c1 = 0, 'sin información',
    como lo que cae fuera del sensor. Un fondo con ceros no da NaN."""
    fondo = np.ones((32, 32))
    fondo[:, :8] = 1e-3
    fondo[0, 0] = 0.0
    c1, raiz, raiz_min = dlhm.contraste_amplitud(1.2 * fondo, fondo, piso=0.05)
    assert np.isfinite(c1).all()
    np.testing.assert_array_equal(c1[:, :8], 0.0)
    np.testing.assert_allclose(c1[:, 8:], 0.2, rtol=1e-12)
    np.testing.assert_array_equal(raiz, np.sqrt(fondo))
    assert raiz_min == pytest.approx(np.sqrt(0.05))


def test_sin_piso_el_contraste_por_amplitud_es_holo_menos_fondo_entre_su_raiz():
    rng = np.random.default_rng(3)
    fondo = 0.5 + rng.random((16, 16))
    holo = 0.5 + rng.random((16, 16))
    c1, raiz, raiz_min = dlhm.contraste_amplitud(holo, fondo, piso=0.0)
    np.testing.assert_allclose(c1, (holo - fondo) / np.sqrt(fondo), rtol=1e-12)
    assert raiz_min == 0.0


@pytest.mark.parametrize("fondo", [None, "media", 2.0])
def test_el_piso_sin_imagen_de_fondo_aborta(fondo):
    with pytest.raises(ValueError, match="piso"):
        dlhm.Reconstructor(np.ones((8, 8)), DELTA, LAMB, L, Z, fondo=fondo, piso=0.05,
                           device="cpu")


@pytest.mark.parametrize("piso", [-0.1, 1.0])
def test_un_piso_fuera_de_rango_aborta(piso):
    with pytest.raises(ValueError, match="piso"):
        dlhm.contraste_amplitud(np.ones((8, 8)), np.ones((8, 8)), piso=piso)


@pytest.mark.parametrize("holo, fondo, mensaje", [
    (np.ones((8, 8)), np.zeros((8, 8)), "finito"),
    (np.ones((8, 8)), np.full((8, 8), np.nan), "finito"),
    (np.ones((8, 8)), np.ones((8, 6)), "fondo es"),
    (np.ones((8, 8), complex), np.linspace(1, 2, 64).reshape(8, 8), "intensidad")])
def test_un_fondo_imagen_mal_hecho_aborta(holo, fondo, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        dlhm.contraste_amplitud(holo, fondo, piso=0.05)


def test_con_fondo_imagen_la_salida_es_la_vuelta_de_c1_entre_la_raiz_proyectada():
    """1 + B{c1}/sqrt(fondo) proyectado, con B la vuelta de siempre: se arma a
    mano con un Reconstructor SIN fondo sobre 1 + c1. En la malla por defecto
    la proyección es píxel a píxel."""
    n = 64
    a = dlhm.fuente_pinhole(20e-3, LAMB)
    t = manchas(n, DO, [(0.0, 0.0), (0.01, -0.01)], 3e-3)
    holo = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, fuente=a, device="cpu"))
    fondo = dlhm.iluminacion((n, n), DELTA, L, a)
    R = dlhm.Reconstructor(holo, DELTA, LAMB, L, Z, fondo=fondo, piso=0.05, device="cpu")
    c1, raiz, raiz_min = dlhm.contraste_amplitud(holo, fondo, piso=0.05)
    B = a_numpy(dlhm.Reconstructor(1 + c1, DELTA, LAMB, L, Z, device="cpu").campo(Z)) - 1
    luz = raiz >= raiz_min
    esperado = 1 + np.where(luz, B / np.where(luz, raiz, 1), 0)
    np.testing.assert_allclose(a_numpy(R.campo(Z)), esperado, rtol=1e-10, atol=1e-12)


def test_el_fondo_proyectado_es_la_amplitud_de_la_fuente_en_la_muestra():
    """sqrt(iluminacion()) llevado a la muestra por (y, x)·L/z -el mismo rayo-
    es |a(s_z)|: en la malla por defecto y en otra más fina y desplazada."""
    n = 64
    a = dlhm.fuente_pinhole(20e-3, LAMB)
    fondo = dlhm.iluminacion((n, n), DELTA, L, a)
    R = dlhm.Reconstructor(fondo.copy(), DELTA, LAMB, L, Z, fondo=fondo, device="cpu")
    for paso, m, centro in ((DO, n, (0.0, 0.0)), (DO / 2, 48, (3 * DO, -5 * DO))):
        y = dlhm.coordenadas(m, paso, centro[0])
        x = dlhm.coordenadas(m, paso, centro[1])
        rho = np.hypot(x[None, :], y[:, None])
        esperado = np.abs(a(rho / np.sqrt(rho**2 + Z**2)))
        np.testing.assert_allclose(R._raiz_proyectada(y, x, Z), esperado, rtol=1e-3)


def test_bajo_el_piso_la_salida_vale_uno_y_un_fondo_con_ceros_no_da_nan():
    """Fuera del disco, sin piso, la salida se multiplica por 1/sqrt de un fondo
    casi nulo: medido |t - 1| hasta 288 (0.54 dentro). Con piso vale 1."""
    n = 256
    a = dlhm.fuente_pinhole(20e-3, LAMB)
    t = manchas(n, DO, [(0.0, 0.0), (0.03, -0.02)], 6e-3)
    fondo = dlhm.iluminacion((n, n), DELTA, L, a)
    fondo[:, :4] = 0.0
    holo = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, fuente=a, device="cpu"))
    tr = a_numpy(dlhm.reconstruir(holo, DELTA, LAMB, Z, L, fondo=fondo, piso=0.05,
                                  device="cpu"))
    assert np.isfinite(tr).all()
    np.testing.assert_array_equal(tr[fondo < 0.05 * fondo.max()], 1.0)


def test_ida_y_vuelta_con_el_airy_en_el_centro_del_disco():
    """Holograma de INTENSIDAD con la emisión del pinhole, reconstruido
    dividiendo por amplitud con piso. Pinhole de 20 um: el disco (|a|² >= 0.05,
    138 px de radio) cubre la malla de 256 salvo las esquinas. En el centro del
    disco (|a|² >= 0.5) rinde lo mismo que sin fuente; dividiendo por la
    media, no. Medido: 0.888 con fuente, 0.893 sin fuente, 0.313 dividiendo
    por la media. Hacia el borde de la luz la división de la salida amplifica
    la gemela (0.656 en |a|² >= 0.05, contra 0.817 sin fuente): por eso se mide
    en el centro."""
    n = 256
    a = dlhm.fuente_pinhole(20e-3, LAMB)
    t = manchas(n, DO, [(0.0, 0.0), (0.03, -0.02), (-0.025, 0.03), (0.02, 0.035)], 6e-3)
    fondo = dlhm.iluminacion((n, n), DELTA, L, a)
    centro = fondo >= 0.5
    sin = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, device="cpu"))
    con = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, fuente=a, device="cpu"))

    def corr(holo, **kw):
        tr = a_numpy(dlhm.reconstruir(holo, DELTA, LAMB, Z, L, device="cpu", **kw))
        return correlacion((tr - 1)[centro], (t - 1)[centro])

    c_sin, c_con = corr(sin), corr(con, fondo=fondo, piso=0.05)
    assert c_con >= c_sin - 0.02, (c_con, c_sin)
    assert corr(con, fondo="media") < 0.5
```

Y al final del archivo, después de `test_gpu_da_lo_mismo_que_cpu_con_fuente` y antes de `if __name__ == "__main__":`:

```python
@pytest.mark.gpu
@pytest.mark.skipif(not gpu_disponible(), reason="sin GPU CUDA")
def test_gpu_da_lo_mismo_que_cpu_con_fondo_imagen():
    n = 64
    a = dlhm.fuente_pinhole(20e-3, LAMB)
    t = manchas(n, DO, [(0.0, 0.0), (0.01, -0.01)], 3e-3)
    holo = a_numpy(dlhm.holograma(t, DO, LAMB, Z, L, (n, n), DELTA, fuente=a, device="cpu"))
    fondo = dlhm.iluminacion((n, n), DELTA, L, a)
    kw = dict(fondo=fondo, piso=0.05)
    cpu = a_numpy(dlhm.reconstruir(holo, DELTA, LAMB, Z, L, device="cpu", **kw))
    gpu = a_numpy(dlhm.reconstruir(holo, DELTA, LAMB, Z, L, device="gpu", **kw))
    assert error_relativo(gpu - 1, cpu - 1) < 1e-4
```

- [ ] **Step 2: Rojo**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_dlhm.py -q -p no:cacheprovider -k "piso or amplitud or fondo_imagen or proyectado or airy"`
Expected: FAIL con `AttributeError: module 'CamposT.dlhm' has no attribute 'contraste_amplitud'` y `'Reconstructor' object has no attribute '_raiz_proyectada'`.

- [ ] **Step 3: La implementación**

En `CamposT/dlhm.py`:

1. En `__all__`, añadir `"contraste_amplitud"` después de `"iluminacion"`.

2. `_contraste` vuelve a su forma sin piso: la firma pasa a `def _contraste(holo, fondo):`; en su docstring se quita el párrafo de cuatro líneas que empieza `    piso > 0 (con un fondo que sea una imagen):`; y en el cuerpo se quita el bloque que va desde `    if not 0 <= piso < 1:` hasta su `        return c1` (el `if piso > 0:` entero), de modo que `    if fondo is None:` quede justo después de las dos líneas de `holo = a_numpy(holo)`.

3. Justo ANTES de `def _contraste(holo, fondo):`, insertar:

```python
def contraste_amplitud(holo, fondo, piso=0.0):
    """-> (c1, raiz, raiz_min), con fondo la intensidad SIN muestra (una imagen).

    c1 = (holo - fondo)/sqrt|fondo| donde sqrt|fondo| >= raiz_min (y > 0), y 0
    fuera: es lo que se retropropaga. raiz = sqrt|fondo| es por lo que se divide
    la salida, proyectada a la muestra; raiz_min = sqrt(piso·max|fondo|).

    Por qué por amplitud y no c = holo/fondo: con la fuente, fondo = |a_L|² y
    holo = |a_L + V|², así que la parte lineal de c1 es sgn(a_L)(V + V*), que
    se retropropaga sin error a (t - 1)·|a_z|; dividir la salida por |a_z| la
    deja en t - 1. Dividir el holograma por |a_L|² retropropaga V/a_L, que
    deforma el término lineal donde la mancha de un detalle abarca píxeles con
    distinta iluminación: en el USAF del BenchmarkTarget SSIM elegía el plano a
    -0.035 mm del foco. Bajo el piso no se divide ni aquí ni a la salida: sin
    piso, fuera del disco del pinhole |t - 1| llega a 288.

    Con un fondo constante da lo mismo que dividir. Un holograma complejo con
    un fondo que varía aborta: el fondo es una intensidad.
    """
    holo = a_numpy(holo)
    holo = holo.astype(np.result_type(holo.dtype, np.float64), copy=False)
    if not 0 <= piso < 1:
        raise ValueError(f"piso = {piso!r}: va en [0, 1)")
    f = np.asarray(a_numpy(fondo), dtype=np.float64)
    if f.shape != holo.shape:
        raise ValueError(f"el fondo es {f.shape} y el holograma {holo.shape}")
    if not np.all(np.isfinite(f)) or not np.abs(f).max() > 0:
        raise ValueError("el fondo tiene que ser finito y no todo cero")
    if np.iscomplexobj(holo) and np.ptp(f) > 1e-12 * np.abs(f).max():
        raise ValueError("un fondo que varía solo vale con un holograma de "
                         "intensidad: el fondo es la intensidad sin muestra")
    raiz = np.sqrt(np.abs(f))
    raiz_min = float(np.sqrt(piso) * raiz.max())
    luz = (raiz >= raiz_min) & (raiz > 0)
    c1 = np.zeros(holo.shape, dtype=holo.dtype)
    c1[luz] = (holo[luz] - f[luz]) / raiz[luz]
    return c1, raiz, raiz_min


def _bilineal(A, Y, X, yq, xq):
    """A, dada sobre la malla uniforme (Y, X), en la malla producto (yq, xq),
    interpolada bilineal por ejes. Fuera de la malla, 0."""
    def pesos(g, q):
        u = (np.asarray(q, dtype=np.float64) - g[0]) / (g[1] - g[0])
        dentro = (u >= -1e-9) & (u <= g.size - 1 + 1e-9)
        i0 = np.clip(np.floor(u).astype(np.int64), 0, g.size - 2)
        return i0, np.clip(u - i0, 0, 1), dentro
    iy, wy, dy = pesos(Y, yq)
    ix, wx, dx = pesos(X, xq)
    B = A[iy] * (1 - wy)[:, None] + A[iy + 1] * wy[:, None]
    C = B[:, ix] * (1 - wx)[None, :] + B[:, ix + 1] * wx[None, :]
    return C * (dy[:, None] & dx[None, :])
```

4. En `Reconstructor.__init__`, reemplazar la línea `        c1 = _contraste(holo, fondo, piso)` por:

```python
        if not 0 <= piso < 1:
            raise ValueError(f"piso = {piso!r}: va en [0, 1)")
        if fondo is None or isinstance(fondo, str) or np.ndim(fondo) == 0:
            if piso > 0:
                raise ValueError(f"piso = {piso} necesita un fondo que sea una imagen, y "
                                 f"fondo = {fondo!r}")
            c1 = _contraste(holo, fondo)
            self._raiz = None
        else:
            c1, self._raiz, self._raiz_min = contraste_amplitud(holo, fondo, piso)
```

y en el docstring de la clase, los párrafos de `fondo` y `piso` pasan a:

```
    fondo    None si holo ya es de contraste (I/I_ref, media ~1; si no lo es,
             aborta); un escalar por el que dividir; 'media' para dividir por
             la media del propio holograma; o la IMAGEN sin muestra (la
             intensidad, p. ej. iluminacion()): entonces se normaliza por
             amplitud (contraste_amplitud) y la salida se divide por
             sqrt(fondo) proyectado a la muestra.
    piso     solo con fondo imagen: fracción de max|fondo| bajo la que no se
             divide, ni en el sensor ni a la salida (ahí t = 1). 0: se divide
             donde el fondo no es cero.
```

5. Justo ANTES de la línea `    @property` que precede a `def n_frecuencias`, añadir el método:

```python
    def _raiz_proyectada(self, y, x, z):
        """sqrt|fondo| del sensor en la malla (y, x) del plano z: el punto (y, x)
        cae en el sensor en (y, x)·L/z, por el mismo rayo, así que es |a_z|.
        Bilineal sobre la malla del sensor; 0 fuera de ella. NumPy."""
        return _bilineal(self._raiz, self.Y, self.X,
                         np.asarray(y) * self.L / z, np.asarray(x) * self.L / z)

```

6. En `campo`, reemplazar la última línea

```python
        return 1 + U * esferica(y, x, z, lamb, self.xp, self.dtype, parte="inversa")
```

por

```python
        U *= esferica(y, x, z, lamb, self.xp, self.dtype, parte="inversa")
        if self._raiz is not None:
            r = self._raiz_proyectada(y, x, z)
            luz = (r >= self._raiz_min) & (r > 0)
            inv = np.zeros_like(r)
            inv[luz] = 1 / r[luz]
            U *= a_dispositivo(inv, self.xp, np.zeros(0, dtype=self.dtype).real.dtype)
        return 1 + U
```

7. En el docstring del módulo, sección «LA FUENTE»: (a) `es solo 1.17 del centro a la esquina del` + `sensor de 3000x4000)` pasa a decir `es solo 1.17 centro/borde en media radial, 1.21 en la esquina del sensor de 3000x4000)` (reajusta el salto de línea); (b) todo lo que va desde `Para reconstruir basta dividir por iluminacion() = |a_L|²:` hasta el final de la sección (`entero empata (0.7906 frente a 0.7900 de esta división; 0.7990 sin fuente).`) se sustituye por:

```
Para reconstruir se normaliza por amplitud con el fondo iluminacion() =
|a_L|²: c1 = (I - fondo)/sqrt(fondo) en el sensor, cuya parte lineal se
retropropaga sin error a (t - 1)·|a_z|, y la salida se divide por sqrt(fondo)
llevado a la muestra por el mismo rayo, que es |a_z|. Bajo un piso no se
divide en ninguno de los dos planos: sin él, fuera del disco del pinhole la
salida explota (|t - 1| hasta 288 en 256 px). Se midió contra dividir el
holograma por |a_L|²: en el ROI del USAF del BenchmarkTarget entero, con la
fuente del banco, la correlación con el objeto empata (0.7906 frente a
0.7900; 0.7990 sin fuente), pero con esa división SSIM elige el plano a
-0.035 mm del foco, y por amplitud a 0. El precio: hacia el borde de la luz
se amplifica la imagen gemela (en 256 px, centro del disco 0.888 frente a
0.893 sin fuente; disco entero 0.656 frente a 0.817).
```

- [ ] **Step 4: Verde**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_dlhm.py -q -p no:cacheprovider`
Expected: PASS entero (incluida `test_fondo_divide_el_holograma`, que con un fondo constante da lo mismo por amplitud que dividiendo).

- [ ] **Step 5: Commit**

```bash
git add CamposT/dlhm.py tests/test_dlhm.py
git commit -m "dlhm: con un fondo imagen la vuelta normaliza por amplitud, y SSIM deja de irse del foco

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Carlos: recorte propio centrado y entrada por amplitud

**Files:**
- Modify: `scripts/mi_prueba_dlhm.py` (constante `CARLOS_RECORTE`, comentario de `PISO`, `MotorCarlos`, `main()`; se quita `contraste_enmascarado`)
- Test: `tests/test_mi_prueba_dlhm_seleccion.py`

**Interfaces:**
- Consumes (Task 9): `dlhm_exacto.contraste_amplitud(holo, fondo, piso) -> (c1, raiz, raiz_min)`.
- Produces: `MotorCarlos(holo, ventana, lamb, delta, L_fuente, xp=np, dtype=np.complex128, sobremuestreo=None, recorte=None, raiz=None, raiz_min=0.0)`; `CARLOS_RECORTE = 1024`.

- [ ] **Step 1: Las pruebas**

1. En la fixture `config`, añadir al dict `valores`, junto a `FUENTE=None, PISO=0.05,`, la entrada `CARLOS_RECORTE=96,` (en la malla de 128x160 el cuadrado centrado de 96 contiene el ROI (48, 32, 64, 64)).

2. Borrar entera la prueba `test_contraste_enmascarado_divide_solo_donde_hay_luz` (la función se va del script).

3. Al final del archivo añadir:

```python
# ── el recorte propio de Carlos (2026-09-29) ────────────────────────────────

def test_carlos_con_recorte_mide_un_trozo_de_su_recorte():
    """Spec D9: reconstruye el cuadrado centrado de `recorte` y las metricas
    se miden en la ventana, dentro de el."""
    _, c = holograma("amplitud", "intensidad")
    m = mp.MotorCarlos(c, VENTANA, LAMB, DELTA, L, recorte=96)
    assert m.ventana_maxima == (16, 32, 96, 96)
    entera = m.campo(Z, m.ventana_maxima)
    f0, c0, alto, ancho = VENTANA
    np.testing.assert_array_equal(m.campo(Z, VENTANA),
                                  entera[f0 - 16:f0 - 16 + alto, c0 - 32:c0 - 32 + ancho])


@pytest.mark.parametrize("ventana, recorte", [((0, 0, 64, 64), 96), (VENTANA, 200)])
def test_carlos_aborta_si_la_ventana_no_cabe_en_su_recorte(ventana, recorte):
    _, c = holograma("amplitud", "intensidad")
    with pytest.raises(SystemExit):
        mp.MotorCarlos(c, ventana, LAMB, DELTA, L, recorte=recorte)


def test_carlos_con_fondo_divide_su_salida_por_la_raiz():
    """Spec D5bis: con raiz, la entrada 1 + c1 va sin normalizar y la salida
    es 1 + (t - 1)/raiz en cada pixel del recorte, 1 bajo el piso."""
    a = dlhm.fuente_pinhole(PINHOLE_PRUEBA, LAMB)
    t = 1 - 0.8 * objeto_de_prueba()
    h = mp.simular(t, LAMB, DELTA, L, Z, "intensidad", device="cpu", fuente=a)
    fondo = dlhm.iluminacion(FORMA, DELTA, L, a)
    c1, raiz, raiz_min = dlhm.contraste_amplitud(h, fondo, 0.05)
    uno = mp.MotorCarlos(1 + c1, VENTANA, LAMB, DELTA, L, recorte=96,
                         raiz=np.ones_like(raiz))
    m = mp.MotorCarlos(1 + c1, VENTANA, LAMB, DELTA, L, recorte=96, raiz=raiz,
                       raiz_min=raiz_min)
    f0, c0, alto, ancho = VENTANA
    r = raiz[f0:f0 + alto, c0:c0 + ancho]
    luz = r >= raiz_min
    base = uno.campo(Z, VENTANA)
    esperado = 1 + np.where(luz, (base - 1) / np.where(luz, r, 1), 0)
    np.testing.assert_allclose(m.campo(Z, VENTANA), esperado, rtol=1e-10, atol=1e-12)
```

(`PINHOLE_PRUEBA`, `holograma`, `objeto_de_prueba`, `VENTANA`, `FORMA` ya están definidos en el archivo.)

- [ ] **Step 2: Rojo**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q -p no:cacheprovider -k "carlos_con_recorte or carlos_aborta or carlos_con_fondo"`
Expected: FAIL con `TypeError: MotorCarlos.__init__() got an unexpected keyword argument 'recorte'`.

- [ ] **Step 3: La implementación**

En `scripts/mi_prueba_dlhm.py`:

1. Justo ANTES de la línea `#: Solo MOTOR = "carlos". Presupuesto de memoria [GB] para la malla`, insertar:

```python
#: Solo MOTOR = "carlos". Lado del cuadrado CENTRADO en la malla que Carlos
#: reconstruye; las metricas se miden solo en la region medida (ROI), dentro de
#: el. Separado del ROI porque con un recorte chico Carlos no tiene NA: medido
#: el 2026-09-29 sobre el BenchmarkTarget entero (z = 3 mm), con el ROI del USAF
#: como recorte (256 px) la Tamura se iba +0.10 mm y SSIM -0.50 incluso sin
#: fuente, y con 384 centrado igual; con 1024 centrado, a 2 um. Centrado, su
#: fuente queda en el eje. Con FUENTE el recorte tiene que ser MAYOR que el ROI:
#: dividir la salida por la raiz del fondo amplifica el borde del recorte (en
#: 128x160 con recorte = ROI, la Tamura se iba -53 um; con recorte 96, a 1 um).
#: None: el recorte es el ROI, como hasta el 2026-09-29 (cuadrado, y conviene
#: centrado).
CARLOS_RECORTE = 1024

```

2. En el comentario de `PISO`, las cuatro primeras líneas

```
#: Fraccion de max|fondo| bajo la que no se divide por el fondo: ahi c - 1 = 0
#: en el motor exacto y c = 1 en el de Carlos, "sin informacion", como lo que
#: cae fuera del sensor. Dividir en los ceros de Airy explota: medido en 256 px,
#: sin piso la correlacion con el objeto cae de 0.82 a 0.02. Se usa con FUENTE
```

pasan a

```
#: Fraccion de max|fondo| bajo la que no se divide por el fondo, ni en el
#: sensor ni a la salida: ahi t = 1, "sin informacion", como lo que cae fuera
#: del sensor, en los dos motores. Dividir en los ceros de Airy explota: medido
#: en 256 px, sin piso |t - 1| llega a 288 fuera del disco. Se usa con FUENTE
```

3. Borrar entera la función `contraste_enmascarado` (desde `def contraste_enmascarado(holo, fondo, piso):` hasta su `    return c`, y una de las líneas en blanco que la separan de `def zona_iluminada`).

4. En el docstring de `MotorCarlos`, el punto 1 (desde `    1. EL RECORTE ES OBLIGADO, CENTRADO Y CUADRADO.` hasta `rectangular la esferica sale con otra curvatura en x que en y.`) pasa a:

```
    1. EL RECORTE ES OBLIGADO, CUADRADO Y, CON recorte, CENTRADO. Obligado por
       memoria: con el sensor entero la malla remuestreada pide 15 GB por paso
       (ver comprobar_memoria). Cuadrado porque reconstruir() usa el mismo
       paso, delta*P/N, en los dos ejes. Con recorte (CARLOS_RECORTE) es el
       cuadrado de ese lado centrado en la malla -la fuente en el eje- y la
       ventana medida es un trozo de el; sin recorte, la ventana misma, y
       reconstruir() pone la fuente en su centro (main() avisa si no lo esta).
```

y el párrafo final

```
    El holograma entra normalizado a [0, 1], como antes: una escala global no
    cambia ninguna metrica.
```

pasa a

```
    El holograma entra normalizado a [0, 1], como antes: una escala global no
    cambia ninguna metrica. Con raiz (un fondo imagen) entra el contraste por
    amplitud 1 + c1 SIN normalizar -la cadena reconstruye L{1} = 1 exacto,
    porque divide por la referencia propagada- y la salida se corrige como la
    del exacto: t = 1 + (t - 1)/raiz en cada pixel, 1 bajo raiz_min.
```

5. `MotorCarlos.__init__` y `MotorCarlos.campo` se reemplazan enteros (`describir` no cambia) por:

```python
    def __init__(self, holo, ventana, lamb, delta, L_fuente, xp=np,
                 dtype=np.complex128, sobremuestreo=None, recorte=None, raiz=None,
                 raiz_min=0.0):
        holo = np.asarray(holo)
        M, N = holo.shape
        f0, c0, alto, ancho = (int(v) for v in ventana)
        if recorte is None:
            if alto != ancho:
                lado = min(alto, ancho)
                raise SystemExit(
                    f"El motor de Carlos necesita un ROI CUADRADO y la region medida "
                    f"es de {alto}x{ancho} px (alto x ancho): reconstruir() usa el "
                    f"mismo paso en los dos ejes, delta*P/N, y con un recorte "
                    f"rectangular la esferica sale con otra curvatura en x que en y."
                    f"\nUsa un ROI cuadrado y centrado, por ejemplo\n\n"
                    f"    ROI = ({(N - lado) // 2}, {(M - lado) // 2}, {lado}, {lado})\n")
            rf0, rc0, lado = f0, c0, alto
        else:
            lado = int(recorte)
            if lado > min(M, N):
                raise SystemExit(f"CARLOS_RECORTE = {lado} no cabe en la malla {M}x{N}: "
                                 f"bajalo a {min(M, N)} como mucho.")
            rf0, rc0 = (M - lado) // 2, (N - lado) // 2
            if not (rf0 <= f0 and rc0 <= c0 and f0 + alto <= rf0 + lado
                    and c0 + ancho <= rc0 + lado):
                raise SystemExit(
                    f"La region medida (fila {f0}, columna {c0}, {alto}x{ancho}) se "
                    f"sale del recorte centrado de Carlos (fila {rf0}, columna {rc0}, "
                    f"{lado}x{lado}). Sube CARLOS_RECORTE o mueve el ROI.")
        h = holo[rf0:rf0 + lado, rc0:rc0 + lado]
        if raiz is None:
            self.h = h / np.abs(h).max()
            self.raiz = None
        else:
            self.h = h
            self.raiz = np.asarray(raiz, dtype=np.float64)[rf0:rf0 + lado, rc0:rc0 + lado]
            self.raiz_min = float(raiz_min)
        self.ventana = self.ventana_maxima = (rf0, rc0, lado, lado)
        self.lamb, self.delta, self.L = float(lamb), float(delta), float(L_fuente)
        self.xp, self.dtype, self.sobremuestreo = xp, dtype, sobremuestreo

    def campo(self, z, ventana):
        f0, c0, alto, ancho = (int(v) for v in ventana)
        rf0, rc0, lado, _ = self.ventana
        if not (rf0 <= f0 and rc0 <= c0 and f0 + alto <= rf0 + lado
                and c0 + ancho <= rc0 + lado):
            raise ValueError(f"el motor de Carlos solo reconstruye su recorte, "
                             f"{self.ventana}; se pidio {tuple(ventana)}")
        kw = dict(xp=self.xp, dtype=self.dtype, sobremuestreo=self.sobremuestreo)
        Rec = reconstruir(self.h, z, self.lamb, self.delta, self.L, **kw)
        U0 = onda_de_referencia(self.h.shape, z, self.lamb, self.delta, self.L, **kw)
        Rec /= self.xp.abs(U0) ** 2
        del U0
        n = self.h.shape[0]
        t = interpolar_centrado(Rec, self.delta * n / Rec.shape[0], (n, n),
                                self.delta * z / self.L, self.xp)
        if self.raiz is not None:
            r = self.xp.asarray(self.raiz)
            luz = (r >= self.raiz_min) & (r > 0)
            t = 1 + self.xp.where(luz, (t - 1) / self.xp.where(luz, r, 1), 0)
        i, j = f0 - rf0, c0 - rc0
        return t[i:i + alto, j:j + ancho]
```

6. En `main()`, la rama del motor de Carlos: desde la línea `    else:` que sigue al bloque del `MotorExacto(...)` hasta la línea `        motor = MotorCarlos(h_carlos, ventana, LAMB, DELTA, L, xp_, dtype, SOBREMUESTREO)` incluida, se reemplaza por:

```python
    else:
        if CARLOS_RECORTE is None and (abs(f0 + alto / 2 - M / 2) > 1
                                       or abs(c0 + ancho / 2 - N / 2) > 1):
            print(f"AVISO: el ROI no esta centrado en la malla {M}x{N}. El motor de "
                  f"Carlos pone la fuente en el centro\n  del recorte: su fisica ya no "
                  f"es la del banco.")
        if isinstance(fondo, np.ndarray):
            # con una imagen de fondo -la simulada o FONDO- el contraste por
            # amplitud, sin normalizar, y la salida entre la raiz del fondo, como
            # el exacto: releer con RUTA y FONDO da lo mismo que simular
            c1, raiz, raiz_min = dlhm_exacto.contraste_amplitud(holo, fondo, piso)
            motor = MotorCarlos(1 + c1, ventana, LAMB, DELTA, L, xp_, dtype,
                                SOBREMUESTREO, recorte=CARLOS_RECORTE, raiz=raiz,
                                raiz_min=raiz_min)
        else:
            motor = MotorCarlos(holo, ventana, LAMB, DELTA, L, xp_, dtype, SOBREMUESTREO,
                                recorte=CARLOS_RECORTE)
```

(las líneas que siguen, `comprobar_memoria(motor.h, ...)` y los `print` de `describir` y de `comprobar_equivalencia`, no cambian).

- [ ] **Step 4: Verde**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q -p no:cacheprovider`
Expected: PASS entero. Medido con un prototipo en esta malla, con el recorte de 96: Carlos sin fuente a 1.0 um y con fuente a 1.0 um (desfase 0); 1.3 um (desfase 0.0023). Si una prueba de punta a punta con Carlos falla la tolerancia, NO se toca: se informa con los números.

- [ ] **Step 5: Commit**

```bash
git add scripts/mi_prueba_dlhm.py tests/test_mi_prueba_dlhm_seleccion.py
git commit -m "mi_prueba_dlhm: Carlos reconstruye su propio recorte centrado y normaliza por amplitud

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: A escala real otra vez, y dejarlo escrito

Sustituye a la Task 8: mismos archivos y mismo cierre, con el código de las Tasks 9 y 10.

**Files:**
- Modify: `scripts/mi_prueba_dlhm.py` (solo la cabecera)
- Modify: `docs/superpowers/specs/2026-09-29-fuente-dlhm-design.md` (sección final de resultado)
- Modify: `docs/Cronograma_TG_DLHM.xlsx` (tarea 80, fila 83)

- [ ] **Step 1: Las cuatro corridas**

Una por una desde la raíz, en Git Bash (cada una simula ~40 s en la GPU y barre 2.5-3.5 mm), con los valores por defecto: ROI = el USAF central, `CARLOS_RECORTE = 1024`, `PISO = 0.05`.

```bash
MPLBACKEND=Agg Tesis_env/Scripts/python.exe -c "import scripts.mi_prueba_dlhm as mp; mp.MOSTRAR = False; mp.main()"
```

```bash
MPLBACKEND=Agg Tesis_env/Scripts/python.exe -c "import scripts.mi_prueba_dlhm as mp; mp.MOSTRAR = False; mp.FUENTE = 'pinhole'; mp.main()"
```

```bash
MPLBACKEND=Agg Tesis_env/Scripts/python.exe -c "import scripts.mi_prueba_dlhm as mp; mp.MOSTRAR = False; mp.MOTOR = 'carlos'; mp.main()"
```

```bash
MPLBACKEND=Agg Tesis_env/Scripts/python.exe -c "import scripts.mi_prueba_dlhm as mp; mp.MOSTRAR = False; mp.MOTOR = 'carlos'; mp.FUENTE = 'pinhole'; mp.main()"
```

Expected: las tres métricas a menos de medio paso grueso (0.005 mm) de 3.000 con el exacto, y a menos de un paso grueso (0.01 mm) con Carlos. Se anota de cada una la tabla de focos (foco, error, FWHM, alt/riz) y los tiempos. En `campos.png` de las dos corridas con fuente, el holograma tiene que mostrar el disco brillante (los anillos de Airy son del 1.75 % del pico: no se ven en escala lineal, y así se dice). Si algo falla, NO se toca ningún umbral ni código: se informa BLOCKED con los números.

- [ ] **Step 2: La cabecera de mi_prueba_dlhm.py**

Justo ANTES del párrafo que empieza `EL ~0.03 DE ANTES, EXPLICADO.`, una sección nueva en ASCII, en prosa densa como el resto de la cabecera, titulada `LA FUENTE (FUENTE, PISO) Y EL RECORTE DE CARLOS, 2026-09-29.`, que diga: (1) la ida exacta supone una fuente ideal y da I/I_ref, y el aspecto de los hologramas de Lopera y Carlos es la emisión de la fuente (media radial centro/borde: exacto 0.98, esférica ideal sin dividir 1.17, envolvente de dlhm() 58.6, su modelo 54.7); `FUENTE = "pinhole"` mete el Airy del pinhole de 5 um en la ida (primer cero a 710 px); (2) la reconstrucción normaliza por amplitud: dividir por |a_L|^2 empataba en correlación (USAF: 0.7900 contra 0.7906, 0.7990 sin fuente) pero SSIM elegía el plano a -0.035 mm; (3) Carlos reconstruye su propio recorte centrado (`CARLOS_RECORTE = 1024`): con el ROI del USAF como recorte se iba +0.10 mm (Tamura) incluso sin fuente. Y a continuación la tabla del Step 1, con este formato:

```
    motor     fuente     metrica     foco     error    FWHM   alt/riz
    exacto    ninguna    tamura      3.0000   +0.000   ...    ...
```

En la misma cabecera, la frase del cuarto central que dice `Queda por defecto porque da el pico la mitad de ancho` se corrige para decir que eso vale con `ROI = None` (el defecto ahora es el USAF).

- [ ] **Step 3: La spec**

Al final de `docs/superpowers/specs/2026-09-29-fuente-dlhm-design.md`, una sección `## Resultado de la verificación a escala real (2026-09-29)` con la tabla del Step 1 y una línea por cada punto de la Verificación (y de la sección de revisión) con su número.

- [ ] **Step 4: El cronograma**

La tarea 80 en la fila 83 de la hoja `Tareas`, con el script del Step 4 de la Task 8 (copia el estilo de la fila 82, comprueba que los IDs 1..79 no tienen huecos, pone el autofiltro en A3:I83). Las notas, en ASCII puro (sin tildes ni ñ) y densas: el diagnóstico (no era regresión: HEAD regenera el holograma del 28/09 a 1.8e-4; la envolvente de Lopera/Carlos es la emisión de la fuente, 0.98 contra 58.6 centro/borde), RS con fuente 1.1e-5 y 4.6e-6, la ida y vuelta en el centro del disco 0.888/0.893, las dos decisiones del usuario (amplitud en vez de |a_L|^2 por el SSIM a -0.035; recorte propio de Carlos de 1024, que con 256 se iba +0.10 mm), la tabla del Step 1, los hashes de los commits de la rama (`git log --oneline 56f3694..HEAD`) y lo abierto: la NA real del banco sin medir (FUENTE apagada por defecto) y la gemela amplificada hacia el borde de la luz.

Run después: `Tesis_env/Scripts/python.exe -c "import openpyxl; ws = openpyxl.load_workbook('docs/Cronograma_TG_DLHM.xlsx')['Tareas']; print([ws.cell(83, c).value for c in range(1, 8)], ws.auto_filter.ref)"`
Expected: la fila 80 con `Hecho` y el autofiltro `A3:I83`.

- [ ] **Step 5: Verde y commit**

Run: `Tesis_env/Scripts/python.exe -m pytest tests/test_mi_prueba_dlhm_seleccion.py -q -p no:cacheprovider` (la cabecera es un docstring).

```bash
git add scripts/mi_prueba_dlhm.py docs/superpowers/specs/2026-09-29-fuente-dlhm-design.md docs/Cronograma_TG_DLHM.xlsx
git commit -m "Cronograma y cabecera: la fuente en la ida DLHM, medida a escala real

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
