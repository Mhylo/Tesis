# Diapositiva 9 corregida — la errata de Zhao et al. (2020)

Reemplazo de la lámina «HALLAZGO NO PREVISTO» del informe de avance 1
(`CamiloR_Avance_1.pptx`, diapositiva 9) para el informe de avance 2.

Todo lo que sigue está medido con `scripts/kf_evolucion.py`, que imprime las
cifras y regenera las figuras:

```
Tesis_env/Scripts/python.exe -m scripts.kf_evolucion --diapo
```

---

## Por qué hay que cambiarla

La lámina actual dice, como titular:

> «A la distancia de trabajo del montaje, **Kf sale 4.8 veces mayor** de lo que
> indica la fórmula publicada.»

y su figura se titula «K_f sobreestimado hasta 4.8x». **La cifra es correcta y
la figura reproduce exactamente**: con N = 512, λ = 405 nm, δ = 3.45 µm y s = 1,
a z = 150 mm el código da K_f = 15.26 y la Ec. (14) da 3.16. Razón 4.83.

El problema es que **ese 4.8 no es una propiedad de la errata, sino de esa
configuración**, y hay tres consecuencias que invalidan el titular tal como está
escrito.

### 1. La razón entre las dos fórmulas es 1/(s·N·λ)

Lejos del umbral, `sqrt(A² + B) → sqrt(B)` mientras que `sqrt(A² · B) →
A·sqrt(B)`, así que el radicando del código lleva un factor `A = (sNλ)²` de más
y K_f queda dividido por `sqrt(A) = s·N·λ`. Verificado contra `kf_auto()` con un
error relativo máximo de **1.8e-5**.

| escenario | s·N·λ | razón código/paper | |
|---|---|---|---|
| diapositivas del avance 1 (N=512, λ=405 nm, s=1) | 0.207 mm | **4.82** | sobreestima |
| Tabla 1 del paper (N=512, λ=632.8 nm, s=4) | 1.296 mm | **0.77** | subestima |
| holograma DLHM de referencia (N=3000, λ=532 nm, s=1) | 1.596 mm | **0.63** | subestima |

El efecto **cambia de signo** en `s·N·λ = 1 mm`.

### 2. Con nuestro propio holograma el titular se invierte

`Simulated_hologram.png` es de 3000 × 3000 px. A esa N el código publicado
**subestima** K_f en un 37 %, no lo sobreestima. La lámina actual se leería al
revés sobre los datos con los que estamos trabajando.

### 3. El número depende de λ, que todavía no se ha medido

λ es uno de los quince parámetros que `CamposT/montaje.py` tiene como
`SIN_MEDIR` (tarea 17 del cronograma). Con N = 512 y s = 1:

```
405 nm -> 4.8x     532 nm -> 3.7x     632.8 nm -> 3.1x
```

El titular se mueve un 36 % según un número que aún no existe. Y 405 nm era
precisamente una de las tres longitudes de onda sueltas que ese módulo se creó
para eliminar.

### 4. Y en el DLHM real la errata no llega a activarse

Éste es el punto más serio, porque contradice el «a la distancia de trabajo del
montaje». Con la geometría del modelo de referencia —L = 8 mm, z = 2 mm, o sea
propagar **L − z = 6.000 mm**, sobre 3000 px de 1.85 µm a 532 nm— el umbral de
compresión está en

    z_umbral = δ²·s·N/λ = 19.3 mm

y por debajo de él **K_f vale 1 en las dos fórmulas**: la errata no cambia
absolutamente nada. Los 150 mm que la figura etiqueta como «z del montaje DLHM»
son la distancia de las pruebas de onda plana de las diapositivas 3–6, no una
distancia DLHM.

---

## El argumento que sí es sólido, y es más fuerte

`1/(s·N·λ)` **tiene unidades de 1/longitud**. Ésa es la prueba de que la versión
con producto no puede ser correcta, y no depende de ninguna configuración: una
errata dimensionalmente consistente daría un factor fijo y adimensional. Que el
mismo error sobreestime en un montaje y subestime en otro *es* el síntoma.

Dicho de otro modo: el hallazgo no es «sobreestima 4.8×», es **«el error escala
como 1/(s·N·λ), luego es dimensional»**. Esa afirmación sobrevive a que midas λ.

---

## Texto de la lámina

> **HALLAZGO NO PREVISTO**
>
> # Errata dimensional en el código publicado por Zhao et al. (2020)
>
> El operador escrito es `*+`, que Python lee como producto. El K_f resultante
> difiere del de la Ec. (14) en un factor **1/(s·N·λ)** — que tiene unidades, y
> por eso sobreestima o subestima según el montaje.
>
> | | |
> |---|---|
> | **Ec. (14) del artículo** | (Nλ)⁴ **+** (8Nλz)² |
> | **Código de referencia** | (Nλ)⁴ **×** (8Nλz)² |
>
> - **En nuestro escenario de onda plana** (N = 512, λ = 405 nm): sobreestima 4.8×
> - **En nuestro holograma DLHM** (N = 3000, λ = 532 nm): subestima 1.6×
> - **En la geometría DLHM real** (propagar 6 mm, umbral 19.3 mm): K_f = 1 en las
>   dos. La errata no interviene.
>
> *Figura:* `resultados/kf/diapo_kf.png`

### Notas del orador

- El hallazgo se mantiene: la fórmula publicada y el código publicado no son la
  misma fórmula, y el código es el que está mal.
- Lo que cambia respecto al avance 1 es el alcance: antes se presentó como un
  factor fijo de 4.8×, y es un factor que depende del montaje.
- Si preguntan «¿entonces les afecta a ustedes?»: en la reconstrucción DLHM, no,
  porque a 6 mm no hay compresión. Afecta al régimen de larga distancia, que es
  donde MPASM se justifica frente a FFT-ASM, y es exactamente el régimen del
  Objetivo 1.
- Si preguntan por λ: aún sin medir; es la tarea 17 y la hoja de parámetros está
  lista para el laboratorio. El argumento dimensional no depende de ese dato.

### Lo que NO hay que volver a decir

- «K_f sobreestimado hasta 4.8x» como enunciado general.
- «A la distancia de trabajo del montaje», si esa distancia son 150 mm: no lo es.
- Cualquier cifra concreta de la razón sin decir con qué N, λ y s se calculó.
