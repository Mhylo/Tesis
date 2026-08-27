# Retropropagación de hologramas — diseño

Fecha: 2026-08-20
Rama: referencias-y-versionado

## Problema

El paquete propaga campos hacia adelante (`pipeline.propagar`), pero no hay
forma de recorrer el camino inverso: partir de un holograma medido —una
imagen de intensidad— y recuperar el objeto que lo produjo. Es el paso que
convierte los propagadores en una cadena de reconstrucción, y lo que hace
falta para el Objetivo 1 sobre datos reales y no sólo sobre el target
sintético.

El requisito que abre todo lo demás: la imagen a retropropagar la elige el
usuario en cada corrida, sin editar código.

## Alcance

Dentro:

- Cargar una imagen como holograma (intensidad medida) y convertirla en campo.
- Retropropagarla con los tres propagadores del paquete: FFT-ASM, BL-ASM y
  MPASM.
- Barrido de distancias de reconstrucción (pila de foco), porque la distancia
  de enfoque no se conoce de antemano.
- Escribir la intensidad reconstruida, un PNG por método y por distancia.
- CLI con argparse: la imagen y los parámetros físicos entran por línea de
  órdenes.

Fuera, a propósito:

- **Corrección DLHM de fuente puntual.** Un holograma sin lente iluminado por
  un pinhole divergente tiene magnificación M = L/z, así que reconstruirlo
  exige reescalar el paso de píxel y usar una distancia efectiva. Este módulo
  asume iluminación colimada (onda plana, in-line clásico). La geometría real
  del montaje aún no está decidida; cuando lo esté, entra como una capa de
  cambio de coordenadas ANTES de llamar a `retropropagar`, sin tocar lo que
  aquí se define.
- **Supresión de la imagen gemela.** La retropropagación de un holograma
  in-line siempre superpone la imagen gemela desenfocada. Eliminarla
  (Gerchberg-Saxton, resta de fondo, phase-shifting) es un problema aparte.
  El módulo muestra la reconstrucción cruda.
- **Autofoco.** El barrido produce la pila; qué distancia enfoca lo decide
  quien mira. Añadir una métrica de nitidez es una decisión que habría que
  justificar aparte.
- **Fase, CSV de tiempos, CSV de contraste entre métodos.** No se escriben.

## Decisiones

### 1. Un módulo del paquete, no un script

`CamposT/retropropagacion.py`, séptimo módulo, con la misma disciplina que
los demás. El archivo previo `CamposT/Retropopagacion.py` se renombra: el
nombre tenía una errata (falta la `r` de *retro-propagación*) e iba en
mayúscula contra la convención del paquete. Su contenido (cuatro líneas,
`from propagadores import ...` sin el prefijo `CamposT.`, con `U0`, `device`
y `dtype` sin definir) se reescribe entero.

Va en `CamposT/` y no en `scripts/` porque define una operación del dominio
—retropropagar es tan primitivo como propagar—, no un experimento concreto.
`scripts/` es para los barridos que generan las figuras de la tesis.

### 2. El holograma entra por `load_field(mode="holograma")`

Un holograma es intensidad: el campo es `U = sqrt(I)`, no `I`. Ningún modo
actual de `load_field` hace eso — `mode="amplitud"` devuelve la transmitancia
tal cual, que es lo correcto para un objeto y falso para un holograma.

Se añade un cuarto modo a `load_field`, en vez de una función de carga propia
en el módulo nuevo, porque `campos.py` declara ser el único sitio que fabrica
campos de entrada, y una función propia duplicaría el `convert("L")`, el
`resize`, el `invert` y el normalizado que ya están allí.

### 3. Convención de signo: el usuario da distancias positivas

`retropropagar(U_h, delta, lamb, zs, ...)` recibe `zs` **positivos** —la
distancia sensor→objeto, como se mide en el montaje— y propaga internamente a
`−z`. Queda en el docstring y fijado por un test.

**Corrección respecto a lo que se supuso al diseñar esto.** Se pensó que
invertir el signo daría una reconstrucción visiblemente peor, y que ésa sería
la prueba. Es falso, y la implementación lo midió: `U_h = sqrt(I)` es real, y
para entrada real `U(−z) = conj(U(+z))`, luego `|U(−z)|² = |U(+z)|²`
**exactamente** (3.5e-16 sobre el escenario de prueba). Una pila de foco
calculada con el signo invertido sale idéntica imagen por imagen. Es la misma
simetría que genera la imagen gemela.

Consecuencia de diseño: el signo se verifica por contrato contra
`pipeline.propagar`, no por inspección de reconstrucciones, y la suite incluye
una prueba que deja escrita la ambigüedad como límite conocido del método. El
signo sí importa en la fase del campo devuelto y en cuanto se encadene un paso
no real —la corrección DLHM, un filtro complejo, recuperación de fase—, que es
hacia donde va esto.

### 4. No se reimplementa ningún propagador

`retropropagar` delega en `pipeline.propagar(..., z=-z, metodo=...)`. Los tres
métodos ya comparten firma; el módulo sólo orquesta el producto
métodos × distancias y filtra los kwargs específicos de MPASM (`s`, `Kf`,
`r`, `mag`) para no pasárselos a FFT-ASM ni a BL-ASM, que no los aceptan.

### 5. `s = 1` por defecto en la CLI

`mpasm()` tiene `s=10` por defecto. Sobre 512×512 con `pad=2` eso es una
matriz espectral de 10240², ~840 MB en complex64, por cada distancia del
barrido. Con 25 distancias es inviable. La CLI usa `s=1` y `--s` lo sube para
el caso de una sola distancia con sobremuestreo.

## Interfaz

```python
barrido_z(z, pasos=25) -> np.ndarray
    # z escalar o [z0]      -> array de un elemento
    # z = [z0, z1]          -> linspace(z0, z1, pasos)

retropropagar(U_h, delta, lamb, zs, metodos=("fft","blas","mpasm"),
              pad=2, device="auto", **kw)
    # genera (metodo, z, campo, info), z positivo, campo ya retropropagado
```

CLI:

```
python -m CamposT.retropropagacion HOLOGRAMA.png --z 10 60 --pasos 25
       [--delta 3.45e-3] [--lamb 405e-6] [--metodos fft blas mpasm]
       [--device auto|cpu|gpu] [--N 512] [--pad 2] [--s 1]
       [--gamma 0.6] [--invert] [--salida DIR]
```

## Salidas

Siguiendo la convención de `resultados/campos/` —carpeta por propagador para
que el método se lea en la ruta, parámetros en el nombre del archivo:

```
resultados/retropropagacion/<nombre_del_holograma>/
    fft/z0010.000.png  z0012.083.png  ...
    blas/...
    mpasm/...
```

El nombre lleva tres decimales, y no el `z0020.png` entero de
`resultados/campos/`, porque un barrido `linspace` da distancias no enteras y
redondearlas haría colisionar archivos distintos.

Contenido: intensidad |U|² normalizada, PNG de 8 bits, `gamma=0.6`, vía
`pipeline.intensidad` y `pipeline.guardar`.

## Verificación

`tests/test_retropropagacion.py`, en el estilo del repo: propiedades, no
comparación contra salidas guardadas.

1. **Re-localización.** Partícula opaca de 8 px sobre fondo transparente,
   N=128, z=16 mm. Se mide qué fracción del contraste `|I/Ī − 1|` cae dentro
   de la huella del objeto. La huella es el 0.39 % del plano; el holograma
   tiene ahí el 0.10 % de su contraste —está completamente deslocalizado— y la
   reconstrucción devuelve ~7.5 %, veinte veces la huella. Ese salto es lo que
   hace la retropropagación y lo que ningún filtro de suavizado produce.

   No se compara contra el objeto término a término: la imagen gemela hace que
   la reconstrucción nunca sea el objeto, y una métrica de parecido mezclaría
   "funciona la retropropagación" con "cuánto estorba la gemela".
2. **La ambigüedad del signo, medida.** `U(−z) == conj(U(+z))` y las dos
   intensidades coinciden. Deja escrito en la suite el límite descrito arriba.
3. **Contrato de signo.** `retropropagar(..., zs=[z])` da exactamente
   `propagar(..., -z)`, para los tres métodos. Fija la convención.
4. **`load_field(mode="holograma")`** devuelve la raíz de la transmitancia,
   contrastado contra `mode="amplitud"` sobre la misma imagen, y un modo mal
   escrito falla en vez de caer en un defecto silencioso.
5. **`barrido_z`** con un valor, con dos, y el error con tres. Más los nombres
   de archivo: 25 distancias de un `linspace` dan 25 nombres distintos, en
   orden alfabético igual al del barrido.
6. **Filtrado de kwargs.** `retropropagar(..., s=2)` no revienta con FFT-ASM
   ni BL-ASM; un kwarg ajeno (`ss=2`) falla en vez de tragarse en silencio.

Las pruebas 1-3 corren en CPU y GPU vía el fixture `device` de `conftest.py`,
como el resto de la suite.

## Estado

Implementado y verificado: 19 pruebas nuevas, suite completa en 177 pasando.
Comprobado además de extremo a extremo con un holograma sintético del target
USAF grabado a z = 30 mm: el barrido `--z 20 40 --pasos 5` reconstruye el
target reconocible en z = 30 y desenfocado en el resto, con los tres métodos.
