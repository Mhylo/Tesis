"""malla_remuestreada() de scripts/mi_prueba_dlhm.py devuelve tamanos PARES.

REGRESION. La malla era int(lado*of), y con los numeros del montaje (528 nm,
1.83 um, z = 3, L = 10) un sensor de 1024 daba 2027: impar. Con P impar, fts()
deja la continua en el indice (P+1)/2 y la rejilla de angular_spectrum(),

    linspace(-P/2*dfx, (P/2 - 1)*dfx, P)

le asigna ahi +dfx/2: no pasa por f = 0. H se evalua medio paso corrida y la
reconstruccion sale desplazada lambda*(L - z)/(2W) = 0.99 um en el plano de la
muestra, del orden del detalle del objeto. Es el fallo que documenta
frecuencias_fft() en CamposT/propagadores.py. Medido con
scripts/verif_dlhm_esferico.py (parte carlos): correlacion con la
reconstruccion exacta de CamposT.dlhm de 0.875 con la malla impar y 0.990 con
la par.

El arreglo va en la malla, no en angular_spectrum(): esa es copia literal del
dlhm.py de Carlos y no se toca.

Lo que se fija:

- que las dos dimensiones salen pares, en malla cuadrada y en rectangular,
  donde cada eje se redondea por su cuenta;
- que es el par SIGUIENTE: 2027 sube a 2028 y no baja a 2026, y lo que ya era
  par no se mueve;
- que comprobar_memoria() presupuesta la malla par, que es la que va a pedir
  reconstruir();
- y la consecuencia fisica: un punto fuera del eje se reconstruye en su sitio.
"""

import matplotlib
matplotlib.use("Agg")           # antes de importar el script, que trae pyplot

import re

import numpy as np
import pytest

from scripts import mi_prueba_dlhm as mp

#: Los numeros del montaje, los de CamposT/montaje.py. Van escritos y no
#: importados: los valores que se fijan abajo salen de ellos, y el dia que el
#: montaje se mida de verdad esta prueba no tiene por que caerse.
LAMB, DELTA, Z, L = 528e-6, 1.83e-3, 3.0, 10.0


# ── la paridad ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("lado_x, lado_y", [
    (256, 256), (512, 512), (750, 750), (1024, 1024), (2048, 2048),
    (3000, 3000), (1024, 768), (4000, 3000)])
def test_la_malla_remuestreada_es_par(lado_x, lado_y):
    """Con el montaje, int(lado*of) salia IMPAR en todos los cuadrados de la
    lista -137, 541, 1131, 2027, 6645, 11635-, y en los rectangulares un eje
    par y el otro impar -1548x2065 y 13912x18549-."""
    N, M = mp.malla_remuestreada(lado_x, lado_y, Z, LAMB, DELTA)
    assert N % 2 == 0 and M % 2 == 0, (N, M)


@pytest.mark.parametrize("lado_x, lado_y, malla", [
    (1024, 1024, (2028, 2028)),         # 2027.9: int da 2027
    (1024, 768, (1548, 2066))])         # 1548.8 ya par; 2065.1 sube
def test_redondea_al_par_siguiente_eje_por_eje(lado_x, lado_y, malla):
    """Hacia ARRIBA, porque subir no le quita nunca un punto a la malla: el
    paso W/N solo puede afinarse. Bajar a 2026 lo engrosaria por encima de
    W/2027.9, el que pide el muestreo.

    Y eje por eje: en el rectangular N ya era par y tiene que quedarse igual
    mientras M sube.
    """
    assert mp.malla_remuestreada(lado_x, lado_y, Z, LAMB, DELTA) == malla


@pytest.mark.parametrize("lado, malla", [(3000, 13144), (1024, 2802),
                                         (750, 1624)])
def test_las_mallas_que_ya_eran_pares_no_se_mueven(lado, malla):
    """La tabla de la cabecera de mi_prueba_dlhm.py (532 nm, 1.85 um, z = 2).

    Ahi int() ya salia par en los tres recortes, y el arreglo no puede tocar
    esas mallas: con ellas estan medidas la memoria y los tiempos que la tabla
    cita. Es lo que separa "el par siguiente" de redondear x a 2*ceil(x/2),
    que tambien da pares pero en 1624.4 da 1626.
    """
    assert mp.malla_remuestreada(lado, lado, 2.0, 532e-6, 1.85e-3) == (malla, malla)


def test_comprobar_memoria_presupuesta_la_malla_par(monkeypatch):
    """El aviso y la tabla de recortes cuentan la malla que reconstruir() va a
    pedir, no la de int().

    comprobar_memoria() no calcula la malla por su cuenta: llama a
    malla_remuestreada(), y por eso hereda el redondeo. Esto fija que siga
    siendo asi. Con el montaje, un holograma de 1024 y un tope de 0.01 GB
    aborta con la tabla de 1024, 750 y 512, que con int() salian 2027, 1131 y
    541.
    """
    monkeypatch.setattr(mp, "SOBREMUESTREO", None)
    with pytest.raises(SystemExit) as fallo:
        mp.comprobar_memoria(np.zeros((1024, 1024)), [Z, 4.0], LAMB, DELTA,
                             tope_gb=0.01)
    texto = str(fallo.value)
    assert "malla de 2028x2028" in texto
    mallas = [int(n) for par in re.findall(r"(\d+)x(\d+)", texto) for n in par]
    assert len(mallas) == 8, texto              # el aviso y tres filas
    assert all(n % 2 == 0 for n in mallas), mallas


# ── la consecuencia: el punto cae en su sitio ───────────────────────────────

def holograma_de_un_punto(n, y0, x0):
    """V/S_L de un absorbente puntual en (y0, x0) de la muestra, sobre un
    sensor de n x n centrado en el eje.

    V es el nucleo RS-I (el de rs_directo en tests/test_dlhm.py) y S_L la
    esferica en el sensor. La iluminacion en el punto es una constante y no
    mueve nada, asi que no va. Complejo y sin el 1 del fondo: el unico pico
    de la reconstruccion es el del punto.
    """
    g = (np.arange(n) - n / 2) * DELTA
    X, Y = np.meshgrid(g, g)
    k, d = 2 * np.pi / LAMB, L - Z
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + d * d)
    V = d / (2 * np.pi * r * r) * (1 / r - 1j * k) * np.exp(1j * k * r)
    R = np.sqrt(X ** 2 + Y ** 2 + L ** 2)
    return V / (np.exp(1j * k * R) / R)


def test_un_punto_desplazado_se_reconstruye_en_su_sitio():
    """El caso medido: el montaje y un sensor de 1024, con la malla que elige
    reconstruir() por geometria.

    Con la malla impar el punto se corria lambda*d/(2W) = 0.99 um en los dos
    ejes -medido 0.86, porque cv.resize le devuelve 0.14 en sentido
    contrario-. Con la par queda solo ese residuo de cv.resize, que alinea
    los centros de pixel y no las coordenadas: (delta - p)/(2M) = 0.14 um,
    medido 0.14 y 0.15. Es de resize(), copia literal de Carlos, y no de la
    malla.

    Un tercio del corrimiento deja margen de dos a uno a cada lado. El punto
    va distinto en y que en x para que un eje cruzado no pase. Cuesta ~3 s.
    """
    n, y0, x0 = 1024, -0.030, 0.050
    Rec = mp.reconstruir(holograma_de_un_punto(n, y0, x0), Z, LAMB, DELTA, L,
                         sobremuestreo=None)
    filas, cols = Rec.shape
    p = n * DELTA / filas               # el sensor entero en la malla nueva
    # centroide de |Rec|^2 alrededor del maximo: el lobulo central del punto
    # mide lambda*d/W = 2 um, unos dos pixeles a cada lado
    w = np.abs(Rec) ** 2
    i, j = np.unravel_index(np.argmax(w), w.shape)
    h = 6
    w = w[i - h:i + h + 1, j - h:j + h + 1]
    yc = w.sum(1) @ ((np.arange(i - h, i + h + 1) - filas / 2) * p) / w.sum()
    xc = w.sum(0) @ ((np.arange(j - h, j + h + 1) - cols / 2) * p) / w.sum()
    tol = LAMB * (L - Z) / (2 * n * DELTA) / 3
    assert abs(yc - y0) < tol, f"y: {(yc - y0) * 1e3:+.3f} um"
    assert abs(xc - x0) < tol, f"x: {(xc - x0) * 1e3:+.3f} um"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
