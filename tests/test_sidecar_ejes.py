"""La guarda del .txt: lambda, delta y la CONVENCION DE EJES.

EL FALLO QUE MOTIVA EL FICHERO. El 08/09 se retropropagó un holograma escrito
por FFT-ASM con `mi_prueba_mpasm`, y no enfocaba. No fallaba: devolvía un campo
plausible y un barrido con el pico en cualquier sitio.

La causa no era MPASM. `angularSpectrum` de pyDHM construye la rejilla de
frecuencias con dfx del número de FILAS y dfy del de COLUMNAS —los ejes
CRUZADOS—, mientras que `espectro_angular_bl` y `mpasm_bloques` los llevan cada
uno a su longitud. En malla CUADRADA las dos convenciones son la misma rejilla;
en RECTANGULAR son dos transformadas distintas y la vuelta no deshace la ida.
Medido sobre el holograma 3000x4000 de resultados/hologramas/BenchmarkTarget:

    espectro_angular con cruzados=True   1.0000   <- el que lo escribió
    espectro_angular con cruzados=False  0.8018
    mpasm_bloques                        0.8018   <- idéntico al anterior

o sea que MPASM hacía exactamente lo que debía. Los dos 0.8018 iguales son la
prueba de que la diferencia era entera de la convención de ejes.

El .txt del holograma YA decía `EJES_CRUZADOS = True` y `malla = 3000x4000`, y
nadie lo miraba: la guarda sólo comprobaba lambda y delta. Estas pruebas fijan
que ahora también compruebe la convención, que es la misma decisión que ya se
había tomado con lambda —cuando el archivo DICE con qué se hizo, no hay razón
para adivinarlo.
"""

import matplotlib
matplotlib.use("Agg")           # antes de importar los scripts, que traen pyplot

import importlib
import inspect

import numpy as np
import pytest

#: Los tres barridos puntuados de onda plana, con su convención declarada.
CONVENCION = {
    "scripts.mi_prueba_angular": True,      # angularSpectrum los cruza
    "scripts.mi_prueba_blas": False,        # espectro_angular_bl no
    "scripts.mi_prueba_mpasm": False,       # mpasm_bloques tampoco
}

LAMB, DELTA = 633e-6, 3.45e-3


@pytest.fixture(params=sorted(CONVENCION))
def mod(request):
    m = importlib.import_module(request.param)
    m._nombre_test = request.param
    return m


def sidecar(tmp_path, **claves):
    """Escribe un .txt de holograma y devuelve la ruta del .npy hermano.

    El .npy no hace falta que exista: _sidecar() sólo abre el .txt. Y el nombre
    lleva decimales a propósito -"z0010.000.npy"-, que es el caso que rompía a
    with_suffix() y que _sidecar() esquiva cortando la extensión a mano.
    """
    base = {"lambda [mm]": LAMB, "delta [mm]": DELTA, "malla": "3000x4000",
            "propagador": "el que sea"}
    base.update(claves)
    (tmp_path / "z0010.000.txt").write_text(
        "\n".join(f"{k} = {v}" for k, v in base.items() if v is not None),
        encoding="utf-8")
    return tmp_path / "z0010.000.npy"


# ── declaración de la convención ────────────────────────────────────────────

def test_cada_script_declara_su_convencion(mod):
    assert isinstance(mod.EJES_CRUZADOS, bool)
    assert mod.EJES_CRUZADOS is CONVENCION[mod._nombre_test]


def test_la_guarda_es_la_misma_en_los_tres():
    """Se permite la copia —cada script tiene que poder leerse solo— pero no
    que diverja. Si hay que cambiar una, se cambian las tres."""
    ms = [importlib.import_module(n) for n in sorted(CONVENCION)]
    for f in ("_sidecar", "_comprobar_con_el_sidecar"):
        fuentes = {inspect.getsource(getattr(m, f)) for m in ms}
        assert len(fuentes) == 1, f"{f} ha divergido entre los tres scripts"


# ── lo que ya comprobaba, que debe seguir comprobando ───────────────────────

def test_sin_txt_no_dice_nada(mod, tmp_path):
    """Los hologramas de terceros no traen .txt: la guarda calla y deja pasar."""
    mod._comprobar_con_el_sidecar(tmp_path / "suelto.png", LAMB, DELTA)


def test_lambda_distinta_aborta(mod, tmp_path):
    ruta = sidecar(tmp_path, **{"lambda [mm]": 532e-6})
    with pytest.raises(SystemExit, match="lambda"):
        mod._comprobar_con_el_sidecar(ruta, LAMB, DELTA)


def test_delta_distinto_aborta(mod, tmp_path):
    ruta = sidecar(tmp_path, **{"delta [mm]": 1.85e-3})
    with pytest.raises(SystemExit, match="delta"):
        mod._comprobar_con_el_sidecar(ruta, LAMB, DELTA)


# ── la convención de ejes ───────────────────────────────────────────────────

def test_ejes_iguales_pasa(mod, tmp_path):
    ruta = sidecar(tmp_path, EJES_CRUZADOS=mod.EJES_CRUZADOS)
    mod._comprobar_con_el_sidecar(ruta, LAMB, DELTA)


def test_ejes_distintos_en_malla_rectangular_aborta(mod, tmp_path):
    """El caso real: un holograma de FFT-ASM reconstruido con MPASM."""
    ruta = sidecar(tmp_path, EJES_CRUZADOS=not mod.EJES_CRUZADOS,
                   malla="3000x4000")
    with pytest.raises(SystemExit, match="ejes"):
        mod._comprobar_con_el_sidecar(ruta, LAMB, DELTA)


def test_ejes_distintos_en_malla_cuadrada_pasa(mod, tmp_path):
    """En malla cuadrada las dos convenciones son la MISMA rejilla.

    Abortar aquí sería un falso positivo que prohibiría un caso correcto: es la
    mitad de la guarda que hace que no estorbe.
    """
    ruta = sidecar(tmp_path, EJES_CRUZADOS=not mod.EJES_CRUZADOS,
                   malla="2048x2048")
    mod._comprobar_con_el_sidecar(ruta, LAMB, DELTA)


def test_sin_la_clave_se_asume_en_su_sitio(mod, tmp_path):
    """retro_blas y retro_mpasm no anotan la clave, y no cruzan los ejes.

    Así que su ausencia significa False, y sólo debe abortar en el script que
    SÍ los cruza.
    """
    ruta = sidecar(tmp_path, EJES_CRUZADOS=None, malla="3000x4000")
    if mod.EJES_CRUZADOS:
        with pytest.raises(SystemExit, match="ejes"):
            mod._comprobar_con_el_sidecar(ruta, LAMB, DELTA)
    else:
        mod._comprobar_con_el_sidecar(ruta, LAMB, DELTA)


def test_sin_la_malla_se_comprueba_igual(mod, tmp_path):
    """Si no consta la forma, no se puede descartar que sea rectangular.

    Se prefiere la parada de más a la reconstrucción muda, que es la misma
    política que el resto del repo: fallar fuerte antes que devolver un número
    que parece bueno.
    """
    ruta = sidecar(tmp_path, EJES_CRUZADOS=not mod.EJES_CRUZADOS, malla=None)
    with pytest.raises(SystemExit, match="ejes"):
        mod._comprobar_con_el_sidecar(ruta, LAMB, DELTA)


def test_la_convencion_se_puede_forzar_por_argumento(mod, tmp_path):
    """La salida de emergencia que el propio mensaje ofrece.

    Sirve para cruzar propagadores a propósito, que es un experimento legítimo:
    lo que la guarda impide es hacerlo SIN SABERLO.
    """
    ruta = sidecar(tmp_path, EJES_CRUZADOS=True, malla="3000x4000")
    mod._comprobar_con_el_sidecar(ruta, LAMB, DELTA, cruzados=True)
    with pytest.raises(SystemExit, match="ejes"):
        mod._comprobar_con_el_sidecar(ruta, LAMB, DELTA, cruzados=False)


def test_el_mensaje_dice_que_propagador_lo_escribio(mod, tmp_path):
    """Un mensaje que no dice de dónde viene el holograma no ayuda a arreglarlo."""
    ruta = sidecar(tmp_path, EJES_CRUZADOS=not mod.EJES_CRUZADOS,
                   propagador="FFT-ASM (espectro angular por FFT)")
    with pytest.raises(SystemExit, match="FFT-ASM"):
        mod._comprobar_con_el_sidecar(ruta, LAMB, DELTA)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
