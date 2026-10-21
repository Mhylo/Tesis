"""Pruebas de scripts/hologramas_carlos: foco paraxial, geometria DLHM y USAF.

Los hologramas de Carlos no estan en el repo (3.2 GB): aqui todo es sintetico.
"""

import numpy as np
import pytest

from CamposT.propagadores import blas
from scripts.hologramas_carlos import foco, geometria, usaf

LAMB = 528e-6


# ------------------------------------------------------------------ geometria
@pytest.mark.parametrize("L", [10.0, 18.229])
@pytest.mark.parametrize("z", [0.4, 1.0, 2.0])
def test_D_z_ida_y_vuelta(z, L):
    assert geometria.z_de(geometria.D_de(z, L), L) == pytest.approx(z)


def test_relaciones_de_la_geometria():
    z, L = 1.2032, 18.229
    M = geometria.M_de(z, L)
    assert M == pytest.approx(L / z)
    # D es M veces la distancia muestra -> sensor, y M^2 veces z_eq
    assert geometria.D_de(z, L) == pytest.approx(M * (L - z))
    assert geometria.D_de(z, L) == pytest.approx(M**2 * geometria.zeq_de(z, L))


def test_cociente_ventana():
    c = geometria.cociente_ventana(0.4146, 18.229, 3.8e-3, 4640, LAMB)
    assert c == pytest.approx(6.17, rel=0.01)


def _sinteticos(p, L, z1, dz, ruido, semilla=0):
    rng = np.random.default_rng(semilla)
    ks = np.arange(1, 18)
    z = z1 + (ks - 1) * dz
    D = L * (L - z) / z / p**2 * (1 + ruido * rng.standard_normal(ks.size))
    kM = np.array([1, 5, 9, 17])
    zM = z1 + (kM - 1) * dz
    DM = L * (L - zM) / zM / p**2
    M = L / zM / p * (1 + ruido * rng.standard_normal(kM.size))
    return ks, D, kM, M, DM


def test_ajuste_conjunto_recupera():
    p, L, z1, dz = 2.0765, 18.229, 0.4167, 0.09906
    ks, D, kM, M, DM = _sinteticos(p, L, z1, dz, 1e-3)
    eD, eM = 1e-3 * D, 1e-3 * M
    r = geometria.ajuste_conjunto(ks, D, eD, kM, M, eM, DM, delta_supuesto=1.83e-3)
    assert r["p"] == pytest.approx(p, rel=0.03)
    assert r["L"] == pytest.approx(L, rel=0.05)
    assert r["dz"] == pytest.approx(dz, rel=0.01)
    assert r["delta"] == pytest.approx(p * 1.83e-3, rel=0.03)
    fijo = geometria.ajuste_conjunto(ks, D, eD, kM, M, eM, DM, delta_supuesto=1.83e-3,
                                     p_fijo=1.0)
    assert fijo["chi2_dof"] > 20 * r["chi2_dof"]


# ------------------------------------------------------------------ USAF
def test_lp_mm():
    assert usaf.lp_mm(4, 2) == pytest.approx(2 ** (4 + 1 / 6))
    assert usaf.lp_mm(7, 1) == pytest.approx(128.0)


def test_barras_con_anillos():
    """Las franjas de la gemela dentro de las barras no engañan al ajuste."""
    rng = np.random.default_rng(0)
    x = np.arange(380, dtype=float)
    dentro = usaf.barras(x, 20.0, 67.0, 1.5)
    p = 1 - 0.6 * dentro + 0.3 * np.cos(2 * np.pi * x / 9) * dentro
    p += 0.05 * rng.standard_normal(x.size)
    w, s = usaf.ajustar_barras(p)
    assert w == pytest.approx(67.0, abs=1.0)


def test_magnificacion():
    # g4e2: periodo 55.68 um en la muestra; 2w = 134.35 px de 1.83 um -> M = 4.4156
    assert usaf.magnificacion(67.18, 4, 2, 1.83) == pytest.approx(4.4156, rel=1e-3)


# ------------------------------------------------------------------ foco
def test_contraste_piso():
    holo = np.full((64, 64), 50.0)
    ref = np.full((64, 64), 100.0)
    ref[:8, :8] = 1.0                      # esquina a 0.01 del maximo
    c = foco.contraste(holo, ref, piso=0.05)
    assert np.all(np.isfinite(c))
    assert np.all(c[:8, :8] == 0)
    assert c[32, 32] == pytest.approx(-0.5)


def _barras_objeto(n=256, ancho=3):
    g = np.zeros((n, n))
    c = n // 2
    for j in range(-1, 2):
        x0 = c + 2 * j * ancho - ancho // 2
        g[c - 15:c + 15, x0:x0 + ancho] = 1
        y0 = c + 40 + 2 * j * ancho
        g[y0:y0 + ancho, c - 15:c + 15] = 1
    return g


DELTA_SINT = 10e-3      # con 256 px, z_lim = 48.5 mm: D = 40 cabe en la ventana
D_SINT = 40.0


def _holograma(t):
    U = blas(t.astype(np.complex128), DELTA_SINT, LAMB, D_SINT, device="cpu")
    return np.abs(U) ** 2


def test_foco_gabor_sintetico():
    I = _holograma(1 - _barras_objeto())
    P = foco.Paraxial(I - 1, DELTA_SINT, LAMB, xp=np)
    Ds = np.arange(20.0, 60.01, 0.5)
    curva = foco.barrer(P, Ds, None, "max")
    assert foco.pico(Ds, curva) == pytest.approx(D_SINT, abs=1.0)


def test_foco_fase_es_minimo():
    I = _holograma(np.exp(1j * 0.8 * _barras_objeto()))
    P = foco.Paraxial(I - 1, DELTA_SINT, LAMB, xp=np)
    Ds = np.arange(20.0, 60.01, 0.5)
    assert foco.pico(Ds, foco.barrer(P, Ds, None, "min")) == pytest.approx(D_SINT, abs=1.0)
    assert foco.pico(Ds, foco.barrer(P, Ds, None, "max")) != pytest.approx(D_SINT, abs=1.0)


def test_pico_fino_quita_rampa():
    """En la ventana fina una rampa fuerte no corre el pico."""
    Ds = np.arange(33.0, 47.01, 0.1)
    curva = 0.02 * np.exp(-0.5 * ((Ds - 40.0) / 0.8) ** 2) - 0.01 * (Ds - 33.0)
    assert foco.pico_fino(Ds, curva) == pytest.approx(40.0, abs=0.1)


def test_centros_robustos_corrige_saltos():
    """Los D gruesos siguen D = a/(k + b) - c; dos que saltan se reemplazan."""
    ks = np.arange(1, 18)
    D = 3000.0 / (ks + 15.0) - 20.0
    malo = D.copy()
    malo[0] = 78.0          # como el z1 de fase en el barrido grueso
    malo[10] = 25.0
    c = foco.centros_robustos(ks, malo, tolerancia=8.0)
    assert c == pytest.approx(D, abs=1.5)
    bueno = foco.centros_robustos(ks, D, tolerancia=8.0)
    assert np.array_equal(bueno, D)       # los que no saltan quedan tal cual


# ------------------------------------------------------------------ comparacion con Carlos
def test_michelson_barras():
    from scripts.hologramas_carlos import comparar
    x = np.arange(300, dtype=float)
    p = 1 - 0.8 * usaf.barras(x, 30.0, 50.0, 1.0)      # barras de 0.2 sobre 1
    assert comparar.michelson(p) == pytest.approx(0.667, abs=0.01)


def test_snr_plano():
    from scripts.hologramas_carlos import comparar
    rng = np.random.default_rng(1)
    A = 1 + 0.1 * rng.standard_normal((200, 200))
    assert comparar.snr(A, (20, 180, 20, 180)) == pytest.approx(10.0, rel=0.05)


def test_recorte_de_carlos_cabe():
    """El lado del recorte de Carlos es el mayor de los candidatos cuya malla
    remuestreada (6 arrays) cabe en el presupuesto, a la z mas chica."""
    from scripts.hologramas_carlos import comparar
    lado = comparar.recorte_carlos(0.41, 3.8e-3, LAMB, presupuesto_gb=2.5)
    assert lado in (1024, 768, 512, 384)
    assert comparar.memoria_carlos(lado, 0.41, 3.8e-3, LAMB) <= 2.5
    mayor = {1024: None, 768: 1024, 512: 768, 384: 512}[lado]
    assert mayor is None or comparar.memoria_carlos(mayor, 0.41, 3.8e-3, LAMB) > 2.5


def test_barrido_grueso_fino_encuentra_foco_fuera_del_3pct():
    """El foco de un motor puede caer a mas del 3 % del de otro: el barrido
    grueso cubre +-10 % y el fino se centra en su maximo."""
    from scripts.hologramas_carlos import comparar
    z0, zf = 1.0, 0.94                      # foco real 6 % por debajo
    rng = np.random.default_rng(2)
    base = rng.random((64, 64))

    def campo(z):                           # contraste maximo en zf
        return 1 + np.exp(-((z - zf) / 0.02) ** 2) * base

    z, tam, s, borde = comparar.barrer_grueso_fino(campo, z0, "max", np)
    assert z == pytest.approx(zf, abs=0.005)
    assert s >= 0 and not borde


def test_barrido_grueso_fino_marca_el_borde():
    """Si el maximo esta fuera de +-10 %, lo marca y no extrapola: la z
    devuelta queda dentro de lo barrido (el Reconstructor aborta fuera)."""
    from scripts.hologramas_carlos import comparar
    base = np.random.default_rng(3).random((32, 32))

    def campo(z):                           # contraste creciente con z
        return 1 + z * base

    z, tam, s, borde = comparar.barrer_grueso_fino(campo, 1.0, "max", np)
    assert borde
    assert z <= 1.0 * (1 + comparar.ANCHO_REL) + 1e-12
