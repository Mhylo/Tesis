"""Geometria DLHM: D, z, M, z_eq, el criterio de ventana y el ajuste conjunto.

Fuente puntual a z de la muestra y a L del sensor:
    M   = L/z                magnificacion
    z_eq = z(L - z)/L        propagacion equivalente en la malla delta/M
    D   = L(L - z)/z = M (L - z) = M^2 z_eq
                             la misma propagacion en la malla del sensor
El criterio de ventana (Sec. 4.4 del manuscrito): el campo se sale de la
ventana cuando z_eq supera N (delta/M)^2/lamb, o sea D > N delta^2/lamb.
"""

import numpy as np
from scipy.optimize import least_squares


def D_de(z, L):
    return L * (L - z) / z


def z_de(D, L):
    return L**2 / (D + L)


def M_de(z, L):
    return L / z


def zeq_de(z, L):
    return z * (L - z) / L


def cociente_ventana(z, L, delta, N, lamb):
    """z_eq / (N (delta/M)^2/lamb), igual a D / (N delta^2/lamb)."""
    return zeq_de(z, L) / (N * (delta / M_de(z, L)) ** 2 / lamb)


def ajuste_conjunto(ks, D_medido, eD, kM, M_medido, eM, D_de_M, delta_supuesto,
                    p_fijo=None):
    """El pixel real, L y la platina, con los focos y las magnificaciones.

    D_medido y M_medido salen con el pixel supuesto. Con p = delta_real/delta_supuesto:
        D_medido = L(L - z)/z / p^2     (el foco fija lamb D/delta^2)
        M_medido = (L/z) / p            (M sale de periodo_px * delta)
        z_k      = z1 + (k - 1) dz      (platina a paso fijo)
    La z de cada M se toma de su propio D (D_de_M), sin suponer el paso.

    -> dict con p, L, z1, dz, sus errores (e_*), delta = p*delta_supuesto,
       chi2_dof y n (datos).
    """
    ks = np.asarray(ks, float)
    D_medido, eD = np.asarray(D_medido, float), np.asarray(eD, float)
    M_medido, eM = np.asarray(M_medido, float), np.asarray(eM, float)
    D_de_M = np.asarray(D_de_M, float)

    def residuos(p, L, z1, dz):
        z = z1 + (ks - 1) * dz
        rD = (L * (L - z) / z / p**2 - D_medido) / eD
        zM = L**2 / (p**2 * D_de_M + L)
        rM = (L / zM / p - M_medido) / eM
        return np.concatenate([rD, rM])

    if p_fijo is None:
        f = lambda q: residuos(*q)
        x0, lo, hi = [2.0, 15.0, 0.4, 0.1], [0.3, 0.5, 1e-3, 1e-4], [6.0, 300.0, 50.0, 10.0]
        nombres = ("p", "L", "z1", "dz")
    else:
        f = lambda q: residuos(p_fijo, *q)
        x0, lo, hi = [8.0 * p_fijo**2, 0.4, 0.1], [0.5, 1e-3, 1e-4], [400.0, 50.0, 10.0]
        nombres = ("L", "z1", "dz")
    r = least_squares(f, x0, bounds=(lo, hi), x_scale="jac")
    n = r.fun.size
    dof = max(n - len(r.x), 1)
    chi2 = float(np.sum(r.fun**2))
    cov = np.linalg.pinv(r.jac.T @ r.jac) * max(chi2 / dof, 1.0)
    e = np.sqrt(np.diag(cov))
    out = dict(zip(nombres, map(float, r.x)))
    out.update({f"e_{k}": float(v) for k, v in zip(nombres, e)})
    if p_fijo is not None:
        out["p"], out["e_p"] = float(p_fijo), 0.0
    out["delta"] = out["p"] * delta_supuesto
    out["chi2_dof"] = chi2 / dof
    out["n"] = n
    return out
