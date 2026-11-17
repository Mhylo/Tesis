# -*- coding: utf-8 -*-
"""Figuras del manuscrito con letra para proyectar.

Llama a las mismas funciones de dibujo de scripts/figuras_manuscrito.py, que
no recalculan nada (leen los CSV de resultados/ o evaluan kf_auto), pero con la
figura ESCALA veces mas chica. La letra y los trazos van en puntos, asi que,
puesta la figura al ancho de la diapositiva, quedan 1/ESCALA veces mas grandes
que en el manuscrito. El manuscrito no se toca: escribe en
presentacion/figuras_charla/ el PDF y su PNG, y generar_defensa.py los prefiere
a los del manuscrito.

Solo van las figuras que aguantan el cambio. En obj1_sam_vs_z, tiempos y
esquema_montaje la leyenda y los rotulos, puestos a mano para su tamano, se
montan sobre las curvas. El esquema se usa tal cual; de las otras dos hay aqui
una version para la diapositiva (sam_vs_z y tiempos_N), con los mismos CSV y
colores, letra de 13 pt y el nombre de cada curva al final en vez de leyenda.

    Tesis_env/Scripts/python.exe -m presentacion.figuras_charla
"""
import pathlib
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts import exactitud as ex
from scripts import figuras_manuscrito as fm

ESCALA = 0.8         # mas chica y los rotulos se montan
DPI = 275            # el PNG conserva los pixeles de la version del manuscrito
SALIDA = pathlib.Path(__file__).resolve().parent / "figuras_charla"
#: kf_vs_z y errata_kf van en las versiones de abajo, con z en cm; la de
#: exp_z_frente_a_L, que iba aqui al 0.8, tiene la suya desde el 13/11.
# Las de la guia de lectura van tal cual: solo se usan en las de respaldo, a un
# tamano en que su letra ya se lee.
GUIA = pathlib.Path(__file__).resolve().parents[1] / "docs" / "guia_manuscrito" / "figuras"
DE_LA_GUIA = ("g_s1", "g_rayos", "g_tres_lecturas", "g_esferica")


#: Del manuscrito, a su tamano (al 0.8 se rompen) pero con ASM en los rotulos.
#: Solo van en el respaldo; fresnel_reconstrucciones tiene la suya desde el 13/11.
TAL_CUAL = [fm.figura_mapas_banco, fm.figura_interferometria]


def solo_asm():
    """En la charla el espectro angular con FFT se llama ASM, no FFT-ASM (pedido
    del 12/11). Todo texto que dibuja matplotlib pasa por Text.set_text, tambien
    los rotulos que traen los scripts del manuscrito, que no se tocan."""
    import matplotlib.text as mtext
    original = mtext.Text.set_text
    if getattr(original, "solo_asm", False):
        return

    def set_text(self, s):
        if isinstance(s, str):
            s = s.replace("FFT-ASM", "ASM")
        return original(self, s)

    set_text.solo_asm = True
    mtext.Text.set_text = set_text


def kf_vs_z():
    """fm.figura_kf_vs_z con el eje z en cm (pedido del 12/11). El calculo es el
    mismo, con las longitudes en mm y la Ec. (14) del articulo, que es
    dimensionalmente consistente: no depende de la unidad."""
    from CamposT.montaje import TABLA1
    from CamposT.propagadores import kf_auto
    N, d, lamb = TABLA1.N, TABLA1.delta, TABLA1.lamb
    zs = np.logspace(1, np.log10(3e5), 400)           # mm
    eses = (1, 2, 4, 10)
    tonos = plt.cm.viridis(np.linspace(0.15, 0.8, len(eses)))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    fm.limpiar(ax, rejilla="both")
    ax.axhline(1.0, color=fm.SUAVE, lw=0.8, zorder=1)
    for s, tono in zip(eses, tonos):
        kf = [kf_auto(N, d, lamb, z, s=s) for z in zs]
        ax.plot(zs * 0.1, kf, color=tono, lw=1.8, label=f"s = {s}", zorder=3)
        asintota = np.sqrt(lamb * zs / (s * N)) / d
        sube = asintota < 1
        ax.plot(zs[sube] * 0.1, asintota[sube], color=tono, lw=0.9, ls=(0, (2, 2)),
                zorder=2)
    kf200 = kf_auto(N, d, lamb, 2e5, s=10)            # s = 10 a 20 000 cm
    ax.plot(2e4, kf200, "o", color=tonos[-1], ms=5, zorder=4)
    ax.annotate(f"{kf200:.1f}", (2e4, kf200), xytext=(-6, 6), textcoords="offset points",
                ha="right", fontsize=8, color=fm.TINTA)
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("$K_f$ automático")
    ax.set_ylim(0.08, 70)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fm.guardar(fig, "kf_vs_z.pdf")
    print(f"     s = 10 a 20 000 cm: K_f = {kf200:.2f}")


def errata_kf(s=10, z_marca=80e3):
    """fm.figura_errata_kf con el eje z en cm. El calculo sigue en mm, como en el
    ejemplo del codigo publicado (Time_test.py: L0 = 5, lamb = 632.8e-6, d = 12000):
    su Ec. (14) no es dimensionalmente consistente y su K_f depende de la unidad."""
    from CamposT.montaje import TABLA1
    from CamposT.propagadores import kf_auto
    N, d, lamb = TABLA1.N, TABLA1.delta, TABLA1.lamb
    zs = np.logspace(2, np.log10(3e5), 300)            # mm, como el codigo
    paper = [kf_auto(N, d, lamb, z, s=s) for z in zs]
    codigo = [kf_auto(N, d, lamb, z, s=s, formula="codigo") for z in zs]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    fm.limpiar(ax, rejilla="both")
    ax.axhline(1.0, color=fm.SUAVE, lw=0.8, zorder=1)
    ax.plot(zs * 0.1, paper, color=fm.COLOR["mpasm"], lw=1.8, label="Ec. (14) del artículo")
    ax.plot(zs * 0.1, codigo, color=fm.COLOR["fft"], lw=1.8, ls=(0, (4, 2)),
            label="código publicado")
    kp = kf_auto(N, d, lamb, z_marca, s=s)
    kc = kf_auto(N, d, lamb, z_marca, s=s, formula="codigo")
    for k in (kp, kc):
        ax.plot(z_marca * 0.1, k, "o", color=fm.TINTA, ms=4, zorder=4)
        ax.annotate(f"{k:.1f}", (z_marca * 0.1, k), xytext=(6, -2),
                    textcoords="offset points", fontsize=8, color=fm.TINTA)
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("$K_f$ automático")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fm.guardar(fig, "errata_kf.pdf")
    # la prueba de que la unidad importa: lo mismo con las longitudes en metros
    kp_m = kf_auto(N, d * 1e-3, lamb * 1e-3, z_marca * 1e-3, s=s)
    kc_m = kf_auto(N, d * 1e-3, lamb * 1e-3, z_marca * 1e-3, s=s, formula="codigo")
    print(f"     z = {z_marca / 10:.0f} cm, calculo en mm: articulo {kp:.2f}, codigo {kc:.2f}; "
          f"en m: articulo {kp_m:.2f}, codigo {kc_m:.2f}")


def estilo_charla():
    fm.estilo()
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 13})


def rotulo(ax, x, y, texto, color):
    ax.annotate(texto, xy=(x, y), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=13, color=color)


def sam_vs_z():
    """La Figura 4.1 (scripts/exactitud.py), desde resultados/exactitud/sam_vs_z.csv."""
    filas = fm.leer_csv(fm.RESULTADOS / "exactitud" / "sam_vs_z.csv")
    # en cm desde el 12/11 (pedido del autor): de 50 a 20 000 cm
    z = [float(f["z_mm"]) * 0.1 for f in filas]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.set_xscale("log")
    ax.axhline(0, color=fm.SUAVE, lw=0.8, zorder=1)
    z_v = ex.z_llena_la_ventana() * 0.1
    ax.axvline(z_v, color=fm.SUAVE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"el haz llena la\nventana ({z_v:.0f} cm)", xy=(z_v, 52), xytext=(6, 0),
                textcoords="offset points", va="top", fontsize=12, color=fm.SUAVE)
    for nombre, clave, color, ls, _ in ex.METODOS:
        v = [float(f[clave]) for f in filas]
        ax.plot(z, v, color=color, ls=ls, lw=2.2, zorder=3)
        rotulo(ax, z[-1], v[-1], nombre, color)
    ax.set_ylim(-5, 52)
    ax.set_xlim(right=z[-1] * 4.5)
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("SAM [dB]")
    fm.limpiar(ax)
    fig.tight_layout()
    fm.guardar(fig, "sam_vs_z.pdf")


def tiempos_N():
    """El panel (a) de figura_tiempos: tiempo contra N con s = 1."""
    filas = fm.leer_csv(fm.RESULTADOS / "escalado_N.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for metodo in ("fft", "blas", "mpasm"):
        sel = [f for f in filas if f["metodo"] == metodo]
        n = [int(f["N"]) for f in sel]
        for col, ls, disp in (("cpu_128", "-", "CPU"), ("gpu_64", "--", "GPU")):
            t = [1e3 * float(f[col]) for f in sel]
            ax.plot(n, t, color=fm.COLOR[metodo], ls=ls, lw=2.2, marker="o", ms=5,
                    markeredgecolor="white", markeredgewidth=0.7)
            if metodo != "blas":   # el BLAS va pegado al ASM: un rotulo para los dos
                nombre = "MPASM" if metodo == "mpasm" else "ASM y BLAS"
                rotulo(ax, n[-1], t[-1], f"{nombre}, {disp}", fm.COLOR[metodo])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(right=2048 * 5.5)
    ax.set_xticks([256, 512, 1024, 2048], ["256", "512", "1024", "2048"])
    ax.set_xlabel("N (puntos por lado), s = 1")
    ax.set_ylabel("tiempo por propagación [ms]")
    fm.limpiar(ax, rejilla="y")
    fig.tight_layout()
    fm.guardar(fig, "tiempos_N.pdf")


# Las tres de abajo van al tamano de su caja en la diapositiva, para que entren
# sin estirarse (pedido del 13/11): la del manuscrito al 0.8 quedaba x1.9 en la
# diapositiva del foco, y las otras dos, con letra de 7 a 10 pt.
def exp_z_frente_a_L():
    """fm.figura_exp_z_frente_a_L para la caja de 9.5 x 3.3 in: el foco en z del
    MPASM esferico para varias L (amp z9/1) y la nitidez en el foco. Mismos CSV."""
    filas = fm.leer_csv(fm.EXP / "z_frente_a_L.csv")
    D = next(float(f["D"]) for f in fm.leer_csv(fm.EXP / "focos_D.csv")
             if f["conjunto"] == "amp" and f["k"] == "9" and f["captura"] == "1")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.3),
                               gridspec_kw=dict(width_ratios=[1.5, 1]))
    ll = np.linspace(4, 31, 200)
    borde = dict(markeredgecolor="white", markeredgewidth=0.6)
    for dum, color, marca in (("1.83", fm.COLOR["fft"], "o"), ("3.80", fm.COLOR["mpasm"], "s")):
        sel = [f for f in filas if f["delta_um"] == dum]
        Ls = np.array([float(f["L"]) for f in sel])
        esc = (float(dum) / 1.83) ** 2
        a.plot(ll, ll**2 / (D * esc + ll), color=color, lw=1.4, alpha=0.6)
        a.plot(Ls, [float(f["z"]) for f in sel], marca, color=color, ms=6.5,
               label=f"píxel {dum} µm", **borde)
        b.plot(Ls, [float(f["tamura"]) for f in sel], marca + "-", color=color, ms=6,
               lw=1.6, **borde)
    a.set_xlabel(r"$L$ supuesto [mm]")
    a.set_ylabel(r"$z$ del foco [mm]")
    a.legend(frameon=False, loc="upper left", fontsize=12)
    a.set_title(r"(a) el foco sigue $z = L^2/(D + L)$", fontsize=13, color=fm.TINTA,
                loc="left")
    b.set_ylim(0.50, 0.55)
    b.set_xlabel(r"$L$ supuesto [mm]")
    b.set_ylabel("Tamura en el foco")
    b.set_title("(b) la nitidez no elige $L$", fontsize=13, color=fm.TINTA, loc="left")
    fm.limpiar(a)
    fm.limpiar(b)
    fig.tight_layout()
    fm.guardar(fig, "exp_z_frente_a_L.pdf")


def mpasm_esferico():
    """fm.figura_mpasm_esferico para la caja de 8.3 x 3.85 in: correlacion con el
    objeto por tesela contra el angulo desde la fuente. Mismo CSV; los nombres de
    las curvas, los de la diapositiva."""
    filas = fm.leer_csv(fm.RESULTADOS / "dlhm_esferico" / "campo_N3000.csv")
    series = (("exacto", "MPASM esférico", fm.COLOR["mpasm"], "-", "o"),
              ("escalado", "escalado paraxial", fm.COLOR["fft"], "--", "s"))
    entradas = (("campo complejo", "(a) entrada: campo complejo"),
                ("intensidad", "(b) entrada: intensidad"))
    fig, ejes = plt.subplots(1, 2, figsize=(8.3, 3.85), sharey=True)
    for ax, (clave, titulo) in zip(ejes, entradas):
        sel = [f for f in filas if f["entrada"] == clave]
        x = [(float(f["grados_desde"]) + float(f["grados_hasta"])) / 2 for f in sel]
        for col, nombre, color, ls, marca in series:
            ax.plot(x, [float(f[col]) for f in sel], color=color, ls=ls, lw=2.2,
                    marker=marca, ms=6, markeredgecolor="white", markeredgewidth=0.7,
                    label=nombre)
        fm.limpiar(ax)
        ax.set_title(titulo, fontsize=13, color=fm.TINTA, loc="left")
        ax.set_xlabel("ángulo desde la fuente [grados]")
        ax.set_xlim(0, 21)
        ax.set_ylim(0.55, 1.02)
    ejes[0].set_ylabel("correlación con el objeto")
    ejes[0].legend(frameon=False, loc="lower left", fontsize=12)
    fig.tight_layout()
    fm.guardar(fig, "mpasm_esferico.pdf")


def fresnel_reconstrucciones(nfs=(10, 1)):
    """fm.figura_fresnel_reconstrucciones para la caja de 7.3 x 4.6 in: las
    reconstrucciones a N_F = 10 y 1. Lee los mismos .npy. Sin las letras de los
    paneles, que la diapositiva no cita: a 12 pt los titulos se montaban."""
    filas = fm.leer_csv(fm.FRESNEL / "resumen.csv")
    columnas = (("ida_fft", None, "ida ASM:\ncampo en el sensor"),
                ("ida_fft_vuelta_fft", ("fft", "fft"), "ASM → ASM"),
                ("ida_mpasm_vuelta_mpasm", ("mpasm", "mpasm"),
                 "MPASM → MPASM,\nventana fija"),
                ("ida_mpasm_amp_vuelta_mpasm", ("mpasm_amp", "mpasm"),
                 "MPASM → MPASM,\nventana ampliada"))
    fig, ejes = plt.subplots(len(nfs), len(columnas), figsize=(7.3, 4.6))
    for k, nf in enumerate(nfs):
        z = next(float(f["z_mm"]) for f in filas if round(float(f["N_F"])) == nf)
        carpeta = fm.FRESNEL / f"z{z:08.3f}"
        for j, (archivo, combinacion, titulo) in enumerate(columnas):
            ax = ejes[k, j]
            I = np.abs(np.load(carpeta / f"{archivo}.npy")) ** 2
            ax.imshow(I / I.max(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for lado in ax.spines.values():
                lado.set_color(fm.SUAVE)
                lado.set_linewidth(0.5)
            if k == 0:
                ax.set_title(titulo, fontsize=12, color=fm.TINTA)
            if combinacion is not None:
                sam = float(fm._fila(filas, z, *combinacion)["sam_db"])
                ax.set_xlabel(f"SAM {sam:.1f} dB", fontsize=12, color=fm.TINTA)
        ejes[k, 0].set_ylabel(f"$N_F$ = {nf}\nz = {z:g} mm", fontsize=12, color=fm.TINTA)
    fig.tight_layout()
    fm.guardar(fig, "fresnel_reconstrucciones.pdf")


_subplots, _figure = plt.subplots, plt.figure


def _chica(figsize):
    return None if figsize is None else (figsize[0] * ESCALA, figsize[1] * ESCALA)


def subplots(*a, figsize=None, **k):
    return _subplots(*a, figsize=_chica(figsize), **k)


def figure(*a, figsize=None, **k):
    return _figure(*a, figsize=_chica(figsize), **k)


def mapas_dos():
    """Los paneles (a) y (e) de figura_mapas_banco: ASM con relleno x2 y
    MPASM con s = 3 y K_f automatico, desde resultados/barrido_obj2/barrido.csv."""
    from matplotlib.colors import LinearSegmentedColormap
    from scripts.barrido_obj2 import RAMPA, _mapa, _panel, leer
    cmap = LinearSegmentedColormap.from_list("azul", RAMPA)
    cmap.set_bad("#ffffff")
    filas = leer()
    fig, ejes = plt.subplots(1, 2, figsize=(8.2, 3.9), sharey=True)
    for ax, (titulo, m, s, f) in zip(ejes, [("ASM, relleno ×2", "fft_asm", None, None),
                                            ("MPASM, s = 3", "mpasm", 3, 1.0)]):
        im = _panel(ax, *_mapa(filas, m, s, f), "", cmap)
        fm._curva_cociente(ax)
        ax.set_title(titulo, fontsize=14, color=fm.TINTA, loc="left")
        ax.set_xlabel("z, fuente a muestra [mm]", fontsize=13)
        ax.tick_params(labelsize=12)
        for t in ax.texts:         # los rotulos de las curvas de nivel
            t.set_fontsize(11)
    ejes[0].set_ylabel("NA de la ventana", fontsize=13)
    barra = fig.colorbar(im, ax=ejes, shrink=0.92, pad=0.02)
    barra.set_label("log₁₀ del error frente a RS", fontsize=13)
    barra.ax.tick_params(labelsize=12)
    fm.guardar(fig, "mapas_dos.pdf")


def celda_ejemplo(z=3.0, na=0.06):
    """El error en una celda fuera de la ventana: ASM con relleno x2 y
    MPASM con s = 1, 2 y 3 y K_f automatico (el ejemplo de la Seccion 4.2.3)."""
    from scripts.barrido_obj2 import leer
    filas = [f for f in leer() if abs(float(f["z"]) - z) < 1e-9
             and abs(float(f["NA"]) - na) < 0.005]
    barras = [("ASM\n×2", next(f for f in filas if f["metodo"] == "fft_asm"),
               fm.COLOR["fft"])]
    for s in (1, 2, 3):
        f = next(f for f in filas if f["metodo"] == "mpasm" and f["s"] == str(s)
                 and float(f["kf_factor"]) == 1.0)
        barras.append((f"MPASM\ns = {s}", f, fm.COLOR["mpasm"]))
    err = [float(f["err_rel"]) for _, f, _ in barras]
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    x = range(len(barras))
    ax.bar(x, err, color=[c for *_, c in barras], width=0.62, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(3e-6, 6e-2)
    ax.axhline(1e-3, color=fm.TINTA, lw=1.0, ls="--", zorder=2)
    ax.annotate("precisión aceptable", xy=(len(barras) - 0.6, 1e-3), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=11,
                color=fm.TINTA)
    for xi, e in zip(x, err):
        m, p = f"{e:.1e}".split("e")
        ax.annotate(f"{m}×10$^{{{int(p)}}}$", xy=(xi, e), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=12)
    ax.set_xticks(list(x), [b[0] for b in barras], fontsize=12)
    ax.set_ylabel("error frente a RS")
    fm.limpiar(ax)
    fig.tight_layout()
    fm.guardar(fig, "celda_ejemplo.pdf")
    print("     celda z = %g mm, NA ~ %g:" % (z, na),
          ", ".join(f"{b[0].split()[0]} {e:.2g}" for b, e in zip(barras, err)))


def ida_vuelta():
    """figura_ida_vuelta_cuatro con tres entradas de leyenda en vez de cinco:
    las tres combinaciones cruzadas van con un mismo rotulo."""
    from matplotlib.lines import Line2D
    zl = fm._z_lim_plana()
    plana, amp = {}, {}
    for carpeta in fm.CRUZADA_PLANA:
        for f in fm.leer_csv(fm.CRUZADA / carpeta / "resumen.csv"):
            z = float(f["z_mm"])
            if f["ida"] == "mpasm_amp":
                amp[z] = float(f["sam_db"])
            else:
                plana.setdefault((f["ida"], f["vuelta"]), {})[z] = float(f["sam_db"])
    esf = fm.leer_csv(fm.CRUZADA_DLHM)
    nombre_esf = {"asm": "fft", "mpasm": "mpasm"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
    for ida, vuelta, color, ls, marca, _ in fm.COMBINACIONES:
        d = plana[(ida, vuelta)]
        zs = sorted(d)
        a.plot(np.array(zs) / zl, [d[z] for z in zs], color=color, ls=ls, lw=2,
               marker=marca, ms=5, markeredgecolor="white", markeredgewidth=0.6)
        sel = sorted((f for f in esf if nombre_esf[f["ida"]] == ida
                      and nombre_esf[f["vuelta"]] == vuelta),
                     key=lambda f: float(f["cociente"]))
        b.plot([float(f["cociente"]) for f in sel], [float(f["sam_db"]) for f in sel],
               color=color, ls=ls, lw=2, marker=marca, ms=5, markeredgecolor="white",
               markeredgewidth=0.6)
    za = sorted(amp)
    a.plot(np.array(za) / zl, [amp[z] for z in za], color=fm.SUAVE, ls=(0, (1, 1.5)),
           lw=2, marker="^", ms=5)
    for ax, titulo, xlab in ((a, "(a) onda plana", r"$z\,/\,z_\mathrm{lim}$"),
                             (b, "(b) banco DLHM, onda esférica",
                              r"$z_\mathrm{eq}\,/\,[N(\delta/M)^2/\lambda]$")):
        ax.set_xscale("log")
        ax.axvline(1.0, color=fm.SUAVE, lw=0.9, zorder=1)
        ax.set_xlabel(xlab)
        ax.set_title(titulo, fontsize=13, color=fm.TINTA, loc="left")
        fm.limpiar(ax, rejilla="y")
    a.set_ylabel("SAM frente al objeto [dB]")
    a.set_ylim(0, 72)
    linea = lambda c, ls, m: Line2D([], [], color=c, ls=ls, lw=2, marker=m, ms=5)
    a.legend(handles=[linea(fm.COLOR["fft"], "-", "o"), linea(fm.COLOR["mpasm"], "-", "s"),
                      linea(fm.SUAVE, (0, (1, 1.5)), "^")],
             labels=["ASM → ASM", "las tres cruzadas", "ventana ampliada"],
             loc="center left", bbox_to_anchor=(0.5, 0.52), frameon=False, fontsize=12)
    b.legend(handles=[linea(fm.COLOR["mpasm"], "-", "s"), linea(fm.COLOR["fft"], "-", "o")],
             labels=["MPASM esférico → MPASM esférico",
                     "método del laboratorio y cruzadas"],
             loc="upper left", bbox_to_anchor=(0.22, 1.0), frameon=False, fontsize=12)
    for f in esf:
        if f["ida"] == "mpasm" and f["vuelta"] == "mpasm" and float(f["z_mm"]) in (1, 3, 7):
            b.annotate(f"z = {float(f['z_mm']):g} mm", (float(f["cociente"]),
                                                       float(f["sam_db"])),
                       textcoords="offset points", xytext=(4, 8), fontsize=11,
                       color=fm.TINTA)
    fig.tight_layout()
    fm.guardar(fig, "ida_vuelta.pdf")


def _exp_dos_filas(nombre, filas):
    """La Figura 4.14 en dos filas y tres columnas, con letra para proyectar: la
    misma zona del sensor en z1, z9 y z17. filas = [(columna del CSV, rotulo)]."""
    import json
    from matplotlib.patches import Rectangle
    from scripts.hologramas_carlos.comparar import zona_grande
    cajas = json.load(open(fm.RAIZ / "scripts" / "hologramas_carlos" / "cajas_metricas.json",
                           encoding="utf-8"))
    met = fm.leer_csv(fm.EXP / "metricas_carlos.csv")
    posiciones = (1, 9, 17)
    fig, ejes = plt.subplots(len(filas), len(posiciones), figsize=(7.1, 4.85))
    for j, k in enumerate(posiciones):
        del_k = [f for f in met if int(f["k"]) == k]
        ejes[0, j].set_title(f"posición {k},  M = {float(del_k[0]['M']):.1f}", fontsize=13,
                             color=fm.TINTA)
        for i, (col, rotulo_fila) in enumerate(filas):
            ax = ejes[i, j]
            if col == "grande":
                A = np.abs(zona_grande(k, 512))
            else:
                A = np.abs(np.load(fm.EXP / f"t_{col}_z{k}.npy"))
            fm._mostrar(ax, A)
            for n, (_, x0, x1, y0, y1) in enumerate(cajas[str(k)]["elementos"]):
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       ec=fm.COLOR["fft"], lw=1.3))
                c = float(del_k[n][f"contraste_{col}"])
                ax.text(x0 + 4, y0 + 4, f"{c:.2f}", color=fm.COLOR["fft"], fontsize=11,
                        va="top", ha="left",
                        bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
            if j == 0:
                ax.set_ylabel(rotulo_fila, fontsize=13, color=fm.TINTA)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.4)
    fm.guardar(fig, nombre + ".pdf")


DELTA_EXP = 3.80e-3   # mm, el pixel adoptado (scripts/hologramas_carlos/correr.py)


def barra_escala(ax, um_por_px, forma):
    """Barra de escala abajo a la derecha del panel, sobre un fondo blanco: la
    longitud redonda mas larga que no pase de un cuarto del ancho."""
    from matplotlib.patches import Rectangle
    alto, ancho = forma
    largo = max(v for v in (10, 20, 50, 100, 200, 500) if v <= ancho * um_por_px / 4)
    n = largo / um_por_px
    texto = f"{largo} µm"
    m, t, h_txt = 0.035 * ancho, 0.018 * alto, 0.11 * alto
    w = max(n, 0.04 * ancho * len(texto))     # el fondo cubre la barra y el rotulo
    cx, y1 = ancho - m - w / 2, alto - m
    ax.add_patch(Rectangle((cx - w / 2 - 0.5 * m, y1 - t - h_txt - 0.5 * m), w + m,
                           t + h_txt + m, fc="white", ec="none", alpha=0.85, zorder=4))
    ax.add_patch(Rectangle((cx - n / 2, y1 - t), n, t, fc=fm.TINTA, ec="none", zorder=5))
    ax.text(cx, y1 - t - 0.012 * alto, texto, ha="center", va="bottom", fontsize=11,
            color=fm.TINTA, zorder=5)


def exp_reconstrucciones():
    """La Figura 4.13 con letra para proyectar: |t| del USAF de amplitud y la fase
    de t del de fase, en el foco del MPASM esferico. Cada panel lleva su barra de
    escala: el pixel reconstruido es delta*z/L, y el panel se muestra cada 2."""
    zex = {(f["conjunto"], int(f["k"])): float(f["z"])
           for f in fm.leer_csv(fm.EXP / "focos_exactos.csv")}
    fig, ejes = plt.subplots(2, 4, figsize=(9.9, 4.5))
    for j, k in enumerate((1, 5, 9, 17)):
        t = np.load(fm.EXP / f"imagen_amp_z{k}.npy")
        A = np.abs(t)[::2, ::2]
        fm._mostrar(ejes[0, j], A)
        z = zex[("amp", k)]
        barra_escala(ejes[0, j], 2e3 * DELTA_EXP * z / fm.L_EXP, A.shape)
        ejes[0, j].set_title(f"posición {k},  M = {fm.L_EXP / z:.1f}",
                             fontsize=13, color=fm.TINTA)
    for j, k in enumerate((1, 9, 17)):
        t = np.load(fm.EXP / f"imagen_phase_z{k}.npy")
        P = fm._fase_sin_rampa(t)[::2, ::2]
        fm._mostrar(ejes[1, j], P)
        z = zex[("phase", k)]
        barra_escala(ejes[1, j], 2e3 * DELTA_EXP * z / fm.L_EXP, P.shape)
        ejes[1, j].set_title(f"posición {k},  M = {fm.L_EXP / z:.1f}",
                             fontsize=13, color=fm.TINTA)
    ejes[0, 0].set_ylabel("USAF de amplitud", fontsize=13, color=fm.TINTA)
    ejes[1, 0].set_ylabel("USAF de fase", fontsize=13, color=fm.TINTA)
    ejes[1, 3].axis("off")
    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.5)
    fm.guardar(fig, "exp_reconstrucciones.pdf")


def fresnel_resolucion():
    """El panel (b) de figura_fresnel_metricas (Figura 4.9) para la diapositiva: el
    periodo mas fino cuya luz llega a la ventana, la fija del sensor (la unica del
    ASM sin relleno) y la ampliada del MPASM, contra z. Mismos datos:
    resultados/comparacion_cruzada/entrada_r4."""
    from CamposT.campos import periodos_usaf
    filas = fm.leer_csv(fm.FRESNEL / "resumen.csv")
    zs = sorted({float(f["z_mm"]) for f in filas})
    nf_z = float(filas[0]["N_F"]) * float(filas[0]["z_mm"])   # N_F * z, constante
    p_fija = [float(fm._fila(filas, z, "mpasm", "mpasm")["periodo_min_um"]) for z in zs]
    p_amp = [float(fm._fila(filas, z, "mpasm_amp", "mpasm")["periodo_min_um"]) for z in zs]
    delta_um = 1e3 * float(next(
        l.split("=")[1] for l in open(fm.FRESNEL / "parametros.txt", encoding="utf-8")
        if l.startswith("delta")))
    periodos = np.array(periodos_usaf()) * delta_um
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axhspan(periodos.min(), periodos.max(), color=fm.COLOR["mpasm"], alpha=0.08, lw=0)
    ax.text(zs[0], periodos.max() * 0.92, "periodos de las barras", fontsize=12,
            color=fm.TINTA, va="top")
    estilo = dict(lw=2.2, ms=6, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.plot(zs, p_fija, color=fm.COLOR["fft"], ls="--", marker="s", **estilo)
    ax.plot(zs, p_amp, color=fm.COLOR["mpasm"], ls="-", marker="o", **estilo)
    rotulo(ax, zs[-1], p_fija[-1], "ventana fija\n(la del ASM)", fm.COLOR["fft"])
    rotulo(ax, zs[-1], p_amp[-1], "MPASM, ventana\nampliada", fm.COLOR["mpasm"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    marcas = [10, 20, 50, 100, 200, 500, 1000]
    ax.set_yticks(marcas, [str(m) for m in marcas])
    ax.set_ylim(9, 1300)
    ax.set_xlim(zs[0] / 1.25, zs[-1] * 3.4)
    ax.set_xticks(zs, [f"{z:g}" for z in zs])
    ax.minorticks_off()
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("periodo más fino resuelto [µm]")
    arriba = ax.secondary_xaxis("top", functions=(lambda z: nf_z / z, lambda n: nf_z / n))
    arriba.set_xticks([nf_z / z for z in zs], [f"{nf_z / z:.0f}" for z in zs])
    arriba.set_xlabel("número de Fresnel, $N_F$", fontsize=12, color=fm.TINTA)
    arriba.tick_params(colors=fm.SUAVE, labelcolor=fm.TINTA)
    arriba.minorticks_off()
    fm.limpiar(ax, rejilla="y")
    fig.tight_layout()
    fm.guardar(fig, "fresnel_resolucion.pdf")
    print("     periodo minimo [um]: fija", np.round(p_fija, 1), "ampliada", np.round(p_amp, 1))


def asm_vs_mpasm_lejos(z=1926.0):
    """Objeto -> holograma -> reconstruccion a z = 1.93 m (N_F = 1): una fila con
    el ASM en la ventana del sensor y otra con el MPASM en la ventana ampliada
    (r = 4). El holograma de la fila del ASM es el campo real en la ventana
    del sensor (la ida con el MPASM, que coincide con la ventana ampliada a 1 %):
    el ASM por si solo lo daria con copias que reentran por los bordes. Mismos
    datos que fresnel_reconstrucciones y fresnel_metricas del manuscrito:
    resultados/comparacion_cruzada/entrada_r4."""
    from matplotlib.patches import ConnectionPatch, Rectangle
    from PIL import Image
    filas = fm.leer_csv(fm.FRESNEL / "resumen.csv")
    d = fm.FRESNEL / f"z{z:08.3f}"
    objeto = np.asarray(Image.open(fm.RAIZ / "resultados" / "campos" / "entrada.png")
                        .convert("L"), dtype=float) / 255.0
    fija = fm._fila(filas, z, "mpasm", "mpasm")
    amp = fm._fila(filas, z, "mpasm_amp", "mpasm")
    L, La = float(fija["ventana_mm"]), float(amp["ventana_mm"])
    sam_fft = float(fm._fila(filas, z, "mpasm", "fft")["sam_db"])

    def panel(ax, I, lado, g=1.0):
        ax.imshow((I / I.max()) ** g, cmap="gray", vmin=0, vmax=1,
                  extent=(-lado / 2, lado / 2, -lado / 2, lado / 2))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(fm.SUAVE)
            s.set_linewidth(0.6)

    fig, ejes = plt.subplots(2, 3, figsize=(7.2, 5.3))
    filas_fig = [
        ("ASM", fm.COLOR["fft"], np.abs(np.load(d / "ida_mpasm.npy")) ** 2, L, 1.0,
         f"{L:.2f} mm, {float(fija['energia']) * 100:.0f} % de la luz",
         np.abs(np.load(d / "ida_mpasm_vuelta_fft.npy")) ** 2, sam_fft),
        ("MPASM", fm.COLOR["mpasm"], np.abs(np.load(d / "ida_mpasm_amp.npy")) ** 2, La, 0.5,
         f"{La:.1f} mm, {float(amp['energia']) * 100:.0f} % de la luz",
         np.abs(np.load(d / "ida_mpasm_amp_vuelta_mpasm.npy")) ** 2, float(amp["sam_db"]))]
    for (a, b, c), (nombre, color, H, lado, g, ley, R, sam) in zip(ejes, filas_fig):
        panel(a, objeto, L)
        panel(b, H, lado, g)
        panel(c, R, L)
        if lado > L:   # el sensor real dentro de la ventana ampliada
            b.add_patch(Rectangle((-L / 2, -L / 2), L, L, fill=False, ec=fm.COLOR["fft"],
                                  lw=2.0))
        a.set_xlabel(f"{L:.2f} mm", fontsize=12, color=fm.TINTA)
        b.set_xlabel(ley, fontsize=12, color=fm.TINTA)
        c.set_xlabel(f"SAM {sam:.1f} dB", fontsize=12, color=fm.TINTA)
        a.set_ylabel(nombre, fontsize=14, color=color, fontweight="bold")
        for ax1, ax2 in ((a, b), (b, c)):
            fig.add_artist(ConnectionPatch(xyA=(1.04, 0.5), coordsA=ax1.transAxes,
                                           xyB=(-0.04, 0.5), coordsB=ax2.transAxes,
                                           arrowstyle="-|>", color=fm.TINTA, lw=1.4,
                                           mutation_scale=15))
    for ax, t in zip(ejes[0], ("objeto", f"holograma a {z / 10:.0f} cm",
                               "reconstrucción")):
        ax.set_title(t, fontsize=13, color=fm.TINTA)
    fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=1.8)
    fm.guardar(fig, "asm_vs_mpasm_lejos.pdf")
    print(f"     a {z:g} mm: ASM, ventana del sensor {sam_fft:.2f} dB; MPASM, "
          f"ventana ampliada {float(amp['sam_db']):.2f} dB")


def exp_misma_ventana():
    _exp_dos_filas("exp_misma_ventana", [("exacto", "MPASM esférico\n512 px"),
                                         ("carlos", "laboratorio\n512 px")])


def exp_mas_ventana():
    _exp_dos_filas("exp_mas_ventana", [("exacto", "MPASM esférico\n512 px"),
                                       ("grande", "MPASM esférico\nrecorte grande")])


def memoria_cadena():
    """La memoria que pide la cadena del laboratorio con la geometria de estos
    hologramas y z = 0.41 mm (Secciones 3.4 y 4.3 del manuscrito)."""
    lado = ["512", "768", "1024", "2048"]
    gb = [1.1, 2.6, 4.8, 19]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(lado))
    ax.bar(x, gb, width=0.5, color=fm.COLOR["fft"], zorder=3)
    ax.axhline(4, color=fm.TINTA, lw=1.1, ls="--", zorder=4)
    ax.annotate("GPU de 4 GB", xy=(-0.42, 4), xytext=(0, 5), textcoords="offset points",
                fontsize=12, color=fm.TINTA)
    for xi, g in zip(x, gb):
        ax.annotate(f"{g:g} GB", xy=(xi, g), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=13, color=fm.TINTA, zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.8))
    ax.set_xticks(x, lado)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 22)
    ax.set_xlabel("lado del recorte [píxeles]")
    ax.set_ylabel("memoria que pide [GB]")
    fm.limpiar(ax)
    fig.tight_layout()
    fm.guardar(fig, "memoria_cadena.pdf")


def recta_MD():
    """La Ec. 3.9: magnificacion medida contra foco medido, con el pixel nominal,
    desde resultados/hologramas_carlos/L_implicito.csv. El grupo 5 es el de la
    Seccion 4.3.2 del manuscrito."""
    filas = fm.leer_csv(fm.EXP / "L_implicito.csv")
    D = np.array([float(f["D"]) for f in filas])
    M = np.array([float(f["M"]) for f in filas])
    e = np.array([float(f["eM"]) for f in filas])
    p, L = 3.80 / 1.83, 18.23
    x = np.array([0.0, 200.0])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(x, 1 + x / 10, color=fm.SUAVE, lw=2, ls=(0, (4, 2)),
            label="nominal: δ = 1.83 µm, L = 10 mm")
    ax.plot(x, 1 / p + p / L * x, color=fm.COLOR["mpasm"], lw=2.2,
            label="adoptada: δ = 3.80 µm, L = 18.23 mm")
    ax.plot(35.142, 4.46, "o", mfc="white", mec=fm.TINTA, ms=11, zorder=3,
            label="grupo 5, fuera del ajuste")
    ax.errorbar(D, M, yerr=e, fmt="o", color=fm.TINTA, ms=6, lw=1.2, capsize=3, zorder=4,
                label="medida con los grupos 4, 6 y 7")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 24)
    ax.set_xlabel("foco con el píxel nominal, $D_\\mathrm{medido}$ [mm]")
    ax.set_ylabel("magnificación, $M_\\mathrm{medida}$")
    ax.legend(frameon=False, fontsize=11.5, loc="lower right")
    fm.limpiar(ax, rejilla="both")
    fig.tight_layout()
    fm.guardar(fig, "recta_MD.pdf")


def resumen_q():
    """Donde cae cada ensayo sobre el cociente. Los extremos son los del manuscrito;
    los del barrido, de resultados/barrido_obj2/barrido.csv."""
    from scripts.barrido_obj2 import leer
    lamb, d, L = 0.528e-3, 1.83e-3, 10.0
    qs = [lamb * L * (L - float(f["z"])) / (int(f["N"]) * d * d * float(f["z"]))
          for f in leer()]
    casos = [("DHM, reenfoque de 1 a 10 mm", 1 / 45.9, 10 / 45.9),
             ("Barrido del banco", min(qs), max(qs)),
             ("Ida y vuelta, banco DLHM", 0.66, 13.9),
             ("Ida y vuelta, onda plana", 0.33, 128),
             ("Hologramas, sensor entero", 1.2, 6.2),
             ("Hologramas, recorte común", 10.6, 56),
             ("Interferometría simulada", 0.5, 20),
             ("Holografía de Fresnel a 100 cm", 21.8, 21.8),
             ("Escenario de Zhao et al.", 500 / 77.4, 200000 / 77.4)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xscale("log")
    y = np.arange(len(casos))[::-1]
    for yi, (_, q0, q1) in zip(y, casos):
        if q1 > q0:
            ax.plot([q0, q1], [yi, yi], color="#4a4a4a", lw=3.6, solid_capstyle="round")
        else:
            ax.plot(q0, yi, "o", color="#4a4a4a", ms=7)
    for x, txt, ha in ((1, "1", "center"), (3.2, "≈ 3", "center")):
        ax.vlines(x, -0.6, len(casos) - 0.5, color=fm.COLOR["fft"], lw=1.1,
                  ls=(0, (4, 3)), zorder=1)
        ax.text(x, len(casos) - 0.42, txt, fontsize=13, color=fm.COLOR["fft"], ha=ha,
                va="bottom")
    ax.set_yticks(y, [c[0] for c in casos], fontsize=12.5)
    ax.tick_params(axis="y", length=0, labelcolor=fm.TINTA)
    ax.set_ylim(-0.6, len(casos) + 0.15)
    ax.set_xlim(0.01, 5000)
    ax.set_xlabel("cociente: distancia equivalente frente a la ventana")
    fm.limpiar(ax, rejilla="x")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fm.guardar(fig, "resumen_q.pdf")


def main():
    SALIDA.mkdir(exist_ok=True)
    solo_asm()
    plt.subplots, plt.figure = subplots, figure
    fm.SALIDA = SALIDA
    fm.estilo()
    try:
        kf_vs_z()
        errata_kf()
    finally:
        plt.subplots, plt.figure = _subplots, _figure
    for dibujar in TAL_CUAL:
        fm.estilo()
        dibujar()
    estilo_charla()
    sam_vs_z()
    tiempos_N()
    exp_z_frente_a_L()
    mpasm_esferico()
    fresnel_reconstrucciones()
    mapas_dos()
    celda_ejemplo()
    ida_vuelta()
    exp_reconstrucciones()
    fresnel_resolucion()
    asm_vs_mpasm_lejos()
    exp_misma_ventana()
    exp_mas_ventana()
    memoria_cadena()
    recta_MD()
    resumen_q()
    for nombre in DE_LA_GUIA:
        shutil.copy2(GUIA / (nombre + ".pdf"), SALIDA / (nombre + ".pdf"))
    for pdf in sorted(SALIDA.glob("*.pdf")):
        subprocess.run([shutil.which("pdftoppm"), "-png", "-r", str(DPI), "-singlefile",
                        str(pdf), str(pdf.with_suffix(""))], check=True)


if __name__ == "__main__":
    main()
