"""Figuras del manuscrito que salen de datos ya guardados o de una fórmula.

Ninguna vuelve a medir ni a propagar nada: leen los CSV que dejaron los
scripts que hicieron el cálculo, o evalúan kf_auto(), y dibujan la versión de
la tesis (sin título ni pie, que los pone el caption de LaTeX; mismo estilo que
la Figura 4.1 de scripts/exactitud.py).

    kf_vs_z.pdf             K_f automático contra z, escenario de la Tabla 1.
                            Sección 2.4.
    mpasm_esferico.pdf      correlación con el objeto contra el ángulo, MPASM
                            esférico frente al escalado. Datos:
                            resultados/dlhm_esferico/campo_N3000.csv, de
                            scripts/verif_dlhm_esferico.py campo --n 3000.
                            Sección 3.2.2.
    errata_kf.pdf           la Ec. (14) impresa frente a la del código
                            publicado, s = 10. Sección 4.1.
    tiempos.pdf             tiempo contra N y contra s. Datos:
                            resultados/escalado_N.csv y escalado_s.csv, de
                            scripts/comparacion.py. Los mismos de la Tabla 4.1.
    fresnel_reconstrucciones.pdf
                            ida y vuelta a N_F = 10 y 1: la ida FFT-ASM, su
                            propia vuelta, la ventana fija y la ampliada.
                            Sección 4.2.3.
    fresnel_metricas.pdf    SAM y periodo mínimo contra z, ventana fija
                            frente a ampliada. Sección 4.2.3. Las dos leen
                            resultados/comparacion_cruzada/entrada_r4/, de
                            scripts/comparacion_cruzada.py con sus valores
                            por defecto.

    Tesis_env/Scripts/python.exe -m scripts.figuras_manuscrito

Escribe en manuscrito/figuras/.
"""

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from CamposT.montaje import TABLA1
from CamposT.propagadores import kf_auto

RAIZ = pathlib.Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "manuscrito" / "figuras"
RESULTADOS = RAIZ / "resultados"

# Paleta de la Figura 4.1 (scripts/exactitud.py): un color por método.
TINTA, SUAVE = "#1a1a1a", "#8a8a8a"
COLOR = {"mpasm": "#1f4e79", "fft": "#c1543a", "blas": "#2a9d8f"}
NOMBRE = {"mpasm": "MPASM", "fft": "FFT-ASM", "blas": "BLAS"}


def estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": SUAVE, "axes.labelcolor": TINTA,
        "xtick.color": SUAVE, "ytick.color": SUAVE,
        "axes.linewidth": 0.8, "figure.facecolor": "white",
    })


def limpiar(ax, rejilla="y"):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis=rejilla, which="major", color=SUAVE, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)


def guardar(fig, nombre):
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / nombre, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {SALIDA / nombre}")


def leer_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- K_f contra z
def figura_kf_vs_z():
    """K_f automático contra z, un trazo por s, con su asíntota.

    A distancias largas la Ec. (kf-auto) tiende a (1/delta)·sqrt(lambda z/(sN)),
    que cruza K_f = 1 en z = s N delta²/lambda.
    """
    N, d, lamb = TABLA1.N, TABLA1.delta, TABLA1.lamb
    zs = np.logspace(1, np.log10(3e5), 400)           # mm
    eses = (1, 2, 4, 10)
    tonos = plt.cm.viridis(np.linspace(0.15, 0.8, len(eses)))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    limpiar(ax, rejilla="both")
    ax.axhline(1.0, color=SUAVE, lw=0.8, zorder=1)
    for s, tono in zip(eses, tonos):
        kf = [kf_auto(N, d, lamb, z, s=s) for z in zs]
        ax.plot(zs * 1e-3, kf, color=tono, lw=1.8, label=f"s = {s}", zorder=3)
        asintota = np.sqrt(lamb * zs / (s * N)) / d
        sube = asintota < 1
        ax.plot(zs[sube] * 1e-3, asintota[sube], color=tono, lw=0.9,
                ls=(0, (2, 2)), zorder=2)
    # el punto que cita la Sección 4.1: s = 10 a 200 m
    kf200 = kf_auto(N, d, lamb, 2e5, s=10)
    ax.plot(200, kf200, "o", color=tonos[-1], ms=5, zorder=4)
    ax.annotate(f"{kf200:.1f}", (200, kf200), xytext=(-6, 6),
                textcoords="offset points", ha="right", fontsize=8,
                color=TINTA)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("$K_f$ automático")
    ax.set_ylim(0.08, 70)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    guardar(fig, "kf_vs_z.pdf")
    print(f"     s = 10 a 200 m: K_f = {kf200:.2f}; "
          f"umbral s N delta²/lambda = {10 * N * d**2 / lamb / 1e3:.2f} m")


# ------------------------------------------------- MPASM esférico vs escalado
def figura_mpasm_esferico():
    """Correlación con el objeto por tesela contra el ángulo desde la fuente."""
    filas = leer_csv(RESULTADOS / "dlhm_esferico" / "campo_N3000.csv")
    series = (("exacto", "MPASM con la esférica exacta", COLOR["mpasm"], "-", "o"),
              ("escalado", "escalado paraxial", COLOR["fft"], "--", "s"))
    entradas = (("campo complejo", "(a) entrada: campo complejo"),
                ("intensidad", "(b) entrada: intensidad"))

    fig, ejes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, (clave, titulo) in zip(ejes, entradas):
        sel = [f for f in filas if f["entrada"] == clave]
        x = [(float(f["grados_desde"]) + float(f["grados_hasta"])) / 2 for f in sel]
        for col, nombre, color, ls, marca in series:
            ax.plot(x, [float(f[col]) for f in sel], color=color, ls=ls, lw=1.8,
                    marker=marca, ms=4.5, markeredgecolor="white",
                    markeredgewidth=0.6, label=nombre)
        limpiar(ax)
        ax.set_title(titulo, fontsize=9, color=TINTA)
        ax.set_xlabel("ángulo desde la fuente [grados]")
        ax.set_xlim(0, 21)
        ax.set_ylim(0.55, 1.02)
        print(f"     {clave}: exacto {float(sel[0]['exacto']):.3f} -> "
              f"{float(sel[-1]['exacto']):.3f}; escalado "
              f"{float(sel[0]['escalado']):.3f} -> {float(sel[-1]['escalado']):.3f}")
    ejes[0].set_ylabel("correlación con el objeto (por tesela)")
    ejes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    guardar(fig, "mpasm_esferico.pdf")


# ----------------------------------------------------------- errata Ec. (14)
def figura_errata_kf(s=10, z_marca=80e3):
    """La Ec. (14) impresa (A² + B) frente a la del código publicado (A²·B)."""
    N, d, lamb = TABLA1.N, TABLA1.delta, TABLA1.lamb
    zs = np.logspace(2, np.log10(3e5), 300)            # mm, como el código
    paper = [kf_auto(N, d, lamb, z, s=s) for z in zs]
    codigo = [kf_auto(N, d, lamb, z, s=s, formula="codigo") for z in zs]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    limpiar(ax, rejilla="both")
    ax.axhline(1.0, color=SUAVE, lw=0.8, zorder=1)
    ax.plot(zs * 1e-3, paper, color=COLOR["mpasm"], lw=1.8,
            label="Ec. (14) del artículo")
    ax.plot(zs * 1e-3, codigo, color=COLOR["fft"], lw=1.8, ls=(0, (4, 2)),
            label="código publicado")
    kp = kf_auto(N, d, lamb, z_marca, s=s)
    kc = kf_auto(N, d, lamb, z_marca, s=s, formula="codigo")
    for k in (kp, kc):
        ax.plot(z_marca * 1e-3, k, "o", color=TINTA, ms=4, zorder=4)
        ax.annotate(f"{k:.1f}", (z_marca * 1e-3, k), xytext=(6, -2),
                    textcoords="offset points", fontsize=8, color=TINTA)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("$K_f$ automático")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    guardar(fig, "errata_kf.pdf")
    print(f"     s = {s}, z = {z_marca / 1e3:.0f} m: artículo {kp:.2f}, código {kc:.2f}, "
          f"cociente {kc / kp:.3f}; 1/(sN lambda) = {1 / (s * N * lamb):.3f}")


# ----------------------------------------------------------------- tiempos
def figura_tiempos():
    """(a) tiempo contra N con s = 1, CPU complex128 y GPU complex64;
    (b) MPASM contra s con N fijo. Los mismos CSV que la Tabla 4.1."""
    filas_N = leer_csv(RESULTADOS / "escalado_N.csv")
    filas_s = leer_csv(RESULTADOS / "escalado_s.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for metodo in ("fft", "blas", "mpasm"):
        sel = [f for f in filas_N if f["metodo"] == metodo]
        n = [int(f["N"]) for f in sel]
        for col, ls, marca, disp in (("cpu_128", "-", "o", "CPU"),
                                     ("gpu_64", "--", "s", "GPU")):
            a.plot(n, [1e3 * float(f[col]) for f in sel], color=COLOR[metodo],
                   ls=ls, lw=1.6, marker=marca, ms=4, markeredgecolor="white",
                   markeredgewidth=0.6, label=f"{NOMBRE[metodo]}, {disp}")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("N (puntos por lado)")
    a.set_ylabel("tiempo por propagación [ms]")
    a.set_title("(a) s = 1", fontsize=9, color=TINTA)
    a.legend(frameon=False, fontsize=7, loc="upper left", ncol=1)
    limpiar(a, rejilla="both")

    s = [int(f["s"]) for f in filas_s]
    for col, ls, marca, disp in (("cpu_128", "-", "o", "CPU"),
                                 ("gpu_64", "--", "s", "GPU")):
        b.plot(s, [1e3 * float(f[col]) for f in filas_s], color=COLOR["mpasm"],
               ls=ls, lw=1.6, marker=marca, ms=4, markeredgecolor="white",
               markeredgewidth=0.6, label=f"MPASM, {disp}")
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xticks(s, [str(v) for v in s])
    b.set_xlabel("s")
    b.set_title(f"(b) MPASM, N = {filas_s[0]['N']}", fontsize=9, color=TINTA)
    b.legend(frameon=False, fontsize=7, loc="upper left")
    limpiar(b, rejilla="both")
    fig.tight_layout()
    guardar(fig, "tiempos.pdf")
    g = [float(f["gpu_64"]) for f in filas_s]
    print(f"     MPASM GPU, s = {s[0]} -> {s[-1]}: {1e3 * g[0]:.1f} -> {1e3 * g[-1]:.1f} ms "
          f"(x{g[-1] / g[0]:.1f})")


# ------------------------------------------------------ zona de Fresnel (4.2.3)
#: la corrida por defecto de scripts/comparacion_cruzada.py (R = 4)
FRESNEL = RESULTADOS / "comparacion_cruzada" / "entrada_r4"


def _fila(filas, z, ida, vuelta):
    return next(f for f in filas if float(f["z_mm"]) == z
                and f["ida"] == ida and f["vuelta"] == vuelta)


def figura_fresnel_reconstrucciones(nfs=(10, 1)):
    """Las reconstrucciones a N_F = 10 y 1: la ida FFT-ASM en el sensor, su
    propia vuelta, la ventana fija y la ampliada. Lee los .npy de la corrida."""
    filas = leer_csv(FRESNEL / "resumen.csv")
    columnas = (("ida_fft", None, "(a) ida FFT-ASM:\ncampo en el sensor"),
                ("ida_fft_vuelta_fft", ("fft", "fft"), "(b) FFT-ASM → FFT-ASM"),
                ("ida_mpasm_vuelta_mpasm", ("mpasm", "mpasm"),
                 "(c) MPASM → MPASM,\nventana fija"),
                ("ida_mpasm_amp_vuelta_mpasm", ("mpasm_amp", "mpasm"),
                 "(d) MPASM → MPASM,\nventana ampliada"))

    fig, ejes = plt.subplots(len(nfs), len(columnas), figsize=(7.2, 4.6))
    for k, nf in enumerate(nfs):
        z = next(float(f["z_mm"]) for f in filas if round(float(f["N_F"])) == nf)
        carpeta = FRESNEL / f"z{z:08.3f}"
        for j, (archivo, combinacion, titulo) in enumerate(columnas):
            ax = ejes[k, j]
            I = np.abs(np.load(carpeta / f"{archivo}.npy")) ** 2
            ax.imshow(I / I.max(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for lado in ax.spines.values():
                lado.set_color(SUAVE)
                lado.set_linewidth(0.5)
            if k == 0:
                ax.set_title(titulo, fontsize=8.5, color=TINTA)
            if combinacion is not None:
                sam = float(_fila(filas, z, *combinacion)["sam_db"])
                ax.set_xlabel(f"SAM {sam:.1f} dB", fontsize=8, color=TINTA)
                print(f"     N_F = {nf}: {archivo:28s} SAM {sam:.2f} dB")
        ejes[k, 0].set_ylabel(f"$N_F$ = {nf}\nz = {z:g} mm", fontsize=8.5,
                              color=TINTA)
    fig.tight_layout()
    guardar(fig, "fresnel_reconstrucciones.pdf")


def figura_fresnel_metricas():
    """(a) SAM de la reconstrucción contra z: las tres combinaciones cruzadas
    con la ventana fija, como banda, frente a la ventana ampliada. (b) El
    periodo más fino que llega al sensor, frente a los periodos del objeto."""
    from CamposT.campos import periodos_usaf

    filas = leer_csv(FRESNEL / "resumen.csv")
    zs = sorted({float(f["z_mm"]) for f in filas})
    nf_z = float(filas[0]["N_F"]) * float(filas[0]["z_mm"])   # N_F * z, constante
    cruzadas = (("fft", "mpasm"), ("mpasm", "fft"), ("mpasm", "mpasm"))
    sam_fija = np.array([[float(_fila(filas, z, *c)["sam_db"]) for c in cruzadas]
                         for z in zs])
    sam_amp = [float(_fila(filas, z, "mpasm_amp", "mpasm")["sam_db"]) for z in zs]
    p_fija = [float(_fila(filas, z, "mpasm", "mpasm")["periodo_min_um"]) for z in zs]
    p_amp = [float(_fila(filas, z, "mpasm_amp", "mpasm")["periodo_min_um"]) for z in zs]
    delta_um = 1e3 * float(next(
        l.split("=")[1] for l in open(FRESNEL / "parametros.txt", encoding="utf-8")
        if l.startswith("delta")))
    periodos = np.array(periodos_usaf()) * delta_um

    fija = dict(color=SUAVE, ls=(0, (4, 2)), lw=1.6, marker="s", ms=4.5,
                markeredgecolor="white", markeredgewidth=0.6,
                label="ventana fija (1.77 mm)")
    ampliada = dict(color=COLOR["mpasm"], ls="-", lw=1.8, marker="o", ms=4.5,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label="MPASM, ventana ampliada")

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.5))
    a.fill_between(zs, sam_fija.min(axis=1), sam_fija.max(axis=1), color=SUAVE,
                   alpha=0.25, lw=0)
    a.plot(zs, sam_fija[:, 2], **fija)
    a.plot(zs, sam_amp, **ampliada)
    a.set_ylabel("SAM de la reconstrucción [dB]")
    a.set_title("(a) exactitud", fontsize=9, color=TINTA)
    a.set_ylim(0, 7)
    a.legend(frameon=False, fontsize=7.5, loc="upper right")

    b.axhspan(periodos.min(), periodos.max(), color=COLOR["mpasm"], alpha=0.08, lw=0)
    b.text(zs[0], periodos.max() * 0.93, "periodos de las barras", fontsize=7.5,
           color=TINTA, va="top")
    b.plot(zs, p_fija, **fija)
    b.plot(zs, p_amp, **ampliada)
    b.set_yscale("log")
    marcas = [10, 20, 50, 100, 200, 500, 1000]
    b.set_yticks(marcas, [str(m) for m in marcas])
    b.set_ylim(9, 1200)
    b.set_ylabel("periodo más fino que llega al sensor [µm]")
    b.set_title("(b) resolución", fontsize=9, color=TINTA)

    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xlabel("z [mm]")
        ax.set_xticks(zs, [f"{z:g}" for z in zs])
        ax.minorticks_off()
        arriba = ax.secondary_xaxis("top", functions=(lambda z: nf_z / z,
                                                      lambda n: nf_z / n))
        arriba.set_xticks([nf_z / z for z in zs],
                          [f"{nf_z / z:.0f}" for z in zs])
        arriba.set_xlabel("$N_F$", fontsize=8.5, color=TINTA)
        arriba.tick_params(colors=SUAVE, labelcolor=TINTA)
        arriba.minorticks_off()
        limpiar(ax, rejilla="y")
    fig.tight_layout()
    guardar(fig, "fresnel_metricas.pdf")
    print(f"     SAM ventana fija (cruzadas): {sam_fija.min(axis=1).round(2)} a "
          f"{sam_fija.max(axis=1).round(2)}; ampliada {np.round(sam_amp, 2)}")
    print(f"     periodo minimo [um]: fija {np.round(p_fija, 1)}, ampliada "
          f"{np.round(p_amp, 1)}; barras {periodos.min():.1f}-{periodos.max():.1f}")


# ------------------------------------------- ida y vuelta: las cuatro combinaciones
#: las corridas de scripts/comparacion_cruzada.py (onda plana) y de
#: scripts/cruzada_dlhm.py (banco DLHM con onda esferica)
CRUZADA = RESULTADOS / "comparacion_cruzada"
CRUZADA_PLANA = ("entrada_r4_cercano", "entrada_r4_transicion", "entrada", "entrada_r4")
CRUZADA_DLHM = RESULTADOS / "cruzada_dlhm" / "resumen.csv"
COMBINACIONES = (("fft", "fft", COLOR["fft"], "-", "o", "FFT-ASM → FFT-ASM"),
                 ("fft", "mpasm", COLOR["fft"], (0, (4, 2)), "s", "FFT-ASM → MPASM"),
                 ("mpasm", "fft", COLOR["mpasm"], (0, (4, 2)), "o", "MPASM → FFT-ASM"),
                 ("mpasm", "mpasm", COLOR["mpasm"], "-", "s", "MPASM → MPASM"))


def _z_lim_plana():
    """N delta^2/lambda del ensayo de onda plana: 512 px de 3.45 um a 405 nm."""
    return 512 * 3.45e-3 ** 2 / 405e-6


def figura_ida_vuelta_cuatro():
    """SAM de las cuatro combinaciones de ida y vuelta contra el cociente
    distancia / limite de la ventana. (a) Onda plana, de 5 mm a la zona de
    Fresnel, con la ventana ampliada del MPASM. (b) El banco DLHM con onda
    esferica: la cadena de Carlos (ASM) y el MPASM esferico, el cociente es
    D/(N delta^2/lambda) = z_eq/(N (delta/M)^2/lambda)."""
    zl = _z_lim_plana()
    plana = {}
    amp = {}
    for carpeta in CRUZADA_PLANA:
        for f in leer_csv(CRUZADA / carpeta / "resumen.csv"):
            z = float(f["z_mm"])
            if f["ida"] == "mpasm_amp":
                amp[z] = float(f["sam_db"])
            else:
                plana.setdefault((f["ida"], f["vuelta"]), {})[z] = float(f["sam_db"])
    esf = leer_csv(CRUZADA_DLHM)
    nombre_esf = {"asm": "fft", "mpasm": "mpasm"}

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ida, vuelta, color, ls, marca, etiqueta in COMBINACIONES:
        d = plana[(ida, vuelta)]
        zs = sorted(d)
        a.plot(np.array(zs) / zl, [d[z] for z in zs], color=color, ls=ls, lw=1.4,
               marker=marca, ms=3.8, markeredgecolor="white", markeredgewidth=0.5,
               label=etiqueta)
        sel = [f for f in esf if nombre_esf[f["ida"]] == ida and nombre_esf[f["vuelta"]] == vuelta]
        sel.sort(key=lambda f: float(f["cociente"]))
        b.plot([float(f["cociente"]) for f in sel], [float(f["sam_db"]) for f in sel],
               color=color, ls=ls, lw=1.4, marker=marca, ms=3.8, markeredgecolor="white",
               markeredgewidth=0.5)
    za = sorted(amp)
    a.plot(np.array(za) / zl, [amp[z] for z in za], color=SUAVE, ls=(0, (1, 1.5)), lw=1.4,
           marker="^", ms=3.8, label="MPASM → MPASM, ventana ampliada")
    for ax, titulo, xlab in (
            (a, "(a) onda plana, 405 nm, 512 px de 3.45 µm",
             r"$z\,/\,z_\mathrm{lim}$"),
            (b, "(b) banco DLHM, 528 nm, 1024 px de 1.83 µm, $L$ = 10 mm",
             r"$z_\mathrm{eq}\,/\,[N(\delta/M)^2/\lambda]$")):
        ax.set_xscale("log")
        ax.axvline(1.0, color=SUAVE, lw=0.8, zorder=1)
        ax.set_xlabel(xlab, fontsize=8.5)
        ax.set_title(titulo, fontsize=8.5, color=TINTA, loc="left")
        limpiar(ax, rejilla="y")
    a.set_ylabel("SAM frente al objeto [dB]")
    a.set_ylim(0, 72)
    for f in esf:
        if f["ida"] == "mpasm" and f["vuelta"] == "mpasm":
            b.annotate(f"z = {float(f['z_mm']):g} mm", (float(f["cociente"]), float(f["sam_db"])),
                       textcoords="offset points", xytext=(4, 6), fontsize=7, color=TINTA)
    a.legend(fontsize=7, frameon=False, loc="center left")
    fig.tight_layout()
    guardar(fig, "ida_vuelta_cuatro.pdf")


# -------------------------------------------------------- esquema del montaje DLHM
def figura_esquema_montaje():
    """Fuente puntual, muestra a z y sensor a L: el cono, la magnificacion y la
    huella de la muestra que el sensor ve."""
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    L, z, W = 10.0, 3.0, 4.6           # unidades del dibujo, no del banco
    h = W / 2
    ax.fill([0, L, L], [0, h, -h], color=COLOR["mpasm"], alpha=0.08, lw=0)
    ax.plot([0, L], [0, h], color=COLOR["mpasm"], lw=0.9)
    ax.plot([0, L], [0, -h], color=COLOR["mpasm"], lw=0.9)
    ax.plot(0, 0, "o", color=TINTA, ms=5)
    ax.annotate("fuente puntual\n(pinhole)", (0, 0), xytext=(-0.2, -1.55), ha="left",
                fontsize=8, color=TINTA)
    hz = h * z / L
    ax.plot([z, z], [-1.35 * hz, 1.35 * hz], color=COLOR["fft"], lw=2.2)
    ax.annotate("muestra", (z, 1.35 * hz), xytext=(z, 1.75 * hz + 0.15), ha="center",
                fontsize=8, color=TINTA)
    ax.plot([L, L], [-h, h], color=TINTA, lw=3.2)
    ax.annotate("sensor\n$N$ píxeles de paso $\\delta$", (L, h), xytext=(L + 0.25, h - 0.55),
                fontsize=8, color=TINTA)
    # un detalle de la muestra y su sombra magnificada
    y0 = 0.55 * hz
    ax.plot([0, L], [0, y0 * L / z], color=SUAVE, lw=0.8, ls=(0, (3, 2)))
    ax.plot(z, y0, "o", color=COLOR["fft"], ms=3.5)
    ax.plot(L, y0 * L / z, "o", color=TINTA, ms=3.5)
    ax.annotate("$x$", (z, y0), xytext=(z - 0.35, y0 + 0.05), fontsize=8, color=TINTA)
    ax.annotate("$Mx$", (L, y0 * L / z), xytext=(L - 0.75, y0 * L / z + 0.12), fontsize=8,
                color=TINTA)
    ya = -h - 0.45
    for x0, x1, txt in ((0, z, "$z$"), (z, L, "$L - z$")):
        ax.annotate("", (x0, ya), (x1, ya), arrowprops=dict(arrowstyle="<->", color=SUAVE, lw=0.8))
        ax.text((x0 + x1) / 2, ya - 0.32, txt, ha="center", fontsize=8.5, color=TINTA)
    ax.annotate("", (0, ya - 0.85), (L, ya - 0.85), arrowprops=dict(arrowstyle="<->", color=SUAVE, lw=0.8))
    ax.text(L / 2, ya - 1.17, "$L$", ha="center", fontsize=8.5, color=TINTA)
    ax.text(5.6, 0.25, "$M = L/z$", fontsize=9, color=TINTA)
    ax.set_xlim(-0.6, L + 2.6)
    ax.set_ylim(-h - 1.75, h + 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    guardar(fig, "esquema_montaje.pdf")


# ------------------------------------------- hologramas experimentales (4.3)
#: scripts/hologramas_carlos/correr.py y comparar.py
EXP = RESULTADOS / "hologramas_carlos"
NOMBRE_CONJ = {"amp": "USAF de amplitud", "phase": "USAF de fase"}
COLOR_CONJ = {"amp": COLOR["mpasm"], "phase": COLOR["fft"]}
#: el barrido de foco se hizo con el pixel nominal (1.83 um); con el adoptado
#: (3.80 um, inferido) el mismo foco es D (3.80/1.83)^2, y entonces
#: D = L(L - z)/z con L = 18.23 mm
ESC_PIXEL = (3.80 / 1.83) ** 2


def _focos_por_carpeta(conjunto):
    filas = [f for f in leer_csv(EXP / "focos_D.csv") if f["conjunto"] == conjunto]
    ks = sorted({int(f["k"]) for f in filas})
    D = [np.array([float(f["D"]) for f in filas if int(f["k"]) == k]) for k in ks]
    return (np.array(ks), ESC_PIXEL * np.array([np.median(d) for d in D]),
            ESC_PIXEL * np.array([np.percentile(d, 25) for d in D]),
            ESC_PIXEL * np.array([np.percentile(d, 75) for d in D]))


def figura_exp_foco():
    """Las curvas de enfoque gruesas (captura 1), sin su tendencia lenta, y el
    foco fino (mediana de las capturas) de cada posicion."""
    from scipy.ndimage import median_filter
    fig, ejes = plt.subplots(1, 2, figsize=(7.2, 6.4), sharex=True)
    for ax, conjunto in zip(ejes, ("amp", "phase")):
        d = np.load(EXP / f"curvas_{conjunto}.npz")
        D = ESC_PIXEL * d["D_grueso"]
        ks, med, _, _ = _focos_por_carpeta(conjunto)
        for i, k in enumerate(ks):
            y = d[f"grueso_z{k}"]
            r = y - median_filter(y, size=int(20 / (D[1] - D[0])) | 1, mode="nearest")
            r = r / (np.abs(r).max() + 1e-12)
            ax.plot(D, r - 1.1 * i, color=COLOR_CONJ[conjunto], lw=0.7)
            ax.plot([med[i]], [np.interp(med[i], D, r) - 1.1 * i], "o", ms=3, color=TINTA)
            ax.text(D[-1] * 1.015, -1.1 * i, f"z{k}", va="center", fontsize=6.5, color=TINTA)
        ax.set_title(NOMBRE_CONJ[conjunto] + (" (máximo)" if conjunto == "amp" else " (mínimo)"),
                     fontsize=8.5, color=TINTA, loc="left")
        ax.set_xlabel(r"$D$ en la malla del sensor, píxel de 3.80 µm [mm]", fontsize=8.5)
        ax.set_yticks([])
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
    fig.tight_layout()
    guardar(fig, "exp_foco.pdf")


def figura_exp_geometria():
    """(a) D del foco contra la posicion, con el modelo de platina a paso fijo
    y la geometria adoptada. (b) L = D/(M - 1) implicito en cada posicion donde
    se midio M, con el pixel de 1.83 um y con el de 3.80 um."""
    import json
    from scipy.optimize import least_squares
    cal = json.load(open(EXP / "calibracion.json", encoding="utf-8"))
    g = cal["pixel_380"]
    L = g["L"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    kk = np.linspace(1, 17, 200)
    for conjunto in ("amp", "phase"):
        ks, med, lo, hi = _focos_por_carpeta(conjunto)
        if conjunto == "amp":
            z1, dz = g["z1"], g["dz"]
        else:
            r = least_squares(lambda q: L**2 / (q[0] + (ks - 1) * q[1]) - L - med,
                              [0.45, 0.05], loss="soft_l1", f_scale=2.0)
            z1, dz = r.x
        z = z1 + (kk - 1) * dz
        a.plot(kk, L * (L - z) / z, color=COLOR_CONJ[conjunto], lw=1.0, alpha=0.7)
        a.errorbar(ks, med, yerr=[med - lo, hi - med], fmt="o", ms=3.5, lw=0.9,
                   color=COLOR_CONJ[conjunto], label=f"{NOMBRE_CONJ[conjunto]}, "
                   f"$\\Delta z$ = {1e3 * dz:.1f} µm")
    a.set_xlabel("posición $k$ de la platina", fontsize=8.5)
    a.set_ylabel(r"$D$ del foco, píxel de 3.80 µm [mm]", fontsize=8.5)
    a.set_xticks(range(1, 18, 2))
    a.legend(fontsize=7, frameon=False)
    a.set_title(f"(a) focos y platina a paso fijo ($L$ = {L:.2f} mm)", fontsize=8.5,
                color=TINTA, loc="left")
    limpiar(a)
    filas = leer_csv(EXP / "L_implicito.csv")
    k = np.array([int(f["k"]) for f in filas])
    for col, ecol, color, marca, nombre in (("L_183", "eL_183", COLOR["fft"], "o", "1.83 µm"),
                                            ("L_380", "eL_380", COLOR["mpasm"], "s", "3.80 µm")):
        Li = np.array([float(f[col]) for f in filas])
        e = np.array([float(f[ecol]) for f in filas])
        b.errorbar(np.arange(k.size), Li / Li.mean(), yerr=e / Li.mean(), fmt=marca + "-",
                   ms=4, lw=1.0, capsize=2, color=color,
                   label=f"píxel {nombre}: {Li.min():.2f} a {Li.max():.2f} mm")
    b.axhline(1.0, color=SUAVE, lw=0.8)
    b.set_xticks(np.arange(k.size))
    b.set_xticklabels([f"z{v}" for v in k])
    b.set_ylim(0.85, 1.15)
    b.set_ylabel(r"$L$ implícito / su media", fontsize=8.5)
    b.set_xlabel("posición donde se midió $M$", fontsize=8.5)
    b.legend(fontsize=7, frameon=False, loc="upper left")
    b.set_title(r"(b) $L = D/(M - 1)$ en cada posición", fontsize=8.5, color=TINTA, loc="left")
    limpiar(b)
    fig.tight_layout()
    guardar(fig, "exp_geometria.pdf")


def figura_exp_z_frente_a_L():
    """El foco en z del MPASM esferico para varias L (amp z9/1): la cresta
    z = L^2/(D + L), y la nitidez en el foco, que no elige L."""
    filas = leer_csv(EXP / "z_frente_a_L.csv")
    D = next(float(f["D"]) for f in leer_csv(EXP / "focos_D.csv")
             if f["conjunto"] == "amp" and f["k"] == "9" and f["captura"] == "1")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ll = np.linspace(4, 31, 200)
    for dum, color, marca in (("1.83", COLOR["fft"], "o"), ("3.80", COLOR["mpasm"], "s")):
        sel = [f for f in filas if f["delta_um"] == dum]
        Ls = np.array([float(f["L"]) for f in sel])
        esc = (float(dum) / 1.83) ** 2
        a.plot(ll, ll**2 / (D * esc + ll), color=color, lw=0.9, alpha=0.6)
        a.plot(Ls, [float(f["z"]) for f in sel], marca, color=color, ms=4.5,
               label=f"píxel {dum} µm")
        b.plot(Ls, [float(f["tamura"]) for f in sel], marca + "-", color=color, ms=4, lw=1)
    a.set_xlabel(r"$L$ supuesto [mm]", fontsize=8.5)
    a.set_ylabel(r"$z$ del foco [mm]", fontsize=8.5)
    a.legend(fontsize=7.5, frameon=False)
    a.set_title(r"(a) el foco sigue $z = L^2/(D + L)$", fontsize=8.5, color=TINTA, loc="left")
    b.set_ylim(0.50, 0.55)
    b.set_xlabel(r"$L$ supuesto [mm]", fontsize=8.5)
    b.set_ylabel("Tamura en el foco", fontsize=8.5)
    b.set_title("(b) la nitidez no elige $L$", fontsize=8.5, color=TINTA, loc="left")
    limpiar(a)
    limpiar(b)
    fig.tight_layout()
    guardar(fig, "exp_z_frente_a_L.pdf")


def _fase_sin_rampa(t):
    from scipy.ndimage import gaussian_filter
    ph = np.angle(t)
    return ph - gaussian_filter(ph, 60)


def _mostrar(ax, A):
    lo, hi = np.percentile(A, [1, 99])
    ax.imshow(A, cmap="gray", vmin=lo, vmax=hi, interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_color(SUAVE)
        lado.set_linewidth(0.5)


def figura_exp_reconstrucciones():
    """|t| (amplitud) y la fase de t (fase) en el foco del MPASM esferico."""
    zex = {(f["conjunto"], int(f["k"])): float(f["z"]) for f in leer_csv(EXP / "focos_exactos.csv")}
    L = 18.229
    fig, ejes = plt.subplots(2, 4, figsize=(7.4, 3.9))
    for j, k in enumerate((1, 5, 9, 17)):
        t = np.load(EXP / f"imagen_amp_z{k}.npy")
        _mostrar(ejes[0, j], np.abs(t)[::2, ::2])
        z = zex[("amp", k)]
        ejes[0, j].set_title(f"amplitud z{k}\n$z$ = {z:.3f} mm, $M$ = {L / z:.1f}",
                             fontsize=7, color=TINTA)
    for j, k in enumerate((1, 9, 17)):
        t = np.load(EXP / f"imagen_phase_z{k}.npy")
        _mostrar(ejes[1, j], _fase_sin_rampa(t)[::2, ::2])
        z = zex[("phase", k)]
        ejes[1, j].set_title(f"fase z{k}\n$z$ = {z:.3f} mm, $M$ = {L / z:.1f}",
                             fontsize=7, color=TINTA)
    ejes[1, 3].axis("off")
    fig.tight_layout()
    guardar(fig, "exp_reconstrucciones.pdf")


def figura_exp_carlos():
    """La misma zona del sensor (recorte comun de 512 px, centrado) en z1, z9 y
    z17 del USAF de amplitud: el MPASM esferico y la cadena del laboratorio en
    ese recorte, y el MPASM esferico con su recorte grande (2048x2560). Las
    cajas son los elementos donde se midio el contraste."""
    import json
    from matplotlib.patches import Rectangle
    from scripts.hologramas_carlos.comparar import zona_grande
    cajas = json.load(open(RAIZ / "scripts" / "hologramas_carlos" / "cajas_metricas.json",
                           encoding="utf-8"))
    met = leer_csv(EXP / "metricas_carlos.csv")
    fig, ejes = plt.subplots(3, 3, figsize=(7.2, 7.4))
    titulos = ("MPASM esférico, 512 px", "laboratorio, 512 px", "MPASM esférico, recorte grande")
    for i, k in enumerate((1, 9, 17)):
        imgs = (np.abs(np.load(EXP / f"t_exacto_z{k}.npy")),
                np.abs(np.load(EXP / f"t_carlos_z{k}.npy")),
                np.abs(zona_grande(k, 512)))
        for j, (A, col) in enumerate(zip(imgs, ("exacto", "carlos", "grande"))):
            ax = ejes[i, j]
            _mostrar(ax, A)
            for n, (o, x0, x1, y0, y1) in enumerate(cajas[str(k)]["elementos"]):
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       ec=COLOR["fft"], lw=0.8))
                c = float([f for f in met if int(f["k"]) == k][n][f"contraste_{col}"])
                ax.text(x0 + 4, y0 + 4, f"{c:.2f}", color=COLOR["fft"], fontsize=6.5,
                        va="top", ha="left",
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
            if i == 0:
                ax.set_title(titulos[j], fontsize=7.5, color=TINTA)
            if j == 0:
                M = float([f for f in met if int(f["k"]) == k][0]["M"])
                ax.set_ylabel(f"z{k}, $M$ = {M:.1f}", fontsize=8, color=TINTA)
    fig.tight_layout()
    guardar(fig, "exp_carlos.pdf")

def main():
    estilo()
    figura_kf_vs_z()
    figura_mpasm_esferico()
    figura_errata_kf()
    figura_tiempos()
    figura_fresnel_reconstrucciones()
    figura_fresnel_metricas()
    figura_ida_vuelta_cuatro()
    figura_esquema_montaje()
    figura_exp_foco()
    figura_exp_geometria()
    figura_exp_z_frente_a_L()
    figura_exp_reconstrucciones()
    figura_exp_carlos()


if __name__ == "__main__":
    main()
