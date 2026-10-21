"""Los hologramas experimentales del USAF que entrego Carlos (objetivo 3).

LOS DATOS. Dos zip, fuera del repo por tamano (3.2 GB):
Downloads/4_Amplitude_USAF.zip y Downloads/10_Phase_USAF.zip. Cada uno trae 17
carpetas z1..z17 -la platina mueve la muestra a paso fijo- con 16 capturas
(1..16.jpg, el USAF en 16 posiciones laterales puestas a mano) y ref.jpg, la
imagen sin muestra. JPEG de 8 bits, 4640 x 3506. Se descomprimen en
<DATOS>/datos_amp y <DATOS>/datos_phase (ver datos.DATOS).

LO QUE HAY AQUI:
    datos      leer un holograma y su referencia
    foco       retropropagacion paraxial en la malla del sensor y el barrido
               de enfoque en D = L(L - z)/z, que no necesita L
    geometria  D, z, M, z_eq, el cociente de la ventana y el ajuste conjunto
               que fija el pixel y L con las magnificaciones del USAF
    usaf       ajuste de tres barras a un elemento del USAF -> M
    correr     las corridas que escriben en resultados/hologramas_carlos/
    comparar   el MPASM esferico (CamposT.dlhm) frente a la cadena de Carlos

LA GEOMETRIA ES INFERIDA. El enfoque solo fija D: con cualquier L hay una z que
enfoca igual de nitido. L y el pixel salen de la escala del USAF. Con el pixel
de CamposT/montaje.py (1.83 um) el modelo DLHM no cierra (chi2/dof 115); con
~3.8 um si (chi2/dof ~8), L ~18.2 mm y la platina a pasos de 0.1 y 0.05 mm.
Hay que confirmarlo con el laboratorio. Por eso aqui delta y L van SIEMPRE por
argumento, nunca desde MONTAJE.

UNIDADES: milimetros.
"""
