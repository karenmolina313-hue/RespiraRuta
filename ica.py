"""
ica.py
Índice de Calidad del Aire (ICA) según la Resolución 2254 de 2017 (MinAmbiente), artículo 19-20.

El ICA se calcula por interpolación lineal dentro del rango de la tabla en que cae la concentración:

    ICA = (I_alto - I_bajo) / (PC_alto - PC_bajo) · (C - PC_bajo) + I_bajo

    C          : concentración (promedio de 24 h para PM10 y PM2.5), truncada como indica la norma
    PC_bajo/alto : puntos de corte de concentración del rango
    I_bajo/alto  : valores del índice de ese rango

Niveles máximos permisibles de la misma resolución (art. 2):
    PM2.5: 37 µg/m³ (24 h) y 25 µg/m³ (anual)
    PM10 : 75 µg/m³ (24 h) y 50 µg/m³ (anual)
"""

import math

import numpy as np

# Rangos del índice y su categoría (color oficial del ICA)
CATEGORIAS = [
    # I_bajo, I_alto, nombre,                                   color,     recomendación corta
    (0,   50,  "Buena",                                         "#00E400", "Calidad del aire satisfactoria, sin riesgo."),
    (51,  100, "Aceptable",                                     "#FFFF00", "Personas muy sensibles deberían reducir el esfuerzo prolongado al aire libre."),
    (101, 150, "Dañina a la salud de grupos sensibles",         "#FF7E00", "Niños, adultos mayores y personas con enfermedades respiratorias o cardiacas deben reducir la actividad física al aire libre."),
    (151, 200, "Dañina a la salud",                             "#FF0000", "Todos deben reducir la actividad física prolongada al aire libre; grupos sensibles, evitarla."),
    (201, 300, "Muy dañina a la salud",                         "#8F3F97", "Todos deben evitar la actividad física al aire libre."),
    (301, 500, "Peligrosa",                                     "#7E0023", "Emergencia: evitar cualquier actividad al aire libre."),
]

# Puntos de corte de concentración (µg/m³, promedio 24 h), uno por categoría
PUNTOS_CORTE = {
    "PM2.5": [(0, 12), (13, 37), (38, 55), (56, 150), (151, 250), (251, 500)],
    "PM10":  [(0, 54), (55, 154), (155, 254), (255, 354), (355, 424), (425, 604)],
}

LIMITE_24H = {"PM2.5": 37, "PM10": 75}
LIMITE_ANUAL = {"PM2.5": 25, "PM10": 50}
GUIA_OMS_24H = {"PM2.5": 15, "PM10": 45}     # Guías de calidad del aire de la OMS (2021)
GUIA_OMS_ANUAL = {"PM2.5": 5, "PM10": 15}


def calcular_ica(concentracion, contaminante="PM2.5"):
    """Devuelve el ICA (número entero) de una concentración. NaN si no hay dato."""
    if concentracion is None or (isinstance(concentracion, float) and math.isnan(concentracion)):
        return np.nan
    if contaminante not in PUNTOS_CORTE:
        raise ValueError(f"ICA no definido aquí para {contaminante}")
    # La norma trabaja con concentraciones enteras para PM: se trunca
    c = math.floor(concentracion)
    cortes = PUNTOS_CORTE[contaminante]
    if c > cortes[-1][1]:
        return 500
    for (pc_bajo, pc_alto), (i_bajo, i_alto, *_ ) in zip(cortes, CATEGORIAS):
        if pc_bajo <= c <= pc_alto:
            return round((i_alto - i_bajo) / (pc_alto - pc_bajo) * (c - pc_bajo) + i_bajo)
    return np.nan


def categoria(ica):
    """Devuelve (nombre, color, recomendación) de un valor de ICA."""
    if ica is None or (isinstance(ica, float) and math.isnan(ica)):
        return ("Sin dato", "#BDBDBD", "No hay datos suficientes.")
    for i_bajo, i_alto, nombre, color, rec in CATEGORIAS:
        if ica <= i_alto:
            return (nombre, color, rec)
    return CATEGORIAS[-1][2:]


def categoria_concentracion(concentracion, contaminante="PM2.5"):
    """Atajo: concentración -> (ICA, nombre, color, recomendación)."""
    i = calcular_ica(concentracion, contaminante)
    return (i, *categoria(i))


def ica_vector(concentraciones, contaminante="PM2.5"):
    """Versión para arreglos (numpy) de calcular_ica, sin redondear: se usa para colorear el mapa."""
    c = np.floor(np.asarray(concentraciones, dtype=float))
    resultado = np.full(c.shape, np.nan)
    for (pc_bajo, pc_alto), (i_bajo, i_alto, *_ ) in zip(PUNTOS_CORTE[contaminante], CATEGORIAS):
        en_rango = (c >= pc_bajo) & (c <= pc_alto)
        resultado[en_rango] = (i_alto - i_bajo) / (pc_alto - pc_bajo) * (c[en_rango] - pc_bajo) + i_bajo
    resultado[c > PUNTOS_CORTE[contaminante][-1][1]] = 500
    return resultado
