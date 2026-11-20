"""
idw.py
Interpolación por Distancia Inversa Ponderada (IDW, Inverse Distance Weighting).

Idea: la concentración en un punto cualquiera se estima como un promedio ponderado de lo que
miden las estaciones; las estaciones más cercanas pesan más.

    C(x) = Σ wᵢ·Cᵢ / Σ wᵢ        con   wᵢ = 1 / dᵢ^p

    Cᵢ : concentración medida en la estación i
    dᵢ : distancia (km) entre el punto x y la estación i
    p  : exponente de potencia. p grande -> manda la estación más cercana;
         p = 0 -> todas pesan igual (promedio simple de las estaciones).

Propiedades que se usan para validar el código (ver test_idw.py):
  * Si el punto coincide con una estación, el resultado es el valor de esa estación.
  * El resultado siempre queda entre el mínimo y el máximo de las estaciones
    (es un promedio ponderado con pesos positivos).
  * Con p = 0 el resultado es el promedio aritmético de las estaciones.
"""

import numpy as np

RADIO_TIERRA_KM = 6371.0


def distancia_km(lat1, lon1, lat2, lon2):
    """
    Distancia sobre la superficie terrestre (fórmula de haversine), en km.
    Acepta números o arreglos (numpy hace el cálculo para todos a la vez).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(a))


def idw(lat_puntos, lon_puntos, lat_est, lon_est, valores, p=2.0, d_min=0.01):
    """
    Interpola valores de estaciones a uno o varios puntos.

    lat_puntos, lon_puntos : coordenadas de los puntos a estimar (número o arreglo)
    lat_est, lon_est       : coordenadas de las estaciones (arreglos)
    valores                : concentración en cada estación (arreglo; NaN = sin dato)
    p                      : exponente de potencia
    d_min                  : distancia (km) por debajo de la cual se toma el valor de la estación

    Devuelve un arreglo con la concentración estimada en cada punto
    (NaN si ninguna estación tiene dato).
    """
    lat_puntos = np.atleast_1d(np.asarray(lat_puntos, dtype=float))
    lon_puntos = np.atleast_1d(np.asarray(lon_puntos, dtype=float))
    lat_est = np.asarray(lat_est, dtype=float)
    lon_est = np.asarray(lon_est, dtype=float)
    valores = np.asarray(valores, dtype=float)

    # Solo se usan estaciones con dato en esa hora
    ok = ~np.isnan(valores)
    if not ok.any():
        return np.full(lat_puntos.shape, np.nan)
    lat_est, lon_est, valores = lat_est[ok], lon_est[ok], valores[ok]

    # Matriz de distancias: una fila por punto, una columna por estación
    d = distancia_km(lat_puntos[:, None], lon_puntos[:, None], lat_est[None, :], lon_est[None, :])

    resultado = np.empty(lat_puntos.shape)
    muy_cerca = d < d_min
    hay_cerca = muy_cerca.any(axis=1)

    # Puntos sobre una estación: se toma el valor medido
    if hay_cerca.any():
        idx_est = np.argmin(d[hay_cerca], axis=1)
        resultado[hay_cerca] = valores[idx_est]

    # Resto de puntos: promedio ponderado por 1/d^p
    resto = ~hay_cerca
    if resto.any():
        w = 1.0 / d[resto] ** p
        resultado[resto] = (w * valores).sum(axis=1) / w.sum(axis=1)
    return resultado


def malla(lat_min, lat_max, lon_min, lon_max, n=60):
    """Crea una malla regular de puntos (para dibujar el mapa de toda la ciudad)."""
    lats = np.linspace(lat_min, lat_max, n)
    lons = np.linspace(lon_min, lon_max, n)
    LON, LAT = np.meshgrid(lons, lats)
    return LAT, LON
