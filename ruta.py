"""
ruta.py
Exposición personal a lo largo de una ruta y dosis inhalada.

Modelo:
  1. La ruta (lista de puntos lat/lon) se divide en tramos cortos (100 m).
  2. Cada tramo se recorre a la velocidad del modo de transporte -> se conoce a qué hora
     pasa la persona por cada tramo.
  3. La concentración en cada tramo se estima con IDW usando los datos horarios de las
     estaciones de la hora en que se pasa por ahí.
  4. Dosis inhalada:
         D = Σ  Cᵢ · F · IR · Δtᵢ          [µg]
     Cᵢ  : concentración estimada en el tramo i            (µg/m³)
     F   : factor de microambiente (relación interior del vehículo / aire exterior; 1 = igual)
     IR  : tasa de inhalación según la actividad física      (m³/h)
     Δtᵢ : tiempo que se tarda en recorrer el tramo i        (h)
  5. Concentración promedio de la exposición (ponderada por tiempo):
         C̄ = Σ Cᵢ·Δtᵢ / Σ Δtᵢ

Tasas de inhalación (IR): U.S. EPA (2011), Exposure Factors Handbook, capítulo 6, tabla 6-2,
valores medios de corto plazo para adultos de 21 a <31 años:
    sedentaria/pasiva 4.2e-3 m³/min = 0.25 m³/h
    ligera            1.2e-2 m³/min = 0.72 m³/h
    moderada          2.6e-2 m³/min = 1.56 m³/h
    alta              5.0e-2 m³/min = 3.00 m³/h

Las velocidades son valores típicos para Bogotá y se pueden cambiar en la app (supuesto).
"""

import numpy as np
import pandas as pd

from idw import distancia_km, idw

IR_EPA = {               # m³/h
    "Sedentaria": 4.2e-3 * 60,
    "Ligera": 1.2e-2 * 60,
    "Moderada": 2.6e-2 * 60,
    "Alta": 5.0e-2 * 60,
}

MODOS = {
    # nombre:        velocidad (km/h), tasa de inhalación (m³/h),                     actividad (texto)
    "A pie":         {"velocidad": 4.5, "IR": IR_EPA["Moderada"],                                  "actividad": "Moderada (caminar rápido)"},
    "Bicicleta":     {"velocidad": 14.0, "IR": (IR_EPA["Moderada"] + IR_EPA["Alta"]) / 2,         "actividad": "Entre moderada y alta"},
    "TransMilenio":  {"velocidad": 23.0, "IR": IR_EPA["Ligera"],                                   "actividad": "Ligera (de pie)"},
    "Carro":         {"velocidad": 20.0, "IR": IR_EPA["Sedentaria"],                               "actividad": "Sedentaria (sentado)"},
}


def longitud_ruta_km(lats, lons):
    """Longitud total de una polilínea (suma de tramos), en km."""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    if len(lats) < 2:
        return 0.0
    return float(distancia_km(lats[:-1], lons[:-1], lats[1:], lons[1:]).sum())


def densificar(lats, lons, paso_km=0.1):
    """
    Agrega puntos intermedios para que ningún tramo mida más de paso_km.
    Devuelve (lats, lons) de los puntos.
    """
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    out_lat, out_lon = [lats[0]], [lons[0]]
    for i in range(len(lats) - 1):
        d = distancia_km(lats[i], lons[i], lats[i + 1], lons[i + 1])
        n = max(1, int(np.ceil(d / paso_km)))
        f = np.arange(1, n + 1) / n
        out_lat.extend(lats[i] + f * (lats[i + 1] - lats[i]))
        out_lon.extend(lons[i] + f * (lons[i + 1] - lons[i]))
    return np.array(out_lat), np.array(out_lon)


def exposicion_ruta(lats, lons, salida, velocidad_kmh, ir_m3h, horario, tabla_est,
                    p=1.0, factor_micro=1.0, paso_km=0.1):
    """
    Calcula la exposición tramo por tramo.

    lats, lons     : puntos de la ruta (en orden)
    salida         : pd.Timestamp de la hora de salida
    velocidad_kmh  : velocidad del modo
    ir_m3h         : tasa de inhalación (m³/h)
    horario        : DataFrame horas x estaciones con la concentración horaria
    tabla_est      : tabla de estaciones (lat, lon)

    Devuelve (tramos, resumen):
      tramos  : DataFrame con un fila por tramo (coordenadas, hora, C, Δt, dosis)
      resumen : diccionario con distancia, duración, C promedio, C máxima y dosis total
    """
    la, lo = densificar(lats, lons, paso_km)
    d = distancia_km(la[:-1], lo[:-1], la[1:], lo[1:])          # km de cada tramo
    dt_h = d / velocidad_kmh                                     # horas en cada tramo
    t_ini = np.concatenate([[0.0], np.cumsum(dt_h)[:-1]])        # hora de inicio de cada tramo (h desde la salida)
    t_mid = salida + pd.to_timedelta((t_ini + dt_h / 2) * 3600, unit="s")
    lat_mid, lon_mid = (la[:-1] + la[1:]) / 2, (lo[:-1] + lo[1:]) / 2

    # La RMCAB marca cada dato al final de la hora: el momento 07:20 corresponde al dato de las 08:00
    etiqueta_hora = t_mid.ceil("h")
    est = tabla_est.loc[horario.columns]
    C = np.full(len(d), np.nan)
    for h in etiqueta_hora.unique():
        sel = etiqueta_hora == h
        if h in horario.index:
            C[sel] = idw(lat_mid[sel], lon_mid[sel], est["lat"], est["lon"], horario.loc[h].to_numpy(float), p=p)

    C_exp = C * factor_micro
    dosis = C_exp * ir_m3h * dt_h                                 # µg en cada tramo
    tramos = pd.DataFrame({
        "lat1": la[:-1], "lon1": lo[:-1], "lat2": la[1:], "lon2": lo[1:],
        "hora": t_mid, "dist_km": d, "dt_h": dt_h, "C": C, "C_exp": C_exp, "dosis_ug": dosis,
    })
    valido = ~np.isnan(C_exp)
    resumen = {
        "distancia_km": float(d.sum()),
        "duracion_min": float(dt_h.sum() * 60),
        "C_promedio": float(np.sum(C_exp[valido] * dt_h[valido]) / np.sum(dt_h[valido])) if valido.any() else np.nan,
        "C_max": float(np.nanmax(C_exp)) if valido.any() else np.nan,
        "dosis_ug": float(np.nansum(dosis)) if valido.any() else np.nan,
        "fraccion_con_dato": float(valido.mean()),
    }
    return tramos, resumen


def comparar_modos(lats, lons, salida, horario, tabla_est, p=1.0, modos=None, factores=None):
    """Misma ruta y hora de salida en todos los modos de transporte."""
    modos = modos or MODOS
    factores = factores or {}
    filas = []
    for nombre, m in modos.items():
        _, r = exposicion_ruta(lats, lons, salida, m["velocidad"], m["IR"], horario, tabla_est,
                               p=p, factor_micro=factores.get(nombre, 1.0))
        filas.append({"modo": nombre, **r})
    return pd.DataFrame(filas).set_index("modo")


def barrido_horas(lats, lons, fecha, velocidad_kmh, ir_m3h, horario, tabla_est, p=1.0,
                  factor_micro=1.0, horas=range(5, 23)):
    """Dosis según la hora de salida a lo largo del día (para encontrar la mejor hora)."""
    filas = []
    for h in horas:
        salida = pd.Timestamp(fecha) + pd.Timedelta(hours=h)
        _, r = exposicion_ruta(lats, lons, salida, velocidad_kmh, ir_m3h, horario, tabla_est,
                               p=p, factor_micro=factor_micro)
        filas.append({"hora_salida": h, **r})
    return pd.DataFrame(filas).set_index("hora_salida")
