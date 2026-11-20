"""
test_idw.py
Pruebas automáticas: comprueban que los cálculos hacen lo que deben.
Se corren con:   python -m pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from datos import leer_fecha, promedio_valido
from estaciones import buscar_estacion, dms_a_decimal, tabla_estaciones
from idw import distancia_km, idw
from validacion import metricas, predicciones_loocv

# Tres estaciones inventadas para las pruebas
LAT = np.array([4.60, 4.65, 4.70])
LON = np.array([-74.10, -74.05, -74.12])
VAL = np.array([10.0, 30.0, 20.0])


def test_dms_a_decimal():
    assert dms_a_decimal(4, 30, 0) == pytest.approx(4.5)


def test_distancia_un_grado_de_latitud():
    # Un grado de latitud mide ~111.2 km
    assert distancia_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.05)


def test_distancia_conocida_bogota():
    # Guaymaral (norte) a Usme (sur): ~28 km en línea recta
    t = tabla_estaciones()
    d = distancia_km(t.loc["GYR", "lat"], t.loc["GYR", "lon"], t.loc["USM", "lat"], t.loc["USM", "lon"])
    assert 26 < d < 30


def test_idw_en_una_estacion_devuelve_su_valor():
    for i in range(3):
        assert idw(LAT[i], LON[i], LAT, LON, VAL)[0] == pytest.approx(VAL[i])


def test_idw_queda_entre_minimo_y_maximo():
    lat = np.random.default_rng(1).uniform(4.5, 4.8, 500)
    lon = np.random.default_rng(2).uniform(-74.2, -74.0, 500)
    r = idw(lat, lon, LAT, LON, VAL, p=2)
    assert r.min() >= VAL.min() - 1e-9 and r.max() <= VAL.max() + 1e-9


def test_idw_p0_es_el_promedio():
    assert idw(4.62, -74.08, LAT, LON, VAL, p=0)[0] == pytest.approx(VAL.mean())


def test_idw_ignora_estaciones_sin_dato():
    v = np.array([10.0, np.nan, 20.0])
    r = idw(4.65, -74.05, LAT, LON, v)[0]
    assert 10 <= r <= 20


def test_idw_sin_ningun_dato_da_nan():
    assert np.isnan(idw(4.65, -74.05, LAT, LON, [np.nan] * 3)[0])


def test_idw_campo_constante():
    # Si todas las estaciones miden lo mismo, en cualquier punto debe dar ese valor
    assert idw(4.63, -74.09, LAT, LON, [25.0] * 3)[0] == pytest.approx(25.0)


def test_loocv_matricial_igual_a_idw_punto_a_punto():
    # La versión rápida (matrices) debe dar exactamente lo mismo que la función idw()
    t = tabla_estaciones().loc[["GYR", "USQ", "SUB", "BOL", "KEN"]]
    rng = np.random.default_rng(0)
    m = pd.DataFrame(rng.uniform(5, 60, (20, 5)), columns=t.index)
    pred = predicciones_loocv(m, t, p=2)
    for s in t.index:
        otras = [o for o in t.index if o != s]
        for h in range(20):
            esperado = idw(t.loc[s, "lat"], t.loc[s, "lon"], t.loc[otras, "lat"], t.loc[otras, "lon"],
                           m.loc[h, otras], p=2)[0]
            assert pred.loc[h, s] == pytest.approx(esperado)


def test_metricas_prediccion_perfecta():
    r = metricas([1, 2, 3, 4], [1, 2, 3, 4])
    assert r["RMSE"] == 0 and r["R2"] == 1


def test_metricas_error_conocido():
    r = metricas([10, 20], [12, 18])      # errores +2 y -2
    assert r["RMSE"] == pytest.approx(2) and r["MAE"] == pytest.approx(2) and r["Sesgo"] == 0


def test_fecha_24_horas():
    f = leer_fecha(pd.Series(["31-12-2025 24:00"]))
    assert f.iloc[0] == pd.Timestamp("2026-01-01 00:00")


def test_promedio_diario_criterio_75():
    idx = pd.date_range("2025-01-01 01:00", periods=48, freq="h")
    s = pd.Series(10.0, index=idx)
    s.iloc[24:24 + 7] = np.nan            # día 2 con 17 de 24 horas (71 %) -> no válido
    prom = promedio_valido(s, "D")
    assert prom.iloc[0] == 10.0 and np.isnan(prom.iloc[1])


def test_reconoce_nombres_de_archivo():
    assert buscar_estacion("Centro_de_alto_rendimiento_2025.xlsx") == "CDAR"
    assert buscar_estacion("Ciudad_Bolivar_2025.xlsx") == "CBV"
    assert buscar_estacion("Kennedy_2025_horario.xlsx") == "KEN"
    assert buscar_estacion("Carvajal-Sevillana_2025.xlsx") == "CSE"


# ---------- ICA (Res. 2254 de 2017) ----------
from ica import calcular_ica, categoria


def test_ica_extremos_de_rango_pm25():
    assert calcular_ica(0, "PM2.5") == 0
    assert calcular_ica(12, "PM2.5") == 50
    assert calcular_ica(13, "PM2.5") == 51
    assert calcular_ica(37, "PM2.5") == 100      # límite 24 h de la norma = tope de "Aceptable"
    assert calcular_ica(38, "PM2.5") == 101
    assert calcular_ica(55, "PM2.5") == 150


def test_ica_calculo_a_mano_pm25():
    # C = 25 µg/m³ -> rango 13-37 / 51-100:  (100-51)/(37-13)·(25-13) + 51 = 75.5 -> 76
    assert calcular_ica(25, "PM2.5") == 76


def test_ica_pm10():
    assert calcular_ica(75, "PM10") == 61        # (100-51)/(154-55)·(75-55)+51 = 60.9
    assert calcular_ica(154, "PM10") == 100


def test_ica_categorias():
    assert categoria(calcular_ica(30, "PM2.5"))[0] == "Aceptable"
    assert categoria(calcular_ica(45, "PM2.5"))[0] == "Dañina a la salud de grupos sensibles"
    assert categoria(calcular_ica(80, "PM2.5"))[0] == "Dañina a la salud"


# ---------- Exposición en la ruta ----------
from ruta import densificar, exposicion_ruta, longitud_ruta_km


def _campo_constante(valor=20.0):
    t = tabla_estaciones().loc[["GYR", "USQ", "SUB", "BOL", "KEN", "USM"]]
    idx = pd.date_range("2025-03-01 01:00", periods=48, freq="h")
    return pd.DataFrame(valor, index=idx, columns=t.index), t


def test_densificar_conserva_longitud():
    lat, lon = [4.60, 4.70], [-74.10, -74.05]
    la, lo = densificar(lat, lon, 0.1)
    assert longitud_ruta_km(la, lo) == pytest.approx(longitud_ruta_km(lat, lon), rel=1e-6)
    assert np.all(distancia_km(la[:-1], lo[:-1], la[1:], lo[1:]) <= 0.1 + 1e-9)


def test_exposicion_campo_constante_calculo_a_mano():
    # Ruta de ~5.56 km hacia el norte (0.05° de latitud), a 10 km/h, IR = 1 m³/h, C = 20 µg/m³
    horario, t = _campo_constante(20.0)
    lat, lon = [4.60, 4.65], [-74.10, -74.10]
    dist = longitud_ruta_km(lat, lon)
    _, r = exposicion_ruta(lat, lon, pd.Timestamp("2025-03-01 07:00"), 10, 1.0, horario, t)
    assert r["C_promedio"] == pytest.approx(20.0)
    assert r["duracion_min"] == pytest.approx(dist / 10 * 60)
    assert r["dosis_ug"] == pytest.approx(20.0 * 1.0 * dist / 10)     # D = C · IR · t


def test_dosis_proporcional_a_la_tasa_de_inhalacion():
    horario, t = _campo_constante(30.0)
    lat, lon = [4.60, 4.66], [-74.12, -74.08]
    _, r1 = exposicion_ruta(lat, lon, pd.Timestamp("2025-03-01 07:00"), 15, 1.0, horario, t)
    _, r2 = exposicion_ruta(lat, lon, pd.Timestamp("2025-03-01 07:00"), 15, 2.0, horario, t)
    assert r2["dosis_ug"] == pytest.approx(2 * r1["dosis_ug"])


def test_mas_lento_mas_dosis():
    horario, t = _campo_constante(30.0)
    lat, lon = [4.60, 4.66], [-74.12, -74.08]
    _, rapido = exposicion_ruta(lat, lon, pd.Timestamp("2025-03-01 07:00"), 20, 1.0, horario, t)
    _, lento = exposicion_ruta(lat, lon, pd.Timestamp("2025-03-01 07:00"), 5, 1.0, horario, t)
    assert lento["dosis_ug"] == pytest.approx(4 * rapido["dosis_ug"])


def test_usa_la_hora_correcta():
    # Concentración 10 hasta las 08:00 y 50 desde las 09:00: salir 08:10 debe dar 50
    horario, t = _campo_constante(10.0)
    horario.loc[pd.Timestamp("2025-03-01 09:00"):] = 50.0
    lat, lon = [4.60, 4.61], [-74.10, -74.10]
    _, r = exposicion_ruta(lat, lon, pd.Timestamp("2025-03-01 08:10"), 10, 1.0, horario, t)
    assert r["C_promedio"] == pytest.approx(50.0)
