"""
validacion.py
Validación cruzada "deja una fuera" (leave-one-out cross-validation, LOOCV) del IDW.

Procedimiento, para cada hora del año y cada estación con dato:
  1. Se "esconde" la estación.
  2. Se estima su concentración con IDW usando solo las demás estaciones.
  3. Se compara lo estimado con lo que la estación midió de verdad.

Con todos esos pares (medido, estimado) se calculan:
  RMSE  = raíz del error cuadrático medio         (µg/m³; mientras más bajo, mejor)
  MAE   = error absoluto medio                    (µg/m³)
  Sesgo = promedio de (estimado - medido)         (>0 sobreestima, <0 subestima)
  R²    = coeficiente de determinación            (1 = perfecto, 0 = no explica nada)

También se compara contra dos métodos de referencia:
  * "Promedio de las demás"  (equivale a IDW con p = 0)
  * "Estación más cercana"   (vecino más cercano)
Si el IDW no le gana a estos métodos simples, no vale la pena usarlo.
"""

import numpy as np
import pandas as pd

from idw import distancia_km

MIN_OTRAS = 3   # mínimo de estaciones vecinas con dato para hacer la estimación


def _matriz_distancias(tabla_est, siglas):
    lat = tabla_est.loc[siglas, "lat"].to_numpy()
    lon = tabla_est.loc[siglas, "lon"].to_numpy()
    return distancia_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def predicciones_loocv(matriz, tabla_est, p=2.0, metodo="idw"):
    """
    matriz    : DataFrame (horas x estaciones) de un contaminante
    tabla_est : tabla de estaciones con columnas lat y lon (índice = sigla)
    metodo    : 'idw' o 'cercana'
    Devuelve un DataFrame con las estimaciones (mismo tamaño que matriz).

    Se calcula con álgebra de matrices para que sea rápido; es exactamente la fórmula del IDW
    (test_idw.py comprueba que da lo mismo que la función idw() punto por punto).
    """
    siglas = list(matriz.columns)
    C = matriz.to_numpy(dtype=float)               # concentraciones (T x n)
    M = ~np.isnan(C)                               # True donde hay dato
    C0 = np.where(M, C, 0.0)
    D = _matriz_distancias(tabla_est, siglas)      # distancias entre estaciones (n x n)

    if metodo == "idw":
        with np.errstate(divide="ignore"):
            W = 1.0 / D ** p
        np.fill_diagonal(W, 0.0)                   # la estación escondida no se usa a sí misma
        numerador = C0 @ W.T                       # Σ wᵢⱼ·Cⱼ  (solo estaciones con dato)
        denominador = M.astype(float) @ W.T        # Σ wᵢⱼ
        with np.errstate(invalid="ignore", divide="ignore"):
            pred = numerador / denominador
    elif metodo == "cercana":
        pred = np.full(C.shape, np.nan)
        orden = np.argsort(D, axis=1)[:, 1:]       # vecinas de cada estación, de la más cercana a la más lejana
        for i in range(len(siglas)):
            for t in range(C.shape[0]):
                for j in orden[i]:
                    if M[t, j]:
                        pred[t, i] = C[t, j]
                        break
    else:
        raise ValueError("metodo debe ser 'idw' o 'cercana'")

    # Solo se evalúa donde la estación tiene dato y hay suficientes vecinas con dato
    otras_con_dato = M.sum(axis=1, keepdims=True) - M
    pred = np.where(M & (otras_con_dato >= MIN_OTRAS), pred, np.nan)
    return pd.DataFrame(pred, index=matriz.index, columns=siglas)


def metricas(medido, estimado):
    """RMSE, MAE, sesgo y R² entre dos arreglos (ignora pares con NaN)."""
    medido = np.asarray(medido, dtype=float).ravel()
    estimado = np.asarray(estimado, dtype=float).ravel()
    ok = ~np.isnan(medido) & ~np.isnan(estimado)
    m, e = medido[ok], estimado[ok]
    error = e - m
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((m - m.mean()) ** 2)
    return {
        "n": int(ok.sum()),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "Sesgo": float(np.mean(error)),
        "R2": float(1 - ss_res / ss_tot),
    }


def comparar_metodos(matriz, tabla_est, potencias=(0, 1, 1.5, 2, 2.5, 3, 4)):
    """Tabla con las métricas de IDW para varios p y del vecino más cercano."""
    filas = []
    for p in potencias:
        pred = predicciones_loocv(matriz, tabla_est, p=p)
        nombre = "Promedio de las demás (p=0)" if p == 0 else f"IDW p={p}"
        filas.append({"metodo": nombre, "p": p, **metricas(matriz, pred)})
    pred = predicciones_loocv(matriz, tabla_est, metodo="cercana")
    filas.append({"metodo": "Estación más cercana", "p": np.nan, **metricas(matriz, pred)})
    return pd.DataFrame(filas).set_index("metodo")


def metricas_por_estacion(matriz, tabla_est, p=2.0):
    """Qué tan bien se predice cada estación cuando se esconde (sirve para ver zonas mal cubiertas)."""
    pred = predicciones_loocv(matriz, tabla_est, p=p)
    filas = []
    for s in matriz.columns:
        filas.append({"sigla": s, "estacion": tabla_est.loc[s, "nombre"],
                      "media_medida": float(np.nanmean(matriz[s])),
                      **metricas(matriz[s], pred[s])})
    return pd.DataFrame(filas).set_index("sigla").sort_values("RMSE")
