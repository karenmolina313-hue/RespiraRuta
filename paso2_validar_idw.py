"""
paso2_validar_idw.py
Paso 2 del proyecto RespiraRuta: carga los datos, revisa su calidad y valida la interpolación IDW.

Uso (desde la carpeta del proyecto):
    python paso2_validar_idw.py                       # PM2.5, archivos en la carpeta 'datos'
    python paso2_validar_idw.py --contaminante PM10
    python paso2_validar_idw.py --carpeta "C:/ruta/a/mis/archivos"

Resultados en la carpeta 'resultados':
    completitud_<cont>.csv        % de datos válidos por estación
    comparacion_metodos_<cont>.csv  RMSE, MAE, sesgo y R² de cada método
    error_por_estacion_<cont>.csv qué tan bien se predice cada estación
    fig1_rmse_vs_p_<cont>.png     cómo cambia el error con el exponente p
    fig2_medido_vs_estimado_<cont>.png
    fig3_mapa_promedio_<cont>.png mapa interpolado del promedio anual
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datos import cargar_carpeta, matriz_contaminante, reporte_completitud
from estaciones import tabla_estaciones
from idw import idw, malla
from validacion import comparar_metodos, metricas_por_estacion, predicciones_loocv

TINTA = "#2b2b2b"
GRIS = "#8a8a8a"
ACENTO = "#c2410c"


def estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRIS)
    ax.tick_params(colors=TINTA, labelsize=9)
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=None,
                    help="Carpeta con los .xlsx. Por defecto: 'datos' si existe; si no, la carpeta que contiene a RespiraRuta")
    ap.add_argument("--contaminante", default="PM2.5")
    args = ap.parse_args()
    cont = args.contaminante
    salida = Path(__file__).resolve().parent / "resultados"
    salida.mkdir(exist_ok=True)
    etiqueta = cont.replace(".", "")

    # 1. Datos
    carpeta = args.carpeta
    if carpeta is None:
        carpeta = "datos" if Path("datos").is_dir() else Path(__file__).resolve().parent.parent
    datos = cargar_carpeta(carpeta)
    tabla = tabla_estaciones()
    matriz = matriz_contaminante(datos, cont)
    print(f"\nEstaciones leídas: {len(datos)}  |  con datos de {cont}: {matriz.shape[1]}")
    print(f"Periodo: {matriz.index.min()}  a  {matriz.index.max()}  ({len(matriz)} horas)\n")

    comp = reporte_completitud(matriz)
    comp.insert(0, "estacion", tabla.loc[comp.index, "nombre"])
    comp.to_csv(salida / f"completitud_{etiqueta}.csv", encoding="utf-8-sig")
    print("== Completitud de datos ==")
    print(comp.to_string(), "\n")

    # 2. Validación cruzada
    tabla_m = comparar_metodos(matriz, tabla)
    tabla_m.round(3).to_csv(salida / f"comparacion_metodos_{etiqueta}.csv", encoding="utf-8-sig")
    print("== Validación cruzada 'deja una fuera' ==")
    print(tabla_m.round(2).to_string(), "\n")

    solo_idw = tabla_m[tabla_m["p"] > 0]
    p_mejor = float(solo_idw["RMSE"].idxmin().split("=")[1])
    print(f"Mejor exponente según RMSE: p = {p_mejor}\n")

    por_est = metricas_por_estacion(matriz, tabla, p=p_mejor)
    por_est.round(2).to_csv(salida / f"error_por_estacion_{etiqueta}.csv", encoding="utf-8-sig")
    print(f"== Error por estación (IDW p={p_mejor}) ==")
    print(por_est.round(2).to_string(), "\n")

    # 3. Figuras
    # Fig 1: RMSE vs p
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(solo_idw["p"], solo_idw["RMSE"], color=ACENTO, linewidth=2, marker="o", markersize=7)
    for nombre, fila in tabla_m.iterrows():
        if not (fila["p"] > 0):
            ax.axhline(fila["RMSE"], color=GRIS, linewidth=1.2, linestyle="--")
            ax.text(solo_idw["p"].min(), fila["RMSE"], f"{nombre}", va="bottom", fontsize=8, color=TINTA)
    ax.set_xlabel("Exponente p del IDW", color=TINTA)
    ax.set_ylabel(f"RMSE de {cont} (µg/m³)", color=TINTA)
    ax.set_title(f"Error de la interpolación según el exponente p — {cont}", color=TINTA, fontsize=11, loc="left")
    estilo(ax)
    fig.tight_layout()
    fig.savefig(salida / f"fig1_rmse_vs_p_{etiqueta}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: medido vs estimado
    pred = predicciones_loocv(matriz, tabla, p=p_mejor)
    m, e = matriz.to_numpy().ravel(), pred.to_numpy().ravel()
    ok = ~np.isnan(m) & ~np.isnan(e)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.hexbin(m[ok], e[ok], gridsize=60, bins="log", cmap="Oranges", mincnt=1)
    lim = np.nanpercentile(np.r_[m[ok], e[ok]], 99.5)
    ax.plot([0, lim], [0, lim], color=TINTA, linewidth=1.2)
    ax.text(lim * 0.97, lim * 0.92, "línea 1:1", ha="right", fontsize=8, color=TINTA)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fila = tabla_m.loc[f"IDW p={p_mejor:g}"]
    ax.text(0.03, 0.97, f"RMSE = {fila['RMSE']:.1f} µg/m³\nR² = {fila['R2']:.2f}\nn = {int(fila['n']):,} horas",
            transform=ax.transAxes, va="top", fontsize=9, color=TINTA)
    ax.set_xlabel(f"{cont} medido (µg/m³)", color=TINTA)
    ax.set_ylabel(f"{cont} estimado con IDW (µg/m³)", color=TINTA)
    ax.set_title("Validación cruzada: medido vs. estimado", color=TINTA, fontsize=11, loc="left")
    estilo(ax)
    fig.tight_layout()
    fig.savefig(salida / f"fig2_medido_vs_estimado_{etiqueta}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: mapa del promedio anual (solo estaciones que cumplen el 75 %)
    validas = comp.index[comp["cumple_75"]]
    prom = matriz[validas].mean()
    t = tabla.loc[validas]
    LAT, LON = malla(t["lat"].min() - 0.02, t["lat"].max() + 0.02, t["lon"].min() - 0.02, t["lon"].max() + 0.02, n=120)
    Z = idw(LAT.ravel(), LON.ravel(), t["lat"], t["lon"], prom.to_numpy(), p=p_mejor).reshape(LAT.shape)
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.contourf(LON, LAT, Z, levels=12, cmap="Oranges")
    ax.scatter(t["lon"], t["lat"], s=40, color="white", edgecolor=TINTA, linewidth=1.2, zorder=3)
    for s, f in t.iterrows():
        desplazamiento = (5, -18) if s == "MOV2" else (5, 4)   # evita que se monte con Fontibón
        ax.annotate(f"{f['nombre']}\n{prom[s]:.1f}", (f["lon"], f["lat"]), xytext=desplazamiento,
                    textcoords="offset points", fontsize=7, color=TINTA)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label(f"{cont} promedio (µg/m³)", color=TINTA)
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
    ax.set_title(f"{cont} promedio del periodo interpolado con IDW (p={p_mejor:g})", color=TINTA, fontsize=11, loc="left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(salida / f"fig3_mapa_promedio_{etiqueta}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Listo. Resultados guardados en la carpeta '{salida}'.")


if __name__ == "__main__":
    main()
