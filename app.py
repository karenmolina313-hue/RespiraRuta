"""
app.py — RespiraRuta
Aplicación web (Streamlit) del proyecto integrador de Control y Simulación Atmosférica.

Para abrirla, en la terminal (dentro de la carpeta RespiraRuta):
    py -m streamlit run app.py
"""

from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from datos import cargar_archivos, cargar_carpeta, matriz_contaminante, reporte_completitud
from estaciones import tabla_estaciones
from ica import CATEGORIAS, LIMITE_24H, categoria_concentracion, ica_vector
from folium.plugins import Draw

from idw import distancia_km, idw
from ruta import MODOS, barrido_horas, comparar_modos, exposicion_ruta
from validacion import comparar_metodos

st.set_page_config(page_title="RespiraRuta", page_icon="🌫️", layout="wide")

CARPETA_PROYECTO = Path(__file__).resolve().parent
RADIO_MAX_KM = 6      # no se interpola a más de 6 km de la estación más cercana (no extrapolar)
CENTRO_BOGOTA = (4.65, -74.10)


# ----------------------------------------------------------------------------------------------
# Datos (se leen una sola vez y quedan en memoria)
# ----------------------------------------------------------------------------------------------
def carpeta_datos():
    local = CARPETA_PROYECTO / "datos"
    return local if local.is_dir() else CARPETA_PROYECTO.parent


def preparar(datos):
    horario = {c: matriz_contaminante(datos, c) for c in ["PM2.5", "PM10"]}
    # Promedio móvil de 24 h: es el tiempo de exposición con el que se define el ICA de PM.
    # Se exige el 75 % de las horas (18 de 24) para que el promedio sea válido.
    movil24 = {c: m.rolling(24, min_periods=18).mean() for c, m in horario.items()}
    return horario, movil24


@st.cache_data(show_spinner="Leyendo los archivos de la RMCAB (solo la primera vez)…", persist="disk")
def cargar(carpeta):
    return cargar_carpeta(carpeta)


@st.cache_data(show_spinner="Procesando los archivos subidos…")
def cargar_subidos(archivos):
    return cargar_archivos(list(archivos))


@st.cache_data(show_spinner="Calculando la validación cruzada con estos datos…")
def validar(_horario, clave, contaminante):
    return comparar_metodos(_horario, tabla_estaciones())


# Degradado continuo sobre los colores oficiales del ICA (cada color en el centro de su rango)
PARADAS_ICA = [0, 25, 75, 125, 175, 250, 400, 500]
COLORES_PARADAS = ["#00E400", "#00E400", "#FFFF00", "#FF7E00", "#FF0000", "#8F3F97", "#7E0023", "#7E0023"]


def colorear_ica(ica):
    """Convierte valores de ICA en colores RGB interpolando entre los colores oficiales."""
    rgb = np.array([hex_a_rgb(c) for c in COLORES_PARADAS], dtype=float)
    salida = np.zeros(ica.shape + (3,))
    for k in range(3):
        salida[..., k] = np.interp(ica, PARADAS_ICA, rgb[:, k])
    return salida.astype(np.uint8)


def hex_a_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def imagen_malla(est, valores, contaminante, p, n=140):
    """Interpola con IDW sobre una malla y devuelve una imagen RGBA coloreada con la escala del ICA."""
    lat_min, lat_max = est["lat"].min() - 0.04, est["lat"].max() + 0.04
    lon_min, lon_max = est["lon"].min() - 0.04, est["lon"].max() + 0.04
    lats = np.linspace(lat_max, lat_min, n)          # de norte a sur: la fila 0 de la imagen es el norte
    lons = np.linspace(lon_min, lon_max, n)
    LON, LAT = np.meshgrid(lons, lats)

    Z = idw(LAT.ravel(), LON.ravel(), est["lat"], est["lon"], valores, p=p).reshape(LAT.shape)

    # Distancia a la estación con dato más cercana: más allá de RADIO_MAX_KM no se muestra
    con_dato = ~np.isnan(valores)
    d = distancia_km(LAT.ravel()[:, None], LON.ravel()[:, None],
                     est["lat"].to_numpy()[con_dato][None, :], est["lon"].to_numpy()[con_dato][None, :])
    lejos = (d.min(axis=1) > RADIO_MAX_KM).reshape(LAT.shape)

    rgba = np.zeros((n, n, 4), dtype=np.uint8)
    validos = ~np.isnan(Z) & ~lejos
    rgba[..., :3] = colorear_ica(ica_vector(np.where(validos, Z, 0), contaminante))
    rgba[..., 3] = np.where(validos, 130, 0)         # transparencia
    return rgba, [[lat_min, lon_min], [lat_max, lon_max]]


# ----------------------------------------------------------------------------------------------
# Pestaña "Mi ruta"
# ----------------------------------------------------------------------------------------------
# Rutas de ejemplo que pasan por estaciones (coordenadas oficiales): sirven para la demostración
RUTAS_EJEMPLO = {
    "Sur industrial: Carvajal-Sevillana → Puente Aranda → MinAmbiente": ["CSE", "PTE", "MAM"],
    "Occidente: Fontibón → Puente Aranda → Móvil 7ma": ["FTB", "MOV2", "PTE", "MAM", "MOV7"],
    "Norte: Suba → Las Ferias → Centro de Alto Rendimiento": ["SUB", "LFR", "CDAR"],
    "Sur–norte: Usme → Tunal → El Jazmín → Las Ferias": ["USM", "TUN", "JAZ", "LFR"],
}
OPCION_DIBUJAR = "✏️ Dibujar mi propia ruta en el mapa"


def color_de_concentracion(c, contaminante):
    if np.isnan(c):
        return "#9e9e9e"
    r, g, b = colorear_ica(ica_vector(np.array([c]), contaminante))[0]
    return f"#{r:02x}{g:02x}{b:02x}"


def ruta_desde_dibujo(dibujos):
    """Toma la última línea dibujada (GeoJSON) y devuelve (lats, lons)."""
    for f in reversed(dibujos or []):
        geom = f.get("geometry", {})
        if geom.get("type") == "LineString" and len(geom.get("coordinates", [])) >= 2:
            coords = geom["coordinates"]                     # GeoJSON viene como [lon, lat]
            return [c[1] for c in coords], [c[0] for c in coords]
    return None


def pestana_ruta():
    st.markdown("Estima **cuánta contaminación respiras en un recorrido** según la ruta, la hora de salida "
                "y el modo de transporte. La concentración de cada tramo se interpola con IDW a partir de las "
                "estaciones, con el dato de la hora en que pasas por ahí.")

    c1, c2, c3 = st.columns([2.2, 1.3, 1])
    with c1:
        eleccion = st.selectbox("Ruta", [OPCION_DIBUJAR] + list(RUTAS_EJEMPLO))
    with c2:
        modo = st.radio("Modo de transporte", list(MODOS), horizontal=True, index=1)
    with c3:
        hora_salida = st.time_input("Hora de salida", value=pd.Timestamp("07:00").time(), step=600)

    with st.expander("⚙️ Supuestos del modelo (se pueden cambiar)"):
        a, b, c = st.columns(3)
        vel = a.number_input("Velocidad (km/h)", 1.0, 60.0, float(MODOS[modo]["velocidad"]), 0.5,
                             key=f"vel_{modo}")
        ir = b.number_input("Tasa de inhalación (m³/h)", 0.1, 5.0, float(round(MODOS[modo]["IR"], 2)), 0.05,
                            key=f"ir_{modo}", help="U.S. EPA (2011), Exposure Factors Handbook, cap. 6, tabla 6-2 "
                                                  f"(adultos 21–31 años). Actividad asumida: {MODOS[modo]['actividad']}.")
        fmic = c.number_input("Factor de microambiente", 0.1, 3.0, 1.0, 0.1, key=f"fm_{modo}",
                              help="Relación entre la concentración dentro del vehículo y la del aire exterior. "
                                   "1 = igual que afuera. Depende de ventanas, filtros y tráfico.")
        st.caption("Las velocidades son valores típicos para Bogotá (supuesto). La ruta se divide en tramos de 100 m.")

    # --- Obtener la ruta ---
    if eleccion == OPCION_DIBUJAR:
        st.caption("Usa el botón de **línea** (arriba a la izquierda del mapa), haz clic punto por punto sobre las "
                   "calles de tu recorrido y termina con doble clic o con *Finish*.")
        m = folium.Map(location=CENTRO_BOGOTA, zoom_start=12, tiles="OpenStreetMap")
        Draw(draw_options={"polyline": {"shapeOptions": {"color": "#1f4e79", "weight": 5}},
                           "polygon": False, "rectangle": False, "circle": False,
                           "marker": False, "circlemarker": False},
             edit_options={"edit": False}).add_to(m)
        for sigla, f in TABLA.loc[HORARIO[contaminante].columns].iterrows():
            folium.CircleMarker((f["lat"], f["lon"]), radius=4, color="#333", fill=True,
                                fill_color="#333", tooltip=f["nombre"]).add_to(m)
        salida_mapa = st_folium(m, height=480, use_container_width=True,
                                returned_objects=["all_drawings"], key="mapa_dibujo")
        nueva = ruta_desde_dibujo((salida_mapa or {}).get("all_drawings"))
        if nueva:
            st.session_state["ruta_dibujada"] = nueva
        ruta = st.session_state.get("ruta_dibujada")
        if not ruta:
            st.info("Dibuja una ruta para ver los resultados.")
            return
    else:
        siglas = RUTAS_EJEMPLO[eleccion]
        ruta = (list(TABLA.loc[siglas, "lat"]), list(TABLA.loc[siglas, "lon"]))

    lats, lons = ruta
    salida = pd.Timestamp(fecha) + pd.Timedelta(hours=hora_salida.hour, minutes=hora_salida.minute)
    horario = HORARIO[contaminante]
    tramos, r = exposicion_ruta(lats, lons, salida, vel, ir, horario, TABLA, p=p, factor_micro=fmic)

    if np.isnan(r["dosis_ug"]):
        st.warning("No hay datos de las estaciones para esa fecha y hora.")
        return

    # --- Resultados principales ---
    st.subheader(f"Resultado: {modo.lower()}, saliendo el {salida:%d/%m/%Y} a las {salida:%H:%M}")
    i_prom, cat_prom, _, rec = categoria_concentracion(r["C_promedio"], contaminante)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Distancia", f"{r['distancia_km']:.1f} km")
    k2.metric("Duración", f"{r['duracion_min']:.0f} min")
    k3.metric(f"{contaminante} promedio respirado", f"{r['C_promedio']:.1f} µg/m³",
              f"escala ICA: {cat_prom}", delta_color="off")
    k4.metric("Dosis inhalada", f"{r['dosis_ug']:.1f} µg")
    if r["fraccion_con_dato"] < 1:
        st.caption(f"⚠️ {100 * (1 - r['fraccion_con_dato']):.0f} % de los tramos quedó sin dato (horas sin medición).")

    col_m, col_g = st.columns([1.3, 1])
    with col_m:
        m2 = folium.Map(location=(np.mean(lats), np.mean(lons)), zoom_start=12, tiles="OpenStreetMap")
        for _, t in tramos.iterrows():
            folium.PolyLine([(t["lat1"], t["lon1"]), (t["lat2"], t["lon2"])], weight=8, opacity=0.9,
                            color=color_de_concentracion(t["C_exp"], contaminante),
                            tooltip=f"{t['hora']:%H:%M} · {t['C_exp']:.1f} µg/m³").add_to(m2)
        folium.Marker((lats[0], lons[0]), tooltip="Salida", icon=folium.Icon(color="green", icon="play")).add_to(m2)
        folium.Marker((lats[-1], lons[-1]), tooltip="Llegada", icon=folium.Icon(color="darkred", icon="flag")).add_to(m2)
        for sigla, f in TABLA.loc[horario.columns].iterrows():
            folium.CircleMarker((f["lat"], f["lon"]), radius=4, color="#333", fill=True,
                                fill_color="#333", tooltip=f["nombre"]).add_to(m2)
        m2.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        st_folium(m2, height=420, use_container_width=True, returned_objects=[], key="mapa_resultado")
    with col_g:
        minutos = np.cumsum(tramos["dt_h"].to_numpy()) * 60
        fig = go.Figure(go.Scatter(x=minutos, y=tramos["C_exp"], mode="lines", line=dict(color="#c2410c", width=2.5),
                                   fill="tozeroy", fillcolor="rgba(194,65,12,0.12)",
                                   hovertemplate="minuto %{x:.0f}: %{y:.1f} µg/m³<extra></extra>"))
        fig.add_hline(y=LIMITE_24H[contaminante], line_dash="dot", line_color="#555",
                      annotation_text=f"norma 24 h ({LIMITE_24H[contaminante]} µg/m³, referencia)",
                      annotation_position="bottom right")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                          title=dict(text="Lo que respiras a lo largo del recorrido", font=dict(size=14)),
                          xaxis_title="Minutos desde la salida", yaxis_title=f"{contaminante} (µg/m³)")
        fig.update_xaxes(gridcolor="#eee"); fig.update_yaxes(gridcolor="#eee", rangemode="tozero")
        st.plotly_chart(fig, width="stretch")
    st.caption(f"**Recomendación según el nivel promedio del recorrido:** {rec} "
               "(La escala ICA oficial se define con promedios de 24 h; aquí es indicativa.)")

    # --- Comparador de modos ---
    st.subheader("¿Y si fueras en otro modo?")
    factores = {modo: fmic}
    modos = {k: dict(v) for k, v in MODOS.items()}
    modos[modo].update(velocidad=vel, IR=ir)
    comp = comparar_modos(lats, lons, salida, horario, TABLA, p=p, modos=modos, factores=factores)
    colores = ["#c2410c" if k == modo else "#d9d9d9" for k in comp.index]
    fig = go.Figure(go.Bar(x=comp.index, y=comp["dosis_ug"], marker_color=colores,
                           text=[f"{v:.1f} µg<br>{d:.0f} min" for v, d in zip(comp["dosis_ug"], comp["duracion_min"])],
                           textposition="outside", hovertemplate="%{x}: %{y:.1f} µg<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      yaxis_title="Dosis inhalada (µg)", showlegend=False)
    fig.update_yaxes(gridcolor="#eee", rangemode="tozero")
    a, b = st.columns([1.4, 1])
    a.plotly_chart(fig, width="stretch")
    tabla_comp = comp[["duracion_min", "C_promedio", "dosis_ug"]].rename(columns={
        "duracion_min": "Duración (min)", "C_promedio": f"{contaminante} promedio (µg/m³)", "dosis_ug": "Dosis (µg)"})
    b.dataframe(tabla_comp.round(1), width="stretch")
    b.caption("La dosis depende de tres cosas: cuánto contaminante hay (C), cuánto aire respiras (IR) "
              "y cuánto tiempo estás expuesto (Δt). Caminar es lento y exige respirar más; el carro es rápido "
              "y sedentario, pero el aire dentro del vehículo puede ser peor que afuera (factor de microambiente).")

    # --- Mejor hora de salida ---
    st.subheader(f"¿A qué hora conviene salir? ({modo.lower()}, {salida:%d/%m/%Y})")
    barrido = barrido_horas(lats, lons, fecha, vel, ir, horario, TABLA, p=p, factor_micro=fmic)
    barrido = barrido.dropna(subset=["dosis_ug"])
    if not barrido.empty:
        mejor_h = barrido["dosis_ug"].idxmin()
        peor_h = barrido["dosis_ug"].idxmax()
        fig = go.Figure(go.Scatter(x=barrido.index, y=barrido["dosis_ug"], mode="lines+markers",
                                   line=dict(color="#c2410c", width=2.5), marker=dict(size=8),
                                   hovertemplate="salida %{x}:00 → %{y:.1f} µg<extra></extra>"))
        fig.add_vline(x=salida.hour + salida.minute / 60, line_dash="dot", line_color="#555",
                      annotation_text="tu hora", annotation_position="top")
        fig.add_annotation(x=mejor_h, y=barrido.loc[mejor_h, "dosis_ug"], text=f"mínimo {mejor_h}:00",
                           showarrow=True, arrowhead=2, ay=-35)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                          xaxis_title="Hora de salida", yaxis_title="Dosis inhalada (µg)")
        fig.update_xaxes(dtick=1, gridcolor="#eee"); fig.update_yaxes(gridcolor="#eee", rangemode="tozero")
        st.plotly_chart(fig, width="stretch")
        ahorro = 100 * (1 - barrido.loc[mejor_h, "dosis_ug"] / r["dosis_ug"])
        if ahorro > 1:
            st.success(f"Si sales a las **{mejor_h}:00** en vez de las {salida:%H:%M}, respiras "
                       f"**{ahorro:.0f} % menos** {contaminante} en este recorrido. "
                       f"La peor hora ese día fue las {peor_h}:00.")
        else:
            st.success("Tu hora de salida ya está entre las de menor exposición de ese día.")


# ----------------------------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------------------------
# --- Origen de los datos: carpeta del proyecto y/o archivos subidos en la app ---
with st.sidebar:
    with st.expander("📂 Cargar datos nuevos", expanded=False):
        subidos = st.file_uploader("Archivos horarios de la RMCAB (.xlsx o .csv), uno por estación",
                                   type=["xlsx", "csv"], accept_multiple_files=True,
                                   help="El nombre del archivo debe decir la estación, p. ej. Kennedy_2024.xlsx")
        solo_subidos = st.checkbox("Usar solo los archivos subidos", value=True,
                                   help="Si se desmarca, los subidos se suman a los de la carpeta "
                                        "(y reemplazan a la misma estación).")

datos_carpeta = {}
try:
    datos_carpeta = cargar(str(carpeta_datos()))
except FileNotFoundError:
    pass

if subidos:
    datos_nuevos, errores = cargar_subidos(tuple((f.name, f.getvalue()) for f in subidos))
    for e in errores:
        st.sidebar.error(f"No pude leer {e}")
    datos_usados = datos_nuevos if solo_subidos else {**datos_carpeta, **datos_nuevos}
    origen = f"{len(datos_nuevos)} archivo(s) subido(s)" + ("" if solo_subidos else " + carpeta")
else:
    datos_usados = datos_carpeta
    origen = "carpeta del proyecto"

if not datos_usados:
    st.error("No hay datos: pon los archivos .xlsx de la RMCAB en la carpeta o súbelos en la barra lateral.")
    st.stop()

HORARIO, MOVIL24 = preparar(datos_usados)
CLAVE_DATOS = (origen, tuple(sorted(datos_usados)), str(HORARIO["PM2.5"].index.min()))

TABLA = tabla_estaciones()

st.title("RespiraRuta")
st.caption("Exposición a la contaminación del aire en Bogotá · datos horarios RMCAB · "
           "ICA según la Resolución 2254 de 2017")

with st.sidebar:
    st.header("Parámetros")
    contaminante = st.radio("Contaminante", ["PM2.5", "PM10"], horizontal=True)
    tipo = st.radio("Valor a mostrar", ["Promedio móvil 24 h (ICA)", "Concentración horaria"],
                    help="El ICA de la norma se define con promedios de 24 horas. "
                         "El valor horario muestra la variación dentro del día.")
    p = st.slider("Exponente p del IDW", 0.5, 4.0, 1.0, 0.5,
                  help="p = 1 tuvo el menor error en la validación cruzada (paso 2).")

    matriz = HORARIO[contaminante]
    fechas = matriz.index
    sugerida = pd.Timestamp("2025-02-14")
    if not (fechas.min() <= sugerida <= fechas.max()):
        sugerida = fechas.min().normalize() + pd.Timedelta(days=1)
    fecha = st.date_input("Fecha", value=sugerida.date(),
                          min_value=fechas.min().date(), max_value=(fechas.max() - pd.Timedelta(hours=1)).date())
    hora = st.slider("Hora", 1, 24, 8,
                     help="La RMCAB marca cada dato al final de la hora: 08:00 es el promedio de 07:00 a 08:00.")
    st.caption(f"Promedio de {hora - 1:02d}:00 a {hora:02d}:00")
    st.divider()
    st.caption(f"**Datos en uso:** {origen} · {len(datos_usados)} estaciones · "
               f"{fechas.min():%d/%m/%Y} a {(fechas.max() - pd.Timedelta(hours=1)):%d/%m/%Y}")

instante = pd.Timestamp(fecha) + pd.Timedelta(hours=hora)
fuente = MOVIL24[contaminante] if tipo.startswith("Promedio") else HORARIO[contaminante]
if instante not in fuente.index:
    st.warning("No hay datos para esa fecha y hora.")
    st.stop()

fila = fuente.loc[instante]
est = TABLA.loc[fila.index]
valores = fila.to_numpy(dtype=float)

tab_mapa, tab_ruta, tab_datos = st.tabs(["🗺️ Mapa de la ciudad", "🚲 Mi ruta", "📋 Datos y validación"])

# ---------------------------------- Mapa ----------------------------------
with tab_mapa:
    col_mapa, col_info = st.columns([3, 1.2])

    with col_mapa:
        m = folium.Map(location=CENTRO_BOGOTA, zoom_start=11, tiles="OpenStreetMap")
        if np.isfinite(valores).sum() >= 3:
            img, limites = imagen_malla(est, valores, contaminante, p)
            folium.raster_layers.ImageOverlay(image=img, bounds=limites, opacity=1,
                                              name="Interpolación IDW").add_to(m)
        for sigla, v in zip(est.index, valores):
            i, nombre_cat, color, _ = categoria_concentracion(v, contaminante)
            texto = (f"<b>{est.loc[sigla, 'nombre']}</b><br>{contaminante}: "
                     + ("sin dato" if np.isnan(v) else f"{v:.1f} µg/m³<br>ICA: {i} — {nombre_cat}"))
            folium.CircleMarker(
                location=(est.loc[sigla, "lat"], est.loc[sigla, "lon"]), radius=9,
                color="#333333", weight=1.5, fill=True, fill_color=color, fill_opacity=0.95,
                tooltip=folium.Tooltip(texto), popup=folium.Popup(texto, max_width=250),
            ).add_to(m)
        st_folium(m, height=620, use_container_width=True, returned_objects=[])

        # Leyenda del ICA
        chips = "".join(
            f'<span style="display:inline-flex;align-items:center;margin-right:14px;font-size:13px">'
            f'<span style="width:14px;height:14px;background:{c[3]};border:1px solid #555;'
            f'border-radius:3px;margin-right:5px"></span>{c[2]} ({c[0]}–{c[1]})</span>'
            for c in CATEGORIAS)
        st.markdown(chips, unsafe_allow_html=True)

    with col_info:
        st.subheader(instante.strftime("%d/%m/%Y %H:%M"))
        validos = fila.dropna()
        if validos.empty:
            st.info("Ninguna estación tiene dato a esta hora.")
        else:
            peor = validos.idxmax()
            mejor = validos.idxmin()
            i_peor, cat_peor, _, rec_peor = categoria_concentracion(validos[peor], contaminante)
            st.metric("Promedio de la red", f"{validos.mean():.1f} µg/m³")
            st.metric(f"Más alta: {TABLA.loc[peor, 'nombre']}", f"{validos[peor]:.1f} µg/m³",
                      f"ICA {i_peor} · {cat_peor}", delta_color="off")
            st.metric(f"Más baja: {TABLA.loc[mejor, 'nombre']}", f"{validos[mejor]:.1f} µg/m³")
            if tipo.startswith("Promedio"):
                sobre = (validos > LIMITE_24H[contaminante]).sum()
                st.metric(f"Estaciones sobre la norma 24 h ({LIMITE_24H[contaminante]} µg/m³)",
                          f"{sobre} de {len(validos)}")
            st.caption(f"**Recomendación (zona más afectada):** {rec_peor}")

    # Perfil del día seleccionado
    st.subheader("Cómo cambió el aire durante el día")
    dia = HORARIO[contaminante].loc[pd.Timestamp(fecha) + pd.Timedelta(hours=1): pd.Timestamp(fecha) + pd.Timedelta(hours=24)]
    if not dia.empty:
        horas = [t.hour if t.hour != 0 else 24 for t in dia.index]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=horas, y=dia.max(axis=1), line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=horas, y=dia.min(axis=1), fill="tonexty", fillcolor="rgba(194,65,12,0.15)",
                                 line=dict(width=0), name="Rango entre estaciones (mín–máx)"))
        fig.add_trace(go.Scatter(x=horas, y=dia.mean(axis=1), line=dict(color="#c2410c", width=2.5),
                                 name="Promedio de la red", mode="lines+markers"))
        fig.add_vline(x=hora, line_dash="dot", line_color="#555")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                          xaxis_title="Hora del día", yaxis_title=f"{contaminante} horario (µg/m³)",
                          legend=dict(orientation="h", y=1.12), plot_bgcolor="white")
        fig.update_xaxes(dtick=2, gridcolor="#eee")
        fig.update_yaxes(gridcolor="#eee", rangemode="tozero")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------- Ruta ----------------------------------
with tab_ruta:
    pestana_ruta()

# ---------------------------------- Datos ----------------------------------
with tab_datos:
    st.subheader(f"Completitud de datos de {contaminante} (criterio del 75 %)")
    comp = reporte_completitud(HORARIO[contaminante])
    comp.insert(0, "Estación", TABLA.loc[comp.index, "nombre"])
    st.dataframe(comp, width="stretch")
    st.caption("Las horas sin dato vienen como 0 en la descarga de la RMCAB y se tratan como faltantes.")
    st.subheader("Validación de la interpolación")
    st.markdown("Validación cruzada *deja una fuera*: se esconde cada estación, se estima con las demás "
                "y se compara con lo que midió. Se puede recalcular con los datos que estén cargados.")
    if st.button(f"▶️ Recalcular validación de {contaminante} con los datos en uso"):
        tabla_viva = validar(HORARIO[contaminante], CLAVE_DATOS, contaminante)
        st.dataframe(tabla_viva.round(3), width="stretch")
        solo = tabla_viva[tabla_viva["p"] > 0]
        st.success(f"Menor error con **{solo['RMSE'].idxmin()}** (RMSE = {solo['RMSE'].min():.2f} µg/m³). "
                   f"Promedio simple: {tabla_viva.iloc[0]['RMSE']:.2f} · estación más cercana: "
                   f"{tabla_viva.iloc[-1]['RMSE']:.2f} µg/m³.")
    st.markdown("**Resultados guardados (datos de la carpeta, script `paso2_validar_idw.py`):**")
    carpeta_res = CARPETA_PROYECTO / "resultados"
    etiqueta = contaminante.replace(".", "")
    tabla_val = carpeta_res / f"comparacion_metodos_{etiqueta}.csv"
    if tabla_val.exists():
        st.dataframe(pd.read_csv(tabla_val, index_col=0), width="stretch")
        c1, c2 = st.columns(2)
        for col, nombre in zip((c1, c2), (f"fig1_rmse_vs_p_{etiqueta}.png", f"fig2_medido_vs_estimado_{etiqueta}.png")):
            if (carpeta_res / nombre).exists():
                col.image(str(carpeta_res / nombre))
    else:
        st.info(f"Corre primero:  py paso2_validar_idw.py --contaminante {contaminante}")
