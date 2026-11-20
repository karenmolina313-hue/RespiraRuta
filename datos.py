"""
datos.py
Carga y limpieza de los archivos horarios descargados de la RMCAB.

Formato esperado de cada archivo (uno por estación, p. ej. 'Kennedy_2025.xlsx'):
    Fecha | PM10 | PM2.5 | Vel Viento | Dir Viento | NO | NO2 | NOX | SO2 | CO | OZONO | ...
    01-01-2025 01:00 | 99.6 | 57.1 | ...

Reglas de limpieza (supuestos que hay que explicar en la sustentación):
  1. La hora "24:00" se convierte en las 00:00 del día siguiente.
     La RMCAB marca cada dato al FINAL de la hora: "01:00" es el promedio de 00:00 a 01:00.
  2. En la descarga, las horas sin dato vienen con 0. Una concentración horaria exactamente
     igual a 0 no es físicamente creíble en Bogotá, así que 0 se trata como DATO FALTANTE.
     (Prueba: estaciones que no miden un contaminante traen 0 en las 8760 horas del año.)
  3. Valores negativos también se tratan como faltantes.
  4. Un promedio (diario, anual) solo es válido si tiene al menos 75 % de datos válidos
     (criterio de representatividad usado en los protocolos de calidad del aire).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from estaciones import buscar_estacion

# Columnas con nombres distintos según la estación -> nombre estándar
_RENOMBRAR = {
    "Vel Viento 10M": "Vel Viento",
    "Dir Viento 10M": "Dir Viento",
    "eBC": "BC",
    "Presion Ba": "Presion Baro",
    "Presión Ba": "Presion Baro",
}

CONTAMINANTES = ["PM10", "PM2.5", "NO2", "OZONO", "CO", "SO2"]
CRITERIO_COMPLETITUD = 0.75


def leer_fecha(serie_texto):
    """Convierte 'dd-mm-aaaa HH:MM' a fecha, manejando la hora '24:00'."""
    texto = serie_texto.astype(str).str.strip()
    es_24 = texto.str.endswith("24:00")
    texto = texto.where(~es_24, texto.str.replace("24:00", "00:00", regex=False))
    fecha = pd.to_datetime(texto, format="%d-%m-%Y %H:%M")
    fecha = fecha.where(~es_24, fecha + pd.Timedelta(days=1))
    return fecha


def leer_archivo(ruta, nombre=None):
    """
    Lee un archivo de una estación y devuelve un DataFrame limpio con índice horario.
    ruta   : ruta del archivo, o el contenido del archivo (por ejemplo, uno subido en la app)
    nombre : nombre del archivo (necesario si 'ruta' es contenido y no una ruta)
    """
    nombre = nombre or Path(ruta).name
    if nombre.lower().endswith(".csv"):
        df = pd.read_csv(ruta, sep=None, engine="python")
    else:
        df = pd.read_excel(ruta)
    df = df.rename(columns=_RENOMBRAR)
    df["Fecha"] = leer_fecha(df["Fecha"])
    df = df.set_index("Fecha").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Regla 2 y 3: 0 o negativo en contaminantes = faltante
    for c in CONTAMINANTES:
        if c in df:
            df.loc[df[c] <= 0, c] = np.nan
    # El viento: si TODA la serie es 0 la estación no lo mide
    for c in ["Vel Viento", "Dir Viento"]:
        if c in df and (df[c].fillna(0) == 0).all():
            df[c] = np.nan
    return df


def cargar_carpeta(carpeta, patron="*.xlsx"):
    """
    Lee todos los archivos de la carpeta.
    Devuelve un diccionario {sigla_estacion: DataFrame}.
    """
    datos = {}
    for ruta in sorted(Path(carpeta).glob(patron)):
        if ruta.name.startswith("~$"):        # archivos temporales de Excel abiertos
            continue
        sigla = buscar_estacion(ruta.name)
        datos[sigla] = leer_archivo(ruta)
    if not datos:
        raise FileNotFoundError(f"No encontré archivos {patron} en {carpeta}")
    return datos


def cargar_archivos(archivos):
    """
    Lee una lista de archivos subidos: [(nombre, contenido_en_bytes), ...].
    Devuelve ({sigla: DataFrame}, [mensajes de error]).
    """
    import io
    datos, errores = {}, []
    for nombre, contenido in archivos:
        try:
            sigla = buscar_estacion(nombre)
            datos[sigla] = leer_archivo(io.BytesIO(contenido), nombre=nombre)
        except Exception as e:          # se informa en la app y se siguen leyendo los demás
            errores.append(f"{nombre}: {e}")
    return datos, errores


def matriz_contaminante(datos, contaminante="PM2.5"):
    """
    Tabla con una fila por hora y una columna por estación, para un contaminante.
    Las estaciones que no miden el contaminante quedan por fuera.
    """
    columnas = {s: df[contaminante] for s, df in datos.items()
                if contaminante in df and df[contaminante].notna().any()}
    return pd.DataFrame(columnas).sort_index()


def reporte_completitud(matriz):
    """Porcentaje de horas con dato válido por estación y si cumple el 75 %."""
    pct = matriz.notna().mean() * 100
    rep = pd.DataFrame({
        "horas_validas": matriz.notna().sum(),
        "porcentaje_valido": pct.round(1),
        "cumple_75": pct >= CRITERIO_COMPLETITUD * 100,
    })
    return rep.sort_values("porcentaje_valido", ascending=False)


def promedio_valido(serie, frecuencia="D"):
    """
    Promedio por periodo (D = diario, ME = mensual, YE = anual) aplicando el criterio del 75 %:
    si el periodo tiene menos del 75 % de horas válidas, el promedio queda como faltante.
    """
    # Se resta 1 hora para que el dato de las 24:00 (00:00 del día siguiente) quede en su día
    s = serie.copy()
    s.index = s.index - pd.Timedelta(hours=1)
    grupos = s.resample(frecuencia)
    media = grupos.mean()
    validos = grupos.count()
    posibles = grupos.size()
    return media.where(validos >= CRITERIO_COMPLETITUD * posibles)
