"""
estaciones.py
Ubicación de las estaciones de la RMCAB (Red de Monitoreo de Calidad del Aire de Bogotá).

Fuente: Secretaría Distrital de Ambiente, "Características generales de las estaciones de la
RMCAB" (https://ambientebogota.gov.co/estaciones-rmcab). Las coordenadas vienen en grados,
minutos y segundos y aquí se convierten a grados decimales:
    decimal = grados + minutos/60 + segundos/3600   (la longitud es negativa porque es Oeste)

OJO: la latitud de Las Ferias no aparece en esa tabla; se usó 4°41'26.52"N.
Verifíquenla antes de la entrega.
"""

import unicodedata

import pandas as pd


def dms_a_decimal(grados, minutos, segundos):
    """Convierte grados-minutos-segundos a grados decimales."""
    return grados + minutos / 60 + segundos / 3600


# clave: nombre normalizado (sin tildes, minúsculas, espacios) tal como sale del nombre del archivo
_TABLA = [
    # clave,                      nombre,                       sigla,  lat (g,m,s),       lon (g,m,s),        tipo de estación
    ("guaymaral",                 "Guaymaral",                  "GYR",  (4, 47, 1.5),      (74, 2, 38.9),      "De fondo"),
    ("usaquen",                   "Usaquén",                    "USQ",  (4, 42, 37.26),    (74, 1, 49.50),     "De fondo"),
    ("suba",                      "Suba",                       "SUB",  (4, 45, 40.49),    (74, 5, 36.46),     "De fondo"),
    ("bolivia",                   "Bolivia",                    "BOL",  (4, 44, 8.9),      (74, 7, 33.2),      "De fondo"),
    ("las ferias",                "Las Ferias",                 "LFR",  (4, 41, 26.52),    (74, 4, 56.94),     "De tráfico"),
    ("centro de alto rendimiento", "Centro de Alto Rendimiento", "CDAR", (4, 39, 30.48),    (74, 5, 2.28),      "De tráfico"),
    ("movil 7ma",                 "Móvil 7ma",                  "MOV7", (4, 38, 42.7),     (74, 3, 41.6),      "De tráfico"),
    ("minambiente",               "MinAmbiente",                "MAM",  (4, 37, 31.75),    (74, 4, 1.13),      "De tráfico"),
    ("fontibon",                  "Fontibón",                   "FTB",  (4, 40, 41.67),    (74, 8, 37.75),     "De tráfico"),
    ("puente aranda",             "Puente Aranda",              "PTE",  (4, 37, 54.36),    (74, 7, 2.94),      "Industrial"),
    ("kennedy",                   "Kennedy",                    "KEN",  (4, 37, 30.18),    (74, 9, 40.80),     "De fondo"),
    ("carvajal sevillana",        "Carvajal-Sevillana",         "CSE",  (4, 35, 45.0),     (74, 8, 54.6),      "Tráfico/industrial"),
    ("tunal",                     "Tunal",                      "TUN",  (4, 34, 34.41),    (74, 7, 51.44),     "De fondo"),
    ("san cristobal",             "San Cristóbal",              "SCR",  (4, 34, 21.19),    (74, 5, 1.73),      "De fondo"),
    ("jazmin",                    "El Jazmín",                  "JAZ",  (4, 36, 30.6),     (74, 6, 53.8),      "Residencial"),
    ("usme",                      "Usme",                       "USM",  (4, 31, 55.4),     (74, 7, 1.7),       "Residencial"),
    ("bosa",                      "Bosa",                       "BOS",  (4, 36, 20.2),     (74, 12, 14.6),     "De fondo"),
    ("ciudad bolivar",            "Ciudad Bolívar",             "CBV",  (4, 34, 40.1),     (74, 9, 58.6),      "Residencial"),
    ("colina",                    "Colina",                     "COL",  (4, 44, 13.9),     (74, 4, 10.1),      "De fondo"),
    ("movil fontibon",            "Móvil Fontibón",             "MOV2", (4, 40, 4.8),      (74, 8, 54.6),      "Tráfico/industrial"),
]

# Otros nombres con los que puede venir un archivo
_ALIAS = {
    "el jazmin": "jazmin",
    "carvajal": "carvajal sevillana",
    "sevillana": "carvajal sevillana",
    "cdar": "centro de alto rendimiento",
    "min ambiente": "minambiente",
    "movil septima": "movil 7ma",
}


def normalizar(texto):
    """'Ciudad_Bolívar' -> 'ciudad bolivar' (sin tildes, minúsculas, sin guiones)."""
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    texto = texto.lower().replace("_", " ").replace("-", " ")
    return " ".join(texto.split())


def tabla_estaciones():
    """DataFrame con clave, nombre, sigla, lat, lon y tipo de cada estación."""
    filas = []
    for clave, nombre, sigla, lat, lon, tipo in _TABLA:
        filas.append({
            "clave": clave,
            "nombre": nombre,
            "sigla": sigla,
            "lat": round(dms_a_decimal(*lat), 6),
            "lon": round(-dms_a_decimal(*lon), 6),   # Oeste -> negativo
            "tipo": tipo,
        })
    return pd.DataFrame(filas).set_index("sigla")


def buscar_estacion(nombre_archivo):
    """Devuelve la sigla de la estación a partir del nombre del archivo, p. ej. 'Las_Ferias_2025.xlsx' -> 'LFR'."""
    base = normalizar(nombre_archivo.rsplit(".", 1)[0])
    # quitar el año y palabras como 'horario' del final
    palabras = [p for p in base.split() if not p.isdigit() and p not in ("horario", "diario", "hora")]
    clave = " ".join(palabras)
    clave = _ALIAS.get(clave, clave)
    tabla = tabla_estaciones()
    coincide = tabla[tabla["clave"] == clave]
    if coincide.empty:
        raise ValueError(f"No reconozco la estación del archivo '{nombre_archivo}' (clave '{clave}'). "
                         f"Agrégala a estaciones.py o cambia el nombre del archivo.")
    return coincide.index[0]
