# RespiraRuta

Exposición personal a la contaminación del aire en Bogotá según ruta, hora y modo de transporte.
Proyecto integrador de Control y Simulación Atmosférica.

## Estado actual: Paso 4 (mapa interactivo + Mi ruta)

Para abrir la app:  `py -m streamlit run app.py`  (se abre en el navegador).

| Archivo | Qué hace |
|---|---|
| `estaciones.py` | Coordenadas oficiales de las 20 estaciones RMCAB (SDA) y reconocimiento del nombre del archivo |
| `datos.py` | Lee los Excel horarios de RMCAB, maneja la hora 24:00, trata 0 y negativos como faltantes, criterio del 75 % |
| `idw.py` | Interpolación por distancia inversa ponderada (IDW) con distancia de haversine |
| `validacion.py` | Validación cruzada "deja una fuera": RMSE, MAE, sesgo y R², comparando contra métodos simples |
| `test_idw.py` | 24 pruebas automáticas de los cálculos |
| `ica.py` | Índice de Calidad del Aire y límites de la Res. 2254 de 2017 |
| `app.py` | Aplicación Streamlit: mapa IDW por fecha y hora, estaciones por ICA, perfil del día |
| `ruta.py` | Exposición en la ruta: tramos de 100 m, IDW por hora, dosis D = Σ C·F·IR·Δt, comparador de modos y de horas |
| `paso2_validar_idw.py` | Corre todo y guarda tablas y figuras en `resultados/` |

## Cómo correrlo (Windows)

En la terminal de VS Code, dentro de la carpeta `RespiraRuta`:

```
pip install -r requirements.txt
python -m pytest -v
python paso2_validar_idw.py
python paso2_validar_idw.py --contaminante PM10
```

Los archivos de datos se buscan en la carpeta `datos/` del proyecto; si no existe, en la carpeta
que contiene a `RespiraRuta` (la de "Reporte Estaciones 2025"). Para agregar una estación basta con
poner su archivo, por ejemplo `Kennedy_2025.xlsx`, con el mismo formato de descarga.

## Supuestos y limitaciones (hasta ahora)

- Tasas de inhalación: U.S. EPA (2011), Exposure Factors Handbook, cap. 6, tabla 6-2 (adultos 21–31 años).
- Velocidades típicas por modo y factor de microambiente = 1 (dentro del vehículo igual que afuera): supuestos editables.
- La ruta dibujada es una línea entre los puntos que marca el usuario; no se ajusta automáticamente a las calles.

- Las horas sin dato vienen como 0 en la descarga: 0 y negativos se tratan como faltantes.
- La latitud de Las Ferias no aparece en la tabla de la SDA (se usó 4°41'26.52"N); hay que verificarla.
- Falta la estación Bosa.
- El mapa no se dibuja a más de 6 km de la estación más cercana (no se extrapola).
- El ICA oficial usa promedios de 24 h; el modo horario es solo indicativo.
- IDW no tiene en cuenta vías, fuentes, topografía ni viento: es una interpolación puramente geométrica.
