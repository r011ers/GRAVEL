# GRAVEL

Calculadora pública de probabilidades para partidos de dardos de la **MODUS Super Series** (Bo7, 1 vs 1).

Este repositorio es una **versión pública / portfolio**. Contiene una aplicación Streamlit de inferencia sobre un **snapshot** de datos y un modelo ya entrenado. **No** incluye el motor privado completo (scrape, entrenamiento, backtest ni la base de datos interna).

> Esto es una herramienta informativa, no de apuestas. No calcula cuotas, valor esperado (EV) ni recomendaciones de stake.

🔗 App en vivo (si está desplegada): [gravel.streamlit.app](https://gravel.streamlit.app)

---

## Qué hace

Eliges dos jugadores y quién saca primero. La app devuelve:

- % de probabilidad de victoria para cada uno
- Comparativa de ELO, win rate, average, checkout, 180s y número de partidos
- Historial de enfrentamientos directos (H2H) entre ambos

Los nombres se muestran con una **identidad canónica** (aliases evidentes de scrape se agrupan; casos dudosos se mantienen separados).

---

## Snapshot de datos

`data/darts_public.sqlite` es una fotografía exportada manualmente. **No se actualiza en tiempo real.**

La interfaz muestra la fecha de corte y el número de partidos del snapshot. Un jugador puede haber competido después de esa fecha y esas actuaciones no estarán aquí.

El snapshot público está filtrado a singles MSS Best of 7. No incluye dobles, otros torneos, Bo5 ni majors PDC.

---

## Cómo se calcula la probabilidad

1. **Features por jugador** a partir del historial del snapshot (ELO almacenado, forma, average, checkout, etc.).
2. **Modelo** (XGBoost calibrado) entrenado fuera de este repositorio.
3. **Calibración adicional** y un ajuste empírico por orden de saque.
4. **H2H** como contexto y como una de las features del modelo.

Las probabilidades se basan en patrones históricos del snapshot, no en el estado del jugador el día del partido.

### Limitaciones

- El snapshot no es live.
- Partidos de grupo ya decididos (“dead rubbers”) no están etiquetados.
- Jugadores con pocos partidos (<5 no se predicen) dan estimaciones menos estables.
- Hay 10 filas (5 partidos) con `fixture_date` nulo. **No están vacías**: tienen jugadores, average y resultado. Cuentan para estadísticas de carrera, no para recencia. No se inventa la fecha ni se borran del snapshot.
- Un registro legado (`Richard Rowland Dont USE`) se oculta del selector; no se borra del snapshot.
- El ELO mostrado es el almacenado en el snapshot para el alias con más partidos. No se recalcula al fusionar grafías.
- El fichero `models/darts_v8.pkl.sha256` fija el hash del modelo; si el pickle cambia, la app no lo carga.

---

## Cómo ejecutar

Requisitos: Python 3.11 (ver `runtime.txt`).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Archivos necesarios en el repo:

- `data/darts_public.sqlite`
- `models/darts_v8.pkl`

### Reexportar el snapshot (opcional)

Si tienes una base origen local, sin hardcodear rutas:

```bash
set GRAVEL_SOURCE_DB=C:\ruta\a\origen.sqlite
python scripts/export_public_db.py
```

---

## Stack

`Python` · `Streamlit` · `XGBoost` · `scikit-learn` · `pandas` · `SQLite`

## Tests

```bash
python -m unittest discover -s tests -v
```

Incluyen identidad/aliases, parseo de fechas, checkout, snapshot de solo lectura, hash del modelo, predicción de referencia (Luke Littler vs Neil Duff) y carga e2e de la app Streamlit.

Auditoría opcional de dependencias (no forma parte del runtime):

```bash
python -m pip audit -r requirements.txt
```

## Licencia

[MIT](./LICENSE)
