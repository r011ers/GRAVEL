# GRAVEL

Calculadora de probabilidades para partidos de dardos de la **MODUS Super Series**, basada en un modelo propio de machine learning entrenado sobre cerca de 19.700 partidos históricos (39.397 registros en la base de datos: dos filas por partido, una por jugador).

> ⚠️ **Esto es una herramienta informativa, no de apuestas.** No calcula cuotas, valor esperado (EV) ni recomendaciones de stake. Su único objetivo es estimar la probabilidad de victoria de cada jugador a partir de su rendimiento histórico en la MODUS SUPER SERIES.

🔗 **App en vivo:** [gravel.streamlit.app](https://gravel.streamlit.app)

---

## Qué hace

Eliges dos jugadores y quién saca primero, y la app devuelve:
- % de probabilidad de victoria para cada uno
- Comparativa de ELO, win rate, average, checkout y 180s
- Historial de enfrentamientos directos (H2H) entre ambos

---

## Cómo se calcula la probabilidad

El cálculo combina varias capas, no es un único número aislado:

**1. Features por jugador.**
Para cada jugador se calculan, a partir de su historial: rating y ranking ELO, forma reciente (con más peso a los últimos partidos que a los antiguos), average de puntuación, % de checkout, cantidad de 180s/140s/100s, racha actual y tendencia (si está mejorando o empeorando en sus últimas actuaciones).

**2. Modelo de predicción.**
Esas features de ambos jugadores se pasan a un modelo (XGBoost) entrenado sobre el histórico completo, que combina la probabilidad base derivada del ELO con el resto de variables para producir una estimación más ajustada que el ELO por sí solo.

**3. Calibración.**
La salida del modelo pasa por una capa de calibración adicional (un segundo modelo de regresión logística) para que el porcentaje mostrado refleje probabilidades reales y no esté sistemáticamente sobreconfiado o infraconfiado — es decir, que de los partidos donde el modelo dice "65%", ese jugador gane realmente alrededor del 65% de las veces.

**4. Ajuste por orden de saque.**
Quién tira primero en cada leg tiene un impacto estadísticamente significativo en el resultado (analizado sobre el histórico completo de la base de datos). Por eso la app pide explícitamente quién saca primero y ajusta la probabilidad en consecuencia — no es una casilla decorativa.

**5. Enfrentamientos directos (H2H).**
Se muestra el récord histórico entre ambos jugadores cuando existe, como contexto adicional. Con muestras muy pequeñas (1-2 partidos), este dato tiene poco valor estadístico por sí solo y debe interpretarse con cautela.

---

## El formato: MODUS Super Series (Bo7)

Todo el modelo está entrenado y pensado específicamente para la **MODUS Super Series**, un circuito semanal de dardos (no PDC Tour) transmitido por Pluto TV, con el siguiente formato:

- Cada semana compiten 12 jugadores repartidos en 3 grupos.
- **Grupo A** (6 jugadores): liga round-robin de lunes a miércoles. El primer clasificado pasa directo a la final del sábado.
- **Grupo B** (jueves y viernes, noche): los 3 primeros clasifican a la final.
- **Grupo C** (jueves y viernes, tarde): los 2 primeros clasifican a la final.
- **Final** el sábado por la noche entre los clasificados de los tres grupos.

Cada partido se juega **al mejor de 7 legs (Bo7)** a 501, con el local sacando primero o en (muy pocas veces) semifinales a veces se hace formato de "cork" (tirar al bull) para decidir quién saca primero en la primera leg, alternando el saque leg a leg. Gana quien llegue primero a 4 legs.

La base de datos y el modelo de GRAVEL están filtrados específicamente a partidos que cumplen este formato exacto — no incluye partidos de otros torneos, formatos Bo5, majors de PDC, ni ligas de sets.

---

## Limitaciones importantes (léelas antes de interpretar cualquier probabilidad)

**1. La base de datos no se actualiza en tiempo real.**
`data/darts_public.sqlite` es una fotografía (snapshot) exportada manualmente desde la base de datos completa del proyecto en una fecha concreta. No se sincroniza automáticamente con los partidos que se juegan después de esa exportación. Un jugador puede haber jugado partidos recientes que todavía no están reflejados en sus estadísticas mostradas aquí.

**2. Partidos sin nada en juego ("dead rubbers") distorsionan la interpretación.**
En el formato round-robin de los Grupos A, B y C, es habitual que un jugador ya haya asegurado su clasificación (o ya esté matemáticamente eliminado) antes de jugar todos sus partidos de grupo. En esos partidos, el nivel real mostrado en pista puede no reflejar su nivel habitual — el modelo no sabe, a partir de los datos históricos, si un partido concreto tuvo esta característica, así que una probabilidad basada en su forma general puede no aplicar bien a ese partido específico.

**3. El modelo no capta el estado físico o de forma del momento.**
Lesiones, fatiga acumulada (el calendario de la MSS puede exigir jugar varios días seguidos), cambios personales, o una racha de mala/buena forma muy reciente que aún no se refleja suficientemente en las métricas históricas, no están representados en el modelo. Las probabilidades se basan en patrones históricos, no en el estado del jugador el día del partido.

**4. Jugadores con poco historial dan predicciones menos fiables.**
Si un jugador tiene muy pocos partidos registrados (la app avisa por debajo de 5), la estimación de sus features (ELO, forma, average) está basada en muy poca información y es inherentemente menos estable que la de un jugador con un historial amplio. Trata estas probabilidades con más cautela que las de jugadores consolidados en la base de datos.

---

## Stack técnico

`Python` · `Streamlit` · `XGBoost` · `scikit-learn` · `pandas` · `SQLite`

## Licencia

Ver [LICENSE](./LICENSE).
