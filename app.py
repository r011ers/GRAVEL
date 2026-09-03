# Punto de entrada de Streamlit
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

for _mod in [k for k in sys.modules if k == "core" or k.startswith("core.")]:
    del sys.modules[_mod]

from core.db import DartsDatabase
from core.features import FeatureExtractor
from core.identities import aliases_of, canonical_player, selectable_players
from core.ml_model import AdvancedDartsModel

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "darts_public.sqlite"


def _core_fingerprint() -> str:
    chunks = []
    core_dir = ROOT / "core"
    if core_dir.is_dir():
        for path in sorted(core_dir.glob("*.py")):
            stat = path.stat()
            chunks.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(chunks)


def _read_snapshot_info(db) -> dict:
    reader = getattr(db, "snapshot_info", None)
    if callable(reader):
        return reader()
    with db.get_connection() as conn:
        n_matches = int(
            conn.execute("SELECT COUNT(DISTINCT match_id) FROM match_stats").fetchone()[0]
        )
        raw_dates = [
            row[0]
            for row in conn.execute(
                "SELECT fixture_date FROM match_stats WHERE fixture_date IS NOT NULL"
            )
        ]
    if raw_dates:
        parsed = pd.to_datetime(
            pd.Series(raw_dates, dtype="object"),
            utc=True,
            format="mixed",
            errors="coerce",
        )
        max_date = parsed.max() if not parsed.isna().all() else pd.NaT
    else:
        max_date = pd.NaT
    return {"n_matches": n_matches, "max_date": max_date}


@st.cache_resource
def load_runtime(code_fp: str):
    db = DartsDatabase(db_path=str(DB_PATH))
    with db.get_connection() as conn:
        raw_players = pd.read_sql_query(
            "SELECT DISTINCT player FROM match_stats ORDER BY player",
            conn,
        )["player"].tolist()
    players = selectable_players(raw_players)
    info = _read_snapshot_info(db)
    return db, FeatureExtractor(db), AdvancedDartsModel(db), players, raw_players, info


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _snapshot_cutoff(info: dict) -> str:
    max_date = info.get("max_date")
    if max_date is None or pd.isna(max_date):
        return "fecha no disponible"
    return pd.Timestamp(max_date).strftime("%Y-%m-%d")


def main() -> None:
    st.set_page_config(page_title="GRAVEL", layout="centered")
    st.title("GRAVEL")
    st.caption(
        "Probabilidades informativas de dardos. Snapshot histórico, no datos en vivo. "
        "No es una herramienta de apuestas."
    )

    db, fe, model, players, raw_players, info = load_runtime(_core_fingerprint())

    st.caption(
        f"Datos hasta {_snapshot_cutoff(info)} · "
        f"{info['n_matches']} partidos · "
        f"{len(players)} jugadores. "
        "Este repositorio público no incluye el motor privado completo."
    )

    if model.v8_model is None:
        st.error(model.load_error or "No se pudo cargar models/darts_v8.pkl")
        return

    if len(players) < 2:
        st.error("No hay suficientes jugadores en data/darts_public.sqlite")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        player_a = st.selectbox("Jugador A", players, index=0)
    with col_b:
        options_b = [p for p in players if p != player_a] or players
        default_b = 0
        player_b = st.selectbox("Jugador B", options_b, index=default_b)

    first_throw_choice = st.radio(
        "¿Quién saca primero?",
        options=[True, False],
        format_func=lambda a_throws: f"{player_a} (Jugador A)" if a_throws else f"{player_b} (Jugador B)",
        horizontal=True,
    )

    if not st.button("Calcular probabilidad", type="primary"):
        return

    canon_a = canonical_player(player_a, raw_players)
    canon_b = canonical_player(player_b, raw_players)
    if canon_a == canon_b:
        st.warning("Elige dos jugadores distintos.")
        return

    feat_a = fe.get_player_features(player_a)
    feat_b = fe.get_player_features(player_b)

    missing = []
    if feat_a is None:
        missing.append(player_a)
    if feat_b is None:
        missing.append(player_b)
    if missing:
        st.warning(
            "No hay suficientes partidos históricos (mínimo 5) para: "
            + ", ".join(missing)
        )
        return

    feat_a = fe.get_player_features(player_a, opp_avg=feat_b.avg_score)
    feat_b = fe.get_player_features(player_b, opp_avg=feat_a.avg_score)
    h2h = fe.get_h2h_summary(player_a, player_b)
    pred = model.predict_match(
        feat_a,
        feat_b,
        h2h_a_winrate=h2h["win_rate_a"],
        first_throw_a=bool(first_throw_choice),
    )

    res_a, res_b = st.columns(2)
    for col, feat, prob in (
        (res_a, feat_a, pred.prob_a),
        (res_b, feat_b, pred.prob_b),
    ):
        with col:
            st.subheader(feat.player_name)
            st.metric("Probabilidad", _pct(prob))
            st.write(f"ELO: {feat.elo_rating:.0f}")
            st.write(f"Win rate: {_pct(feat.win_rate)}")
            st.write(f"Average: {feat.avg_score:.2f}")
            st.write(f"Checkout: {feat.avg_checkout:.1f}%")
            st.write(f"180s: {feat.avg_180s:.2f}")
            st.write(f"Partidos: {feat.n_matches}")

    st.subheader("H2H")
    n_h2h = int(h2h["n_matches"])
    if n_h2h == 0:
        st.write("Sin enfrentamientos directos registrados")
    else:
        aliases_a = aliases_of(canon_a, raw_players)
        aliases_b = aliases_of(canon_b, raw_players)
        df_h2h = db.get_h2h(canon_a, canon_b, aliases_a=aliases_a, aliases_b=aliases_b)
        if not df_h2h.empty and "match_id" in df_h2h.columns:
            df_h2h = df_h2h.drop_duplicates(subset=["match_id"], keep="first")
        wins_a = int(df_h2h["result"].sum()) if not df_h2h.empty else 0
        wins_b = n_h2h - wins_a
        label = "partido" if n_h2h == 1 else "partidos"
        st.write(
            f"Enfrentamientos directos: {n_h2h} {label} — "
            f"{player_a} {wins_a}, {player_b} {wins_b}"
        )
        st.write(
            f"Average en sus enfrentamientos: {h2h['avg_a']:.2f} vs {h2h['avg_b']:.2f}"
        )


if __name__ == "__main__":
    main()
