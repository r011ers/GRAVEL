"""
╔══════════════════════════════════════════════════════════════╗
║         GRAVEL v12 — features.py                            ║
║         Features + live form Last3 (65/35 blend)             ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

from .db import DartsDatabase

@dataclass
class PlayerFeatures:
    player_name: str
    n_matches: int
    
    # Historico
    avg_score: float
    avg_checkout: float
    avg_100s: float
    avg_140s: float
    avg_180s: float
    win_rate: float
    legs_win_rate: float
    
    # Forma Reciente (últimos X partidos)
    form_avg: float
    form_checkout: float
    form_100s: float
    form_140s: float
    form_180s: float
    form_win_rate: float
    form_legs_wr: float
    streak: str  # e.g., 'WWLW'
    
    # Situacionales
    home_wr: float
    away_wr: float
    days_since_last: Optional[int]
    
    # Contexto de Jornada (Novedad v8-v11)
    session_position: int     # 1, 2, 3... partido en el mismo día
    matches_last_7_days: int  # Densidad de juego reciente
    
    # Fatigue Features (v11)
    legs_played_today: int    # Piernas acumuladas hoy
    rest_minutes: float       # Minutos desde el último partido
    hour_of_day: float        # Hora decimal (14.5 = 14:30)
    
    # ELO Systems (v11 Pro)
    elo_rating: float
    elo_rank: Any             # Posición en el ranking (int o 'N/A')
    is_debut: bool

    # NUEVAS FEATURES v8 AUDIT
    form_avg_ewm: float
    form_checkout_ewm: float
    form_wr_ewm: float
    avg_slope: float          # Tendencia (mejora o caída)
    pressure_co: float        # Checkout en legs decisivas
    pressure_delta: float     # Diferencia vs normal
    vs_similar_wr: float      # Win rate contra nivel similar
    avg_std: float            # Desviación estándar del score_avg
    elo_momentum: float = 0.0 # [Novedad v13] ELO Momentum point-in-time




class FeatureExtractor:
    def __init__(self, db: DartsDatabase):
        self.db = db
        self._name_cache = {}
        
    def _calculate_streak(self, series: pd.Series, limit=5) -> str:
        results = series.tail(limit).astype(int).tolist()
        return "".join("W" if r == 1 else "L" for r in results)

    @staticmethod
    def _name_tokens(name: str) -> frozenset:
        return frozenset(t for t in str(name).lower().replace("-", " ").split() if t)

    @staticmethod
    def _name_match_ratio(query: str, candidate: str) -> float:
        """Ratio de nombres; 1.0 si mismos tokens en distinto orden (Lesic Zvonimir ≈ Zvonimir Lesic)."""
        import difflib
        q = str(query).strip()
        c = str(candidate).strip()
        if not q or not c:
            return 0.0
        q_tok, c_tok = FeatureExtractor._name_tokens(q), FeatureExtractor._name_tokens(c)
        if q_tok and q_tok == c_tok:
            return 1.0
        q_clean = q.replace(" ", "").lower()
        c_clean = c.replace(" ", "").lower()
        ratio = difflib.SequenceMatcher(None, q_clean, c_clean).ratio()
        # orden invertido: "A B" vs "B A"
        q_parts, c_parts = q.split(), c.split()
        if len(q_parts) >= 2 and len(c_parts) >= 2:
            q_rev = "".join(q_parts[::-1]).lower()
            c_fwd = c_clean
            ratio = max(ratio, difflib.SequenceMatcher(None, q_rev, c_fwd).ratio())
            ratio = max(
                ratio,
                difflib.SequenceMatcher(
                    None, "".join(q_parts).lower(), "".join(c_parts[::-1]).lower()
                ).ratio(),
            )
        if q_clean in c_clean or c_clean in q_clean:
            ratio = max(ratio, 0.85)
        return ratio

    def _resolve_player_name(self, player_name: str) -> str:
        """Resuelve alias / orden invertido al nombre canónico con más partidos en BD."""
        if player_name in self._name_cache:
            return self._name_cache[player_name]

        conn = self.db.get_connection()
        df_names = pd.read_sql_query(
            "SELECT player, COUNT(*) AS n FROM match_stats GROUP BY player",
            conn,
        )
        conn.close()

        best_name = player_name
        best_ratio = 0.0
        best_n = -1
        q_tok = self._name_tokens(player_name)

        for _, row in df_names.iterrows():
            p_db = row["player"]
            n = int(row["n"])
            ratio = self._name_match_ratio(player_name, p_db)
            # Preferir el alias con más historial a igualdad de match
            if ratio > best_ratio + 1e-9 or (abs(ratio - best_ratio) < 1e-9 and n > best_n):
                best_ratio = ratio
                best_name = p_db
                best_n = n

        if best_ratio >= 0.8:
            # Si hay varios con mismos tokens, canónico = el de más filas
            if q_tok:
                same = df_names[df_names["player"].map(lambda p: self._name_tokens(p) == q_tok)]
                if not same.empty:
                    best_name = same.sort_values("n", ascending=False).iloc[0]["player"]
            self._name_cache[player_name] = best_name
            return best_name

        self._name_cache[player_name] = player_name
        return player_name

    def _load_player_history(self, player_name: str) -> pd.DataFrame:
        """Carga historial uniendo todos los alias con los mismos tokens de nombre."""
        resolved = self._resolve_player_name(player_name)
        tok = self._name_tokens(resolved) or self._name_tokens(player_name)

        df = self.db.load_player_data(resolved)
        if not tok:
            return df

        # Unir alias invertidos (misma persona, otro orden en scrape)
        try:
            conn = self.db.get_connection()
            all_names = pd.read_sql_query("SELECT DISTINCT player FROM match_stats", conn)
            conn.close()
            aliases = [
                p for p in all_names["player"].tolist()
                if p != resolved and self._name_tokens(p) == tok
            ]
            if aliases:
                frames = [df] if not df.empty else []
                for a in aliases:
                    da = self.db.load_player_data(a)
                    if not da.empty:
                        frames.append(da)
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    if "fixture_date" in df.columns:
                        df = df.sort_values(["fixture_date", "match_id"]).reset_index(drop=True)
        except Exception:
            pass
        return df

    def get_player_features(self, player_name: str, form_window: int = 3, before_date: Optional[pd.Timestamp] = None, opp_avg: Optional[float] = None) -> Optional[PlayerFeatures]:
        # Live form: últimos 3 partidos, +10% peso vs blend histórico (0.65/0.35)
        LIVE_FORM_W = 0.65
        HIST_W = 0.35

        player_name = self._resolve_player_name(player_name)
        df = self._load_player_history(player_name)
        if df.empty or len(df) < 5:
            return None
            
        # --- NORMALIZACIÓN DE ZONAS HORARIAS (Fix v11.1) ---
        # Si los datos vienen de SQLite sin zona horaria, los localizamos a UTC
        if not df.empty and df["fixture_date"].dt.tz is None:
            df["fixture_date"] = df["fixture_date"].dt.tz_localize("UTC")
        
        # Si el filtro before_date es naive, lo localizamos a UTC para que coincida
        if before_date is not None and before_date.tzinfo is None:
            before_date = before_date.tz_localize("UTC")
            
        # Filtrado point-in-time para backtesting
        if before_date is not None:
            df = df[df["fixture_date"] < before_date]
            if df.empty or len(df) < 5:
                return None
            
        n = len(df)
        
        # Histórico
        avg_score = df["score_avg"].mean()
        avg_checkout = df["checkout_pct"].mean()
        avg_100s = df["turns_100"].mean()
        avg_140s = df["turns_140"].mean()
        avg_180s = df["turns_180"].mean()
        win_rate = df["result"].mean()
        
        total_legs_won = df["legs_won"].sum()
        total_legs_played = df["total_legs"].sum()
        legs_wr = total_legs_won / total_legs_played if total_legs_played > 0 else 0.5
        
        # Forma Reciente (últimos 3) + blend 65% live / 35% hist
        df_recent = df.tail(form_window)
        
        raw_form_avg = df_recent["score_avg"].mean()
        raw_form_checkout = df_recent["checkout_pct"].mean()
        raw_form_100s = df_recent["turns_100"].mean()
        raw_form_140s = df_recent["turns_140"].mean()
        raw_form_180s = df_recent["turns_180"].mean()
        raw_form_wr = df_recent["result"].mean()
        raw_form_legs_wr = (
            df_recent["legs_won"].sum() / df_recent["total_legs"].sum()
            if df_recent["total_legs"].sum() > 0 else legs_wr
        )

        form_avg = LIVE_FORM_W * raw_form_avg + HIST_W * avg_score
        form_checkout = LIVE_FORM_W * raw_form_checkout + HIST_W * avg_checkout
        form_100s = LIVE_FORM_W * raw_form_100s + HIST_W * avg_100s
        form_140s = LIVE_FORM_W * raw_form_140s + HIST_W * avg_140s
        form_180s = LIVE_FORM_W * raw_form_180s + HIST_W * avg_180s
        form_wr = LIVE_FORM_W * raw_form_wr + HIST_W * win_rate
        form_legs_wr = LIVE_FORM_W * raw_form_legs_wr + HIST_W * legs_wr
        
        streak = self._calculate_streak(df["result"], limit=form_window)
        
        # Saco (Home = 1, Away = 0)
        df_home = df[df["first_throw"] == 1]
        df_away = df[df["first_throw"] == 0]
        home_wr = df_home["result"].mean() if not df_home.empty else win_rate
        away_wr = df_away["result"].mean() if not df_away.empty else win_rate

        # Días desde último partido
        days_since = None
        if pd.notna(df["fixture_date"].max()):
            last_date = df["fixture_date"].max()
            if last_date.tzinfo is None:
                from datetime import timezone
                last_date = last_date.replace(tzinfo=timezone.utc)
            now = pd.Timestamp.now(tz="UTC")
            days_since = (now - last_date).days

        # --- NUEVAS FEATURES DE CONTEXTO (v8-v11) ---
        match_ref_date = before_date if before_date is not None else pd.Timestamp.now(tz="UTC")
        
        # 1. Posición en la sesión (dentro del mismo día)
        matches_today = df[df["fixture_date"].dt.date == match_ref_date.date()]
        session_pos = len(matches_today) + 1 # +1 porque estamos calculando para el SIGUIENTE partido
            
        # 2. Partidos en los últimos 7 días
        seven_days_ago = match_ref_date - pd.Timedelta(days=7)
        matches_7d = len(df[df["fixture_date"] >= seven_days_ago])

        # 3. Fatigue - Piernas jugadas hoy
        legs_today = matches_today["legs_won"].sum() + matches_today["total_legs"].sum() - matches_today["legs_won"].sum() 
        # Simplificando: legs_today es la suma de total_legs de los partidos de hoy
        legs_today = matches_today["total_legs"].sum()

        # 4. Fatigue - Descanso y Hora
        rest_min = 999.0
        if not matches_today.empty:
            last_match_end = matches_today["fixture_date"].max()
            rest_delta = match_ref_date - last_match_end
            rest_min = rest_delta.total_seconds() / 60.0
        
        hour_dec = match_ref_date.hour + match_ref_date.minute / 60.0

        # --- ELO DATA ---
        elo_data = self.db.get_player_elo_at_time(player_name, before_date)

        # --- NUEVAS FEATURES v8 AUDIT ---
        
        # 1. EWM span=form_window (3) + blend 65% live EWM / 35% hist
        ewm_avg = df["score_avg"].ewm(span=form_window, adjust=False).mean().iloc[-1]
        ewm_co = df["checkout_pct"].ewm(span=form_window, adjust=False).mean().iloc[-1]
        ewm_wr = df["result"].ewm(span=form_window, adjust=False).mean().iloc[-1]
        form_avg_ewm = LIVE_FORM_W * ewm_avg + HIST_W * avg_score
        form_checkout_ewm = LIVE_FORM_W * ewm_co + HIST_W * avg_checkout
        form_wr_ewm = LIVE_FORM_W * ewm_wr + HIST_W * win_rate

        # 2. Trend Slope (últimos 7 partidos)
        def _get_trend_slope(series, n=7):
            recent = series.tail(n).dropna()
            if len(recent) < 3: return 0.0
            x = np.arange(len(recent))
            slope, _ = np.polyfit(x, recent.values, 1)
            return float(slope)
        
        avg_slope = _get_trend_slope(df["score_avg"])

        # 3. Pressure Checkout (Clutch) con Suavizado Bayesiano Empírico (Novedad v13)
        decider_legs = df[df["is_decider"] == 1]
        normal_legs = df[df["is_decider"] == 0]
        
        n_deciders = len(decider_legs)
        co_decider_raw = decider_legs["checkout_pct"].mean() if n_deciders > 0 else avg_checkout
        co_decider = (n_deciders * co_decider_raw + 5.0 * avg_checkout) / (n_deciders + 5.0)
        
        co_normal = normal_legs["checkout_pct"].mean() if not normal_legs.empty else avg_checkout
        pressure_delta = co_decider - co_normal

        # 4. VS Similar Profile (Real implementation via SQL)
        vs_similar_wr = self.db.get_vs_similar_wr(player_name, opp_avg, before_date) if opp_avg else win_rate

        # 5. Volatility (Standard deviation of score_avg)
        avg_std_val = df["score_avg"].std()
        avg_std = float(avg_std_val) if pd.notna(avg_std_val) else 0.0

        # 6. ELO Momentum (Novedad v13: diferencia contra el ELO de hace 5 partidos)
        elo_hist = self.db.get_player_elo_history_before(player_name, before_date, limit=6)
        elo_momentum = 0.0
        if len(elo_hist) >= 6:
            elo_momentum = float(elo_data["elo"] - elo_hist[-1])

        return PlayerFeatures(
            player_name=player_name,
            n_matches=n,
            avg_score=float(avg_score),
            avg_checkout=float(avg_checkout),
            avg_100s=float(avg_100s),
            avg_140s=float(avg_140s),
            avg_180s=float(avg_180s),
            win_rate=float(win_rate),
            legs_win_rate=float(legs_wr),
            form_avg=float(form_avg),
            form_checkout=float(form_checkout),
            form_100s=float(form_100s),
            form_140s=float(form_140s),
            form_180s=float(form_180s),
            form_win_rate=float(form_wr),
            form_legs_wr=float(form_legs_wr),
            streak=streak,
            home_wr=float(home_wr),
            away_wr=float(away_wr),
            days_since_last=days_since,
            session_position=session_pos,
            matches_last_7_days=matches_7d,
            legs_played_today=int(legs_today),
            rest_minutes=float(rest_min),
            hour_of_day=float(hour_dec),
            elo_rating=float(elo_data["elo"]),
            elo_rank=elo_data["rank"],
            is_debut=(elo_data["status"] == "DEBUT"),
            form_avg_ewm=float(form_avg_ewm),
            form_checkout_ewm=float(form_checkout_ewm),
            form_wr_ewm=float(form_wr_ewm),
            avg_slope=float(avg_slope),
            pressure_co=float(co_decider),
            pressure_delta=float(pressure_delta),
            vs_similar_wr=float(vs_similar_wr),
            avg_std=avg_std,
            elo_momentum=elo_momentum
        )


    def get_h2h_summary(self, player_a: str, player_b: str, before_date: Optional[pd.Timestamp] = None) -> dict:
        df_a = self.db.get_h2h(player_a, player_b)
        df_b = self.db.get_h2h(player_b, player_a)
        
        # Normalizar zonas horarias (Fix v11.1)
        if not df_a.empty and df_a["fixture_date"].dt.tz is None:
            df_a["fixture_date"] = df_a["fixture_date"].dt.tz_localize("UTC")
        if not df_b.empty and df_b["fixture_date"].dt.tz is None:
            df_b["fixture_date"] = df_b["fixture_date"].dt.tz_localize("UTC")
        
        if before_date is not None:
            if before_date.tzinfo is None:
                before_date = before_date.tz_localize("UTC")
            df_a = df_a[df_a["fixture_date"] < before_date]
            df_b = df_b[df_b["fixture_date"] < before_date]
            
        if df_a.empty and df_b.empty:
            return {
                "n_matches": 0,
                "win_rate_a": 0.5,
                "avg_a": 0.0,
                "avg_b": 0.0,
                "avg_checkout_a": 0.0,
                "avg_checkout_b": 0.0,
                "avg_100s_a": 0.0, "avg_140s_a": 0.0, "avg_180s_a": 0.0,
                "avg_100s_b": 0.0, "avg_140s_b": 0.0, "avg_180s_b": 0.0,
            }
            
        n_h2h = len(df_a)
        raw_wr = float(df_a["result"].mean()) if not df_a.empty else 0.5
        shrunk_wr = (raw_wr * n_h2h + 0.5 * 5.0) / (n_h2h + 5.0)
        return {
            "n_matches": n_h2h,
            "win_rate_a": shrunk_wr,
            "avg_a": float(df_a["score_avg"].mean()) if not df_a.empty else 0.0,
            "avg_b": float(df_b["score_avg"].mean()) if not df_b.empty else 0.0,
            "avg_checkout_a": float(df_a["checkout_pct"].mean()) if not df_a.empty else 0.0,
            "avg_checkout_b": float(df_b["checkout_pct"].mean()) if not df_b.empty else 0.0,
            "avg_100s_a": float(df_a["turns_100"].mean()) if not df_a.empty else 0.0,
            "avg_140s_a": float(df_a["turns_140"].mean()) if not df_a.empty else 0.0,
            "avg_180s_a": float(df_a["turns_180"].mean()) if not df_a.empty else 0.0,
            "avg_100s_b": float(df_b["turns_100"].mean()) if not df_b.empty else 0.0,
            "avg_140s_b": float(df_b["turns_140"].mean()) if not df_b.empty else 0.0,
            "avg_180s_b": float(df_b["turns_180"].mean()) if not df_b.empty else 0.0,
        }

if __name__ == "__main__":
    db = DartsDatabase()
    fe = FeatureExtractor(db)
    print("Extrayendo stats de ejemplo...")
    stats = fe.get_player_features("Luke Littler")
    print(stats)
