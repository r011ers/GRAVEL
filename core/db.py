"""Capa de acceso al snapshot SQLite público (solo lectura)."""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Sequence
import logging
import datetime

logger = logging.getLogger("gravel_db")


def parse_fixture_dates(values) -> pd.Series:
    """Parsea fixture_date a UTC. NULL y valores ilegibles → NaT. No inventa fechas."""
    s = pd.Series(values)
    if s.empty:
        return pd.to_datetime(s, utc=True)
    return pd.to_datetime(s, utc=True, format="mixed", errors="coerce")


def clip_checkout_pct(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "checkout_pct" not in df.columns:
        return df
    out = df.copy()
    out["checkout_pct"] = out["checkout_pct"].clip(lower=0, upper=100)
    return out


def _prepare_match_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = clip_checkout_pct(df)
    if "fixture_date" in out.columns:
        out = out.copy()
        out["fixture_date"] = parse_fixture_dates(out["fixture_date"])
    return out


class DartsDatabase:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            self.db_path = str(
                Path(__file__).resolve().parent.parent / "data" / "darts_public.sqlite"
            )
        else:
            self.db_path = db_path
        path = Path(self.db_path)
        if not path.is_file():
            raise FileNotFoundError(f"SQLite snapshot not found: {self.db_path}")

    def get_connection(self):
        conn = sqlite3.connect(str(Path(self.db_path).resolve()))
        conn.execute("PRAGMA query_only = ON")
        return conn

    def load_player_data(self, player_name: str) -> pd.DataFrame:
        """Historial MSS Bo7 de un jugador."""
        query = f"""
            SELECT * FROM match_stats
            WHERE player = ? AND {self._MSS_BO7_WHERE}
            ORDER BY fixture_date ASC, match_id ASC
        """
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(player_name,))
        return _prepare_match_frame(df)

    def get_h2h(
        self,
        player_a: str,
        player_b: str,
        aliases_a: Optional[Sequence[str]] = None,
        aliases_b: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Enfrentamientos directos MSS Bo7 (acepta aliases canónicos)."""
        names_a = list(aliases_a) if aliases_a else [player_a]
        names_b = list(aliases_b) if aliases_b else [player_b]
        ph_a = ",".join("?" * len(names_a))
        ph_b = ",".join("?" * len(names_b))
        query = f"""
            SELECT * FROM match_stats
            WHERE player IN ({ph_a}) AND opponent IN ({ph_b}) AND {self._MSS_BO7_WHERE}
            ORDER BY fixture_date ASC
        """
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(*names_a, *names_b))
        return _prepare_match_frame(df)

    def get_vs_similar_wr(self, player_name: str, opponent_avg: float, before_date: Optional[datetime.datetime] = None, margin: float = 5.0, player_aliases: Optional[Sequence[str]] = None) -> float:
        """
        Calcula el Win Rate de un jugador contra oponentes de nivel similar (peer performance).
        Lógica: busca partidos del jugador donde el oponente tenía un avg ± margin del actual.
        """
        conn = self.get_connection()
        names = list(player_aliases) if player_aliases else [player_name]
        ph = ",".join("?" * len(names))
        query = f"""
            SELECT ms_player.result, ms_opp.score_avg as opp_avg
            FROM match_stats ms_player
            JOIN match_stats ms_opp 
              ON ms_player.match_id = ms_opp.match_id 
              AND ms_opp.player = ms_player.opponent
            WHERE ms_player.player IN ({ph})
              AND ms_player.fixture_date < ?
              AND ms_opp.score_avg BETWEEN ? AND ?
              AND ms_player.match_id < 100000000
              AND ms_player.player NOT LIKE '%/%'
              AND ms_player.total_legs BETWEEN 4 AND 7
              AND ms_player.legs_won BETWEEN 0 AND 4
        """
        # --- NORMALIZACIÓN DE FECHAS (v11.1) ---
        # Si antes_date es naive, lo hacemos aware (UTC). Si no hay, usamos now aware.
        if before_date is not None:
            if before_date.tzinfo is None:
                ref_date = before_date.replace(tzinfo=datetime.timezone.utc)
            else:
                ref_date = before_date
        else:
            ref_date = datetime.datetime.now(datetime.timezone.utc)
        
        try:
            # Nota: Debido a detect_types=sqlite3.PARSE_DECLTYPES, 
            # SQLite intentará comparar objetos datetime.
            # Sin embargo, si los datos en la BD son naive, 
            # localizamos el DF resultante si es necesario (o comparamos naive vs naive).
            # Para SQL puro, lo más seguro es pasar strings o asegurar que el driver maneje la zona.
            
            # Cargamos con pandas para manejar la zona horaria después si es necesario
            df = pd.read_sql_query(query, conn, params=(
                *names,
                ref_date.strftime('%Y-%m-%d %H:%M:%S'),
                opponent_avg - margin,
                opponent_avg + margin
            ))
            conn.close()
            if df.empty:
                return 0.5 # Neutral si no hay datos
            return float(df["result"].mean())
        except Exception as e:
            logger.error(f"Error en get_vs_similar_wr: {e}")
            conn.close()
            return 0.5

    # Solo MODUS Super Series singles Best of 7 (first to 4 → total_legs 4..7)
    _MSS_BO7_WHERE = """
        match_id < 100000000
        AND player NOT LIKE '%/%'
        AND opponent NOT LIKE '%/%'
        AND total_legs BETWEEN 4 AND 7
        AND legs_won BETWEEN 0 AND 4
    """

    def load_all_data(self) -> pd.DataFrame:
        """Carga solo histórico MSS singles Best of 7."""
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM match_stats WHERE {self._MSS_BO7_WHERE}",
                conn,
            )
        return _prepare_match_frame(df)

    def snapshot_info(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            n_rows = conn.execute("SELECT COUNT(*) FROM match_stats").fetchone()[0]
            n_matches = conn.execute("SELECT COUNT(DISTINCT match_id) FROM match_stats").fetchone()[0]
            n_players = conn.execute("SELECT COUNT(DISTINCT player) FROM match_stats").fetchone()[0]
            n_null = conn.execute(
                "SELECT COUNT(*) FROM match_stats WHERE fixture_date IS NULL"
            ).fetchone()[0]
            raw_dates = [
                row[0]
                for row in conn.execute(
                    "SELECT fixture_date FROM match_stats WHERE fixture_date IS NOT NULL"
                )
            ]
        parsed = parse_fixture_dates(raw_dates)
        max_date = parsed.max() if not parsed.isna().all() else pd.NaT
        return {
            "n_rows": int(n_rows),
            "n_matches": int(n_matches),
            "n_players_raw": int(n_players),
            "n_null_dates": int(n_null),
            "max_date": max_date,
        }

    def get_player_elo(self, player_name: str) -> Dict[str, Any]:
        """Obtiene el rating ELO y el ranking de un jugador."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Obtener ranking absoluto
                cursor.execute("""
                    SELECT elo, status, matches_played,
                    (SELECT COUNT(*) + 1 FROM player_elo p2 WHERE p2.elo > p1.elo AND p2.status = 'PRO') as rank
                    FROM player_elo p1 WHERE player = ?
                """, (player_name,))
                row = cursor.fetchone()
                if row:
                    return {
                        "elo": row[0],
                        "status": row[1],
                        "matches_played": row[2],
                        "rank": row[3] if row[1] == "PRO" else "N/A"
                    }
        except Exception as e:
            logger.error(f"Error leyendo ELO de {player_name}: {e}")
        return {"elo": 1500.0, "status": "DEBUT", "matches_played": 0, "rank": "N/A"}

    def get_player_elo_at_time(self, player_name: str, before_date: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        """Obtiene el rating ELO de un jugador antes de una fecha dada (evitando look-ahead bias)."""
        if before_date is None:
            return self.get_player_elo(player_name)
            
        try:
            # Si before_date tiene zona horaria, lo convertimos a UTC y luego quitamos la zona horaria para formatear
            if before_date.tzinfo is not None:
                before_date = before_date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
            date_str = before_date.strftime('%Y-%m-%d %H:%M:%S')
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT elo, matches_played
                    FROM player_elo_history
                    WHERE player = ? AND fixture_date < ?
                    ORDER BY fixture_date DESC, id DESC LIMIT 1
                """, (player_name, date_str))
                row = cursor.fetchone()
                if row:
                    elo = row[0]
                    n_matches = row[1]
                    status = "PRO" if n_matches >= 5 else "DEBUT"
                    
                    # Calcular ranking relativo a ese instante
                    cursor.execute("""
                        SELECT COUNT(*) + 1 
                        FROM player_elo_history p1
                        WHERE p1.elo > ? 
                          AND p1.fixture_date < ? 
                          AND p1.matches_played >= 5
                          AND p1.id = (
                              SELECT id FROM player_elo_history p2 
                              WHERE p2.player = p1.player AND p2.fixture_date < ?
                              ORDER BY p2.fixture_date DESC, p2.id DESC LIMIT 1
                          )
                    """, (elo, date_str, date_str))
                    rank_row = cursor.fetchone()
                    rank = rank_row[0] if rank_row else "N/A"
                    
                    return {
                        "elo": elo,
                        "status": status,
                        "matches_played": n_matches,
                        "rank": rank if status == "PRO" else "N/A"
                    }
        except Exception as e:
            logger.error(f"Error leyendo ELO histórico de {player_name}: {e}")
            
        return {"elo": 1500.0, "status": "DEBUT", "matches_played": 0, "rank": "N/A"}

    def get_player_elo_history_before(self, player_name: str, before_date: Optional[datetime.datetime] = None, limit: int = 10) -> List[float]:
        """Obtiene los últimos ratings ELO de un jugador antes de una fecha dada, ordenados de más reciente a más antiguo."""
        try:
            date_filter = ""
            params = [player_name]
            if before_date is not None:
                if before_date.tzinfo is not None:
                    before_date = before_date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                date_filter = "AND fixture_date < ?"
                params.append(before_date.strftime('%Y-%m-%d %H:%M:%S'))
            
            params.append(limit)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT elo FROM player_elo_history
                    WHERE player = ? {date_filter}
                    ORDER BY fixture_date DESC, id DESC LIMIT ?
                """, tuple(params))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error leyendo historial ELO previo de {player_name}: {e}")
        return []
