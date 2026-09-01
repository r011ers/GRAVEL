"""
GRAVEL — db.py
Capa de acceso a la base de datos filtrada.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import datetime

logger = logging.getLogger("beta_db")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class DartsDatabase:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Ruta centralizada en la raíz de 02_MOTOR_CORE
            self.db_path = str(Path(__file__).resolve().parent.parent / "darts.sqlite")
        else:
            self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Inicializa la estructura de la base de datos si no existe."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de calibración (Platt Scaling) para los modelos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_calibration (
                model_id TEXT PRIMARY KEY,
                param_a REAL NOT NULL,
                param_b REAL NOT NULL,
                last_updated DATETIME
            )
        """)

        # Tabla de histórico de predicciones para Calibración y Backtest
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                prob_a REAL NOT NULL,
                actual_result INTEGER, -- 1 si ganó A, 0 si perdió, NULL si pendiente
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(match_id, model_version)
            )
        """)
        
        # Tabla de ELO Rankings (v11 Pro)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_elo (
                player TEXT PRIMARY KEY,
                elo REAL NOT NULL DEFAULT 1500.0,
                matches_played INTEGER DEFAULT 0,
                status TEXT DEFAULT 'DEBUT', -- DEBUT (<5 matches) vs PRO
                last_updated DATETIME
            )
        """)

        # Tabla de ELO History (Novedad v13 para evitar look-ahead bias)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_elo_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player TEXT NOT NULL,
                match_id INTEGER NOT NULL,
                elo REAL NOT NULL,
                fixture_date DATETIME,
                matches_played INTEGER,
                UNIQUE(player, match_id)
            )
        """)

        
        # Tabla principal de partidos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL,
                player TEXT NOT NULL,
                opponent TEXT NOT NULL,
                score_avg REAL,
                checkout_pct REAL,
                turns_100 INTEGER,
                turns_140 INTEGER,
                turns_180 INTEGER,
                first_throw INTEGER,
                result INTEGER,
                legs_won INTEGER,
                total_legs INTEGER,
                is_decider INTEGER,
                fixture_date DATETIME,
                UNIQUE(match_id, player)
            )
        """)
        
        # Índices para búsquedas súper rápidas de histórico y h2h
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_player ON match_stats(player)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_opponent ON match_stats(opponent)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON match_stats(fixture_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_elo_hist_player_date ON player_elo_history(player, fixture_date)")

        
        conn.commit()
        conn.close()

    def get_connection(self):
        return sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    def load_player_data(self, player_name: str) -> pd.DataFrame:
        """Historial MSS Bo7 de un jugador."""
        conn = self.get_connection()
        query = f"""
            SELECT * FROM match_stats
            WHERE player = ? AND {self._MSS_BO7_WHERE}
            ORDER BY fixture_date ASC, match_id ASC
        """
        df = pd.read_sql_query(query, conn, params=(player_name,), parse_dates=["fixture_date"])
        conn.close()
        return df

    def get_h2h(self, player_a: str, player_b: str) -> pd.DataFrame:
        """Enfrentamientos directos MSS Bo7."""
        conn = self.get_connection()
        query = f"""
            SELECT * FROM match_stats
            WHERE player = ? AND opponent = ? AND {self._MSS_BO7_WHERE}
            ORDER BY fixture_date ASC
        """
        df = pd.read_sql_query(query, conn, params=(player_a, player_b), parse_dates=["fixture_date"])
        conn.close()
        return df

    def get_vs_similar_wr(self, player_name: str, opponent_avg: float, before_date: Optional[datetime.datetime] = None, margin: float = 5.0) -> float:
        """
        Calcula el Win Rate de un jugador contra oponentes de nivel similar (peer performance).
        Lógica: busca partidos del jugador donde el oponente tenía un avg ± margin del actual.
        """
        conn = self.get_connection()
        query = """
            SELECT ms_player.result, ms_opp.score_avg as opp_avg
            FROM match_stats ms_player
            JOIN match_stats ms_opp 
              ON ms_player.match_id = ms_opp.match_id 
              AND ms_opp.player = ms_player.opponent
            WHERE ms_player.player = ? 
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
                player_name, 
                ref_date.strftime('%Y-%m-%d %H:%M:%S'), # Pasamos como string para evitar líos de TZ en el driver
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
        conn = self.get_connection()
        df = pd.read_sql_query(
            f"SELECT * FROM match_stats WHERE {self._MSS_BO7_WHERE}",
            conn,
            parse_dates=["fixture_date"],
        )
        conn.close()
        return df

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
