"""Exporta data/darts_public.sqlite con match_stats (MSS Bo7), player_elo y player_elo_history."""

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.db import DartsDatabase

_source = os.environ.get("GRAVEL_SOURCE_DB", "").strip()
if not _source:
    raise SystemExit("Set GRAVEL_SOURCE_DB to the source SQLite path before exporting.")
SOURCE = Path(_source)
DEST = ROOT / "data" / "darts_public.sqlite"
TABLES_FULL = ("player_elo", "player_elo_history")
TABLE_FILTERED = "match_stats"
SKIP_TABLES = frozenset({"model_calibration", "prediction_history", "sqlite_sequence"})


def _create_table(dst: sqlite3.Connection, src: sqlite3.Connection, name: str) -> None:
    row = src.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    if row is None or not row[0]:
        raise RuntimeError(f"Tabla no encontrada en origen: {name}")
    dst.execute(row[0])
    indexes = src.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
        (name,),
    ).fetchall()
    for (sql,) in indexes:
        dst.execute(sql)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        DEST.unlink()

    src = sqlite3.connect(str(SOURCE))
    dst = sqlite3.connect(str(DEST))
    try:
        extra = [
            r[0]
            for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if r[0] not in {TABLE_FILTERED, *TABLES_FULL, *SKIP_TABLES}
        ]
        if extra:
            raise RuntimeError(f"Tablas inesperadas no copiadas: {extra}")

        dst.execute(f"ATTACH DATABASE '{SOURCE.as_posix()}' AS orig")

        _create_table(dst, src, TABLE_FILTERED)
        dst.execute(
            f"INSERT INTO {TABLE_FILTERED} SELECT * FROM orig.{TABLE_FILTERED} WHERE {DartsDatabase._MSS_BO7_WHERE}"
        )

        for name in TABLES_FULL:
            _create_table(dst, src, name)
            dst.execute(f"INSERT INTO {name} SELECT * FROM orig.{name}")

        dst.execute("CREATE INDEX IF NOT EXISTS idx_player ON match_stats(player)")
        dst.execute("CREATE INDEX IF NOT EXISTS idx_elo_hist_player ON player_elo_history(player)")

        dst.commit()
    finally:
        dst.close()
        src.close()

    _print_summary(DEST)


def _print_summary(dest: Path) -> None:
    expected = {"match_stats": 39397, "player_elo": 607, "player_elo_history": 39396}
    conn = sqlite3.connect(str(dest))
    try:
        counts = {
            name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in expected
        }
    finally:
        conn.close()
    size_mb = dest.stat().st_size / (1024 * 1024)
    print("data/darts_public.sqlite")
    for name, n in counts.items():
        ok = "OK" if n == expected[name] else f"MISMATCH expected {expected[name]}"
        print(f"  {name}: {n} ({ok})")
    print(f"  size: {size_mb:.2f} MB")
    if counts != expected:
        raise RuntimeError(f"Recuentos no coinciden: {counts} vs {expected}")


if __name__ == "__main__":
    main()
