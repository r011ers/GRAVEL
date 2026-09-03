import sqlite3
import unittest
from pathlib import Path

import pandas as pd

from core.db import DartsDatabase, clip_checkout_pct, parse_fixture_dates
from core.features import FeatureExtractor
from core.ml_model import AdvancedDartsModel


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "darts_public.sqlite"


class DateCheckoutReadonlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DartsDatabase(str(DB_PATH))

    def test_parse_known_formats(self):
        values = [
            "2022-07-25 08:36:16+00:00",
            "2026-08-31 13:19:00",
            "2026-04-06 08:39:39.767000+00:00",
            None,
        ]
        parsed = parse_fixture_dates(values)
        self.assertFalse(pd.isna(parsed.iloc[0]))
        self.assertFalse(pd.isna(parsed.iloc[1]))
        self.assertFalse(pd.isna(parsed.iloc[2]))
        self.assertTrue(pd.isna(parsed.iloc[3]))
        self.assertEqual(str(parsed.dt.tz), "UTC")

    def test_snapshot_null_dates_and_max(self):
        info = self.db.snapshot_info()
        self.assertEqual(info["n_null_dates"], 10)
        self.assertFalse(pd.isna(info["max_date"]))
        max_date = pd.Timestamp(info["max_date"])
        self.assertEqual(max_date.year, 2026)
        self.assertEqual(max_date.month, 8)
        self.assertEqual(max_date.day, 31)

    def test_null_date_rows_are_complete_matches(self):
        """No son filas vacías: 5 partidos Bo7 con stats, solo sin timestamp."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT match_id, player, opponent, score_avg, result, total_legs
                FROM match_stats WHERE fixture_date IS NULL
                ORDER BY match_id, id
                """
            ).fetchall()
        self.assertEqual(len(rows), 10)
        match_ids = {r[0] for r in rows}
        self.assertEqual(len(match_ids), 5)
        for row in rows:
            self.assertTrue(row[1] and row[2])
            self.assertIsNotNone(row[3])
            self.assertIn(row[4], (0, 1))
            self.assertGreaterEqual(row[5], 4)

    def test_load_player_data_parses_recent_dates(self):
        df = self.db.load_all_data()
        parsed_nat = int(df["fixture_date"].isna().sum())
        self.assertEqual(parsed_nat, 10)

    def test_checkout_clipped(self):
        raw = pd.DataFrame({"checkout_pct": [-1.0, 50.0, 200.0, 133.33]})
        clipped = clip_checkout_pct(raw)["checkout_pct"].tolist()
        self.assertEqual(clipped[0], 0.0)
        self.assertEqual(clipped[1], 50.0)
        self.assertEqual(clipped[2], 100.0)
        self.assertEqual(clipped[3], 100.0)
        df = self.db.load_all_data()
        self.assertFalse((df["checkout_pct"] > 100).any())
        self.assertFalse((df["checkout_pct"] < 0).any())

    def test_startup_does_not_write_sqlite(self):
        before = DB_PATH.stat()
        DartsDatabase(str(DB_PATH))
        after = DB_PATH.stat()
        self.assertEqual(before.st_mtime, after.st_mtime)
        self.assertEqual(before.st_size, after.st_size)

    def test_connection_is_read_only(self):
        db = DartsDatabase(str(DB_PATH))
        conn = db.get_connection()
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE gravel_should_not_exist (id INTEGER)")
        conn.close()

    def test_import_and_predict(self):
        fe = FeatureExtractor(self.db)
        model = AdvancedDartsModel(self.db)
        self.assertIsNotNone(model.v8_model)
        feat_a = fe.get_player_features("Luke Littler")
        feat_b = fe.get_player_features("Neil Duff")
        self.assertIsNotNone(feat_a)
        self.assertIsNotNone(feat_b)
        feat_a = fe.get_player_features("Luke Littler", opp_avg=feat_b.avg_score)
        feat_b = fe.get_player_features("Neil Duff", opp_avg=feat_a.avg_score)
        h2h = fe.get_h2h_summary("Luke Littler", "Neil Duff")
        pred = model.predict_match(
            feat_a, feat_b, h2h_a_winrate=h2h["win_rate_a"], first_throw_a=True
        )
        self.assertEqual(len(model._build_feature_vector(feat_a, feat_b, h2h["win_rate_a"], True)), 37)
        self.assertGreater(pred.prob_a, 0.0)
        self.assertLess(pred.prob_a, 1.0)
        self.assertAlmostEqual(pred.prob_a + pred.prob_b, 1.0, places=3)
        self.assertFalse(pd.isna(feat_a.avg_score))
        self.assertFalse(pd.isna(feat_a.elo_rating))
        self.assertFalse(hasattr(model, "_precompute_elo_history"))

    def test_model_sha256_matches_sidecar(self):
        digest = __import__("hashlib").sha256(
            (ROOT / "models" / "darts_v8.pkl").read_bytes()
        ).hexdigest()
        sidecar = (ROOT / "models" / "darts_v8.pkl.sha256").read_text(encoding="utf-8").strip().split()[0]
        self.assertEqual(digest, sidecar)

    def test_golden_prediction_littler_duff(self):
        fe = FeatureExtractor(self.db)
        model = AdvancedDartsModel(self.db)
        feat_b = fe.get_player_features("Neil Duff")
        feat_a = fe.get_player_features("Luke Littler", opp_avg=feat_b.avg_score)
        feat_b = fe.get_player_features("Neil Duff", opp_avg=feat_a.avg_score)
        h2h = fe.get_h2h_summary("Luke Littler", "Neil Duff")
        pred = model.predict_match(
            feat_a, feat_b, h2h_a_winrate=h2h["win_rate_a"], first_throw_a=True
        )
        self.assertEqual(pred.prob_a, 0.754)
        self.assertEqual(pred.prob_b, 0.246)
        pred_b = model.predict_match(
            feat_a, feat_b, h2h_a_winrate=h2h["win_rate_a"], first_throw_a=False
        )
        self.assertEqual(pred_b.prob_a, 0.7298)
        self.assertEqual(pred_b.prob_b, 0.2702)


if __name__ == "__main__":
    unittest.main()
