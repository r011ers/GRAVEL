import unittest

from core.db import DartsDatabase
from core.features import FeatureExtractor
from core.identities import (
    aliases_of,
    canonical_player,
    identity_key,
    is_hidden_player,
    normalize_compare,
    selectable_players,
)


class IdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DartsDatabase()
        conn = cls.db.get_connection()
        cls.names = [
            r[0] for r in conn.execute("SELECT DISTINCT player FROM match_stats")
        ]
        conn.close()
        cls.selectable = selectable_players(cls.names)

    def test_normalize_whitespace_and_underscores(self):
        self.assertEqual(normalize_compare("Andy  Baetens"), "andy baetens")
        self.assertEqual(normalize_compare("Berry van_Peer"), "berry van peer")
        self.assertEqual(normalize_compare("Noa-Lynn van_Leuven_"), "noa-lynn van leuven")

    def test_apostrophe_encoding(self):
        self.assertEqual(normalize_compare("John O'Shea"), normalize_compare("John O\u00b4Shea"))
        self.assertEqual(identity_key("John O'Shea"), identity_key("John O\u00b4Shea"))

    def test_name_order_two_tokens(self):
        self.assertEqual(identity_key("Zvonimir Lesic"), identity_key("Lesic Zvonimir"))

    def test_known_aliases_merge(self):
        pairs = [
            ("Andy Baetens", "Andy  Baetens"),
            ("Zvonimir Lesic", "Lesic Zvonimir"),
            ("Berry Van Peer", "Berry van_Peer"),
            ("Danny van Trijp", "Danny van_Trijp"),
            ("Keanu van Velzen", "Keanu Van_Velzen"),
            ("Jamai van den Herik", "Jamai Van_Den_Herik"),
            ("Noa-Lynn van Leuven", "Noa-Lynn van_Leuven_"),
            ("Bradley O'Connor", "Bradley O Connor"),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertEqual(
                    canonical_player(a, self.names),
                    canonical_player(b, self.names),
                )

    def test_oshea_encoding_alias(self):
        acute = "John O\u00b4Shea"
        if acute not in self.names:
            self.skipTest("encoded O'Shea variant not in snapshot")
        self.assertEqual(
            canonical_player("John O'Shea", self.names),
            canonical_player(acute, self.names),
        )

    def test_do_not_merge_ambiguous(self):
        separated = [
            ("Lee Evans", "Lee (ENG) Evans"),
            ("Lee Evans", "Lee Evans (WAL)"),
            ("Lee (ENG) Evans", "Lee Evans (WAL)"),
            ("John McCarthy", "Josh McCarthy"),
            ("Matt Dennant", "Matthew Dennant"),
            ("Llew Bevan", "Llew-J Bevan"),
            ("Josh Richardson", "Joshua Richardson"),
            ("Steve Johnstone", "Steven Johnstone"),
        ]
        for a, b in separated:
            with self.subTest(a=a, b=b):
                self.assertNotEqual(
                    canonical_player(a, self.names),
                    canonical_player(b, self.names),
                )

    def test_dropdown_unique_lesic_baetens(self):
        self.assertIn("Zvonimir Lesic", self.selectable)
        self.assertNotIn("Lesic Zvonimir", self.selectable)
        self.assertIn("Andy Baetens", self.selectable)
        self.assertNotIn("Andy  Baetens", self.selectable)

    def test_no_self_match_via_alias(self):
        a = canonical_player("Lesic Zvonimir", self.names)
        b = canonical_player("Zvonimir Lesic", self.names)
        self.assertEqual(a, b)

    def test_hidden_player_not_selectable(self):
        self.assertTrue(is_hidden_player("Richard Rowland Dont USE"))
        self.assertNotIn("Richard Rowland Dont USE", self.selectable)
        self.assertIn("Richard Rowland Dont USE", self.names)

    def test_match_1018870_deduped_for_lesic(self):
        fe = FeatureExtractor(self.db)
        df = fe._load_player_history("Zvonimir Lesic")
        n = int((df["match_id"] == 1018870).sum())
        self.assertEqual(n, 1)
        df_alias = fe._load_player_history("Lesic Zvonimir")
        self.assertEqual(int((df_alias["match_id"] == 1018870).sum()), 1)

    def test_h2h_uses_canonical_and_not_self(self):
        fe = FeatureExtractor(self.db)
        h2h_self = fe.get_h2h_summary("Lesic Zvonimir", "Zvonimir Lesic")
        self.assertEqual(h2h_self["n_matches"], 0)
        h2h = fe.get_h2h_summary("Zvonimir Lesic", "Berry Van Peer")
        h2h_alias = fe.get_h2h_summary("Lesic Zvonimir", "Berry van_Peer")
        self.assertEqual(h2h["n_matches"], h2h_alias["n_matches"])
        self.assertGreater(h2h["n_matches"], 0)

    def test_aliases_of_includes_raw_variants(self):
        aliases = aliases_of("Andy Baetens", self.names)
        self.assertTrue(any("Baetens" in a for a in aliases))
        self.assertGreaterEqual(len(aliases), 2)


if __name__ == "__main__":
    unittest.main()
