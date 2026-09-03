import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class AppE2ETests(unittest.TestCase):
    def test_app_script_loads(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        at.run()
        self.assertEqual(len(at.exception), 0)
        titles = [el.value for el in at.title]
        self.assertTrue(any("GRAVEL" in str(v) for v in titles))
        captions = [str(el.value) for el in at.caption]
        self.assertTrue(any("Snapshot" in c or "snapshot" in c.lower() for c in captions))
