import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orqa.normalize_metadata import _clean_html, _clean_text


class CleanTextMojibakeTests(unittest.TestCase):
    def test_repairs_latin1_decoded_utf8(self):
        # Real value observed in Bologna's ODS field labels: "Età singolo"
        # mis-decoded upstream as latin-1/cp1252 before we ever see it.
        self.assertEqual(_clean_text("EtÃ\xa0 singolo"), "Età singolo")

    def test_repairs_smart_quote_mojibake(self):
        self.assertEqual(_clean_text("cafÃ©â€™s report"), "café’s report")

    def test_leaves_correctly_encoded_accents_untouched(self):
        for text in ("bénéficiaires", "Personnes Âgées", "città", "año"):
            self.assertEqual(_clean_text(text), text)

    def test_leaves_ascii_untouched(self):
        self.assertEqual(_clean_text("Quartiere"), "Quartiere")

    def test_clean_html_repairs_mojibake_before_stripping(self):
        self.assertEqual(_clean_html("<p>EtÃ\xa0 grandi</p>"), "Età grandi")


if __name__ == "__main__":
    unittest.main()
