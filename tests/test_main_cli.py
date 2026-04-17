from __future__ import annotations

import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import main  # noqa: E402


def _module_with(attr_name: str):
    calls: list[object] = []

    def _fn(cfg):
        calls.append(cfg)

    module = types.SimpleNamespace()
    setattr(module, attr_name, _fn)
    return module, calls


class MainCliTests(unittest.TestCase):
    def test_build_parser_parses_country_only_target(self):
        args = main.build_parser().parse_args(
            ["--country", "canada", "--steps", "crawl", "index"]
        )

        self.assertEqual(args.country, "canada")
        self.assertIsNone(args.city)
        self.assertEqual(args.steps, ["crawl", "index"])

    def test_build_parser_parses_city_target(self):
        args = main.build_parser().parse_args(
            [
                "--country",
                "italy",
                "--city",
                "bologna",
                "--steps",
                "clean",
                "normalize-metadata",
            ]
        )

        self.assertEqual(args.country, "italy")
        self.assertEqual(args.city, "bologna")
        self.assertEqual(args.steps, ["clean", "normalize-metadata"])

    def test_main_rejects_missing_city_for_city_scoped_country(self):
        with self.assertRaises(SystemExit) as exc:
            main.main(["--country", "italy", "--steps", "clean"])

        self.assertEqual(exc.exception.code, 2)

    def test_main_rejects_city_for_country_only_target(self):
        with self.assertRaises(SystemExit) as exc:
            main.main(
                ["--country", "canada", "--city", "toronto", "--steps", "crawl"]
            )

        self.assertEqual(exc.exception.code, 2)

    def test_parser_requires_steps(self):
        with self.assertRaises(SystemExit) as exc:
            main.build_parser().parse_args(["--country", "canada"])

        self.assertEqual(exc.exception.code, 2)

    def test_resolve_target_rejects_unknown_country_city_combination(self):
        with self.assertRaisesRegex(ValueError, "Unknown city"):
            main.resolve_target("italy", "rome")

    def test_resolve_data_path_matches_existing_layout(self):
        expected = {
            ("canada", None): Path("/tmp/orqa-data/open_data/ckan/canada_small"),
            ("uk", None): Path("/tmp/orqa-data/orqa/ckan/uk"),
            ("italy", "modena"): Path("/tmp/orqa-data/open_data/ckan/modena"),
            ("italy", "bologna"): Path("/tmp/orqa-data/orqa/ods/bologna"),
            ("spain", "madrid"): Path("/tmp/orqa-data/orqa/ckan/madrid"),
            ("france", "paris"): Path("/tmp/orqa-data/orqa/ods/paris"),
            ("usa", "nyc"): Path("/tmp/orqa-data/orqa/socrata/nyc"),
        }

        with patch.dict(os.environ, {"DATADIR": "/tmp/orqa-data"}, clear=False):
            for target_key, expected_path in expected.items():
                with self.subTest(target=target_key):
                    spec = main.resolve_target(*target_key)
                    self.assertEqual(main.resolve_data_path(spec), expected_path)

    def test_resolve_data_path_requires_datadir(self):
        spec = main.resolve_target("canada", None)

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DATADIR is not set"):
                main.resolve_data_path(spec)

    def test_run_steps_imports_only_requested_modules(self):
        spec = main.resolve_target("italy", "bologna")
        cfg = object()

        cleaning_module, cleaning_calls = _module_with("ods_cleaning")
        normalize_module, normalize_calls = _module_with("normalize_metadata")
        imported_modules: list[str] = []

        def fake_import(module_name: str):
            imported_modules.append(module_name)
            modules = {
                "orqa.cleaning": cleaning_module,
                "orqa.normalize_metadata": normalize_module,
            }
            return modules[module_name]

        with patch("main.importlib.import_module", side_effect=fake_import):
            main.run_steps(cfg, spec, ["clean", "normalize-metadata"])

        self.assertEqual(imported_modules, ["orqa.cleaning", "orqa.normalize_metadata"])
        self.assertEqual(cleaning_calls, [cfg])
        self.assertEqual(normalize_calls, [cfg])

    def test_help_smoke_check(self):
        result = subprocess.run(
            [sys.executable, str(SRC_DIR / "main.py"), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--country", result.stdout)
        self.assertIn("--steps", result.stdout)


if __name__ == "__main__":
    unittest.main()
