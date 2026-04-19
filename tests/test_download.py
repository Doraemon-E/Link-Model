from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hy_models import download


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_complete_original(directory: Path) -> None:
    _touch(directory / "config.json", "{}")
    _touch(directory / "tokenizer.json", '{"tokenizer": true}')
    _touch(directory / "generation_config.json", "{}")
    _touch(directory / "model-00001-of-00002.safetensors", "weights-a")
    _touch(directory / "model-00002-of-00002.safetensors", "weights-b")


def _build_complete_mlx(directory: Path) -> None:
    _touch(directory / "config.json", "{}")
    _touch(directory / "tokenizer.json", '{"tokenizer": true}')
    _touch(directory / "chat_template.jinja", "{{ user }}")
    _touch(directory / "model.safetensors.index.json", "{}")
    _touch(directory / "model-00001-of-00002.safetensors", "mlx-a")
    _touch(directory / "model-00002-of-00002.safetensors", "mlx-b")


class DownloadPathsTests(unittest.TestCase):
    def test_default_path_resolution(self) -> None:
        base = Path("/tmp/demo")
        self.assertEqual(
            download.resolve_default_original_dir(base),
            base / download.DEFAULT_ORIGINAL_DIR,
        )
        self.assertEqual(
            download.resolve_default_mlx_8bit_dir(base),
            base / download.DEFAULT_MLX_8BIT_DIR,
        )


class IntegrityTests(unittest.TestCase):
    def test_original_integrity_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_complete_original(root)
            result = download.check_original_integrity(root)
            self.assertTrue(result["is_complete"])
            self.assertEqual(result["missing_requirements"], [])

    def test_mlx_integrity_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_complete_mlx(root)
            result = download.check_mlx_integrity(root)
            self.assertTrue(result["is_complete"])
            self.assertEqual(result["missing_requirements"], [])


class DownloadControlTests(unittest.TestCase):
    def test_skip_download_when_already_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_complete_original(root)
            with patch("hy_models.download._snapshot_download") as mocked:
                result = download.ensure_model_download(
                    repo_id=download.ORIGINAL_REPO_ID,
                    model_type="original",
                    local_dir=root,
                    force=False,
                )
            mocked.assert_not_called()
            self.assertEqual(result["status"], "skipped_already_complete")
            self.assertFalse(result["download_triggered"])

    def test_force_triggers_snapshot_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_complete_original(root)
            with patch("hy_models.download._snapshot_download", return_value=str(root)) as mocked:
                result = download.ensure_model_download(
                    repo_id=download.ORIGINAL_REPO_ID,
                    model_type="original",
                    local_dir=root,
                    force=True,
                )
            mocked.assert_called_once()
            self.assertEqual(result["status"], "downloaded")
            self.assertTrue(result["download_triggered"])


class ReportTests(unittest.TestCase):
    def test_report_contains_required_fields_and_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            original_dir = tmp_root / "original"
            mlx_dir = tmp_root / "mlx"
            _build_complete_original(original_dir)
            _build_complete_mlx(mlx_dir)

            report_one = download.generate_static_compare_report(
                original_dir=original_dir,
                mlx_dir=mlx_dir,
            )
            report_two = download.generate_static_compare_report(
                original_dir=original_dir,
                mlx_dir=mlx_dir,
            )

            self.assertEqual(report_one, report_two)
            self.assertEqual(report_one["original"]["repo_id"], download.ORIGINAL_REPO_ID)
            self.assertEqual(report_one["mlx_8bit"]["repo_id"], download.MLX_8BIT_REPO_ID)
            self.assertEqual(report_one["original"]["directory"], str(original_dir))
            self.assertEqual(report_one["mlx_8bit"]["directory"], str(mlx_dir))

            comparison = report_one["comparison"]
            self.assertIn("size", comparison)
            self.assertIn("weights", comparison)
            self.assertIn("named_files", comparison)
            self.assertIn("file_set_diff", comparison)
            self.assertIn("file_diff_summary", comparison)

            self.assertIn(
                "config_json",
                report_one["original"]["summary"]["key_files"],
            )
            self.assertIn(
                "tokenizer_files",
                report_one["mlx_8bit"]["summary"]["key_files"],
            )
            self.assertFalse(
                any(
                    path.startswith(".cache/")
                    for path in report_one["original"]["summary"]["key_files"]["tokenizer_files"]
                )
            )
            self.assertFalse(
                any(
                    path.startswith(".cache/")
                    for path in report_one["mlx_8bit"]["summary"]["key_files"]["tokenizer_files"]
                )
            )


if __name__ == "__main__":
    unittest.main()
