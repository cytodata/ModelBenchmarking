"""Tests for notebook artifact path discovery."""

from pathlib import Path

import pytest

from model_benchmarking.paths import cache_gcs_file, find_project_root, notebook_artifact_dirs


def test_find_project_root_from_nested_directory() -> None:
    """Find the repository root from a numbered notebook directory."""
    root = find_project_root(Path(__file__).parent)
    assert (root / "pyproject.toml").is_file()
    assert (root / "notebooks").is_dir()


def test_notebook_artifact_dirs_use_expected_layout(tmp_path: Path) -> None:
    """Place raw data at the root and results inside the requested stage."""
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "notebooks").mkdir()
    raw_dir, results_dir = notebook_artifact_dirs("0.download_and_assemble", "test-layout", start=tmp_path)

    assert raw_dir.parts[-3:] == ("data", "raw", "test-layout")
    assert results_dir.parts[-3:] == ("notebooks", "0.download_and_assemble", "results")


def test_cache_gcs_file_rejects_non_gcs_uri(tmp_path: Path) -> None:
    """Reject input locations that are not GCS object URIs."""
    with pytest.raises(ValueError, match="gs://"):
        cache_gcs_file("https://example.com/data.csv", tmp_path / "data.csv")
