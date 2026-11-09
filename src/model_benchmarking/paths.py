"""Filesystem locations shared by the analysis notebooks."""

from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root containing ``pyproject.toml`` and ``notebooks``.

    The search starts at ``start`` (or the current working directory) and walks upward.
    This makes notebook paths independent of whether Jupyter was opened at the repository
    root or inside a numbered notebook folder.
    """
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "notebooks").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find the ModelBenchmarking project root. Start Jupyter from the cloned repository."
    )


def notebook_artifact_dirs(stage: str, raw_dataset: str, start: str | Path | None = None) -> tuple[Path, Path]:
    """Create and return the raw-data and results directories for a notebook stage.

    Parameters
    ----------
    stage : str
        Numbered notebook directory, for example ``"0.download_and_assemble"``.
    raw_dataset : str
        Dataset-specific directory below ``data/raw``.
    start : str or pathlib.Path, optional
        Starting location used to discover the repository root.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        ``(data/raw/<raw_dataset>, notebooks/<stage>/results)``.
    """
    root = find_project_root(start)
    raw_dir = root / "data" / "raw" / raw_dataset
    results_dir = root / "notebooks" / stage / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, results_dir


def cache_gcs_file(gcs_uri: str, destination: str | Path, *, overwrite: bool = False) -> Path:
    """Download one GCS object into ``data/raw`` and return its local path.

    Existing non-empty files are reused unless ``overwrite`` is true.
    """
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
        return destination

    parsed = urlparse(gcs_uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Expected a gs:// URI, received: {gcs_uri}")

    client = storage.Client()
    blob = client.bucket(parsed.netloc).blob(parsed.path.lstrip("/"))
    blob.download_to_filename(str(destination))
    return destination
