"""Device selection, memory release, progress logging and download progress bars."""

import ctypes
import gc
import time
from functools import lru_cache

import psutil
import torch
from tqdm import tqdm


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """Return the CUDA device if it is usable, otherwise the CPU.

    Mirrors the "self-healing" hardware check originally used in the notebooks: a GPU is
    only selected if a small tensor operation actually succeeds on it (a GPU can be
    visible to PyTorch but lack kernels for its architecture).

    Returns
    -------
    torch.device
    """
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda") + 1
            device = torch.device("cuda")
        except Exception:
            device = torch.device("cpu")
    return device


def release_memory() -> None:
    """Run garbage collection and, on Linux (glibc), return freed memory to the operating system."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def log_debug(msg: str, t_start: float | None = None) -> None:
    """Print a message prefixed with elapsed time and current RAM usage.

    Parameters
    ----------
    msg : str
        Message to print.
    t_start : float, optional
        Reference time from ``time.time()``; when given, the elapsed seconds since
        ``t_start`` are shown.
    """
    ram_gb = psutil.virtual_memory().used / 1e9
    ram_pct = psutil.virtual_memory().percent
    elapsed = f"[+{time.time() - t_start:5.1f}s] " if t_start else ""
    print(f"⏱️ {elapsed}🧠 RAM: {ram_gb:4.1f}GB ({ram_pct:2.0f}%) | {msg}")


class DownloadProgressBar(tqdm):
    """``tqdm`` progress bar usable as the ``reporthook`` of ``urllib.request.urlretrieve``."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        """Advance the bar to ``b * bsize`` bytes out of ``tsize``.

        Parameters
        ----------
        b : int
            Number of blocks transferred so far.
        bsize : int
            Size of each block in bytes.
        tsize : int, optional
            Total size in bytes, if known.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
