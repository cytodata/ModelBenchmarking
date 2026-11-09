"""Shared pytest configuration: render figures off-screen and close them after each test."""

from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def no_figure_windows(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replace ``plt.show`` with a no-op and close every figure a test creates."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")
