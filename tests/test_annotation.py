"""Tests for model_benchmarking.annotation against mocked PubChem/ChEMBL responses (no network)."""

from collections.abc import Iterator
from typing import Any

import pytest

from model_benchmarking import annotation


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        """Return the canned JSON payload."""
        return self._payload


@pytest.fixture(autouse=True)
def clear_cache() -> Iterator[None]:
    """Empty the annotation cache before and after each test."""
    annotation.ANNOTATION_CACHE.clear()
    yield
    annotation.ANNOTATION_CACHE.clear()


def test_name_from_pubchem_and_moa_from_chembl(monkeypatch: pytest.MonkeyPatch) -> None:
    """The name comes from PubChem synonyms and the mechanism from ChEMBL's curated mechanisms."""

    def fake_get(url: str, timeout: float) -> FakeResponse:
        if "pubchem" in url:
            return FakeResponse(
                {"InformationList": {"Information": [{"CID": 1, "Synonym": ["50-78-2", "SCHEMBL1", "aspirin"]}]}}
            )
        if "molecule.json" in url:
            return FakeResponse({"molecules": [{"molecule_chembl_id": "CHEMBL25", "pref_name": "ASPIRIN"}]})
        if "mechanism.json" in url:
            return FakeResponse({"mechanisms": [{"mechanism_of_action": "Cyclooxygenase inhibitor"}]})
        raise AssertionError(url)

    monkeypatch.setattr(annotation.requests, "get", fake_get)
    assert annotation.get_rich_compound_annotation("BSYNRYMUTXBXSQ-UHFFFAOYSA-N") == (
        "aspirin",
        "Cyclooxygenase inhibitor",
    )


def test_ic50_fallback_and_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a curated mechanism the IC50 target is used, and repeat lookups hit the cache."""
    calls = []

    def fake_get(url: str, timeout: float) -> FakeResponse:
        calls.append(url)
        if "pubchem" in url:
            return FakeResponse({}, status_code=404)
        if "molecule.json" in url:
            return FakeResponse({"molecules": [{"molecule_chembl_id": "CHEMBL1", "pref_name": "TOOL-1"}]})
        if "mechanism.json" in url:
            return FakeResponse({"mechanisms": []})
        if "activity.json" in url:
            return FakeResponse({"activities": [{"target_pref_name": "Kinase X"}]})
        raise AssertionError(url)

    monkeypatch.setattr(annotation.requests, "get", fake_get)
    key = "AAAAAAAAAAAAAA-BBBBBBBBBB-N"
    assert annotation.get_rich_compound_annotation(key) == ("TOOL-1", "Kinase X inhibitor (IC50 < 500nM)")
    n_calls = len(calls)
    annotation.get_rich_compound_annotation(key)
    assert len(calls) == n_calls  # second lookup served from cache


def test_network_failure_returns_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the network is unavailable the lookup returns "Unknown"/"Unannotated" instead of failing."""

    def failing_get(url: str, timeout: float) -> FakeResponse:
        raise annotation.requests.ConnectionError("offline")

    monkeypatch.setattr(annotation.requests, "get", failing_get)
    assert annotation.get_rich_compound_annotation("CCCCCCCCCCCCCC-DDDDDDDDDD-N") == ("Unknown", "Unannotated")
