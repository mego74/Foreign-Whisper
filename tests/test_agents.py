# tests/test_agents.py — renamed module is now foreign_whispers.reranking
from foreign_whispers.reranking import (
    get_shorter_translations,
    analyze_failures,
    TranslationCandidate,
    FailureAnalysis,
)


def test_get_shorter_returns_empty_stub():
    """Long segments should produce shorter candidates, shortest first."""
    result = get_shorter_translations(
        "Right now we have to do it this way.",
        "En este momento tenemos que hacerlo de esta manera.",
        2.0,
    )
    assert result
    assert isinstance(result[0], TranslationCandidate)
    assert result == sorted(result, key=lambda candidate: candidate.char_count)
    assert result[0].char_count < len("En este momento tenemos que hacerlo de esta manera.")
    assert any("ahora" in candidate.text for candidate in result)


def test_get_shorter_keeps_fitting_baseline_available():
    result = get_shorter_translations("hello", "hola", 2.0)
    assert result
    assert any(candidate.text == "hola" for candidate in result)


def test_analyze_failures_returns_dataclass():
    result = analyze_failures({"mean_abs_duration_error_s": 0.5})
    assert isinstance(result, FailureAnalysis)
    assert result.failure_category == "ok"


def test_analyze_failures_detects_overflow():
    result = analyze_failures({"pct_severe_stretch": 30})
    assert result.failure_category == "duration_overflow"


def test_analyze_failures_detects_drift():
    result = analyze_failures({"total_cumulative_drift_s": 5.0})
    assert result.failure_category == "cumulative_drift"


def test_analyze_failures_detects_stretch_quality():
    result = analyze_failures({"mean_abs_duration_error_s": 1.2})
    assert result.failure_category == "stretch_quality"
