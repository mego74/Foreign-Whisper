"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    align_report: dict | None = None,
) -> dict:
    """Summarize clip quality across multiple normalized dimensions.

    The scorecard is intentionally heuristic and dependency-free so it can run
    during notebook iteration and tests without extra models.
    """
    if not metrics:
        return {
            "timing_accuracy": 0.0,
            "intelligibility": 0.0,
            "semantic_fidelity": 0.0,
            "naturalness": 0.0,
            "overall": 0.0,
        }

    report = align_report or clip_evaluation_report(metrics, aligned)
    n_segments = max(len(metrics), 1)
    failed_or_retried = sum(
        1 for segment in aligned
        if segment.action in {AlignAction.REQUEST_SHORTER, AlignAction.FAIL}
    )

    mean_err = float(report.get("mean_abs_duration_error_s", 0.0))
    severe_pct = float(report.get("pct_severe_stretch", 0.0))
    drift = abs(float(report.get("total_cumulative_drift_s", 0.0)))

    timing_accuracy = max(
        0.0,
        1.0 - min(1.0, (mean_err / 1.5) * 0.5 + (severe_pct / 100.0) * 0.3 + (drift / 5.0) * 0.2),
    )

    intelligibility = max(
        0.0,
        1.0 - min(1.0, (severe_pct / 100.0) * 0.6 + (failed_or_retried / n_segments) * 0.4),
    )

    semantic_fidelity = max(
        0.0,
        1.0 - min(1.0, failed_or_retried / n_segments),
    )

    speed_values = []
    if isinstance(align_report, dict):
        for segment in align_report.get("segments", []):
            speed = segment.get("speed_factor")
            if isinstance(speed, (int, float)):
                speed_values.append(float(speed))
    if not speed_values:
        speed_values = [segment.stretch_factor for segment in aligned]

    if len(speed_values) > 1:
        speed_variance = _stats.pvariance(speed_values)
    else:
        speed_variance = 0.0
    naturalness = max(0.0, 1.0 - min(1.0, speed_variance / 0.25))

    overall = round(
        (timing_accuracy + intelligibility + semantic_fidelity + naturalness) / 4.0,
        3,
    )
    return {
        "timing_accuracy": round(timing_accuracy, 3),
        "intelligibility": round(intelligibility, 3),
        "semantic_fidelity": round(semantic_fidelity, 3),
        "naturalness": round(naturalness, 3),
        "overall": overall,
    }
