from foreign_whispers.alignment import compute_segment_metrics, global_align
from foreign_whispers.evaluation import clip_evaluation_report, dubbing_scorecard


def test_dubbing_scorecard_keys_and_range():
    en = {"segments": [{"start": 0.0, "end": 3.0, "text": "Hello world"}]}
    es = {"segments": [{"start": 0.0, "end": 3.0, "text": "Hola mundo"}]}

    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    scorecard = dubbing_scorecard(
        metrics,
        aligned,
        {"segments": [{"speed_factor": 1.0}], **report},
    )

    assert set(scorecard.keys()) == {
        "timing_accuracy",
        "intelligibility",
        "semantic_fidelity",
        "naturalness",
        "overall",
    }
    assert all(0.0 <= value <= 1.0 for value in scorecard.values())


def test_dubbing_scorecard_penalizes_harder_clips():
    easy_en = {"segments": [{"start": 0.0, "end": 4.0, "text": "Hello"}]}
    easy_es = {"segments": [{"start": 0.0, "end": 4.0, "text": "Hola"}]}
    hard_en = {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}]}
    hard_es = {"segments": [{"start": 0.0, "end": 1.0, "text": "ba" * 12}]}

    easy_metrics = compute_segment_metrics(easy_en, easy_es)
    easy_aligned = global_align(easy_metrics, silence_regions=[])
    hard_metrics = compute_segment_metrics(hard_en, hard_es)
    hard_aligned = global_align(hard_metrics, silence_regions=[])

    easy_score = dubbing_scorecard(easy_metrics, easy_aligned)
    hard_score = dubbing_scorecard(hard_metrics, hard_aligned)

    assert easy_score["overall"] > hard_score["overall"]
