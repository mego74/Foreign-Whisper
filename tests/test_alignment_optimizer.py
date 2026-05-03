from foreign_whispers.alignment import compute_segment_metrics, global_align, global_align_dp


def test_global_align_dp_returns_same_number_of_segments():
    en = {"segments": [{"start": 0.0, "end": 3.0, "text": "Hello world"}]}
    es = {"segments": [{"start": 0.0, "end": 3.0, "text": "Hola mundo"}]}

    metrics = compute_segment_metrics(en, es)
    aligned = global_align_dp(metrics, silence_regions=[])

    assert len(aligned) == len(metrics)


def test_global_align_dp_can_trade_gap_shift_for_stretch():
    en = {"segments": [{"start": 0.0, "end": 1.5, "text": "Hello"}]}
    es = {"segments": [{"start": 0.0, "end": 1.5, "text": "ba" * 10}]}
    silence = [{"start_s": 1.5, "end_s": 2.5, "label": "silence"}]

    metrics = compute_segment_metrics(en, es)
    greedy = global_align(metrics, silence_regions=silence)
    improved = global_align_dp(metrics, silence_regions=silence)

    assert improved[0].gap_shift_s <= greedy[0].gap_shift_s
    assert improved[0].scheduled_end <= greedy[0].scheduled_end
