"""Tests for POST /api/diarize/{video_id} endpoint."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def ui_dir(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "transcriptions" / "whisper").mkdir(parents=True)
    (tmp_path / "diarizations").mkdir()
    return tmp_path


@pytest.fixture()
def client(monkeypatch, ui_dir):
    monkeypatch.setattr("whisper.load_model", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("TTS.api.TTS", lambda *a, **kw: MagicMock())

    from api.src.core.config import settings

    monkeypatch.setattr(settings, "data_dir", ui_dir)
    monkeypatch.setattr(settings, "ui_dir", ui_dir)

    from api.src.main import app

    with TestClient(app) as c:
        yield c


def test_diarize_endpoint_runs_and_merges_transcript(client, monkeypatch, ui_dir):
    (ui_dir / "videos" / "Test Title.mp4").write_bytes(b"fake-video")
    transcript = {
        "text": "Hello world",
        "language": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello"},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "world"},
        ],
    }
    (ui_dir / "transcriptions" / "whisper" / "Test Title.json").write_text(json.dumps(transcript))

    monkeypatch.setattr("api.src.routers.diarize.resolve_title", lambda video_id: "Test Title")
    monkeypatch.setattr("api.src.routers.diarize._extract_audio", lambda video_path, audio_path: audio_path.write_bytes(b"wav"))
    monkeypatch.setattr(
        "api.src.routers.diarize._speaker_profiles",
        lambda audio_path, diar_segments: {"SPEAKER_00": {"gender": "male"}},
    )
    monkeypatch.setattr(
        "api.src.routers.diarize._alignment_service.diarize",
        lambda audio_path: [{"start_s": 0.0, "end_s": 4.0, "speaker": "SPEAKER_00"}],
    )

    resp = client.post("/api/diarize/G3Eup4mfJdA")
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "G3Eup4mfJdA"
    assert body["speakers"] == ["SPEAKER_00"]
    assert body["segments"][0]["speaker"] == "SPEAKER_00"

    saved = json.loads((ui_dir / "diarizations" / "Test Title.json").read_text())
    assert saved["speaker_profiles"]["SPEAKER_00"]["gender"] == "male"

    labeled = json.loads((ui_dir / "transcriptions" / "whisper" / "Test Title.json").read_text())
    assert labeled["segments"][0]["speaker"] == "SPEAKER_00"
    assert labeled["segments"][1]["speaker"] == "SPEAKER_00"


def test_diarize_endpoint_uses_cache_and_backfills_transcript(client, monkeypatch, ui_dir):
    transcript = {
        "text": "Hello world",
        "language": "en",
        "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "Hello"}],
    }
    (ui_dir / "transcriptions" / "whisper" / "Test Title.json").write_text(json.dumps(transcript))
    (ui_dir / "diarizations" / "Test Title.json").write_text(
        json.dumps(
            {
                "speakers": ["SPEAKER_01"],
                "segments": [{"start_s": 0.0, "end_s": 2.0, "speaker": "SPEAKER_01"}],
            }
        )
    )

    monkeypatch.setattr("api.src.routers.diarize.resolve_title", lambda video_id: "Test Title")

    resp = client.post("/api/diarize/G3Eup4mfJdA")
    assert resp.status_code == 200
    assert resp.json()["skipped"] is True

    labeled = json.loads((ui_dir / "transcriptions" / "whisper" / "Test Title.json").read_text())
    assert labeled["segments"][0]["speaker"] == "SPEAKER_01"


def test_diarize_endpoint_returns_404_for_unknown_video(client, monkeypatch):
    monkeypatch.setattr("api.src.routers.diarize.resolve_title", lambda video_id: None)

    resp = client.post("/api/diarize/NONEXISTENT")
    assert resp.status_code == 404
