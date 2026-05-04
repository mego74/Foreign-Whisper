"""POST /api/diarize/{video_id} — speaker diarization (issue fw-lua)."""

import asyncio
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService
from foreign_whispers.diarization import DEFAULT_SPEAKER, assign_speakers

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-y",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )


def _infer_gender_from_pitch(pitch_hz: float | None) -> str | None:
    if pitch_hz is None:
        return None
    if pitch_hz < 165.0:
        return "male"
    if pitch_hz > 185.0:
        return "female"
    return None


def _speaker_profiles(audio_path: Path, diar_segments: list[dict]) -> dict[str, dict]:
    if not diar_segments:
        return {}

    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError:
        return {}

    audio, sr = sf.read(str(audio_path))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for segment in diar_segments:
        grouped[str(segment["speaker"])].append(
            (float(segment["start_s"]), float(segment["end_s"]))
        )

    profiles: dict[str, dict] = {}
    for speaker, intervals in grouped.items():
        snippets = []
        budget_s = 8.0
        for start_s, end_s in intervals:
            if budget_s <= 0:
                break
            start_i = max(0, int(start_s * sr))
            end_i = min(len(audio), int(end_s * sr))
            if end_i <= start_i:
                continue
            max_len = int(budget_s * sr)
            clip = audio[start_i:min(end_i, start_i + max_len)]
            if len(clip) == 0:
                continue
            snippets.append(clip)
            budget_s -= len(clip) / sr

        if not snippets:
            profiles[speaker] = {"gender": None}
            continue

        sample = np.concatenate(snippets).astype(float)
        if len(sample) < int(sr * 0.3):
            profiles[speaker] = {"gender": None}
            continue

        try:
            f0 = librosa.yin(sample, fmin=70, fmax=350, sr=sr)
            voiced = f0[np.isfinite(f0)]
            voiced = voiced[(voiced >= 70) & (voiced <= 350)]
            pitch_hz = float(np.median(voiced)) if voiced.size else None
        except Exception:
            pitch_hz = None

        profiles[speaker] = {
            "gender": _infer_gender_from_pitch(pitch_hz),
            "pitch_hz": round(pitch_hz, 1) if pitch_hz is not None else None,
        }

    return profiles


def _fallback_single_speaker_segments(title: str) -> list[dict]:
    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if not transcript_path.exists():
        return []

    transcript = json.loads(transcript_path.read_text())
    segments = transcript.get("segments", [])
    if not segments:
        return []

    return [
        {
            "start_s": float(segments[0].get("start", 0.0)),
            "end_s": float(segments[-1].get("end", segments[0].get("end", 0.0))),
            "speaker": DEFAULT_SPEAKER,
        }
    ]


def _merge_transcript_speakers(title: str, diar_segments: list[dict]) -> None:
    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if not transcript_path.exists():
        return

    transcript = json.loads(transcript_path.read_text())
    transcript["segments"] = assign_speakers(transcript.get("segments", []), diar_segments)
    transcript_path.write_text(json.dumps(transcript))


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Cache and return speaker segments
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    # Return cached result
    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        _merge_transcript_speakers(title, data.get("segments", []))
        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=data.get("segments", []),
            skipped=True,
        )

    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found for {video_id}")

    audio_path = diar_dir / f"{title}.wav"

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _extract_audio, video_path, audio_path)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise HTTPException(status_code=500, detail=f"ffmpeg audio extraction failed: {detail}") from exc

    diar_segments = await loop.run_in_executor(None, _alignment_service.diarize, str(audio_path))
    if not diar_segments:
        diar_segments = _fallback_single_speaker_segments(title)

    speakers = sorted({segment["speaker"] for segment in diar_segments}) if diar_segments else []
    profiles = _speaker_profiles(audio_path, diar_segments)
    result = {
        "speakers": speakers,
        "segments": diar_segments,
        "speaker_profiles": profiles,
    }
    diar_path.write_text(json.dumps(result))
    _merge_transcript_speakers(title, diar_segments)

    return DiarizeResponse(video_id=video_id, speakers=speakers, segments=diar_segments)
