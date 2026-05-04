"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService
from foreign_whispers.voice_resolution import build_speaker_voice_map

router = APIRouter(prefix="/api")


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


def _load_translation(title: str) -> dict:
    source_path = settings.translations_dir / f"{title}.json"
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Translated transcript not found")
    return json.loads(source_path.read_text())


def _load_speaker_profiles(title: str) -> dict[str, dict]:
    diar_path = settings.diarizations_dir / f"{title}.json"
    if not diar_path.exists():
        return {}
    data = json.loads(diar_path.read_text())
    return data.get("speaker_profiles", {})


def _build_voice_map(title: str, target_language: str) -> dict[str, str] | None:
    translated = _load_translation(title)
    speaker_ids = sorted(
        {
            str(segment["speaker"])
            for segment in translated.get("segments", [])
            if segment.get("speaker")
        }
    )
    if not speaker_ids:
        return None

    return build_speaker_voice_map(
        settings.speakers_dir,
        target_language,
        speaker_ids,
        _load_speaker_profiles(title),
    )


def _translation_has_speakers(translated: dict) -> bool:
    return any(segment.get("speaker") for segment in translated.get("segments", []))


def _tts_cache_is_fresh(
    title: str,
    config: str,
    translated: dict,
    voice_map: dict[str, str] | None,
) -> bool:
    wav_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not wav_path.exists():
        return False

    src_path = settings.translations_dir / f"{title}.json"
    if src_path.exists() and wav_path.stat().st_mtime < src_path.stat().st_mtime:
        return False

    if not _translation_has_speakers(translated):
        return True

    report_path = settings.tts_audio_dir / config / f"{title}.align.json"
    if not report_path.exists():
        return False

    report = json.loads(report_path.read_text())
    report_segments = report.get("segments", [])
    if not report_segments or not any(segment.get("speaker") for segment in report_segments):
        return False

    if voice_map and not any(segment.get("speaker_wav") for segment in report_segments):
        return False

    return True


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    speaker_wav: str | None = Query(default=None),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"
    source_json_path = trans_dir / f"{title}.json"
    if wav_path.exists() and not source_json_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    translated = _load_translation(title)
    source_path = str(source_json_path)
    target_language = translated.get("language", "es")
    voice_map = _build_voice_map(title, target_language)

    if _tts_cache_is_fresh(title, config, translated, voice_map):
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    await _run_in_threadpool(
        None,
        svc.text_file_to_speech,
        source_path,
        str(audio_dir),
        alignment=alignment,
        speaker_wav=speaker_wav,
        voice_map=voice_map,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")
