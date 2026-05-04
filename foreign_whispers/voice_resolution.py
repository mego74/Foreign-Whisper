"""Voice resolution for Chatterbox speaker cloning.

Resolves which reference WAV to use for a given target language
and optional speaker ID. The Chatterbox container expects a filename
relative to its /app/voices/ mount point.
"""

from pathlib import Path
from typing import Iterable


def resolve_speaker_wav(
    speakers_dir: Path,
    target_language: str,
    speaker_id: str | None = None,
    gender: str | None = None,
) -> str:
    """Resolve the reference WAV path for voice cloning.

    Resolution order:
    1. speakers/{lang}/{speaker_id}.wav  (if speaker_id given and file exists)
    2. speakers/{lang}/{gender}.wav      (if gender given and file exists)
    3. speakers/{lang}/default.wav       (language-specific default)
    4. speakers/default.wav              (global fallback)

    Args:
        speakers_dir: Absolute path to the speakers directory.
        target_language: Language code (e.g. "es", "fr").
        speaker_id: Optional speaker identifier (e.g. "SPEAKER_00").
        gender: Optional coarse gender tag ("male" or "female").

    Returns:
        Relative path string for the Chatterbox container (e.g. "es/default.wav").
    """
    lang = target_language.strip().lower()

    candidates: list[tuple[Path, str]] = []
    if speaker_id:
        candidates.append(
            (speakers_dir / lang / f"{speaker_id}.wav", f"{lang}/{speaker_id}.wav")
        )
    if gender:
        normalized_gender = gender.strip().lower()
        candidates.append(
            (speakers_dir / lang / f"{normalized_gender}.wav", f"{lang}/{normalized_gender}.wav")
        )
    candidates.append((speakers_dir / lang / "default.wav", f"{lang}/default.wav"))
    candidates.append((speakers_dir / "default.wav", "default.wav"))

    for abs_path, rel_path in candidates:
        if abs_path.exists():
            return rel_path

    # Final fallback keeps the interface stable even if the file is added later.
    return "default.wav"


def build_speaker_voice_map(
    speakers_dir: Path,
    target_language: str,
    speaker_ids: Iterable[str],
    speaker_profiles: dict[str, dict] | None = None,
) -> dict[str, str]:
    """Resolve one reference WAV per speaker label.

    Uses speaker-specific WAVs when present, otherwise falls back to
    gender-aware language defaults and finally the global default.
    """
    speaker_profiles = speaker_profiles or {}
    voice_map: dict[str, str] = {}

    for speaker_id in sorted(set(speaker_ids)):
        profile = speaker_profiles.get(speaker_id, {})
        voice_map[speaker_id] = resolve_speaker_wav(
            speakers_dir,
            target_language,
            speaker_id=speaker_id,
            gender=profile.get("gender"),
        )

    return voice_map
