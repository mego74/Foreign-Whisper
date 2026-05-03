"""HTTP-agnostic service wrapping Whisper transcription."""

import re
import json
import pathlib
from pathlib import Path
from typing import Any


class TranscriptionService:
    """Thin wrapper around the Whisper model for transcription.

    Accepts *ui_dir* and a pre-loaded *whisper_model* via constructor injection.
    """

    def __init__(self, ui_dir: Path, whisper_model: Any) -> None:
        self.ui_dir = ui_dir
        self.whisper_model = whisper_model

    def transcribe(self, video_path: str) -> dict:
        """Run Whisper transcription on a video file and return the result dict."""
        result = self.whisper_model.transcribe(video_path)
        return self._split_long_segments(result)

    @staticmethod
    def _split_long_segments(
        result: dict,
        *,
        max_duration_s: float = 2.8,
        max_words: int = 5,
    ) -> dict:
        """Split overly long Whisper segments into smaller timed phrases.

        Whisper sometimes emits 5-8 second segments that span multiple spoken
        phrases. That is acceptable for subtitles, but it creates audible dead
        air in dubbing because TTS speaks the phrase early and then waits out
        the rest of the segment window. This post-processing step keeps the
        original total timing while subdividing long segments into shorter
        phrase windows.
        """
        segments = result.get("segments", [])
        if not segments:
            return result

        split_segments: list[dict] = []

        def _chunk_text(text: str, duration: float) -> list[str]:
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                return []

            target_parts = max(1, int(duration / max_duration_s + 0.999))
            dynamic_max_words = max(4, min(max_words, int(len(text.split()) / target_parts + 0.999)))

            clauses = [
                clause.strip()
                for clause in re.split(r"(?<=[,.;:!?])\s+", text)
                if clause.strip()
            ]

            chunks: list[str] = []
            for clause in clauses:
                words = clause.split()
                if len(words) <= dynamic_max_words:
                    chunks.append(clause)
                    continue
                for i in range(0, len(words), dynamic_max_words):
                    chunks.append(" ".join(words[i : i + dynamic_max_words]))
            return chunks or [text]

        next_id = 0
        for seg in segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = str(seg.get("text", "")).strip()
            duration = max(end - start, 0.0)

            parts = _chunk_text(text, duration)
            if (
                len(parts) <= 1
                and duration <= max_duration_s
                and len(text.split()) <= max_words
            ):
                split_segments.append({**seg, "id": next_id, "text": text})
                next_id += 1
                continue

            weights = [max(len(part.split()), 1) for part in parts]
            total_weight = sum(weights) or 1
            cursor = start

            for idx, (part, weight) in enumerate(zip(parts, weights)):
                if idx == len(parts) - 1:
                    part_end = end
                else:
                    part_duration = duration * (weight / total_weight)
                    part_end = min(end, cursor + part_duration)

                split_segments.append(
                    {
                        **seg,
                        "id": next_id,
                        "start": cursor,
                        "end": part_end,
                        "text": part,
                    }
                )
                next_id += 1
                cursor = part_end

        return {
            **result,
            "text": " ".join(seg["text"] for seg in split_segments).strip(),
            "segments": split_segments,
        }

    @staticmethod
    def title_for_video_id(video_id: str, search_dir: pathlib.Path) -> str | None:
        """Find a title by scanning *search_dir* for matching files.

        Returns the stem (title) of the first match, or None.
        """
        for f in search_dir.glob("*.mp4"):
            return f.stem
        return None
