import asyncio
import logging as _logging
import os
import pathlib
import json
import glob
import shutil
import subprocess
import tempfile
import re

import requests
import librosa
import soundfile as sf
import pyrubberband
from pydub import AudioSegment

# ── Chatterbox API configuration ─────────────────────────────────────
CHATTERBOX_API_URL = os.getenv("CHATTERBOX_API_URL", "http://localhost:8020")
# Path to the default speaker reference WAV, relative to pipeline_data/speakers/
CHATTERBOX_SPEAKER_WAV = os.getenv("CHATTERBOX_SPEAKER_WAV", "")

# Set FW_ALIGNMENT=off to use the pre-alignment baseline (legacy unclamped stretch).
# Default is "on" (new clamped path). Useful for A/B comparisons.
_ALIGNMENT_ENABLED = os.getenv("FW_ALIGNMENT", "on").lower() != "off"

SPEED_MIN = 1.0
SPEED_MAX = 1.25
_SPEED_MIN_LEGACY = 0.1
_SPEED_MAX_LEGACY = 10.0
_TRIM_TOLERANCE_MS = 60


class SyncedSegmentAudio(AudioSegment):
    """AudioSegment with unpackable timing metadata for backwards compatibility."""

    speed_factor: float
    raw_duration_s: float

    def __iter__(self):
        yield self
        yield self.speed_factor
        yield self.raw_duration_s


def _attach_sync_metadata(
    segment_audio: AudioSegment,
    speed_factor: float,
    raw_duration_s: float,
) -> SyncedSegmentAudio:
    segment_audio.__class__ = SyncedSegmentAudio
    segment_audio.speed_factor = speed_factor
    segment_audio.raw_duration_s = raw_duration_s
    return segment_audio


class ChatterboxClient:
    """Thin HTTP client for the Chatterbox TTS API server (OpenAI-compatible).

    Uses /v1/audio/speech for default voice and /v1/audio/speech/upload
    when a speaker reference WAV is provided for voice cloning.
    """

    def __init__(self, base_url: str = CHATTERBOX_API_URL,
                 speaker_wav: str = CHATTERBOX_SPEAKER_WAV):
        self.base_url = base_url.rstrip("/")
        self.speaker_wav = speaker_wav  # path relative to pipeline_data/speakers/

    def tts_to_file(self, text: str, file_path: str, **kwargs) -> None:
        """Synthesize *text* via the Chatterbox API and save the WAV to *file_path*.

        If *speaker_wav* is provided (via kwarg or constructor), uses the
        /v1/audio/speech/upload endpoint with the reference WAV for voice cloning.
        Otherwise uses /v1/audio/speech with the server's default voice.
        """
        chunks = self._split_text(text) if len(text) > 200 else [text]
        wav_parts = []

        speaker_wav = kwargs.get("speaker_wav", self.speaker_wav)

        for chunk in chunks:
            if speaker_wav:
                # Voice cloning: upload the reference WAV
                wav_parts.append(self._synthesize_with_voice(chunk, speaker_wav))
            else:
                # Default voice
                wav_parts.append(self._synthesize_default(chunk))

        if len(wav_parts) == 1:
            pathlib.Path(file_path).write_bytes(wav_parts[0])
        else:
            combined = AudioSegment.empty()
            for part in wav_parts:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    tmp.write(part)
                    tmp.flush()
                    combined += AudioSegment.from_wav(tmp.name)
            combined.export(file_path, format="wav")

    def _synthesize_default(self, text: str) -> bytes:
        """Call /v1/audio/speech with the server's default voice."""
        resp = requests.post(
            f"{self.base_url}/v1/audio/speech",
            json={"input": text, "response_format": "wav"},
            timeout=(5, 60),
        )
        resp.raise_for_status()
        return resp.content

    def _synthesize_with_voice(self, text: str, speaker_wav: str) -> bytes:
        """Call /v1/audio/speech/upload with a reference WAV for voice cloning."""
        # Resolve the speaker WAV path — could be relative to speakers dir
        speakers_base = pathlib.Path(__file__).parent.parent.parent.parent / "pipeline_data" / "speakers"
        wav_path = speakers_base / speaker_wav
        if not wav_path.exists():
            # Try as absolute path
            wav_path = pathlib.Path(speaker_wav)
        if not wav_path.exists():
            _logging.getLogger(__name__).warning(
                "[tts] Speaker WAV %s not found, falling back to default voice", speaker_wav
            )
            return self._synthesize_default(text)

        with open(wav_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/v1/audio/speech/upload",
                data={"input": text, "response_format": "wav"},
                files={"voice_file": (wav_path.name, f, "audio/wav")},
                timeout=(5, 60),
            )
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def _split_text(text: str, max_len: int = 200) -> list[str]:
        """Split text at sentence boundaries to stay under max_len chars."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for s in sentences:
            if current and len(current) + len(s) + 1 > max_len:
                chunks.append(current.strip())
                current = s
            else:
                current = f"{current} {s}".strip() if current else s
        if current:
            chunks.append(current.strip())
        return chunks if chunks else [text]


class MacSayClient:
    """Local macOS TTS fallback using the built-in `say` command."""

    def __init__(self, voice: str | None = None):
        self.voice = voice or os.getenv("FW_SAY_VOICE", "Monica")

    @staticmethod
    def _voice_for_speaker_wav(speaker_wav: str | None, fallback_voice: str) -> str:
        if not speaker_wav:
            return fallback_voice

        normalized = speaker_wav.replace("\\", "/").lower()
        if normalized.endswith("/male.wav") or "/male." in normalized:
            return os.getenv("FW_SAY_VOICE_MALE", "Eddy (Spanish (Mexico))")
        if normalized.endswith("/female.wav") or "/female." in normalized:
            return os.getenv("FW_SAY_VOICE_FEMALE", "Flo (Spanish (Mexico))")
        return fallback_voice

    def tts_to_file(self, text: str, file_path: str, **kwargs) -> None:
        if not text or not text.strip():
            AudioSegment.silent(duration=250).export(file_path, format="wav")
            return

        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)

        try:
            voice = self._voice_for_speaker_wav(kwargs.get("speaker_wav"), self.voice)
            cmd = ["say", "-v", voice, "-o", str(tmp_path), text]
            subprocess.run(cmd, check=True, capture_output=True)
            audio, sr = sf.read(str(tmp_path))
            sf.write(file_path, audio, sr)
        finally:
            tmp_path.unlink(missing_ok=True)


class SilentTTSEngine:
    """Last-resort fallback that writes silence instead of crashing."""

    def tts_to_file(self, text: str, file_path: str, **kwargs) -> None:
        duration_ms = max(300, int(max(len(text.strip()), 1) / 12.0 * 1000))
        AudioSegment.silent(duration=duration_ms).export(file_path, format="wav")


def _make_tts_engine():
    """Create TTS engine: Chatterbox API client if server is reachable, else local Coqui.

    Tries Chatterbox with a real /v1/audio/speech test call
    to ensure the model is fully loaded before committing.
    """
    try:
        client = ChatterboxClient()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            client.tts_to_file(text="prueba", file_path=tmp.name)
        print(f"[tts] Using Chatterbox GPU server at {CHATTERBOX_API_URL}")
        return client
    except Exception as exc:
        print(f"[tts] Chatterbox not available ({exc}), falling back to local Coqui")

    # Fallback: local Coqui TTS (for dev/test without Docker)
    try:
        import functools
        import torch
        from TTS.api import TTS as CoquiTTS

        cache_root = (
            pathlib.Path(__file__).resolve().parent.parent.parent.parent
            / "pipeline_data"
            / ".cache"
            / "tts"
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TTS_HOME", str(cache_root))
        os.environ.setdefault("XDG_DATA_HOME", str(cache_root.parent))

        # Coqui TTS checkpoints contain classes (RAdam, defaultdict, etc.) that
        # PyTorch 2.6+ rejects with weights_only=True. Monkey-patch torch.load
        # to default to weights_only=False for these trusted model files.
        _original_torch_load = torch.load

        @functools.wraps(_original_torch_load)
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_load
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[tts] Using local Coqui TTS on {device}")
        return CoquiTTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=False).to(device)
    except Exception as exc:
        print(f"[tts] Coqui not available ({exc}), falling back to macOS say")

    if shutil.which("say"):
        print("[tts] Using macOS say fallback")
        return MacSayClient()

    print("[tts] No local TTS backend available, using silent fallback")
    return SilentTTSEngine()


_tts_engine = None
tts = None


def _get_tts_engine():
    """Lazy singleton — resolved on first call, not at import time."""
    global _tts_engine, tts
    if _tts_engine is None:
        _tts_engine = _make_tts_engine()
        tts = _tts_engine
    return _tts_engine


def text_from_file(file_path) -> str:
    with open(file_path, 'r') as file:
        trans = json.load(file)
    return trans["text"]


def segments_from_file(file_path) -> list[dict]:
    """Load segments with start/end timestamps from a translated JSON file."""
    with open(file_path, 'r') as file:
        trans = json.load(file)
    return trans.get("segments", [])


def files_from_dir(dir_path) -> list:
    SUFFIX = ".json"
    pth = pathlib.Path(dir_path)
    if not pth.exists():
        raise ValueError("provided path does not exist")

    es_files = glob.glob(str(pth) + "/*.json")

    if not es_files:
        raise ValueError(f"no {SUFFIX} files found in {pth}")

    return es_files


def _synthesize_raw(
    tts_engine,
    text: str,
    wav_path: str,
    *,
    speaker_wav: str | None = None,
) -> bytes | None:
    """GPU-bound: call TTS engine and return raw WAV bytes, or None on failure."""
    if tts_engine is None or not text or not text.strip():
        return None

    def _candidate_texts(raw_text: str) -> list[str]:
        candidates: list[str] = []

        def _add(value: str) -> None:
            cleaned = re.sub(r"\s+", " ", value).strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

        _add(raw_text)

        normalized = raw_text.replace("¿", "").replace("¡", "")
        normalized = re.sub(r"^[>\-\s]+", "", normalized)
        normalized = re.sub(r"\.{2,}", ".", normalized)
        normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
        normalized = normalized.strip(" \"'")
        _add(normalized)
        _add(normalized.strip(".,;:!?"))

        return candidates

    kwargs = {}
    if speaker_wav is not None:
        kwargs["speaker_wav"] = speaker_wav

    last_exc: Exception | None = None
    candidates = _candidate_texts(text)

    # Local Coqui fallback is effectively single-speaker on this Mac path.
    # When we have diarization-driven voice hints, prefer macOS say so male
    # and female speakers remain audibly distinct even without Chatterbox.
    if speaker_wav and not isinstance(tts_engine, (ChatterboxClient, MacSayClient)) and shutil.which("say"):
        backup_engine = MacSayClient()
        for candidate in candidates:
            try:
                backup_engine.tts_to_file(
                    text=candidate,
                    file_path=wav_path,
                    speaker_wav=speaker_wav,
                )
                print("[tts] Using macOS say for speaker-aware local fallback")
                return pathlib.Path(wav_path).read_bytes()
            except Exception as exc:
                last_exc = exc

    for candidate in candidates:
        try:
            tts_engine.tts_to_file(text=candidate, file_path=wav_path, **kwargs)
            return pathlib.Path(wav_path).read_bytes()
        except Exception as exc:
            last_exc = exc

    if not isinstance(tts_engine, MacSayClient) and shutil.which("say"):
        backup_engine = MacSayClient()
        for candidate in candidates:
            try:
                backup_engine.tts_to_file(text=candidate, file_path=wav_path)
                print("[tts] Falling back to macOS say for one segment")
                return pathlib.Path(wav_path).read_bytes()
            except Exception as exc:
                last_exc = exc

    print(f"[tts] TTS failed for segment ({last_exc}), using silence")
    return None


def _time_stretch_audio(y, sr: int, speed_factor: float):
    """Stretch audio with rubberband when available, otherwise librosa."""
    try:
        return pyrubberband.time_stretch(y, sr, speed_factor)
    except (FileNotFoundError, OSError, RuntimeError, subprocess.CalledProcessError):
        return librosa.effects.time_stretch(y, rate=speed_factor)


def _postprocess_segment(raw_wav_bytes: bytes | None, target_sec: float,
                         stretch_factor: float, alignment_enabled: bool,
                         work_dir: str) -> tuple:
    """CPU-bound: time-stretch raw TTS audio to match target duration.

    Returns (AudioSegment | None, speed_factor, raw_duration_s).
    """
    if target_sec <= 0:
        return (None, 0.0, 0.0)

    target_ms = int(target_sec * 1000)

    if raw_wav_bytes is None:
        return (AudioSegment.silent(duration=target_ms), 1.0, 0.0)

    work_path = pathlib.Path(work_dir)
    raw_wav = work_path / "raw_segment.wav"
    raw_wav.write_bytes(raw_wav_bytes)

    y, sr = librosa.load(str(raw_wav), sr=None)
    raw_duration = len(y) / sr

    if raw_duration == 0:
        return (AudioSegment.silent(duration=target_ms), 1.0, 0.0)

    duration_ratio = raw_duration / target_sec

    if not alignment_enabled:
        # Baseline mode should preserve the raw voice pacing. Any remaining
        # difference from the source window is handled by silence padding or
        # downstream drift rather than slowing or clipping the speaker.
        speed_factor = 1.0
    else:
        effective_target = target_sec * max(stretch_factor, 0.1)
        # In aligned mode, preserve natural speech when the synthesized segment
        # already fits inside the available window. Padding with silence sounds
        # better than artificially slowing the voice to fill every pause.
        speed_factor = raw_duration / effective_target
        if speed_factor < 1.0:
            speed_factor = 1.0
        speed_factor = max(SPEED_MIN, min(SPEED_MAX, speed_factor))

    if abs(speed_factor - 1.0) > 0.01:
        y_stretched = _time_stretch_audio(y, sr, speed_factor)
    else:
        y_stretched = y

    stretched_wav = work_path / "stretched_segment.wav"
    sf.write(str(stretched_wav), y_stretched, sr)

    segment_audio = AudioSegment.from_wav(str(stretched_wav))

    if len(segment_audio) < target_ms:
        segment_audio += AudioSegment.silent(duration=target_ms - len(segment_audio))
    elif len(segment_audio) > target_ms:
        overflow_ms = len(segment_audio) - target_ms
        if not alignment_enabled:
            # Baseline mode keeps the full utterance even if it drifts.
            pass
        elif overflow_ms <= _TRIM_TOLERANCE_MS:
            # Minor resampling jitter can safely be cropped.
            segment_audio = segment_audio[:target_ms]
        elif speed_factor < SPEED_MAX - 1e-6:
            # We expected this segment to fit the aligned window; crop only when
            # we still had playback headroom and the excess is likely incidental.
            segment_audio = segment_audio[:target_ms]
        else:
            # We hit the safe speed ceiling and still overflowed — keep the full
            # utterance instead of cutting words off.
            pass

    return (segment_audio, speed_factor, raw_duration)


def _synced_segment_audio(
    tts_engine,
    text: str,
    target_sec: float,
    work_dir,
    stretch_factor: float = 1.0,
    alignment_enabled: bool | None = None,
    speaker_wav: str | None = None,
):
    """Generate TTS audio for *text* and time-stretch it to *target_sec*.

    Convenience wrapper kept for callers that don't use the batch path.
    """
    if target_sec <= 0:
        return None
    if alignment_enabled is None:
        alignment_enabled = _ALIGNMENT_ENABLED
    raw_wav = str(pathlib.Path(work_dir) / "raw_segment.wav")
    raw_bytes = _synthesize_raw(tts_engine, text, raw_wav, speaker_wav=speaker_wav)
    segment_audio, speed_factor, raw_duration = _postprocess_segment(
        raw_bytes,
        target_sec,
        stretch_factor,
        alignment_enabled,
        str(work_dir),
    )
    if segment_audio is None:
        return None
    return _attach_sync_metadata(segment_audio, speed_factor, raw_duration)


def text_to_speech(text, output_file_path):
    _get_tts_engine().tts_to_file(text=text, file_path=str(output_file_path))


def _load_en_transcript(es_source_path: str) -> dict:
    """Locate the source-language transcript that corresponds to the translated file.

    Convention: translated JSON lives at .../translations/{model}/<title>.json
    Source transcript lives at .../transcriptions/{model}/<title>.json
    Returns an empty dict (no segments) if the source file is not found.
    """
    es_path = pathlib.Path(es_source_path)
    # Navigate: translations/{model}/ → data_dir → transcriptions/whisper/
    data_dir = es_path.parent.parent.parent
    en_path = data_dir / "transcriptions" / "whisper" / es_path.name
    if not en_path.exists():
        print(f"[tts] EN transcript not found at {en_path}, alignment skipped")
        return {}
    with open(en_path) as f:
        return json.load(f)


def _build_alignment(en_transcript: dict, es_transcript: dict) -> tuple:
    """Run global_align and return (metrics_list, {segment_index: AlignedSegment}).

    Returns ([], {}) if the alignment library is unavailable or fails.
    """
    try:
        from foreign_whispers.alignment import compute_segment_metrics, global_align
    except ImportError:
        return [], {}
    try:
        metrics = compute_segment_metrics(en_transcript, es_transcript)
        aligned = global_align(metrics, silence_regions=[])
        return metrics, {seg.index: seg for seg in aligned}
    except Exception as exc:
        print(f"[tts] alignment failed ({exc}), proceeding without alignment")
        return [], {}


def _shorten_segment_text(en_text: str, es_text: str, target_sec: float) -> str:
    """Try to shorten a Spanish translation to fit *target_sec*.

    Delegates to ``get_shorter_translations()`` (student assignment stub).
    Returns the original *es_text* if no shorter candidate is available.
    """
    try:
        from foreign_whispers.reranking import get_shorter_translations
        candidates = get_shorter_translations(
            source_text=en_text,
            baseline_es=es_text,
            target_duration_s=target_sec,
        )
        if candidates:
            return candidates[0].text
    except Exception as exc:
        _logging.getLogger(__name__).warning("[tts] rerank failed: %s", exc)
    return es_text


def _write_align_report(
    output_path: str,
    stem: str,
    metrics: list,
    aligned: list,
    segment_details: list,
    *,
    alignment_enabled: bool,
) -> None:
    """Write a {stem}.align.json sidecar with evaluation metrics and per-segment detail.

    segment_details is a list of dicts: [{raw_duration_s, speed_factor, action, text}, ...]
    Written next to the WAV so both baseline and aligned runs produce comparable files.
    """
    try:
        from foreign_whispers.evaluation import clip_evaluation_report
        summary = clip_evaluation_report(metrics, aligned)
    except Exception as exc:
        _logging.getLogger(__name__).warning("clip_evaluation_report failed: %s", exc)
        summary = {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch": 0.0,
            "n_gap_shifts": 0,
            "n_translation_retries": 0,
            "total_cumulative_drift_s": 0.0,
        }

    report = {**summary, "alignment_enabled": alignment_enabled, "segments": segment_details}
    sidecar_path = pathlib.Path(output_path) / f"{stem}.align.json"
    sidecar_path.write_text(json.dumps(report, indent=2))


def _compute_speech_offset(source_path: str) -> float:
    """Compute timing offset between YouTube captions and Whisper segments.

    Returns seconds to add to Whisper timestamps so TTS audio aligns with
    the actual speech start in the original video.
    """
    title = pathlib.Path(source_path).stem
    # source_path: .../translations/{model}/{title}.json → data_dir is 3 levels up
    base_dir = pathlib.Path(source_path).parent.parent.parent

    yt_path = base_dir / "youtube_captions" / f"{title}.txt"
    whisper_path = base_dir / "transcriptions" / "whisper" / f"{title}.json"

    if not yt_path.exists() or not whisper_path.exists():
        return 0.0

    first_line = yt_path.read_text().split("\n", 1)[0].strip()
    if not first_line:
        return 0.0
    yt_start = json.loads(first_line).get("start", 0.0)

    whisper_data = json.loads(whisper_path.read_text())
    segs = whisper_data.get("segments", [])
    whisper_start = segs[0]["start"] if segs else 0.0

    return yt_start - whisper_start


def text_file_to_speech(
    source_path,
    output_path,
    tts_engine=None,
    *,
    alignment=None,
    speaker_wav: str | None = None,
    voice_map: dict[str, str] | None = None,
):
    """Read translated JSON with segment timestamps and produce a time-aligned WAV.

    Each segment is individually synthesized and time-stretched to match its
    original timestamp window.  Gaps between segments are filled with silence.
    Applies the YouTube caption timing offset so TTS audio starts when speech
    actually begins in the original video.

    *tts_engine* overrides the module-level ``tts`` instance (used by the
    FastAPI app which loads the model at startup).

    *alignment* overrides the module-level ``_ALIGNMENT_ENABLED`` flag.
    Pass True for aligned mode, False for baseline, or None to use the env var.
    """
    engine = tts_engine if tts_engine is not None else _get_tts_engine()
    use_alignment = alignment if alignment is not None else _ALIGNMENT_ENABLED

    save_name = pathlib.Path(source_path).stem + ".wav"
    print(f"generating {save_name}...", end="")

    segments = segments_from_file(source_path)

    if not segments:
        text = text_from_file(source_path)
        save_path = pathlib.Path(output_path) / pathlib.Path(save_name)
        engine.tts_to_file(text=text, file_path=str(save_path), speaker_wav=speaker_wav)
        print("success!")
        return None

    # Apply YouTube caption timing offset
    offset = _compute_speech_offset(source_path)
    if offset > 0:
        print(f" (applying {offset:.1f}s speech offset)", end="")

    # Pre-compute alignment; also returns flat metrics list for clip_evaluation_report
    with open(source_path) as f:
        es_transcript = json.load(f)
    en_transcript = _load_en_transcript(source_path)
    if use_alignment:
        _metrics_list, align_map = _build_alignment(en_transcript, es_transcript)
    else:
        _metrics_list, align_map = [], {}
    _aligned_list = list(align_map.values())

    # ── Prepare per-segment metadata ────────────────────────────────────
    seg_metas = []
    for i, seg in enumerate(segments):
        aligned_seg = align_map.get(i)
        stretch_factor = aligned_seg.stretch_factor if aligned_seg else 1.0
        target_sec = seg["end"] - seg["start"]
        scheduled_target_sec = target_sec
        scheduled_start = seg["start"]

        if use_alignment and aligned_seg is not None:
            scheduled_start = aligned_seg.scheduled_start
            scheduled_target_sec = max(
                aligned_seg.scheduled_end - aligned_seg.scheduled_start,
                0.0,
            )

        seg_text = seg["text"]
        if aligned_seg is not None:
            from foreign_whispers.alignment import AlignAction
            if aligned_seg.action == AlignAction.REQUEST_SHORTER:
                en_text = ""
                en_segs = en_transcript.get("segments", [])
                if i < len(en_segs):
                    en_text = en_segs[i].get("text", "")
                seg_text = _shorten_segment_text(en_text, seg["text"], target_sec)

        seg_metas.append({
            "index": i,
            "text": seg_text,
            "start": scheduled_start,
            "end": seg["end"],
            "target_sec": target_sec,
            "scheduled_target_sec": scheduled_target_sec,
            "stretch_factor": stretch_factor,
            "aligned_seg": aligned_seg,
            "speaker": seg.get("speaker"),
            "speaker_wav": (
                voice_map.get(seg.get("speaker"), speaker_wav)
                if voice_map and seg.get("speaker") is not None
                else speaker_wav
            ),
        })

    print(f" ({len(segments)} segments synthesized)", end="")

    with tempfile.TemporaryDirectory() as tmpdir:
        combined = AudioSegment.empty()
        cursor_ms = 0
        segment_details = []

        for m in seg_metas:
            i = m["index"]
            start_ms = int((m["start"] + offset) * 1000)

            if start_ms > cursor_ms:
                combined += AudioSegment.silent(duration=start_ms - cursor_ms)
                cursor_ms = start_ms

            if use_alignment and m["speaker_wav"] is None:
                synced = _synced_segment_audio(
                    engine,
                    m["text"],
                    m["scheduled_target_sec"],
                    tmpdir,
                    1.0,
                )
            else:
                synced = _synced_segment_audio(
                    engine,
                    m["text"],
                    m["scheduled_target_sec"],
                    tmpdir,
                    stretch_factor=1.0 if use_alignment else m["stretch_factor"],
                    alignment_enabled=use_alignment,
                    speaker_wav=m["speaker_wav"],
                )
            if synced is None:
                seg_audio = None
                seg_speed_factor = 0.0
                seg_raw_duration = 0.0
            else:
                seg_audio, seg_speed_factor, seg_raw_duration = synced

            aligned_seg = m["aligned_seg"]
            segment_details.append({
                "index": i,
                "text": m["text"],
                "speaker": m["speaker"],
                "speaker_wav": m["speaker_wav"],
                "target_sec": round(m["target_sec"], 3),
                "stretch_factor": round(m["stretch_factor"], 3),
                "raw_duration_s": round(seg_raw_duration, 3),
                "speed_factor": round(seg_speed_factor, 3),
                "action": aligned_seg.action.value if aligned_seg and hasattr(aligned_seg, "action") else "unknown",
            })

            if seg_audio is not None:
                combined += seg_audio
                cursor_ms += len(seg_audio)

        save_path = pathlib.Path(output_path) / save_name
        combined.export(str(save_path), format="wav")

    stem = pathlib.Path(source_path).stem
    _write_align_report(
        str(output_path),
        stem,
        _metrics_list,
        _aligned_list,
        segment_details,
        alignment_enabled=use_alignment,
    )

    print("success!")
    return None


if __name__ == '__main__':
    SOURCE_PATH = "./data/transcriptions/es"
    OUTPUT_PATH = "./audios/"

    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    files = files_from_dir(SOURCE_PATH)
    for file in files:
        text_file_to_speech(file, OUTPUT_PATH)
