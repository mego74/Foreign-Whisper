"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import logging
import re

logger = logging.getLogger(__name__)
DEFAULT_SPEAKER = "SPEAKER_00"


def _patch_torchaudio_for_pyannote() -> None:
    """Backfill ``torchaudio.AudioMetaData`` for newer torchaudio builds.

    pyannote.audio 3.x references this type at import time, but recent
    torchaudio wheels on macOS no longer export it at the package root.
    A lightweight shim is enough because pyannote uses it for annotations.
    """
    try:
        import torchaudio
    except ImportError:
        return

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:  # pragma: no cover - tiny compatibility shim
            def __init__(
                self,
                sample_rate: int,
                num_frames: int,
                num_channels: int,
                bits_per_sample: int,
                encoding: str,
            ) -> None:
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None

    if not hasattr(torchaudio, "info"):
        import soundfile as sf

        def _bits_per_sample(subtype: str | None) -> int:
            if not subtype:
                return 0
            match = re.search(r"(\d+)", subtype)
            return int(match.group(1)) if match else 0

        def info(uri, backend=None):
            with sf.SoundFile(str(uri)) as audio_file:
                return torchaudio.AudioMetaData(
                    sample_rate=audio_file.samplerate,
                    num_frames=len(audio_file),
                    num_channels=audio_file.channels,
                    bits_per_sample=_bits_per_sample(audio_file.subtype),
                    encoding=audio_file.format,
                )

        torchaudio.info = info

    if not getattr(torchaudio.load, "__fw_pyannote_patched__", False):
        import soundfile as sf
        import torch

        def load(
            uri,
            frame_offset: int = 0,
            num_frames: int = -1,
            normalize: bool = True,
            channels_first: bool = True,
            format=None,
            backend=None,
            buffer_size: int = 4096,
        ):
            frames = -1 if num_frames is None else num_frames
            data, sample_rate = sf.read(
                str(uri),
                start=frame_offset,
                frames=frames,
                always_2d=True,
                dtype="float32",
            )
            waveform = torch.from_numpy(data.T if channels_first else data)
            return waveform, sample_rate

        load.__fw_pyannote_patched__ = True
        torchaudio.load = load


def _patch_torch_for_pyannote() -> None:
    """Restore pre-2.6 checkpoint loading behavior for trusted pyannote weights."""
    try:
        import torch
    except ImportError:
        return

    try:
        from torch.serialization import add_safe_globals

        add_safe_globals([torch.torch_version.TorchVersion])
    except Exception:
        pass

    if getattr(torch.load, "__fw_pyannote_patched__", False):
        return

    original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    _patched_load.__fw_pyannote_patched__ = True
    torch.load = _patched_load


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        _patch_torchaudio_for_pyannote()
        _patch_torch_for_pyannote()
        from pyannote.audio import Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is empty
    or there is no overlap, defaults to ``SPEAKER_00``.

    Returns a new list and does not mutate the input segments.
    """
    labeled: list[dict] = []

    for segment in segments:
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))
        best_speaker = DEFAULT_SPEAKER
        best_overlap = 0.0

        for interval in diarization:
            overlap_start = max(seg_start, float(interval.get("start_s", seg_start)))
            overlap_end = min(seg_end, float(interval.get("end_s", seg_end)))
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(interval.get("speaker", DEFAULT_SPEAKER))

        labeled.append({**segment, "speaker": best_speaker})

    return labeled
