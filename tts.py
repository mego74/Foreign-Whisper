"""Backward-compatible top-level TTS helpers.

Older tests and notebooks import ``tts`` directly from the project root.
Reload the maintained implementation module on import so env-driven flags
like ``FW_ALIGNMENT`` are refreshed when callers run ``importlib.reload(tts)``.
"""

import importlib

from api.src.services import tts_engine as _tts_engine_module

_tts_engine_module = importlib.reload(_tts_engine_module)

_ALIGNMENT_ENABLED = _tts_engine_module._ALIGNMENT_ENABLED
_synced_segment_audio = _tts_engine_module._synced_segment_audio
text_file_to_speech = _tts_engine_module.text_file_to_speech
tts = _tts_engine_module.tts
