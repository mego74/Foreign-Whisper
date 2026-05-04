from foreign_whispers.voice_resolution import build_speaker_voice_map, resolve_speaker_wav


def test_resolve_speaker_specific_voice(tmp_path):
    (tmp_path / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    (tmp_path / "es").mkdir()
    (tmp_path / "es" / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    (tmp_path / "es" / "SPEAKER_00.wav").write_bytes(b"RIFF" + b"\x00" * 40)

    assert resolve_speaker_wav(tmp_path, "es", "SPEAKER_00") == "es/SPEAKER_00.wav"


def test_resolve_language_default(tmp_path):
    (tmp_path / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    (tmp_path / "es").mkdir()
    (tmp_path / "es" / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)

    assert resolve_speaker_wav(tmp_path, "es", "SPEAKER_01") == "es/default.wav"


def test_resolve_global_default(tmp_path):
    (tmp_path / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    (tmp_path / "fr").mkdir()

    assert resolve_speaker_wav(tmp_path, "fr", "SPEAKER_00") == "default.wav"


def test_resolve_without_speaker_id(tmp_path):
    (tmp_path / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    (tmp_path / "es").mkdir()
    (tmp_path / "es" / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)

    assert resolve_speaker_wav(tmp_path, "es") == "es/default.wav"


def test_resolve_unknown_language_without_any_file(tmp_path):
    assert resolve_speaker_wav(tmp_path, "xx") == "default.wav"


def test_resolve_gender_specific_voice(tmp_path):
    (tmp_path / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    (tmp_path / "es").mkdir()
    (tmp_path / "es" / "female.wav").write_bytes(b"RIFF" + b"\x00" * 40)

    assert resolve_speaker_wav(tmp_path, "es", "SPEAKER_09", gender="female") == "es/female.wav"


def test_build_speaker_voice_map_uses_profiles(tmp_path):
    (tmp_path / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    (tmp_path / "es").mkdir()
    (tmp_path / "es" / "male.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    (tmp_path / "es" / "female.wav").write_bytes(b"RIFF" + b"\x00" * 40)

    result = build_speaker_voice_map(
        tmp_path,
        "es",
        ["SPEAKER_01", "SPEAKER_00"],
        {
            "SPEAKER_00": {"gender": "male"},
            "SPEAKER_01": {"gender": "female"},
        },
    )

    assert result == {
        "SPEAKER_00": "es/male.wav",
        "SPEAKER_01": "es/female.wav",
    }
