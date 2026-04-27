import os
import pathlib
import tempfile


def write_temp_audio(audio_bytes: bytes, filename: str | None) -> str:
    suffix = pathlib.Path(filename).suffix if filename else ".wav"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp_file.write(audio_bytes)
        temp_file.flush()
    finally:
        temp_file.close()
    return temp_file.name


def cleanup_temp_audio(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass

