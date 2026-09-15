from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

SUPPORTED_MEDIA_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".mp4",
    ".mov",
    ".m4v",
}


def ensure_supported_input_media(input_media: Path) -> None:
    suffix = input_media.suffix.lower()
    if suffix not in SUPPORTED_MEDIA_SUFFIXES:
        raise RuntimeError(
            "Unsupported input format. Supported formats include "
            "wav, mp3, m4a, flac, aac, ogg, mp4, mov, and m4v."
        )


def run_ffmpeg(command: list[str], failure_message: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found. Install it and ensure `ffmpeg` is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"{failure_message}: {message}") from exc


def media_to_mp3(input_media: Path, output_mp3: Path) -> Path:
    ensure_supported_input_media(input_media)
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_media),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_mp3),
    ]
    run_ffmpeg(command, "FFmpeg source MP3 export failed")
    if not output_mp3.exists():
        raise RuntimeError(f"FFmpeg did not produce the expected MP3: {output_mp3}")
    return output_mp3


def media_to_wav(input_media: Path, output_wav: Path, sample_rate: int = 44100) -> Path:
    ensure_supported_input_media(input_media)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_media),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_wav),
    ]
    run_ffmpeg(command, "FFmpeg WAV conversion failed")
    if not output_wav.exists():
        raise RuntimeError(f"FFmpeg did not produce the expected WAV: {output_wav}")
    return output_wav


def midi_to_mp3(input_midi: Path, output_mp3: Path, sample_rate: int = 44100) -> Path:
    try:
        import numpy as np
        import pretty_midi
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pretty_midi and soundfile are required for MP3 export. Run `pip install -e .` first."
        ) from exc

    output_mp3.parent.mkdir(parents=True, exist_ok=True)

    midi = pretty_midi.PrettyMIDI(str(input_midi))
    audio = midi.synthesize(fs=sample_rate)
    audio = np.asarray(audio, dtype="float32")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        wav_path = Path(tmp_file.name)

    try:
        sf.write(str(wav_path), audio, sample_rate)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_mp3),
        ]
        run_ffmpeg(command, "FFmpeg MIDI MP3 export failed")
    finally:
        wav_path.unlink(missing_ok=True)

    if not output_mp3.exists():
        raise RuntimeError(f"FFmpeg did not produce the expected MP3: {output_mp3}")

    return output_mp3
