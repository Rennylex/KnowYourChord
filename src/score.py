from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def midi_to_musicxml(input_midi: Path, output_musicxml: Path) -> Path:
    try:
        from music21 import converter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "music21 is not installed. Run `pip install -e .` first."
        ) from exc

    score = converter.parse(str(input_midi))
    score = score.makeNotation(inPlace=False)
    output_musicxml.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(output_musicxml))
    return output_musicxml


def resolve_musescore_path(explicit_path: Path | None = None) -> str:
    candidates: list[str] = []
    if explicit_path is not None:
        candidates.append(str(explicit_path))

    candidates.extend(
        [
            "musescore",
            "mscore",
            "mscore4",
            "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
            "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
        ]
    )

    for candidate in candidates:
        resolved = shutil.which(candidate) if "/" not in candidate else candidate
        if resolved and Path(resolved).exists():
            return resolved

    raise RuntimeError(
        "MuseScore CLI was not found. Install MuseScore 4 and pass "
        "`--musescore-path /path/to/mscore` if it is not on PATH."
    )


def musicxml_to_pdf(
    input_musicxml: Path,
    output_pdf: Path,
    musescore_path: Path | None = None,
) -> Path:
    executable = resolve_musescore_path(musescore_path)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        str(input_musicxml),
        "-o",
        str(output_pdf),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"MuseScore PDF export failed: {message}") from exc

    if not output_pdf.exists():
        raise RuntimeError(f"MuseScore did not produce the expected PDF: {output_pdf}")

    return output_pdf


def midi_to_pdf(
    input_midi: Path,
    output_pdf: Path,
    musescore_path: Path | None = None,
) -> Path:
    executable = resolve_musescore_path(musescore_path)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-f",
        str(input_midi),
        "-o",
        str(output_pdf),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"MuseScore MIDI PDF export failed: {message}") from exc

    if not output_pdf.exists():
        raise RuntimeError(f"MuseScore did not produce the expected PDF: {output_pdf}")

    return output_pdf
