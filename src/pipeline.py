from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.accompaniment import midi_to_chord_midi
from src.audio import media_to_mp3, media_to_wav, midi_to_mp3
from src.score import midi_to_musicxml, midi_to_pdf, musicxml_to_pdf
from src.transcription import transcribe_audio_to_midi


@dataclass(slots=True)
class PipelineResult:
    job_dir: Path
    input_dir: Path
    original_media_path: Path
    normalized_wav_path: Path
    midi_path: Path
    source_mp3_path: Path | None
    rendered_mp3_path: Path | None
    score_path: Path | None
    pdf_path: Path | None
    chord_midi_path: Path | None
    chord_rendered_mp3_path: Path | None
    chord_score_path: Path | None
    chord_pdf_path: Path | None


def run_pipeline(
    input_media: Path,
    output_dir: Path,
    output_stem: str | None = None,
    instrument: str = "auto",
    generate_score: bool = True,
    generate_pdf: bool = True,
    generate_rendered_mp3: bool = True,
    export_source_mp3: bool = True,
    export_chord_accompaniment: bool = False,
    musescore_path: Path | None = None,
) -> PipelineResult:
    if not input_media.exists():
        raise FileNotFoundError(f"Input media not found: {input_media}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or input_media.stem

    job_dir = output_dir / stem
    input_dir = job_dir / "input"
    generated_dir = job_dir / "generated"
    input_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    original_media_path = input_dir / input_media.name
    normalized_wav_path = input_dir / f"{stem}.normalized.wav"
    midi_path = generated_dir / f"{stem}.mid"
    source_mp3_path = input_dir / f"{stem}.source.mp3" if export_source_mp3 else None
    rendered_mp3_path = (
        generated_dir / f"{stem}.rendered.mp3" if generate_rendered_mp3 else None
    )
    score_path = generated_dir / f"{stem}.musicxml" if generate_score else None
    pdf_path = generated_dir / f"{stem}.pdf" if generate_score and generate_pdf else None
    chord_midi_path = (
        generated_dir / f"{stem}.chords.mid" if export_chord_accompaniment else None
    )
    chord_rendered_mp3_path = (
        generated_dir / f"{stem}.chords.rendered.mp3"
        if export_chord_accompaniment and generate_rendered_mp3
        else None
    )
    chord_score_path = (
        generated_dir / f"{stem}.chords.musicxml"
        if export_chord_accompaniment and generate_score
        else None
    )
    chord_pdf_path = (
        generated_dir / f"{stem}.chords.pdf"
        if export_chord_accompaniment and generate_score and generate_pdf
        else None
    )

    shutil.copy2(input_media, original_media_path)

    if source_mp3_path is not None:
        media_to_mp3(input_media=original_media_path, output_mp3=source_mp3_path)

    media_to_wav(input_media=original_media_path, output_wav=normalized_wav_path)
    transcribe_audio_to_midi(
        input_audio=normalized_wav_path,
        output_midi=midi_path,
        instrument=instrument,
    )

    if rendered_mp3_path is not None:
        midi_to_mp3(input_midi=midi_path, output_mp3=rendered_mp3_path)

    if score_path is not None:
        midi_to_musicxml(input_midi=midi_path, output_musicxml=score_path)
        if pdf_path is not None:
            try:
                musicxml_to_pdf(
                    input_musicxml=score_path,
                    output_pdf=pdf_path,
                    musescore_path=musescore_path,
                )
            except RuntimeError:
                midi_to_pdf(
                    input_midi=midi_path,
                    output_pdf=pdf_path,
                    musescore_path=musescore_path,
                )

    if chord_midi_path is not None:
        midi_to_chord_midi(input_midi=midi_path, output_midi=chord_midi_path)
        if chord_rendered_mp3_path is not None:
            midi_to_mp3(input_midi=chord_midi_path, output_mp3=chord_rendered_mp3_path)
        if chord_score_path is not None:
            midi_to_musicxml(input_midi=chord_midi_path, output_musicxml=chord_score_path)
            if chord_pdf_path is not None:
                try:
                    musicxml_to_pdf(
                        input_musicxml=chord_score_path,
                        output_pdf=chord_pdf_path,
                        musescore_path=musescore_path,
                    )
                except RuntimeError:
                    midi_to_pdf(
                        input_midi=chord_midi_path,
                        output_pdf=chord_pdf_path,
                        musescore_path=musescore_path,
                    )

    return PipelineResult(
        job_dir=job_dir,
        input_dir=input_dir,
        original_media_path=original_media_path,
        normalized_wav_path=normalized_wav_path,
        midi_path=midi_path,
        source_mp3_path=source_mp3_path,
        rendered_mp3_path=rendered_mp3_path,
        score_path=score_path,
        pdf_path=pdf_path,
        chord_midi_path=chord_midi_path,
        chord_rendered_mp3_path=chord_rendered_mp3_path,
        chord_score_path=chord_score_path,
        chord_pdf_path=chord_pdf_path,
    )
