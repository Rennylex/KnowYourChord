from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe piano/guitar performance media to MIDI, score, PDF, and MP3."
    )
    parser.add_argument(
        "input_media",
        type=Path,
        help="Path to the input media file. Supports wav, mp4, mov, m4v, and common audio formats.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write MIDI and score files into.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Optional base filename for outputs. Defaults to the input stem.",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        choices=["auto", "guitar", "piano"],
        default="auto",
        help="Optional instrument-specific MIDI cleanup profile.",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Only export MIDI and MP3, skip MusicXML/PDF generation.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Export MusicXML but skip PDF score generation.",
    )
    parser.add_argument(
        "--skip-mp3",
        action="store_true",
        help="Skip MP3 rendering from the generated MIDI.",
    )
    parser.add_argument(
        "--skip-source-mp3",
        action="store_true",
        help="Skip extracting the source media audio to MP3.",
    )
    parser.add_argument(
        "--musescore-path",
        type=Path,
        default=None,
        help="Optional explicit path to the MuseScore CLI executable for PDF export.",
    )
    parser.add_argument(
        "--export-chord-accompaniment",
        action="store_true",
        help="Export an additional chord-only accompaniment arrangement without replacing the original outputs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = run_pipeline(
        input_media=args.input_media,
        output_dir=args.output_dir,
        output_stem=args.stem,
        instrument=args.instrument,
        generate_score=not args.skip_score,
        generate_pdf=not args.skip_score and not args.skip_pdf,
        generate_rendered_mp3=not args.skip_mp3,
        export_source_mp3=not args.skip_source_mp3,
        export_chord_accompaniment=args.export_chord_accompaniment,
        musescore_path=args.musescore_path,
    )

    print(f"Job Dir: {result.job_dir}")
    print(f"Input Dir: {result.input_dir}")
    print(f"Original Media: {result.original_media_path}")
    print(f"Normalized WAV: {result.normalized_wav_path}")
    print(f"MIDI: {result.midi_path}")
    if result.source_mp3_path is not None:
        print(f"Source MP3: {result.source_mp3_path}")
    if result.rendered_mp3_path is not None:
        print(f"Rendered MP3: {result.rendered_mp3_path}")
    if result.score_path is not None:
        print(f"Score: {result.score_path}")
    if result.pdf_path is not None:
        print(f"PDF: {result.pdf_path}")
    if result.chord_midi_path is not None:
        print(f"Chord MIDI: {result.chord_midi_path}")
    if result.chord_rendered_mp3_path is not None:
        print(f"Chord Rendered MP3: {result.chord_rendered_mp3_path}")
    if result.chord_score_path is not None:
        print(f"Chord Score: {result.chord_score_path}")
    if result.chord_pdf_path is not None:
        print(f"Chord PDF: {result.chord_pdf_path}")


if __name__ == "__main__":
    main()
