from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GuitarPostprocessConfig:
    min_pitch: int = 40
    max_pitch: int = 88
    min_duration_s: float = 0.09
    merge_gap_s: float = 0.08
    overlap_group_s: float = 0.05
    harmonic_interval_min: int = 7
    max_simultaneous_notes: int = 6
    harmonic_velocity_ratio: float = 0.82


def postprocess_midi(input_midi: Path, output_midi: Path, instrument: str) -> Path:
    if instrument != "guitar":
        return input_midi

    try:
        import pretty_midi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pretty_midi is required for guitar postprocessing. Run `pip install -e .` first."
        ) from exc

    midi = pretty_midi.PrettyMIDI(str(input_midi))
    config = GuitarPostprocessConfig()

    for instrument_track in midi.instruments:
        cleaned_notes = _clean_guitar_notes(instrument_track.notes, config)
        instrument_track.notes = cleaned_notes

    midi.write(str(output_midi))
    return output_midi


def _clean_guitar_notes(notes: list, config: GuitarPostprocessConfig) -> list:
    filtered = [
        note
        for note in notes
        if config.min_pitch <= note.pitch <= config.max_pitch
        and (note.end - note.start) >= config.min_duration_s
    ]
    filtered.sort(key=lambda note: (note.start, note.pitch, note.end))

    merged = _merge_adjacent_same_pitch(filtered, config)
    suppressed = _suppress_harmonics(merged, config)
    return suppressed


def _merge_adjacent_same_pitch(notes: list, config: GuitarPostprocessConfig) -> list:
    if not notes:
        return []

    merged: list = []
    for note in notes:
        if (
            merged
            and merged[-1].pitch == note.pitch
            and note.start - merged[-1].end <= config.merge_gap_s
        ):
            merged[-1].end = max(merged[-1].end, note.end)
            merged[-1].velocity = max(merged[-1].velocity, note.velocity)
            continue
        merged.append(note)
    return merged


def _suppress_harmonics(notes: list, config: GuitarPostprocessConfig) -> list:
    if not notes:
        return []

    kept: list = []
    current_group: list = []
    group_start = notes[0].start

    for note in notes:
        if note.start - group_start <= config.overlap_group_s:
            current_group.append(note)
            continue
        kept.extend(_select_group_notes(current_group, config))
        current_group = [note]
        group_start = note.start

    kept.extend(_select_group_notes(current_group, config))
    kept.sort(key=lambda note: (note.start, note.pitch, note.end))
    return kept


def _select_group_notes(notes: list, config: GuitarPostprocessConfig) -> list:
    if len(notes) <= 1:
        return notes

    notes = sorted(
        notes,
        key=lambda note: (
            note.start,
            -note.velocity,
            -(note.end - note.start),
            note.pitch,
        ),
    )

    selected: list = []
    for note in notes:
        harmonic_of_existing = False
        for existing in selected:
            is_higher = note.pitch > existing.pitch
            close_in_time = abs(note.start - existing.start) <= config.overlap_group_s
            related_interval = (
                note.pitch - existing.pitch >= config.harmonic_interval_min
            )
            weaker = note.velocity <= existing.velocity * config.harmonic_velocity_ratio
            shorter = (note.end - note.start) <= (existing.end - existing.start) * 1.1
            if is_higher and close_in_time and related_interval and weaker and shorter:
                harmonic_of_existing = True
                break
        if harmonic_of_existing:
            continue
        selected.append(note)
        selected.sort(key=lambda existing: existing.pitch)

    selected = sorted(
        selected,
        key=lambda note: (-note.velocity, -(note.end - note.start), note.pitch),
    )[: config.max_simultaneous_notes]
    selected.sort(key=lambda note: (note.start, note.pitch, note.end))
    return selected
