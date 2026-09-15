from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PITCH_CLASS_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
CHORD_PATTERNS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "5": [0, 7],
}


@dataclass(frozen=True, slots=True)
class ChordEvent:
    start: float
    end: float
    root_pc: int
    quality: str
    bass_pc: int

    @property
    def symbol(self) -> str:
        suffix = {
            "maj": "",
            "min": "m",
            "dim": "dim",
            "sus2": "sus2",
            "sus4": "sus4",
            "5": "5",
        }[self.quality]
        return f"{PITCH_CLASS_NAMES[self.root_pc]}{suffix}"


def midi_to_chord_midi(input_midi: Path, output_midi: Path) -> Path:
    try:
        import pretty_midi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pretty_midi is required for chord accompaniment export. Run `pip install -e .` first."
        ) from exc

    source = pretty_midi.PrettyMIDI(str(input_midi))
    chord_events = extract_chord_events(source)

    chord_midi = pretty_midi.PrettyMIDI(initial_tempo=_initial_tempo(source))
    pad = pretty_midi.Instrument(program=0, name="Chord Accompaniment")
    bass = pretty_midi.Instrument(program=32, name="Bass")

    for event in chord_events:
        _add_chord_notes(pretty_midi, pad, bass, event)

    chord_midi.instruments.append(pad)
    chord_midi.instruments.append(bass)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    chord_midi.write(str(output_midi))
    return output_midi


def extract_chord_events(midi) -> list[ChordEvent]:
    notes = [note for instrument in midi.instruments for note in instrument.notes]
    if not notes:
        return []

    beats = list(midi.get_beats())
    end_time = max(note.end for note in notes)
    if len(beats) < 2:
        beats = _fallback_beats(midi, end_time)
    if beats[-1] < end_time:
        beat_step = beats[-1] - beats[-2] if len(beats) >= 2 else 0.5
        beats.append(end_time + max(beat_step, 0.5))

    events: list[ChordEvent] = []
    previous_signature: tuple[int, str, int] | None = None

    for start, end in zip(beats, beats[1:]):
        window_notes = _notes_in_window(notes, start, end)
        if not window_notes:
            continue
        chord = _detect_chord(window_notes)
        if chord is None:
            continue
        signature = (chord.root_pc, chord.quality, chord.bass_pc)
        if previous_signature == signature and events:
            events[-1] = ChordEvent(
                start=events[-1].start,
                end=end,
                root_pc=events[-1].root_pc,
                quality=events[-1].quality,
                bass_pc=events[-1].bass_pc,
            )
            continue
        events.append(
            ChordEvent(
                start=start,
                end=end,
                root_pc=chord.root_pc,
                quality=chord.quality,
                bass_pc=chord.bass_pc,
            )
        )
        previous_signature = signature

    return events


def _initial_tempo(midi) -> float:
    tempo_times, tempi = midi.get_tempo_changes()
    if len(tempi) == 0:
        return 120.0
    return float(tempi[0])


def _fallback_beats(midi, end_time: float) -> list[float]:
    tempo = _initial_tempo(midi)
    beat_length = 60.0 / max(tempo, 1.0)
    beats: list[float] = []
    current = 0.0
    while current <= end_time + beat_length:
        beats.append(current)
        current += beat_length
    return beats


def _notes_in_window(notes: list, start: float, end: float) -> list:
    window_notes = []
    for note in notes:
        overlap = min(note.end, end) - max(note.start, start)
        if overlap > 0.04:
            window_notes.append((note, overlap))
    return window_notes


@dataclass(frozen=True, slots=True)
class DetectedChord:
    root_pc: int
    quality: str
    bass_pc: int


def _detect_chord(window_notes: list[tuple]) -> DetectedChord | None:
    pc_weights = [0.0] * 12
    bass_pitch = None
    for note, overlap in window_notes:
        weight = overlap * max(note.velocity, 1)
        pc_weights[note.pitch % 12] += weight
        if bass_pitch is None or note.pitch < bass_pitch:
            bass_pitch = note.pitch

    active_pcs = {pc for pc, weight in enumerate(pc_weights) if weight > 3}
    if not active_pcs:
        return None

    best_score = -1.0
    best_root = None
    best_quality = None
    for root_pc in active_pcs:
        for quality, intervals in CHORD_PATTERNS.items():
            score = 0.0
            for interval in intervals:
                score += pc_weights[(root_pc + interval) % 12]
            penalty = 0.0
            for pc in active_pcs:
                if (pc - root_pc) % 12 not in intervals:
                    penalty += pc_weights[pc] * 0.35
            score -= penalty
            if score > best_score:
                best_score = score
                best_root = root_pc
                best_quality = quality

    if best_root is None or best_quality is None:
        return None

    return DetectedChord(
        root_pc=best_root,
        quality=best_quality,
        bass_pc=bass_pitch % 12 if bass_pitch is not None else best_root,
    )


def _add_chord_notes(pretty_midi, pad, bass, event: ChordEvent) -> None:
    chord_pitches = []
    root_pitch = 60 + event.root_pc
    while root_pitch > 72:
        root_pitch -= 12
    while root_pitch < 48:
        root_pitch += 12

    for interval in CHORD_PATTERNS[event.quality]:
        chord_pitches.append(root_pitch + interval)

    bass_pitch = 36 + event.bass_pc
    while bass_pitch > 52:
        bass_pitch -= 12
    while bass_pitch < 28:
        bass_pitch += 12

    duration = max(event.end - event.start, 0.1)
    for pitch in chord_pitches:
        pad.notes.append(
            pretty_midi.Note(
                velocity=78,
                pitch=pitch,
                start=event.start,
                end=event.start + duration,
            )
        )

    bass.notes.append(
        pretty_midi.Note(
            velocity=72,
            pitch=bass_pitch,
            start=event.start,
            end=event.start + min(duration, 0.9),
        )
    )
