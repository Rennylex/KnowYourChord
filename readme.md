# KnowYourChord

Yoho, musician! KnowYourChord is a small Python CLI for turning piano or guitar performance audio/video into editable music artifacts.

It takes common media files such as `wav`, `mp3`, `mp4`, or `mov`, extracts the audio, transcribes it to MIDI, and can export MusicXML, PDF sheet music, a source MP3, and a rendered MP3 preview.

## What It Does

```text
audio/video performance
-> normalized mono WAV
-> MIDI transcription
-> MusicXML score
-> PDF sheet music
-> MP3 previews
```

The default transcription backend is Spotify's `basic-pitch`. The repository also includes an optional Docker-based MT3 runner for trying Google's Magenta MT3 checkpoints without installing the MT3 dependency chain directly on macOS.

## Features

- Accepts common audio and video formats: `wav`, `mp3`, `m4a`, `flac`, `aac`, `ogg`, `mp4`, `mov`, and `m4v`
- Generates structured per-job output folders
- Exports MIDI from performance audio
- Converts MIDI to MusicXML with `music21`
- Exports PDF sheet music through MuseScore CLI
- Extracts the original media audio to MP3
- Renders generated MIDI back to MP3 for quick listening
- Includes optional guitar cleanup rules
- Can generate an additional chord-only accompaniment arrangement
- Includes an optional Docker runner for MT3 / ISMIR 2021 transcription models

## Requirements

- Python `3.10` or `3.11`
- `ffmpeg` available on `PATH`
- MuseScore CLI if you want PDF export
- Docker Desktop if you want to use the MT3 runner

`basic-pitch` currently does not install cleanly on Python `3.13`, so use Python `3.10` or `3.11`.

On macOS, MuseScore is commonly available at:

```bash
/Applications/MuseScore\ 4.app/Contents/MacOS/mscore
```

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Run the full pipeline:

```bash
know-your-chord path/to/performance.mov --output-dir outputs
```

The old command name is also kept as an alias:

```bash
music-transcriber path/to/performance.mov --output-dir outputs
```

For guitar recordings, enable guitar-specific cleanup:

```bash
know-your-chord path/to/performance.mov \
  --output-dir outputs \
  --instrument guitar
```

Generate an additional chord accompaniment arrangement:

```bash
know-your-chord path/to/performance.mov \
  --output-dir outputs \
  --instrument guitar \
  --export-chord-accompaniment
```

Skip PDF export:

```bash
know-your-chord path/to/performance.mp4 --output-dir outputs --skip-pdf
```

Only generate MIDI:

```bash
know-your-chord path/to/performance.mov \
  --output-dir outputs \
  --skip-score \
  --skip-mp3 \
  --skip-source-mp3
```

Pass a MuseScore CLI path explicitly:

```bash
know-your-chord path/to/performance.wav \
  --output-dir outputs \
  --musescore-path "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
```

## Output Layout

For an input named `performance.mov`, the pipeline writes:

```text
outputs/performance/
  input/
    performance.mov
    performance.normalized.wav
    performance.source.mp3
  generated/
    performance.mid
    performance.rendered.mp3
    performance.musicxml
    performance.pdf
    performance.chords.mid
    performance.chords.rendered.mp3
    performance.chords.musicxml
    performance.chords.pdf
```

The `.chords.*` files are only created when `--export-chord-accompaniment` is enabled.

## Docker MT3 Runner

The default CLI uses `basic-pitch`. If you want to experiment with Magenta MT3 instead, use the Docker runner:

```bash
chmod +x scripts/run_mt3_docker.sh
./scripts/run_mt3_docker.sh path/to/performance.mov outputs_mt3 mt3
```

Use the piano-focused ISMIR 2021 model:

```bash
./scripts/run_mt3_docker.sh path/to/performance.mov outputs_mt3_ismir ismir2021
```

The first run downloads checkpoints into `.docker-model-cache/`.

## Project Structure

```text
src/cli.py            CLI argument parsing
src/pipeline.py       End-to-end transcription pipeline
src/audio.py          ffmpeg conversion and MIDI-to-MP3 rendering
src/transcription.py  basic-pitch audio-to-MIDI transcription
src/postprocess.py    guitar-oriented MIDI cleanup
src/accompaniment.py  chord accompaniment extraction/export
src/score.py          MIDI -> MusicXML -> PDF conversion
docker/mt3/           optional MT3 Docker runner
scripts/              helper scripts
```

## Limitations

Automatic transcription is imperfect. Clean solo piano or solo guitar recordings work best. Dense mixes, vocals, percussion, reverb-heavy audio, and noisy videos will usually need manual score correction after export.

Generated PDF scores should be treated as a starting point, not a final publication-ready engraving.
