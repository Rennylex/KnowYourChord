from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.postprocess import postprocess_midi


def transcribe_audio_to_midi(
    input_audio: Path,
    output_midi: Path,
    instrument: str = "auto",
) -> Path:
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict_and_save
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "basic-pitch is not installed. Run `pip install -e .` first."
        ) from exc

    output_midi.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="basic-pitch-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        predict_and_save(
            audio_path_list=[str(input_audio)],
            output_directory=str(tmp_path),
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )

        generated_files = sorted(tmp_path.glob("*.mid")) + sorted(
            tmp_path.glob("*.midi")
        )
        if not generated_files:
            raise RuntimeError("basic-pitch did not produce a MIDI file.")

        shutil.move(str(generated_files[0]), str(output_midi))

    postprocess_midi(
        input_midi=output_midi,
        output_midi=output_midi,
        instrument=instrument,
    )

    return output_midi
