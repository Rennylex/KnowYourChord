from __future__ import annotations

import argparse
import functools
import os
import subprocess
from pathlib import Path

import gin
import jax
import librosa
import nest_asyncio
import note_seq
import numpy as np
import seqio
import t5
import t5x
import tensorflow.compat.v2 as tf

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies

nest_asyncio.apply()

SAMPLE_RATE = 16000


class InferenceModel:
    """Wrapper around the MT3 colab inference flow."""

    def __init__(self, checkpoint_path: Path, model_type: str = "mt3") -> None:
        if model_type == "ismir2021":
            num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec
            self.inputs_length = 512
            gin_files = ["/opt/mt3/mt3/gin/model.gin", "/opt/mt3/mt3/gin/ismir2021.gin"]
        elif model_type == "mt3":
            num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            self.inputs_length = 256
            gin_files = ["/opt/mt3/mt3/gin/model.gin", "/opt/mt3/mt3/gin/mt3.gin"]
        else:
            raise ValueError(f"unknown model_type: {model_type}")

        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {
            "inputs": self.inputs_length,
            "targets": self.outputs_length,
        }
        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)
        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(
                num_velocity_bins=num_velocity_bins
            )
        )
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            "targets": seqio.Feature(vocabulary=self.vocabulary),
        }

        self._parse_gin(gin_files)
        self.model = self._load_model()
        self.restore_from_checkpoint(checkpoint_path)

    @property
    def input_shapes(self) -> dict[str, tuple[int, int]]:
        return {
            "encoder_input_tokens": (self.batch_size, self.inputs_length),
            "decoder_input_tokens": (self.batch_size, self.outputs_length),
        }

    def _parse_gin(self, gin_files: list[str]) -> None:
        gin_bindings = [
            "from __gin__ import dynamic_registration",
            "from mt3 import vocabularies",
            "VOCAB_CONFIG=@vocabularies.VocabularyConfig()",
            "vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS",
        ]
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                gin_files, gin_bindings, finalize_config=False
            )

    def _load_model(self):
        model_config = gin.get_configurable(network.T5Config)()
        module = network.Transformer(config=model_config)
        return models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.output_features["inputs"].vocabulary,
            output_vocabulary=self.output_features["targets"].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config),
        )

    def restore_from_checkpoint(self, checkpoint_path: Path) -> None:
        train_state_initializer = t5x.utils.TrainStateInitializer(
            optimizer_def=self.model.optimizer_def,
            init_fn=self.model.get_initial_variables,
            input_shapes=self.input_shapes,
            partitioner=self.partitioner,
        )

        restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
            path=str(checkpoint_path), mode="specific", dtype="float32"
        )
        train_state_axes = train_state_initializer.train_state_axes
        self._predict_fn = self._get_predict_fn(train_state_axes)
        self._train_state = train_state_initializer.from_checkpoint_or_scratch(
            [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0)
        )

    @functools.lru_cache()
    def _get_predict_fn(self, train_state_axes):
        def partial_predict_fn(params, batch, decode_rng):
            del decode_rng
            return self.model.predict_batch_with_aux(
                params, batch, decoder_params={"decode_rng": None}
            )

        return self.partitioner.partition(
            partial_predict_fn,
            in_axis_resources=(
                train_state_axes.params,
                t5x.partitioning.PartitionSpec("data"),
                None,
            ),
            out_axis_resources=t5x.partitioning.PartitionSpec("data"),
        )

    def predict_tokens(self, batch, seed: int = 0) -> np.ndarray:
        prediction, _ = self._predict_fn(
            self._train_state.params, batch, jax.random.PRNGKey(seed)
        )
        return self.vocabulary.decode_tf(prediction).numpy()

    def __call__(self, audio: np.ndarray):
        ds = self.audio_to_dataset(audio)
        ds = self.preprocess(ds)
        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
            ds, task_feature_lengths=self.sequence_length
        )
        model_ds = model_ds.batch(self.batch_size)
        inferences = (
            tokens
            for batch in model_ds.as_numpy_iterator()
            for tokens in self.predict_tokens(batch)
        )

        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), inferences):
            predictions.append(self.postprocess(tokens, example))

        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=self.encoding_spec
        )
        return result["est_ns"]

    def audio_to_dataset(self, audio: np.ndarray):
        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors(
            {
                "inputs": frames,
                "input_times": frame_times,
            }
        )

    def _audio_to_frames(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode="constant")
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times

    def preprocess(self, ds):
        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                feature_key="inputs",
                additional_feature_keys=["input_times"],
            ),
            preprocessors.add_dummy_targets,
            functools.partial(
                preprocessors.compute_spectrograms,
                spectrogram_config=self.spectrogram_config,
            ),
            functools.partial(
                preprocessors.tokenize_transcription_example,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                codec=self.codec,
                onsets_only=False,
                include_ties=True,
            ),
        ]
        for pp in pp_chain:
            ds = pp(ds)
        return ds

    def postprocess(self, tokens: np.ndarray, example: dict) -> dict:
        tokens = np.array(tokens, np.int32)
        if vocabularies.DECODED_EOS_ID in tokens:
            tokens = tokens[: np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
        return {
            "est_tokens": tokens,
            "start_time": example["input_times"][0],
            "raw_inputs": example["inputs"],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Docker MT3 transcription runner")
    parser.add_argument("input_media", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", type=str, default=None)
    parser.add_argument("--model", choices=["mt3", "ismir2021"], default="mt3")
    return parser


def media_to_wav(input_media: Path, output_wav: Path) -> None:
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
        str(SAMPLE_RATE),
        "-ac",
        "1",
        str(output_wav),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = build_parser().parse_args()
    stem = args.stem or args.input_media.stem
    job_dir = args.output_dir / stem
    input_dir = job_dir / "input"
    generated_dir = job_dir / "generated"
    input_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    original_media_path = input_dir / args.input_media.name
    normalized_wav_path = input_dir / f"{stem}.mt3.normalized.wav"
    midi_path = generated_dir / f"{stem}.mt3.mid"

    if original_media_path.resolve() != args.input_media.resolve():
        original_media_path.write_bytes(args.input_media.read_bytes())
    media_to_wav(original_media_path, normalized_wav_path)

    audio, _ = librosa.load(normalized_wav_path, sr=SAMPLE_RATE, mono=True)
    checkpoint_path = Path("/models/checkpoints") / args.model
    inference_model = InferenceModel(checkpoint_path=checkpoint_path, model_type=args.model)
    est_ns = inference_model(audio)
    note_seq.sequence_proto_to_midi_file(est_ns, str(midi_path))

    print(f"Job Dir: {job_dir}")
    print(f"Input Dir: {input_dir}")
    print(f"Original Media: {original_media_path}")
    print(f"Normalized WAV: {normalized_wav_path}")
    print(f"MT3 MIDI: {midi_path}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
