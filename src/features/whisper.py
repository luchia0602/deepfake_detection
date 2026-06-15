import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel
from .base import FeatureExtractor


class WhisperExtractor(FeatureExtractor):
    """
    Whisper encoder feature extractor. Audio is resampled to 16 kHz and converted to Whisper log-mel features using the pretrained Whisper processor. A fixed-length utterance representation is obtained by mean-pooling the encoder hidden states.
    """

    name = "whisper"

    def __init__(self, model_name: str = "openai/whisper-small", target_sr: int = 16000, device: str | None = None):
        self.target_sr = target_sr

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.processor = (
            WhisperProcessor.from_pretrained(
                model_name
            )
        )

        self.model = (
            WhisperModel.from_pretrained(
                model_name
            )
        )

        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _pool(hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool hidden states across the time dimension.
        """
        return hidden_states.mean(dim=1)

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract a Whisper embedding from a waveform. Takes audio(np.ndarray) and sample_rate(int), returns Whisper embedding(np.ndarray)
        """

        if sample_rate != self.target_sr:

            waveform = torch.from_numpy(
                audio
            ).float()

            waveform = (
                torchaudio.functional.resample(
                    waveform,
                    sample_rate,
                    self.target_sr,
                )
            )

            audio = (waveform.cpu().numpy())

        inputs = self.processor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )

        input_features = (
            inputs.input_features.to(
                self.device
            )
        )

        with torch.no_grad():
            encoder_outputs = (
                self.model.encoder(
                    input_features
                )
            )

        embedding = self._pool(
            encoder_outputs.last_hidden_state
        )

        return (embedding.squeeze(0).cpu().numpy().astype(np.float32))