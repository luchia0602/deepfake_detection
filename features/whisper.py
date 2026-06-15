import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel
from extractors.base import FeatureExtractor


class WhisperExtractor(FeatureExtractor):
    """
    Whisper encoder feature extractor. Produces a fixed-length embedding by mean-pooling the encoder hidden states.
    """

    name = "whisper"

    def __init__(self, model_name: str = "openai/whisper-small", target_sr: int = 16000, device: str | None = None,):
        self.target_sr = target_sr

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.processor = WhisperProcessor.from_pretrained(
            model_name
        )

        self.model = WhisperModel.from_pretrained(
            model_name
        )

        self.model.eval()
        self.model.to(self.device)

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling across time dimension.
        """
        return hidden.mean(dim=1)

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:

        if sample_rate != self.target_sr:
            audio = (
                torchaudio.functional.resample(
                    torch.from_numpy(audio).float(),
                    sample_rate,
                    self.target_sr,
                )
                .cpu()
                .numpy()
            )

        inputs = self.processor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(
            self.device
        )

        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_features
            )

        embedding = self._pool(
            encoder_outputs.last_hidden_state
        )

        return (embedding.squeeze(0).cpu().numpy().astype(np.float32))