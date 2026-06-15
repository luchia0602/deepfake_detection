import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Model
from extractors.base import FeatureExtractor


class XLSRExtractor(FeatureExtractor):
    """
    XLS-R (wav2vec2-xls-r-300m) feature extractor. Produces a fixed-length embedding by mean-pooling the final hidden states.
    """

    name = "xlsr"

    def __init__(self, model_name: str = "facebook/wav2vec2-xls-r-300m", target_sr: int = 16000, device: str | None = None):
        self.target_sr = target_sr

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling across time dimension.
        """
        return hidden.mean(dim=1)

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:

        waveform = torch.from_numpy(audio).float().unsqueeze(0)

        if sample_rate != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.target_sr,
            )

        if waveform.shape[1] < 400:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, 400 - waveform.shape[1]),
            )

        waveform = waveform.to(self.device)

        with torch.no_grad():
            outputs = self.model(waveform)

        embedding = self._pool(outputs.last_hidden_state)

        return (embedding.squeeze(0).cpu().numpy().astype(np.float32))