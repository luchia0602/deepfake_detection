import numpy as np
import librosa
from amfm_decompy import pYAAPT
from amfm_decompy import basic_tools
from .base import FeatureExtractor


class ProsodicExtractor(FeatureExtractor):
    """
    Prosodic feature extractor. Features: F0 mean, F0 std, F0 min, F0 max, RMS mean, RMS std, RMS min, RMS max
    """

    name = "prosodic"

    def __init__(self, frame_length: int = 400, hop_length: int = 160):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def _extract_pitch(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:

        try:
            signal = basic_tools.SignalObj(
                audio,
                sample_rate
            )

            pitch = pYAAPT.yaapt(
                signal,
                frame_length=25.0,
                frame_space=10.0
            )

            voiced = pitch.samp_values
            voiced = voiced[voiced > 0]

            if len(voiced) == 0:
                voiced = np.array([0.0])

        except Exception:
            voiced = np.array([0.0], dtype=np.float32)

        return voiced

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:

        if len(audio) < self.frame_length:
            audio = np.pad(
                audio,
                (0, self.frame_length - len(audio))
            )

        voiced = self._extract_pitch(
            audio,
            sample_rate
        )

        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        features = np.array(
            [
                voiced.mean(),
                voiced.std(),
                voiced.min(),
                voiced.max(),

                rms.mean(),
                rms.std(),
                rms.min(),
                rms.max(),
            ],
            dtype=np.float32,
        )

        return features