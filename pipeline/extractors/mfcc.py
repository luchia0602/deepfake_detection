import numpy as np
import librosa
from python_speech_features import mfcc, delta

from extractors.base import FeatureExtractor


class MFCCExtractor(FeatureExtractor):
    """
    Mel-Frequency Cepstral Coefficients with deltas.
    13 static + delta + delta-delta = 39 dims per frame, mean-pooled to 39-d.
    """

    name = "mfcc"

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if sample_rate != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate,
                                     target_sr=self.target_sr)

        static = mfcc(audio, self.target_sr)       # (n_frames, 13)
        d1 = delta(static, N=2)
        d2 = delta(d1, N=2)
        frames = np.concatenate([static, d1, d2], axis=1)  # (n_frames, 39)
        return frames.mean(axis=0).astype(np.float32)
