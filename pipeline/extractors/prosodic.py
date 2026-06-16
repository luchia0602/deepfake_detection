import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import librosa
from amfm_decompy import pYAAPT, basic_tools
from .base import FeatureExtractor

TARGET_SR = 16_000


class ProsodicExtractor(FeatureExtractor):
    """
    Extracts an 8-dimensional prosodic feature vector from a waveform.
    """

    name: str = "prosodic"

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = audio.astype(np.float32)

        if sample_rate != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SR)

        if len(audio) < 400:
            audio = np.pad(audio, (0, 400 - len(audio)))

        try:
            signal = basic_tools.SignalObj(audio, TARGET_SR)
            pitch = pYAAPT.yaapt(signal, frame_length=25.0, frame_space=10.0)
            voiced = pitch.samp_values
            voiced = voiced[voiced > 0]
            if len(voiced) == 0:
                voiced = np.array([0.0])
        except (IndexError, ValueError):
            voiced = np.array([0.0])

        rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=160)[0]

        return np.array(
            [
                voiced.mean(), voiced.std(), voiced.min(), voiced.max(),
                rms.mean(),    rms.std(),    rms.min(),    rms.max(),
            ],
            dtype=np.float32,
        )