from abc import ABC, abstractmethod
import numpy as np


class FeatureExtractor(ABC):
    """
    Base class for all feature extractors.
    """
    name: str = "base"

    @abstractmethod
    def extract(self, audio: np.ndarray,  sample_rate: int) -> np.ndarray:
        """
        Extract a fixed-length feature vector from audio. Input: audio waveform (np.ndarray) and sampling rate (int). Output: feature vector (np.ndarray)
        """
        raise NotImplementedError

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.extract(audio, sample_rate)