import sys
import numpy as np
import librosa
from extractors.base import FeatureExtractor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CQCC_PYTHON_DIR = _REPO_ROOT / "2021" / "LA" / "Baseline-CQCC-GMM" / "python"
sys.path.insert(0, str(_CQCC_PYTHON_DIR))

from CQCC.cqcc import cqcc  # noqa: E402

class CQCCExtractor(FeatureExtractor):
    """
    Constant-Q Cepstral Coefficients, ASVspoof 2021 baseline config.
    60 dims per frame, mean-pooled to 60-d.
    """

    name = "cqcc"

    def __init__(self, target_sr: int = 16000, B: int = 96, d: int = 16,
                 cf: int = 19, ZsdD: str = "ZsdD"):
        self.target_sr = target_sr
        self.B = B          # bins per octave
        self.d = d          # uniform samples in first octave
        self.cf = cf        # cepstral coefficients (excluding 0th)
        self.ZsdD = ZsdD

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if sample_rate != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate,
                                     target_sr=self.target_sr)

        n_coeffs = (self.cf + 1) * 3  # 60
        try:
            x_col = audio.reshape(-1, 1)
            fmax = self.target_sr / 2
            fmin = fmax / 2 ** 9
            frames, *_ = cqcc(
                x_col, self.target_sr, self.B, fmax, fmin,
                self.d, self.cf, self.ZsdD,
            )
            # cqcc.py already returns (n_frames, n_coeffs)
        except (ValueError, TypeError):
            return np.zeros(n_coeffs, dtype=np.float32)
        return frames.mean(axis=0).astype(np.float32)
