import numpy as np
import librosa
from scipy.fft import dct

from extractors.base import FeatureExtractor


class LFCCExtractor(FeatureExtractor):
    """
    Linear Frequency Cepstral Coefficients, following the ASVspoof 2021
    LFCC-LCNN baseline configuration: pre-emphasis, a Hamming-windowed STFT
    (20 ms / 10 ms framing, 1024-point FFT), 20 linear triangular filters over
    the lower half of the spectrum, 20 cepstral coefficients via a type-II DCT,
    and delta + delta-delta coefficients. Mean-pooled to a 60-d vector.
    """

    name = "lfcc"

    def __init__(self, target_sr: int = 16000, n_filters: int = 20,
                 n_ceps: int = 20, n_fft: int = 1024,
                 win_length: int = 320, hop_length: int = 160,
                 pre_emph: float = 0.97):
        self.target_sr = target_sr
        self.n_filters = n_filters
        self.n_ceps = n_ceps
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.pre_emph = pre_emph

    @staticmethod
    def _linear_filterbank(n_filters, n_fft, low_freq, high_freq):
        """Triangular filterbank with linearly (uniformly) spaced centres."""
        bin_freqs = np.linspace(0, high_freq, n_fft // 2 + 1)
        center_freqs = np.linspace(low_freq, high_freq, n_filters + 2)
        fb = np.zeros((n_filters, n_fft // 2 + 1))
        for m in range(1, n_filters + 1):
            lo, mid, hi = center_freqs[m - 1], center_freqs[m], center_freqs[m + 1]
            rising = (bin_freqs >= lo) & (bin_freqs <= mid)
            falling = (bin_freqs > mid) & (bin_freqs <= hi)
            fb[m - 1, rising] = (bin_freqs[rising] - lo) / (mid - lo)
            fb[m - 1, falling] = (hi - bin_freqs[falling]) / (hi - mid)
        return fb

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if sample_rate != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate,
                                     target_sr=self.target_sr)

        # pre-emphasis (0.97), matching the baseline
        audio = np.append(audio[0],
                          audio[1:] - self.pre_emph * audio[:-1]).astype(np.float32)

        # LCNN baseline limits the frequency range to 0.5 * Nyquist
        high_freq = (self.target_sr / 2) * 0.5

        # Hamming-windowed power spectrogram; 320-sample window zero-padded to a
        # 1024-point FFT, 160-sample hop (20 ms / 10 ms at 16 kHz)
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft,
                                win_length=self.win_length,
                                hop_length=self.hop_length,
                                window="hamming")) ** 2

        fb = self._linear_filterbank(self.n_filters, self.n_fft, 0.0, high_freq)

        # filterbank -> log10 -> DCT
        log_fb = np.log10(fb @ S + 1e-10).T
        static = dct(log_fb, type=2, axis=1, norm="ortho")[:, : self.n_ceps]

        # delta width must be odd and >= 3; for very short clips with too few
        # frames, deltas are undefined, so fall back to zeros (keeps output 60-d)
        n_frames = static.shape[0]
        width = min(9, n_frames)
        if width % 2 == 0:
            width = max(1, width - 1)

        if n_frames >= 3 and width >= 3:
            d1 = librosa.feature.delta(static.T, width=width).T
            d2 = librosa.feature.delta(static.T, width=width, order=2).T
        else:
            d1 = np.zeros_like(static)
            d2 = np.zeros_like(static)

        frames = np.concatenate([static, d1, d2], axis=1)  # (n_frames, 60)
        return frames.mean(axis=0).astype(np.float32)
