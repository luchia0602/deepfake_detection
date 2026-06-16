from .base import FeatureExtractor
from .xlsr import XLSRExtractor
from .whisper import WhisperExtractor
from .prosodic import ProsodicExtractor

EXTRACTORS = {
    "xlsr": XLSRExtractor,
    "whisper": WhisperExtractor,
    "prosodic": ProsodicExtractor,
}

__all__ = [
    "FeatureExtractor",
    "XLSRExtractor",
    "WhisperExtractor",
    "ProsodicExtractor",
    "EXTRACTORS",
]