from .base import FeatureExtractor
from .xlsr import XLSRExtractor
from .whisper import WhisperExtractor
from .prosodic import ProsodicExtractor
from .cqcc import CQCCExtractor
from .lfcc import LFCCExtractor
from .mfcc import MFCCExtractor

EXTRACTORS = {
    "xlsr": XLSRExtractor,
    "whisper": WhisperExtractor,
    "prosodic": ProsodicExtractor,
    "cqcc": CQCCExtractor,
    "lfcc": LFCCExtractor,
    "mfcc": MFCCExtractor

}

__all__ = [
    "FeatureExtractor",
    "XLSRExtractor",
    "WhisperExtractor",
    "ProsodicExtractor",
    "CQCCExtractor",
    "LFCCExtractor",
    "MFCCExtractor",
    "EXTRACTORS",
]
