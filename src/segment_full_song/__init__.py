"""Python Package Template"""
from __future__ import annotations

import sys
import loguru
from .models.factory import create_model
from .models.segment_full_song import SegmentFullSongModel

loguru.logger.remove()
loguru.logger.add(
    sys.stdout, format="{time:HH:mm:ss} [{level}] {message}", level="INFO"
)


__version__ = "0.0.1"
__all__ = ["create_model", "SegmentFullSongModel"]
