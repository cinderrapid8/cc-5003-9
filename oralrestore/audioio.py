"""Audio file I/O.

The source file is opened read-only and its bytes are never touched.  All
processing works on an in-memory float32 copy in the range [-1.0, 1.0].
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioData:
    """An in-memory mono audio buffer plus its sample rate and source info."""

    samples: np.ndarray  # float32, shape (n,), range nominally [-1, 1]
    sample_rate: int
    source_path: str
    source_sha256: str
    original_dtype: str

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_audio(path: str) -> AudioData:
    """Load an audio file as mono float32 without modifying the file on disk."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    info = sf.info(path)
    if info.channels > 1:
        # Oral-history sources are mono; refuse rather than silently mixing.
        raise ValueError(
            f"{path} has {info.channels} channels; only mono recordings are "
            "supported. Down-mix the file explicitly first."
        )
    # soundfile opens the file read-only; we keep no writable handle.
    data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    digest = sha256_file(path)
    return AudioData(
        samples=np.ascontiguousarray(data, dtype=np.float32),
        sample_rate=sample_rate,
        source_path=path,
        source_sha256=digest,
        original_dtype=info.subtype,
    )


def write_wav(path: str, samples: np.ndarray, sample_rate: int,
              subtype: str = "PCM_16") -> None:
    """Write a render to a new WAV file (never overwrites the source)."""
    out = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    sf.write(path, out, sample_rate, subtype=subtype)