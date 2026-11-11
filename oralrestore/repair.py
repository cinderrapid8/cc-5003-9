"""Local repair algorithms.

Two interpolation algorithms are available for accepted short click windows,
plus a parameter-recorded IIR notch for mains hum.  Nothing here decides which
events to repair: it only executes operations an expert has accepted, and it
refuses to interpolate gaps longer than the approved limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal as sps
from scipy.interpolate import CubicSpline


@dataclass
class RepairResult:
    samples: np.ndarray
    operation: dict[str, Any]


# ---------------------------------------------------------------------------
# Click interpolation
# ---------------------------------------------------------------------------


def _expand_window(start: int, end: int, n: int, pad: int) -> tuple[int, int]:
    """Grow the window slightly so splines anchor on clean neighbours."""
    return max(0, start - pad), min(n, end + pad)


def interpolate_ar(
    samples: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
    *,
    order: int = 24,
    context_ms: float = 20.0,
) -> RepairResult:
    """Fill a short gap with a two-sided autoregressive model.

    AR coefficients are fitted on clean context outside the gap.  Two forward
    predictors (left context, right context reversed) each estimate the gap;
    the estimates are blended with a linear cross-fade so both edges match.
    """
    x = np.array(samples, dtype=np.float64, copy=True)
    n = len(x)
    if not (0 <= start < end <= n):
        raise ValueError("invalid repair window")
    gap = end - start
    ctx = min(n - gap, max(gap * 4, int(sample_rate * context_ms / 1000.0)))

    l0, l1 = max(0, start - ctx), start
    r0, r1 = end, min(n, end + ctx)
    left, right = x[l0:l1], x[r0:r1]
    if left.size < order + 2 or right.size < order + 2:
        raise ValueError("not enough clean context around the gap")

    def fit_predict(seg: np.ndarray, count: int, reverse: bool) -> np.ndarray:
        src = seg[::-1] if reverse else seg
        rows = len(src) - order
        X = np.column_stack([src[order - 1 - k: rows + order - 1 - k]
                            for k in range(order)])
        y = src[order:]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        out = np.empty(count, dtype=np.float64)
        mem = list(src[-order:])
        for i in range(count):
            v = float(np.dot(coef, mem[::-1]))
            out[i] = v
            mem = mem[1:] + [v]
        return out[::-1] if reverse else out

    from_left = fit_predict(left, gap, reverse=False)
    from_right = fit_predict(right, gap, reverse=True)
    w = np.linspace(0.0, 1.0, gap)
    x[start:end] = (1.0 - w) * from_left + w * from_right

    op = {
        "op": "click_repair",
        "algorithm": "ar_interpolation",
        "start": int(start),
        "end": int(end),
        "params": {"order": order, "context_ms": context_ms},
    }
    return RepairResult(x.astype(np.float32), op)


def interpolate_spline(
    samples: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
    *,
    pad_ms: float = 1.0,
) -> RepairResult:
    """Fill a short gap with a natural cubic spline through clean neighbours."""
    x = np.array(samples, dtype=np.float64, copy=True)
    n = len(x)
    if not (0 <= start < end <= n):
        raise ValueError("invalid repair window")
    pad = max(1, int(sample_rate * pad_ms / 1000.0))
    s, e = _expand_window(start, end, n, pad)
    anchor_idx = np.concatenate((np.arange(s, start), np.arange(end, e)))
    if anchor_idx.size < 2:
        raise ValueError("not enough anchor samples for spline interpolation")
    cs = CubicSpline(anchor_idx, x[anchor_idx], bc_type="natural")
    x[start:end] = cs(np.arange(start, end))

    op = {
        "op": "click_repair",
        "algorithm": "cubic_spline",
        "start": int(start),
        "end": int(end),
        "params": {"pad_ms": pad_ms},
    }
    return RepairResult(x.astype(np.float32), op)


def repair_click(
    samples: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
    *,
    algorithm: str = "ar",
    max_gap_ms: float = 10.0,
    **kwargs: Any,
) -> RepairResult:
    """Dispatch with the hard safety limit on interpolated gap length."""
    gap = end - start
    max_gap = max(1, int(round(sample_rate * max_gap_ms / 1000.0)))
    if gap > max_gap:
        raise ValueError(
            f"refusing to interpolate {gap}-sample gap "
            f"({1000.0 * gap / sample_rate:.1f} ms); limit is {max_gap_ms} ms. "
            "Long damage is marked unrepairable, not guessed."
        )
    if algorithm == "ar":
        return interpolate_ar(samples, start, end, sample_rate, **kwargs)
    if algorithm == "spline":
        return interpolate_spline(samples, start, end, sample_rate, **kwargs)
    raise ValueError(f"unknown interpolation algorithm: {algorithm}")


# ---------------------------------------------------------------------------
# Mains-frequency notch
# ---------------------------------------------------------------------------


def notch_hum(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frequency: float,
    bandwidth_hz: float = 2.0,
    quality_scale: float = 1.0,
) -> RepairResult:
    """Apply a narrow IIR notch at one mains frequency/harmonic.

    The actual centre frequency and -3 dB bandwidth are stored on the
    operation so the filter choice can be audited.  Zero-phase filtfilt keeps
    the waveform phase intact; the notch is narrow by construction so nearby
    useful content is left alone.
    """
    if not (0 < frequency < sample_rate / 2.0):
        raise ValueError("notch frequency outside the Nyquist range")
    if not (0 < bandwidth_hz < frequency):
        raise ValueError("bandwidth must be positive and below the frequency")
    q = frequency / (bandwidth_hz * quality_scale)
    b, a = sps.iirnotch(w0=frequency, Q=q, fs=sample_rate)
    # padlen guard for very short clips
    padlen = min(3 * max(len(a), len(b)), max(0, len(samples) - 1))
    out = sps.filtfilt(b, a, np.asarray(samples, dtype=np.float64),
                       padlen=padlen).astype(np.float32)
    w, h = sps.freqz(b, a, worN=[2 * np.pi * frequency / sample_rate],
                     fs=sample_rate)
    op = {
        "op": "hum_notch",
        "algorithm": "iirnotch_filtfilt",
        "start": 0,
        "end": len(samples),
        "params": {
            "frequency_hz": float(frequency),
            "bandwidth_hz": float(bandwidth_hz),
            "Q": round(float(q), 3),
            "gain_at_centre_db": round(
                float(20 * np.log10(np.maximum(np.abs(h[0]), 1e-12))), 2),
        },
    }
    return RepairResult(out, op)
