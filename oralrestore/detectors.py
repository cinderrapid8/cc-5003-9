"""Candidate detectors for impulsive pops and mains-frequency hum.

Detectors never modify audio.  They return *candidates* only: real plosives,
percussion and sharp consonants can look identical to a narrow injected pop, so
an expert must accept or exclude each one before anything is interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Candidate records
# ---------------------------------------------------------------------------


@dataclass
class ClickCandidate:
    """A single detected impulsive event (candidate, not yet acted upon)."""

    cand_id: str
    start: int          # inclusive sample index
    end: int            # exclusive sample index
    peak_sample: int
    peak_amplitude: float
    score: float        # detector score in units of robust sigma
    detector: str
    detector_params: dict[str, Any]
    repairable: bool
    reject_reason: str | None = None
    # Expert decision: "pending" | "accepted" | "excluded"
    decision: str = "pending"

    @property
    def width(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "cand_id": self.cand_id,
            "kind": "click",
            "start": self.start,
            "end": self.end,
            "width": self.width,
            "peak_sample": self.peak_sample,
            "peak_amplitude": self.peak_amplitude,
            "score": round(float(self.score), 3),
            "detector": self.detector,
            "detector_params": self.detector_params,
            "repairable": self.repairable,
            "reject_reason": self.reject_reason,
            "decision": self.decision,
        }


@dataclass
class HumCandidate:
    """A suspected mains-frequency component (50/60 Hz line and harmonics)."""

    cand_id: str
    frequency: float    # Hz
    bandwidth: float    # Hz, suggested notch bandwidth
    peak_amplitude: float
    snr_db: float       # peak level vs. local median spectrum
    decision: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cand_id": self.cand_id,
            "kind": "hum",
            "frequency": round(self.frequency, 2),
            "bandwidth": round(self.bandwidth, 2),
            "peak_amplitude": round(float(self.peak_amplitude), 6),
            "snr_db": round(float(self.snr_db), 2),
            "decision": self.decision,
        }


# ---------------------------------------------------------------------------
# Robust statistics helpers
# ---------------------------------------------------------------------------


def _mad_sigma(x: np.ndarray) -> float:
    """Median absolute deviation scaled to a Gaussian-consistent sigma."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-12)


def _merge_regions(mask: np.ndarray, min_gap: int) -> np.ndarray:
    """Merge True regions separated by fewer than min_gap False samples."""
    if min_gap <= 0 or not mask.any():
        return mask.copy()
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return mask.copy()
    # gaps between consecutive flagged samples smaller than min_gap are bridged
    gaps = np.diff(idx)
    bridge = np.flatnonzero(gaps <= min_gap)
    out = mask.copy()
    for b in bridge:
        out[idx[b]: idx[b + 1] + 1] = True
    return out


def _regions(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    padded = np.concatenate(([0], mask.view(np.uint8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2], edges[1::2]))


# ---------------------------------------------------------------------------
# Click / pop detectors
# ---------------------------------------------------------------------------


def _median_residual(x: np.ndarray, half_win: int) -> np.ndarray:
    """Residual against a running median, which tracks slow speech dynamics."""
    n = len(x)
    base = np.empty(n, dtype=np.float32)
    # Sliding median via cumulative windows; windows are tiny so a direct
    # loop over samples is fast enough at audio rates and dependency-free.
    for i in range(n):
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        base[i] = np.median(x[lo:hi])
    return x - base


def _ar_residual(x: np.ndarray, order: int) -> np.ndarray:
    """One-step linear-prediction error from a global AR(order) model."""
    n = len(x)
    if n <= order * 3:
        return np.zeros_like(x)
    # Build the Toeplitz-style design matrix and solve in least squares.
    rows = n - order
    X = np.empty((rows, order), dtype=np.float64)
    for k in range(order):
        X[:, k] = x[order - 1 - k: n - 1 - k]
    y = x[order:].astype(np.float64)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = np.zeros(n, dtype=np.float64)
    resid[order:] = y - pred
    return resid.astype(np.float32)


def detect_clicks_with_trace(
    samples: np.ndarray,
    sample_rate: int,
    *,
    method: str = "ar",
    threshold_sigma: float = 8.0,
    half_win: int = 3,
    ar_order: int = 16,
    merge_gap_ms: float = 1.0,
    max_gap_ms: float = 10.0,
    window_pad_ms: float = 0.75,
) -> tuple[list[ClickCandidate], dict[str, Any]]:
    """Locate impulsive events and return both candidates and a mask trace.

    The trace is the auditable link between the raw threshold mask and the
    padded candidate windows later offered for interpolation: it records the
    robust sigma, every threshold-crossing span and how padding/merging grew
    it into the candidate region.  Detection still never modifies audio.
    """
    x = np.asarray(samples, dtype=np.float32)
    n = len(x)
    pad = max(0, int(round(sample_rate * window_pad_ms / 1000.0)))
    if method == "ar":
        residual = _ar_residual(x, ar_order)
        params = {
            "method": "ar",
            "threshold_sigma": threshold_sigma,
            "ar_order": ar_order,
            "merge_gap_ms": merge_gap_ms,
            "max_gap_ms": max_gap_ms,
            "window_pad_ms": window_pad_ms,
        }
        det_name = "ar_residual"
    elif method == "median":
        residual = _median_residual(x, half_win)
        params = {
            "method": "median",
            "threshold_sigma": threshold_sigma,
            "half_win": half_win,
            "merge_gap_ms": merge_gap_ms,
            "max_gap_ms": max_gap_ms,
            "window_pad_ms": window_pad_ms,
            "caveat": "median residual over-fires on periodic/voiced signals",
        }
        det_name = "median_residual"
    else:
        raise ValueError(f"unknown click detector method: {method}")

    sigma = _mad_sigma(residual)
    score = np.abs(residual) / sigma
    raw_mask = score >= threshold_sigma
    merge_gap = max(1, int(sample_rate * merge_gap_ms / 1000.0))
    mask = _merge_regions(raw_mask, merge_gap)

    max_gap = max(1, int(round(sample_rate * max_gap_ms / 1000.0)))
    candidates: list[ClickCandidate] = []
    region_trace: list[dict[str, Any]] = []
    for k, (rs, re_) in enumerate(_regions(mask)):
        # Symmetric padding covers the full ringing transient, not just the
        # samples that crossed the threshold.
        s = max(0, rs - pad)
        e = min(n, re_ + pad)
        region_score = score[rs:re_]
        peak_off = int(np.argmax(region_score))
        width = e - s
        repairable = width <= max_gap
        reason = None if repairable else (
            f"window {width} samples ({1000.0 * width / sample_rate:.1f} ms) "
            f"exceeds max_gap_ms={max_gap_ms}; long damage cannot be "
            "reliably interpolated"
        )
        cand = ClickCandidate(
            cand_id=f"click-{k + 1:04d}",
            start=int(s),
            end=int(e),
            peak_sample=int(rs + peak_off),
            peak_amplitude=float(x[rs + peak_off]),
            score=float(region_score[peak_off]),
            detector=det_name,
            detector_params=dict(params),
            repairable=repairable,
            reject_reason=reason,
        )
        candidates.append(cand)
        # Mask-to-window linkage: the crossing span is what the threshold
        # actually flagged; the padded window is what interpolation would
        # touch.  Their ratio is the core "detection too wide" evidence.
        region_trace.append({
            "cand_id": cand.cand_id,
            "threshold_span": [int(rs), int(re_)],
            "threshold_span_ms": [
                round(1000.0 * rs / sample_rate, 3),
                round(1000.0 * re_ / sample_rate, 3)],
            "candidate_window": [int(s), int(e)],
            "candidate_window_ms": [
                round(1000.0 * s / sample_rate, 3),
                round(1000.0 * e / sample_rate, 3)],
            "n_threshold_samples": int(re_ - rs),
            "n_window_samples": int(width),
            "window_to_threshold_ratio": round(
                float(width) / float(max(1, re_ - rs)), 3),
            "edge_pad_samples": [int(rs - s), int(e - re_)],
            "peak_score": round(float(region_score[peak_off]), 3),
            "edge_scores": [
                round(float(score[rs]), 3),
                round(float(score[re_ - 1]), 3)],
            "repairable": repairable,
        })
    trace = {
        "stage": "click_detection",
        "detector": det_name,
        "params": dict(params),
        "n_samples": int(n),
        "robust_sigma": round(float(sigma), 8),
        "merge_gap_samples": merge_gap,
        "pad_samples": int(pad),
        "n_raw_threshold_samples": int(raw_mask.sum()),
        "n_mask_regions": len(region_trace),
        "regions": region_trace,
    }
    return candidates, trace


def detect_clicks(
    samples: np.ndarray,
    sample_rate: int,
    *,
    method: str = "ar",
    threshold_sigma: float = 8.0,
    half_win: int = 3,
    ar_order: int = 16,
    merge_gap_ms: float = 1.0,
    max_gap_ms: float = 10.0,
    window_pad_ms: float = 0.75,
) -> list[ClickCandidate]:
    """Locate narrow impulsive events and return them as review candidates.

    method: "ar" (default; linear-prediction residual.  The predictor models
            voiced speech so only un-predictable narrow transients score
            highly) or "median" (running-median residual; useful for rough,
            aperiodic material but it over-fires on periodic waveforms and
            should be used cautiously).
    threshold_sigma: robust residual threshold; higher => fewer, sharper hits.
    max_gap_ms: events wider than this are flagged unrepairable rather than
                presented as routine interpolation jobs.
    """
    candidates, _ = detect_clicks_with_trace(
        samples, sample_rate, method=method,
        threshold_sigma=threshold_sigma, half_win=half_win,
        ar_order=ar_order, merge_gap_ms=merge_gap_ms,
        max_gap_ms=max_gap_ms, window_pad_ms=window_pad_ms)
    return candidates


# ---------------------------------------------------------------------------
# Mains hum detector (advisory only)
# ---------------------------------------------------------------------------


def detect_hum_with_trace(
    samples: np.ndarray,
    sample_rate: int,
    *,
    line_frequencies: tuple[float, ...] = (50.0, 60.0),
    max_harmonic: int = 3,
    snr_threshold_db: float = 12.0,
    bandwidth_hz: float = 2.0,
    search_tolerance_hz: float = 1.5,
) -> tuple[list[HumCandidate], dict[str, Any]]:
    """Find mains-frequency energy, returning candidates plus a spectrum trace.

    The trace keeps the local-baseline SNR at every searched grid bin and the
    exact bins chosen, so a later notch audit can tie filter centre/bandwidth
    back to the evidence that produced the candidate.
    """
    x = np.asarray(samples, dtype=np.float32)
    n = len(x)
    if n < sample_rate:  # need at least ~1 s for 1 Hz resolution
        return [], {
            "stage": "hum_detection",
            "n_samples": int(n),
            "skipped": "clip shorter than 1 s; insufficient frequency resolution",
            "grid_searches": [],
        }
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(x * window)) * 2.0 / np.sum(window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    # Local spectral baseline: median over a +/-25 Hz neighbourhood.
    baseline = np.empty_like(spectrum)
    for i in range(len(spectrum)):
        lo = max(0, i - 25)
        hi = min(len(spectrum), i + 26)
        baseline[i] = np.median(spectrum[lo:hi]) + 1e-12
    snr = 20.0 * np.log10(spectrum / baseline)

    out: list[HumCandidate] = []
    seen: set[float] = set()
    k = 0
    grid_searches: list[dict[str, Any]] = []
    for f0 in line_frequencies:
        for h in range(1, max_harmonic + 1):
            target = f0 * h
            if target >= sample_rate / 2.0:
                break
            near = np.where(np.abs(freqs - target) <= search_tolerance_hz)[0]
            if near.size == 0:
                continue
            i = near[np.argmax(spectrum[near])]
            f = float(freqs[i])
            key = round(f, 1)
            grid_searches.append({
                "target_frequency_hz": round(float(target), 2),
                "selected_bin_hz": round(f, 3),
                "bin_index": int(i),
                "bin_snr_db": round(float(snr[i]), 2),
                "bin_amplitude": round(float(spectrum[i]), 6),
                "local_baseline_amplitude": round(float(baseline[i]), 6),
                "snr_threshold_db": snr_threshold_db,
                "already_seen": key in seen,
                "passes_threshold": bool(
                    key not in seen and snr[i] >= snr_threshold_db),
            })
            if key in seen:
                continue
            if snr[i] >= snr_threshold_db:
                seen.add(key)
                k += 1
                out.append(
                    HumCandidate(
                        cand_id=f"hum-{k:04d}",
                        frequency=f,
                        bandwidth=bandwidth_hz,
                        peak_amplitude=float(spectrum[i]),
                        snr_db=float(snr[i]),
                    )
                )
    out.sort(key=lambda c: c.frequency)
    trace = {
        "stage": "hum_detection",
        "n_samples": int(n),
        "line_frequencies": list(line_frequencies),
        "max_harmonic": max_harmonic,
        "snr_threshold_db": snr_threshold_db,
        "bandwidth_hz": bandwidth_hz,
        "search_tolerance_hz": search_tolerance_hz,
        "frequency_resolution_hz": round(float(sample_rate / n), 3),
        "n_grid_searches": len(grid_searches),
        "n_candidates": len(out),
        "grid_searches": grid_searches,
    }
    return out, trace


def detect_hum(
    samples: np.ndarray,
    sample_rate: int,
    *,
    line_frequencies: tuple[float, ...] = (50.0, 60.0),
    max_harmonic: int = 3,
    snr_threshold_db: float = 12.0,
    bandwidth_hz: float = 2.0,
    search_tolerance_hz: float = 1.5,
) -> list[HumCandidate]:
    """Find energy concentrated at mains frequencies and their harmonics.

    Returns suggestions only.  The narrow search tolerance and explicit
    bandwidth keep the subsequent notch from swallowing nearby speech energy.
    """
    candidates, _ = detect_hum_with_trace(
        samples, sample_rate, line_frequencies=line_frequencies,
        max_harmonic=max_harmonic, snr_threshold_db=snr_threshold_db,
        bandwidth_hz=bandwidth_hz,
        search_tolerance_hz=search_tolerance_hz)
    return candidates
