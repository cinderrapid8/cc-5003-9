"""Synthetic fixtures and quantitative evaluation.

A known-clean baseline is generated first, then contaminated with recorded
interference (narrow pops and 50/60 Hz hum).  Because both signals are known,
we can measure exactly what a restoration preserves, removes and alters.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audioio import write_wav
from .project import Project

# ---------------------------------------------------------------------------
# Clean baseline: voiced-like formant signal + gentle breathing gaps
# ---------------------------------------------------------------------------


def synthesize_clean(
    sample_rate: int = 16000,
    duration: float = 6.0,
    *,
    f0: float = 120.0,
    seed: int = 20261111,
) -> np.ndarray:
    """Speech-like clean signal: vibrato glottal tone through formants."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sample_rate)) / sample_rate
    vibrato = f0 * (1.0 + 0.03 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(vibrato) / sample_rate
    glottis = (np.sin(phase) + 0.35 * np.sin(2 * phase)
               + 0.12 * np.sin(3 * phase))
    # Three slowly drifting formants give the tone a speech-like spectrum.
    out = np.zeros_like(t)
    for k, (fc, bw, gain) in enumerate(
        [(720, 90, 1.0), (1240, 130, 0.55), (2800, 200, 0.25)]
    ):
        drift = 40 * np.sin(2 * np.pi * (0.3 + 0.11 * k) * t + k)
        out += gain * np.sin(2 * np.pi * (fc + drift) * t)
    sig = glottis * (0.55 + 0.45 * out / (np.max(np.abs(out)) + 1e-9))
    # Breath-like low-level noise floor, realistically present in recordings.
    sig += 0.004 * rng.standard_normal(len(t))
    # Gentle amplitude envelope mimicking phrases and pauses.
    env = 0.75 + 0.2 * np.sin(2 * np.pi * 0.25 * t)
    pause = np.ones_like(t)
    for centre in (2.1, 4.3):
        pause *= 1.0 - 0.85 * np.exp(-0.5 * ((t - centre) / 0.12) ** 2)
    sig *= env * pause
    return (sig / np.max(np.abs(sig)) * 0.85).astype(np.float32)


# ---------------------------------------------------------------------------
# Interference injection
# ---------------------------------------------------------------------------


@dataclass
class InjectedPops:
    centres_ms: list[float]
    widths_ms: list[float]
    amplitudes: list[float]


@dataclass
class InjectedHum:
    fundamental_hz: float
    harmonics: list[int]
    amplitudes: list[float]


def inject_pops(
    clean: np.ndarray,
    sample_rate: int,
    *,
    centres_ms: list[float],
    widths_ms: list[float],
    amplitudes: list[float],
    ring_freq_hz: float = 2500.0,
    seed: int = 7,
) -> tuple[np.ndarray, InjectedPops]:
    """Add narrow damped impulse bursts (the synthetic 'pop' defect).

    Each pop is a sharp leading spike (one sample at its centre) followed by
    a short exponentially decaying ringing, matching a clipped/contact pop
    rather than a smooth tone.  ``widths_ms`` sets the decay time constant,
    so the detectable event stays a narrow 1-3 ms transient.
    """
    rng = np.random.default_rng(seed)
    out = clean.copy()
    n = len(out)
    t_step = 1.0 / sample_rate
    for c_ms, w_ms, amp in zip(centres_ms, widths_ms, amplitudes):
        c = int(c_ms / 1000.0 * sample_rate)
        tau = max(t_step, w_ms / 1000.0 / 3.0)
        half = min(n - c, c, int(6 * tau * sample_rate) + 2)
        lo, hi = max(0, c - half), min(n, c + half)
        tt = (np.arange(lo, hi) - c) * t_step
        # Delta-like centre spike + causal damped ringing, tiny pre-ring.
        spike = np.zeros(hi - lo, dtype=np.float64)
        centre_idx = c - lo
        spike[centre_idx] = 1.0
        ring = np.exp(-np.abs(tt) / tau) * np.sin(
            2 * np.pi * ring_freq_hz * tt)
        pulse = amp * (0.7 * spike + 0.5 * ring)
        out[lo:hi] += pulse.astype(np.float32)
    return out, InjectedPops(list(centres_ms), list(widths_ms),
                             list(amplitudes))


def inject_hum(
    clean: np.ndarray,
    sample_rate: int,
    *,
    fundamental_hz: float = 50.0,
    harmonics: list[int] | None = None,
    amplitudes: list[float] | None = None,
) -> tuple[np.ndarray, InjectedHum]:
    harmonics = harmonics or [1, 2, 3]
    amplitudes = amplitudes or [0.08, 0.04, 0.02]
    t = np.arange(len(clean)) / sample_rate
    out = clean.copy()
    for h, amp in zip(harmonics, amplitudes):
        # Deterministic phase so injection is reproducible across runs.
        out += (amp * np.sin(2 * np.pi * fundamental_hz * h * t
                             + 0.3 * h)).astype(np.float32)
    return out, InjectedHum(fundamental_hz, list(harmonics),
                            list(amplitudes))


def make_long_defect(
    clean: np.ndarray,
    sample_rate: int,
    *,
    centre_ms: float = 3000.0,
    length_ms: float = 60.0,
    amplitude: float = 0.9,
) -> np.ndarray:
    """Overwrite a long span with a burst; it must be flagged unrepairable."""
    out = clean.copy()
    c = int(centre_ms / 1000.0 * sample_rate)
    half = int(length_ms / 1000.0 * sample_rate / 2)
    lo, hi = max(0, c - half), min(len(out), c + half)
    rng = np.random.default_rng(99)
    out[lo:hi] = amplitude * rng.choice([-1, 1], size=hi - lo).astype(
        np.float32)
    return out


def add_legitimate_transients(
    clean: np.ndarray,
    sample_rate: int,
    *,
    centres_ms: list[float],
    seed: int = 31,
) -> np.ndarray:
    """Add real plosive/percussion-like transients that must NOT be auto-deleted.

    These mimic sharp consonants and genuine table taps: short broadband
    bursts with natural decay, physically indistinguishable at the sample
    level from some defect pops.  A correct workbench surfaces them as
    candidates and lets the expert exclude them.
    """
    rng = np.random.default_rng(seed)
    out = clean.copy()
    n = len(out)
    for c_ms in centres_ms:
        c = int(c_ms / 1000.0 * sample_rate)
        # 6-10 ms decaying noise burst, like a /k/ or a finger tap.
        length = int(sample_rate * rng.uniform(0.006, 0.010))
        lo, hi = max(0, c - length // 2), min(n, c + length // 2)
        tt = np.arange(hi - lo)
        env = np.exp(-tt / (length * 0.35))
        burst = 0.55 * env * rng.standard_normal(hi - lo)
        out[lo:hi] += burst.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rms_db(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-20)
    return float(20 * np.log10(rms))


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Signal-to-error ratio of candidate against a known reference."""
    ref = reference.astype(np.float64)
    err = candidate.astype(np.float64) - ref
    p_sig = np.mean(ref ** 2)
    p_err = np.mean(err ** 2) + 1e-20
    return float(10 * np.log10(p_sig / p_err))


def band_energy(x: np.ndarray, sample_rate: int, f_lo: float,
                f_hi: float) -> float:
    X = np.abs(np.fft.rfft(x.astype(np.float64) * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    sel = (f >= f_lo) & (f <= f_hi)
    return float(np.sum(X[sel] ** 2))


def fidelity_outside_guards_db(reference: np.ndarray, candidate: np.ndarray,
                               sample_rate: int,
                               notch_centres: list[float],
                               guard_hz: float = 25.0) -> float:
    """Fidelity over spectrum excluding each notch plus a guard interval.

    The guard absorbs both the filter's transition band and Hann-window
    spectral leakage, so the metric reflects changes to genuinely useful
    content rather than penalising the intended narrow removal.
    """
    n = len(reference)
    R = np.fft.rfft(reference.astype(np.float64) * np.hanning(n))
    C = np.fft.rfft(candidate.astype(np.float64) * np.hanning(n))
    f = np.fft.rfftfreq(n, 1.0 / sample_rate)
    keep = np.ones(len(f), dtype=bool)
    for fc in notch_centres:
        keep &= ~((f >= fc - guard_hz) & (f <= fc + guard_hz))
    p_sig = np.sum(np.abs(R[keep]) ** 2)
    p_err = np.sum(np.abs(C[keep] - R[keep]) ** 2) + 1e-20
    return float(10 * np.log10(p_sig / p_err))


def outside_windows_maxdiff(a: np.ndarray, b: np.ndarray,
                            windows: list[tuple[int, int]]) -> float:
    """Largest sample difference outside every repaired window.

    Must be zero (up to float32 rounding) for click interpolation, proving the
    repair touches only the approved short windows.
    """
    mask = np.ones(len(a), dtype=bool)
    for s, e in windows:
        mask[s:e] = False
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(a[mask] - b[mask])))


# ---------------------------------------------------------------------------
# End-to-end evaluation harness
# ---------------------------------------------------------------------------


def _detector_hits(candidates, truth_centres, sample_rate,
                  tolerance_ms: float = 3.0) -> tuple[int, int, list]:
    """Match candidates to injected events by centre proximity."""
    tol = tolerance_ms / 1000.0 * sample_rate
    truth = list(truth_centres)
    matched: list[tuple[Any, float]] = []
    used: set[int] = set()
    for c in candidates:
        centre = (c.start + c.end) / 2.0
        best, best_d = None, tol
        for i, tc in enumerate(truth):
            if i in used:
                continue
            d = abs(centre - tc)
            if d <= best_d:
                best, best_d = i, d
        if best is not None:
            used.add(best)
            matched.append((c, truth[best]))
    matched_ids = {id(c) for c, _ in matched}
    false_alarms = [c for c in candidates if id(c) not in matched_ids]
    misses = len(truth) - len(matched)
    return len(matched), misses, false_alarms


def evaluate(project: Project, clean: np.ndarray,
             pop_centres_ms: list[float],
             hum: InjectedHum | None) -> dict[str, Any]:
    """Compute the full metric set for a rendered restoration project."""
    sr = project.audio.sample_rate
    rendered = project.render()
    original = project.render_original()
    click_windows = [(o["start"], o["end"]) for o in project.operations
                     if o["op"] == "click_repair"]

    truth_centres = [ms / 1000.0 * sr for ms in pop_centres_ms]
    click_cands = project.click_candidates
    hits, misses, false_alarms = _detector_hits(
        click_cands, truth_centres, sr)

    report: dict[str, Any] = {
        "sample_rate": sr,
        "n_samples": len(clean),
        "detection": {
            "click_candidates_total": len(click_cands),
            "injected_pops": len(truth_centres),
            "true_positives": hits,
            "misses": misses,
            "false_alarms": len(false_alarms),
            "false_alarm_ids": [c.cand_id for c in false_alarms],
            "false_alarm_details": [
                {
                    "cand_id": c.cand_id,
                    "at_ms": round(1000.0 * c.peak_sample / sr, 2),
                    "score_sigma": round(c.score, 2),
                    "repairable": c.repairable,
                }
                for c in false_alarms
            ],
            "unrepairable_candidates": [
                c.cand_id for c in project.click_candidates
                if not c.repairable
            ],
        },
        "hum_detection": {
            "candidates_total": len(project.hum_candidates),
            "candidates": [c.to_dict() for c in project.hum_candidates],
        },
        "preservation": {
            "max_abs_diff_outside_repair_windows": outside_windows_maxdiff(
                original, rendered, click_windows),
            "note": "click interpolation must be exactly zero here; hum "
                    "notch legitimately alters the whole signal",
        },
        "signal_fidelity": {
            "snr_rendered_vs_clean_db": round(snr_db(clean, rendered), 2),
            "snr_damaged_vs_clean_db": round(
                snr_db(clean, original), 2),
        },
        "repair_graph": {
            "n_operations": len(project.operations),
            "overlapping_pairs": project.overlapping_operations(),
            "operations": project.repair_list(),
        },
    }

    if hum is not None:
        rows = []
        notch_centres: list[float] = []
        for h, amp in zip(hum.harmonics, hum.amplitudes):
            f = hum.fundamental_hz * h
            bw = 2.0
            notch_centres.append(f)
            e_clean = band_energy(clean, sr, f - bw, f + bw)
            e_damaged = band_energy(original, sr, f - bw, f + bw)
            e_rendered = band_energy(rendered, sr, f - bw, f + bw)
            rows.append({
                "frequency_hz": f,
                "injected_amplitude": amp,
                "band_energy_clean": round(e_clean, 6),
                "band_energy_damaged": round(e_damaged, 6),
                "band_energy_rendered": round(e_rendered, 6),
                "residual_vs_clean_db": round(
                    10 * np.log10((e_rendered + 1e-20)
                                  / (e_clean + 1e-20)), 2),
                "suppression_db": round(
                    10 * np.log10((e_damaged + 1e-20)
                                  / (e_rendered + 1e-20)), 2),
            })
        report["hum_residual"] = rows
        # Fidelity measured only where the repair promised not to touch.
        report["signal_fidelity"]["snr_outside_notch_guards_db"] = round(
            fidelity_outside_guards_db(clean, rendered, sr,
                                       notch_centres), 2)
        # Useful signal away from the notch frequencies must survive.
        voice_mask_energy_before = band_energy(original, sr, 300, 3000)
        voice_mask_energy_after = band_energy(rendered, sr, 300, 3000)
        report["signal_fidelity"]["voice_band_change_db"] = round(
            10 * np.log10((voice_mask_energy_after + 1e-20)
                          / (voice_mask_energy_before + 1e-20)), 3)

    return report


def write_report(report: dict[str, Any], out_dir: str) -> str:
    path = os.path.join(out_dir, "evaluation_report.json")

    def default(o: Any) -> Any:
        import numpy as _np
        if isinstance(o, _np.bool_):
            return bool(o)
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)!r}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=default)
    return os.path.abspath(path)


def build_fixture_bundle(out_dir: str, sample_rate: int = 16000) -> dict[str, str]:
    """Render every fixture WAV used by the evaluation scenarios to disk."""
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}

    clean = synthesize_clean(sample_rate=sample_rate)
    p = os.path.join(out_dir, "clean_baseline.wav")
    write_wav(p, clean, sample_rate)
    paths["clean"] = p

    pops, _ = inject_pops(
        clean, sample_rate,
        centres_ms=[600.0, 1500.0, 2750.0, 3600.0, 5100.0],
        widths_ms=[1.0, 2.0, 1.5, 3.0, 0.8],
        amplitudes=[0.6, 0.8, 0.5, 0.9, 0.7],
    )
    p = os.path.join(out_dir, "pops_only.wav")
    write_wav(p, pops, sample_rate)
    paths["pops"] = p

    hummed, _ = inject_hum(clean, sample_rate)
    p = os.path.join(out_dir, "hum_only.wav")
    write_wav(p, hummed, sample_rate)
    paths["hum"] = p

    both = inject_hum(inject_pops(
        clean, sample_rate,
        centres_ms=[600.0, 1500.0, 2750.0, 3600.0, 5100.0],
        widths_ms=[1.0, 2.0, 1.5, 3.0, 0.8],
        amplitudes=[0.6, 0.8, 0.5, 0.9, 0.7])[0], sample_rate)[0]
    p = os.path.join(out_dir, "pops_and_hum.wav")
    write_wav(p, both, sample_rate)
    paths["both"] = p

    long_d = make_long_defect(clean, sample_rate)
    p = os.path.join(out_dir, "long_defect.wav")
    write_wav(p, long_d, sample_rate)
    paths["long"] = p

    legit = add_legitimate_transients(
        clean, sample_rate, centres_ms=[900.0, 2400.0, 4700.0])
    p = os.path.join(out_dir, "legitimate_transients.wav")
    write_wav(p, legit, sample_rate)
    paths["legitimate_transients"] = p

    return paths
