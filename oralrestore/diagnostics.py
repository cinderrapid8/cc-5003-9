"""Collateral-damage diagnostics for pop repair and mains-hum suppression.

The restoration engine itself is never instrumented: diagnostics run as an
observer beside it and only read buffers before/after each processing stage.
Everything here is therefore switchable, and a diagnostic replay renders
byte-identical audio to :meth:`Project.render` (the CLI verifies this).

Three failure families are kept separate on purpose.  None is assumed to be
the root cause; each gets its own evidence chain:

  H1 detection too wide  threshold mask vs. padded candidate window, and the
                         input content the window would overwrite;
  H2 boundary interpolation  edge continuity and local-energy change of the
                         accepted interpolation window;
  H3 filter parameter mismatch  measured notch response, target-interference
                         suppression and useful-band loss of each IIR notch.

Every record links four things: the candidate, the processing stage, the
input segment (sample range + content hash, never a retained audio copy) and
the measured quantities.  Sampling is bounded by :class:`DiagnosticBudget`
so a long recording cannot turn diagnostics into an unbounded log or memory
cost.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal as sps

from .detectors import detect_clicks_with_trace, detect_hum_with_trace
from .repair import notch_hum, repair_click

# ---------------------------------------------------------------------------
# Bounded sampling budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticBudget:
    """Hard caps on diagnostic sampling and log size.

    max_segments bounds how many input segments are referenced; max_evidence
    bounds the number of evidence records; max_window_ms bounds how much of a
    segment is actually hashed/analysed (analysis runs on a centred crop, so
    cost is independent of total recording length).
    """

    max_segments: int = 128
    max_evidence: int = 512
    max_window_ms: float = 40.0
    max_grid_bins: int = 256

    def exceeded_segments(self, used: int) -> bool:
        return used >= self.max_segments

    def exceeded_evidence(self, used: int) -> bool:
        return used >= self.max_evidence


# ---------------------------------------------------------------------------
# Input-segment references and the evidence collector
# ---------------------------------------------------------------------------


def segment_hash(samples: np.ndarray, start: int, end: int) -> str:
    """Content hash of an input segment so evidence can be re-bound later."""
    view = np.ascontiguousarray(samples[start:end], dtype=np.float32)
    return hashlib.sha256(view.tobytes()).hexdigest()


def make_segment_ref(samples: np.ndarray, start: int, end: int,
                     sample_rate: int, budget: DiagnosticBudget,
                     *, role: str, cand_id: str | None = None,
                     ) -> dict[str, Any]:
    """Reference an input segment by range + hash; no audio is retained."""
    start = max(0, int(start))
    end = min(len(samples), int(end))
    cap = max(1, int(round(sample_rate * budget.max_window_ms / 1000.0)))
    if end - start > cap:  # centre the bounded analysis crop
        mid = (start + end) // 2
        start = max(0, mid - cap // 2)
        end = min(len(samples), start + cap)
    return {
        "role": role,
        "cand_id": cand_id,
        "start": start,
        "end": end,
        "n_samples": end - start,
        "start_ms": round(1000.0 * start / sample_rate, 3),
        "end_ms": round(1000.0 * end / sample_rate, 3),
        "content_sha256": segment_hash(samples, start, end),
        "capped_to_budget": end - start >= cap,
    }


@dataclass
class DiagnosticCollector:
    """Accumulates a bounded, stage-linked evidence chain."""

    sample_rate: int
    source_sha256: str
    budget: DiagnosticBudget = field(default_factory=DiagnosticBudget)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def take_segment(self, samples: np.ndarray, start: int, end: int,
                     *, role: str, cand_id: str | None = None
                     ) -> dict[str, Any] | None:
        if self.budget.exceeded_segments(len(self.segments)):
            self.skipped.append({
                "role": role, "cand_id": cand_id,
                "reason": f"segment budget {self.budget.max_segments} reached",
            })
            return None
        ref = make_segment_ref(samples, start, end, self.sample_rate,
                               self.budget, role=role, cand_id=cand_id)
        self.segments.append(ref)
        return ref

    def add_evidence(self, record: dict[str, Any]) -> bool:
        if self.budget.exceeded_evidence(len(self.evidence)):
            self.skipped.append({
                "role": record.get("stage"),
                "cand_id": record.get("cand_id"),
                "reason": f"evidence budget {self.budget.max_evidence} reached",
            })
            return False
        self.evidence.append(record)
        return True

    def to_report(self) -> dict[str, Any]:
        return {
            "diagnostics_enabled": True,
            "source_sha256": self.source_sha256,
            "sample_rate": self.sample_rate,
            "budget": {
                "max_segments": self.budget.max_segments,
                "max_evidence": self.budget.max_evidence,
                "max_window_ms": self.budget.max_window_ms,
                "max_grid_bins": self.budget.max_grid_bins,
            },
            "sampling": {
                "segments_recorded": len(self.segments),
                "evidence_records": len(self.evidence),
                "skipped": list(self.skipped),
            },
            "segments": self.segments,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Consonant-transient vs. sustained-vowel classification
# ---------------------------------------------------------------------------

# Thresholds are fixed constants, never tuned per file, so the same rule is
# applied to every segment and its behaviour is auditable.
FLATNESS_VOICED_MAX = 0.12
ZERO_CROSSING_VOICED_MAX = 0.35
ONSET_RATIO_CONSONANT_MIN = 9.0
MIN_ANALYSIS_MS = 1.0

# H3 analytic limits.  A correctly dimensioned notch (bw = 2 Hz) attenuates
# the closest useful band by ~2 dB via filtfilt; a 12 Hz mismatch reaches
# ~-5.3 dB at +1 Hz and stays below -1.4 dB at +10 Hz, so -3 dB separates the
# two parameter choices without relying on short-clip energy ratios.
SIDE_BAND_ANALYTIC_LIMIT_DB = -3.0
VOICE_BAND_OBSERVED_LIMIT_DB = -0.5


def classify_segment(samples: np.ndarray, start: int, end: int,
                     sample_rate: int) -> dict[str, Any]:
    """Distinguish a consonant-like transient from a sustained voiced vowel.

    The decision uses four deterministic acoustic descriptors of the segment
    itself (spectral flatness, zero-crossing rate, energy concentration in the
    first quarter, onset energy ratio).  Both hypotheses' descriptors are
    returned with the label so reviewers can see exactly which rule fired;
    nothing about the defect or any later stage enters the classification.
    """
    lo = max(0, int(start))
    hi = min(len(samples), int(end))
    x = np.asarray(samples[lo:hi], dtype=np.float64)
    if x.size < max(8, int(sample_rate * MIN_ANALYSIS_MS / 1000.0)):
        return {"label": "undetermined", "reason": "segment too short",
                "features": {}}
    # Spectral flatness: geometric/arithmetic mean of the magnitude spectrum.
    # Voiced vowels are harmonic (low flatness); consonant bursts are noise-like.
    X = np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-12
    flatness = float(np.exp(np.mean(np.log(X))) / np.mean(X))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    quarter = max(1, x.size // 4)
    first_energy = float(np.mean(x[:quarter] ** 2) + 1e-20)
    rest_energy = float(np.mean(x[quarter:] ** 2) + 1e-20)
    onset_ratio = float(first_energy / rest_energy)
    total = float(np.sum(x ** 2) + 1e-20)
    first_share = float(np.sum(x[:quarter] ** 2) / total)
    features = {
        "spectral_flatness": round(flatness, 4),
        "zero_crossing_rate": round(zcr, 4),
        "onset_energy_ratio": round(onset_ratio, 3),
        "first_quarter_energy_share": round(first_share, 4),
        "rms": round(float(np.sqrt(np.mean(x ** 2) + 1e-20)), 6),
    }
    consonant = (flatness > FLATNESS_VOICED_MAX
                 or zcr > ZERO_CROSSING_VOICED_MAX
                 or onset_ratio >= ONSET_RATIO_CONSONANT_MIN)
    label = "consonant_transient" if consonant else "sustained_voiced"
    fired = []
    if flatness > FLATNESS_VOICED_MAX:
        fired.append(f"flatness>{FLATNESS_VOICED_MAX}")
    if zcr > ZERO_CROSSING_VOICED_MAX:
        fired.append(f"zcr>{ZERO_CROSSING_VOICED_MAX}")
    if onset_ratio >= ONSET_RATIO_CONSONANT_MIN:
        fired.append(f"onset_ratio>={ONSET_RATIO_CONSONANT_MIN}")
    return {"label": label, "rules_fired": fired, "features": features}


# ---------------------------------------------------------------------------
# H1: detection width (mask -> candidate window -> input content)
# ---------------------------------------------------------------------------


def diagnose_click_detection(samples: np.ndarray, sample_rate: int,
                             collector: DiagnosticCollector,
                             **detector_kwargs: Any) -> list[dict[str, Any]]:
    """Run click detection and bind each mask region to its input segment."""
    candidates, trace = detect_clicks_with_trace(
        samples, sample_rate, **detector_kwargs)
    bounded_trace = dict(trace)
    # Bound per-region detail even if the detector itself returned many spans.
    bounded_trace["regions_shown"] = min(
        len(trace["regions"]), collector.budget.max_grid_bins)
    bounded_trace["regions"] = trace["regions"][:collector.budget.max_grid_bins]
    collector.add_evidence(bounded_trace)
    for cand in candidates:
        if collector.budget.exceeded_evidence(len(collector.evidence)):
            break
        ref = collector.take_segment(
            samples, cand.start, cand.end, role="click_candidate_window",
            cand_id=cand.cand_id)
        if ref is None:
            break
        content = classify_segment(
            samples, cand.start, cand.end, sample_rate)
        region = next((r for r in trace["regions"]
                       if r["cand_id"] == cand.cand_id), None)
        collector.add_evidence({
            "hypothesis": "H1_detection_too_wide",
            "stage": "click_detection",
            "cand_id": cand.cand_id,
            "segment_ref": ref,
            "input_content_classification": content,
            "mask_to_window": region,
            "detector_score_sigma": round(float(cand.score), 3),
            "repairable": cand.repairable,
            "reject_reason": cand.reject_reason,
        })
    return [c.to_dict() for c in candidates]


def diagnose_hum_detection(samples: np.ndarray, sample_rate: int,
                           collector: DiagnosticCollector,
                           **detector_kwargs: Any) -> list[dict[str, Any]]:
    """Run hum detection and keep the bounded grid search trace."""
    candidates, trace = detect_hum_with_trace(
        samples, sample_rate, **detector_kwargs)
    trace = dict(trace)
    trace["grid_searches_shown"] = min(
        len(trace["grid_searches"]), collector.budget.max_grid_bins)
    trace["grid_searches"] = trace["grid_searches"][
        :collector.budget.max_grid_bins]
    collector.add_evidence(trace)
    return [c.to_dict() for c in candidates]


# ---------------------------------------------------------------------------
# H2: boundary interpolation
# ---------------------------------------------------------------------------


def _edge_jump(prev_vals: np.ndarray, next_vals: np.ndarray) -> float:
    """Mean absolute step across a boundary, normalised by local RMS."""
    scale = float(np.sqrt(
        np.mean(np.concatenate((prev_vals, next_vals)) ** 2) + 1e-20))
    return float(np.mean(np.abs(next_vals - prev_vals)) / scale)


def diagnose_click_interpolation(before: np.ndarray, after: np.ndarray,
                                 start: int, end: int, sample_rate: int,
                                 op: dict[str, Any],
                                 collector: DiagnosticCollector) -> None:
    """Measure edge continuity and energy change of one repaired window.

    Absolute edge steps are meaningless on periodic voiced material (a normal
    sample-to-sample step already normalises to ~1.0 there), so continuity is
    judged by the *delta* the repair introduces at each edge versus the same
    boundary in the untouched input.  A good interpolation leaves that delta
    near zero; a boundary fault makes the repaired edge substantially worse
    than the original rather than merely large in absolute terms.
    """
    cand_id = op.get("cand_id")
    ref_before = collector.take_segment(
        before, start, end, role="pre_interpolation_window",
        cand_id=cand_id)
    ref_after = collector.take_segment(
        after, start, end, role="post_interpolation_window",
        cand_id=cand_id)
    if ref_before is None or ref_after is None:
        return
    guard = max(2, min(8, (end - start) // 2))

    def jumps(buf: np.ndarray) -> tuple[float, float]:
        return (
            _edge_jump(buf[max(0, start - guard):start],
                       buf[start:min(len(buf), start + guard)]),
            _edge_jump(buf[max(0, end - guard):end],
                       buf[end:min(len(buf), end + guard)]))

    left_before, right_before = jumps(before)
    left_after, right_after = jumps(after)
    e_before = float(np.mean(before[start:end].astype(np.float64) ** 2)
                     + 1e-20)
    e_after = float(np.mean(after[start:end].astype(np.float64) ** 2)
                    + 1e-20)
    energy_change_db = float(10.0 * np.log10(e_after / e_before))
    outside = np.ones(len(before), dtype=bool)
    outside[start:end] = False
    outside_identical = bool(np.array_equal(
        before[outside], after[outside]))
    collector.add_evidence({
        "hypothesis": "H2_boundary_interpolation",
        "stage": "click_repair",
        "cand_id": cand_id,
        "op_id": op.get("op_id"),
        "algorithm": op.get("algorithm"),
        "segment_ref_before": ref_before,
        "segment_ref_after": ref_after,
        "window_ms": [round(1000.0 * start / sample_rate, 3),
                      round(1000.0 * end / sample_rate, 3)],
        "edge_guard_samples": guard,
        "left_edge_jump_before": round(left_before, 4),
        "left_edge_jump_after": round(left_after, 4),
        "right_edge_jump_before": round(right_before, 4),
        "right_edge_jump_after": round(right_after, 4),
        "left_edge_jump_delta": round(left_after - left_before, 4),
        "right_edge_jump_delta": round(right_after - right_before, 4),
        "window_energy_change_db": round(energy_change_db, 3),
        "outside_window_samples_identical": outside_identical,
    })


# ---------------------------------------------------------------------------
# H3: filter parameter mismatch
# ---------------------------------------------------------------------------


def _band_energy(x: np.ndarray, sample_rate: int, f_lo: float,
                 f_hi: float) -> float:
    X = np.abs(np.fft.rfft(x.astype(np.float64) * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    sel = (f >= f_lo) & (f <= f_hi)
    return float(np.sum(X[sel] ** 2))


def diagnose_notch(before: np.ndarray, after: np.ndarray,
                   sample_rate: int, op: dict[str, Any],
                   collector: DiagnosticCollector) -> None:
    """Measure the real response of one notch and the bands it affected."""
    p = op["params"]
    fc = float(p["frequency_hz"])
    bw = float(p["bandwidth_hz"])
    q_scale = float(p.get("quality_scale", 1.0))
    q = fc / (bw * q_scale)
    b, a = sps.iirnotch(w0=fc, Q=q, fs=sample_rate)

    # Measured filter state: response at centre, at the nominal -3 dB band
    # edges, and one octave away where useful content must be untouched.
    # freqz takes frequencies in Hz when fs is supplied; filtfilt applies the
    # filter twice (zero phase), so the realised gain is the squared magnitude.
    probe = np.array([fc, fc - bw / 2, fc + bw / 2,
                      fc / 2.0, fc * 2.0])
    probe = probe[(probe > 0) & (probe < sample_rate / 2.0)]
    _, h = sps.freqz(b, a, worN=probe, fs=sample_rate)
    response = [
        {"frequency_hz": round(float(f), 2),
         "single_pass_gain_db": round(float(20 * np.log10(
             np.maximum(np.abs(g), 1e-12))), 3),
         "filtfilt_gain_db": round(float(20 * np.log10(
             np.maximum(np.abs(g) ** 2, 1e-12))), 3)}
        for f, g in zip(probe, h)
    ]
    target_before = _band_energy(before, sample_rate, fc - bw / 2, fc + bw / 2)
    target_after = _band_energy(after, sample_rate, fc - bw / 2, fc + bw / 2)
    target_suppression_db = float(10 * np.log10(
        (target_before + 1e-20) / (target_after + 1e-20)))
    # Effective band: frequencies around fc at which the realised (filtfilt)
    # response is more than 3 dB down.  A dense analytic sweep is used instead
    # of the signal's FFT bins because the latter are spaced ~1 Hz apart for
    # short clips, coarser than the requested 2 Hz notch width.
    sweep = np.linspace(max(1.0, fc - 4.0 * bw),
                        min(sample_rate / 2.0 - 1.0, fc + 4.0 * bw), 4001)
    _, hs = sps.freqz(b, a, worN=sweep, fs=sample_rate)
    realised = np.abs(hs) ** 2
    down3 = realised <= 10 ** (-3.0 / 20.0)
    flagged = sweep[down3]
    if flagged.size:
        effective_band_hz = [round(float(flagged.min()), 2),
                             round(float(flagged.max()), 2)]
        effective_width_hz = round(float(flagged.max() - flagged.min()), 2)
    else:
        effective_band_hz = [round(fc, 2), round(fc, 2)]
        effective_width_hz = 0.0
    # Analytic attenuation of the useful 10 Hz bands immediately outside the
    # requested notch.  This is a property of the filter alone, so it is
    # immune to the spectral leakage that corrupts energy-ratio measurements
    # on short clips.
    side_offsets = np.array([1.0, 2.0, 5.0, 10.0])
    side_freqs = np.concatenate([fc - bw / 2 - side_offsets,
                                 fc + bw / 2 + side_offsets])
    side_freqs = side_freqs[(side_freqs > 0)
                            & (side_freqs < sample_rate / 2.0)]
    _, h_side = sps.freqz(b, a, worN=side_freqs, fs=sample_rate)
    analytic_side_gain_db = [
        {"frequency_hz": round(float(f), 2),
         "filtfilt_gain_db": round(float(20 * np.log10(
             np.maximum(np.abs(g) ** 2, 1e-12))), 4)}
        for f, g in zip(side_freqs, h_side)]
    worst_side_gain_db = float(min(
        row["filtfilt_gain_db"] for row in analytic_side_gain_db))
    # Observed energy ratios supplement the analytic response: they capture
    # what actually happened to this recording at each band.
    side_before = _band_energy(before, sample_rate,
                               fc - bw / 2 - 10.0, fc - bw / 2) + \
        _band_energy(before, sample_rate,
                     fc + bw / 2, fc + bw / 2 + 10.0)
    side_after = _band_energy(after, sample_rate,
                              fc - bw / 2 - 10.0, fc - bw / 2) + \
        _band_energy(after, sample_rate,
                     fc + bw / 2, fc + bw / 2 + 10.0)
    side_change_db = float(10 * np.log10(
        (side_after + 1e-20) / (side_before + 1e-20)))
    voice_before = _band_energy(before, sample_rate, 300.0, 3000.0)
    voice_after = _band_energy(after, sample_rate, 300.0, 3000.0)
    voice_change_db = float(10 * np.log10(
        (voice_after + 1e-20) / (voice_before + 1e-20)))
    # filtfilt doubles the filter order at both signal ends; measure the
    # actual edge energy shift so edge effects are not mistaken for a
    # centre-frequency problem.
    edge = max(1, sample_rate // 100)
    edge_before = float(np.mean(before[:edge].astype(np.float64) ** 2))
    edge_after = float(np.mean(after[:edge].astype(np.float64) ** 2))
    edge_change_db = float(10 * np.log10(
        (edge_after + 1e-20) / (edge_before + 1e-20)))
    anomalies = []
    if effective_width_hz > 2.0 * bw:
        anomalies.append("effective_band_more_than_2x_requested_bandwidth")
    if worst_side_gain_db < SIDE_BAND_ANALYTIC_LIMIT_DB:
        anomalies.append("analytic_adjacent_band_over_attenuated")
    if voice_change_db < VOICE_BAND_OBSERVED_LIMIT_DB:
        anomalies.append("observed_voice_band_attenuated")
    ref = collector.take_segment(
        before, 0, len(before), role="notch_input_clip",
        cand_id=op.get("cand_id"))
    collector.add_evidence({
        "hypothesis": "H3_filter_parameter_mismatch",
        "stage": "hum_notch",
        "cand_id": op.get("cand_id"),
        "op_id": op.get("op_id"),
        "segment_ref": ref,
        "requested": {
            "centre_hz": fc, "bandwidth_hz": bw,
            "quality_scale": q_scale, "Q": round(q, 3)},
        "measured_response_db": response,
        "measured_minus3db_band_hz": effective_band_hz,
        "measured_effective_width_hz": effective_width_hz,
        "target_interference": {
            "band_hz": [round(fc - bw / 2, 2), round(fc + bw / 2, 2)],
            "suppression_db": round(target_suppression_db, 3)},
        "analytic_side_band_gain_db": analytic_side_gain_db,
        "worst_analytic_side_gain_db": round(worst_side_gain_db, 4),
        "observed_adjacent_10hz_band_change_db": round(side_change_db, 4),
        "observed_voice_band_300_3000_change_db": round(voice_change_db, 4),
        "filtfilt_edge_energy_change_db": round(edge_change_db, 4),
        "filter_state": {
            "sos_or_ba": "ba",
            "b": [round(float(v), 9) for v in b],
            "a": [round(float(v), 9) for v in a],
            "application": "filtfilt_zero_phase",
            "padlen": min(3 * max(len(a), len(b)),
                          max(0, len(before) - 1))},
        "anomalies": anomalies,
    })


# ---------------------------------------------------------------------------
# Stage-localised diagnostic replay (audio path identical to Project.render)
# ---------------------------------------------------------------------------


def diagnostic_replay(project: Any, collector: DiagnosticCollector,
                      *, detect_kwargs_click: dict[str, Any] | None = None,
                      detect_kwargs_hum: dict[str, Any] | None = None,
                      ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Replay a project's graph with an observer, returning the same audio.

    Detection is re-run only to attach H1/H3 input-side evidence; the render
    itself replays exactly the recorded operations through the unmodified
    repair functions.  The caller is expected to assert the returned buffer
    equals :meth:`Project.render`.
    """
    samples = project.audio.samples
    sr = project.audio.sample_rate
    click_diag: list[dict[str, Any]] = []
    hum_diag: list[dict[str, Any]] = []
    if detect_kwargs_click is not None:
        click_diag = diagnose_click_detection(
            samples, sr, collector, **detect_kwargs_click)
    if detect_kwargs_hum is not None:
        hum_diag = diagnose_hum_detection(
            samples, sr, collector, **detect_kwargs_hum)

    y = np.array(samples, dtype=np.float32, copy=True)
    stages: list[dict[str, Any]] = []
    for op in project.operations:
        before = y
        if op["op"] == "click_repair":
            res = repair_click(
                y, op["start"], op["end"], sr,
                algorithm=op["algorithm"],
                max_gap_ms=op.get("max_gap_ms", 10.0),
                **op.get("params", {}))
            y = res.samples
            diagnose_click_interpolation(
                before, y, op["start"], op["end"], sr, op, collector)
        elif op["op"] == "hum_notch":
            p = op["params"]
            res = notch_hum(
                y, sr, frequency=p["frequency_hz"],
                bandwidth_hz=p["bandwidth_hz"],
                quality_scale=p.get("quality_scale", 1.0))
            y = res.samples
            diagnose_notch(before, y, sr, op, collector)
        else:
            raise RuntimeError(f"unknown operation {op['op']}")
        stages.append({"op_id": op["op_id"], "op": op["op"],
                       "cand_id": op.get("cand_id"), "stage_index": len(stages)})
    collector.add_evidence({
        "stage": "replay_summary",
        "stages": stages,
        "n_operations": len(project.operations),
        "detection_evidence": {
            "click_candidates_seen": len(click_diag),
            "hum_candidates_seen": len(hum_diag)},
    })
    return y, stages
