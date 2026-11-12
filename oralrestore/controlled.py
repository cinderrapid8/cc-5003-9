"""Controlled-record diagnosis scenarios.

A single controlled recording (annotated clear onsets, sustained stable
harmonics and consonant bursts) is run through the real detection + repair
path.  Each of the three damage hypotheses is evaluated independently from
measured evidence; the orchestrator never asserts a root cause, it reports
which hypothesis the bounded evidence supports and which stage any target-
interference or effective-band anomaly maps to.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .diagnostics import (
    DiagnosticBudget,
    DiagnosticCollector,
    diagnostic_replay,
)
from .project import Project
from .repair import notch_hum, repair_click
from .synthetic import (
    build_controlled_recording,
    inject_controlled_hum,
    inject_controlled_pops,
)

# Evidence rule thresholds.  They are deliberately explicit constants so a
# reviewer can audit the verdict rather than trust an opaque score.
WIDTH_RATIO_WIDE = 2.0          # padded window vs. threshold-crossing span
EDGE_JUMP_DELTA_FAULT = 0.5     # repair-induced edge step beyond the original
WINDOW_ENERGY_LOSS_DB = -12.0   # repaired window lost content beyond the pop
EFFECTIVE_BAND_OVERSHOOT = 2.0  # measured notch width vs. requested width
SIDE_BAND_ANALYTIC_LIMIT_DB = -3.0  # analytic filtfilt gain beside the notch
VOICE_BAND_OBSERVED_LIMIT_DB = -0.5


def _matching_segment(annotations: list[dict[str, Any]], start: int,
                      end: int) -> dict[str, Any] | None:
    """Ground-truth annotation overlapping a candidate window, if any."""
    for ann in annotations:
        if ann["start"] < end and start < ann["end"]:
            return ann
    return None


def evaluate_detection_width(evidence: list[dict[str, Any]],
                             annotations: dict[str, Any]
                             ) -> dict[str, Any]:
    """H1: compare threshold spans, padded windows and known content."""
    records = [e for e in evidence
               if e.get("hypothesis") == "H1_detection_too_wide"]
    rows, support = [], 0
    for rec in records:
        m = rec["mask_to_window"]
        ratio = m["window_to_threshold_ratio"]
        label = rec["input_content_classification"]["label"]
        ann = _matching_segment(
            annotations["onset_segments"] + annotations["vowel_segments"],
            m["candidate_window"][0], m["candidate_window"][1])
        wide = ratio >= WIDTH_RATIO_WIDE
        support += int(wide)
        rows.append({
            "cand_id": rec["cand_id"],
            "threshold_span_ms": m["threshold_span_ms"],
            "candidate_window_ms": m["candidate_window_ms"],
            "window_to_threshold_ratio": ratio,
            "edge_pad_samples": m["edge_pad_samples"],
            "classified_content": label,
            "ground_truth_segment": ann["kind"] if ann else None,
            "rule": f"wide if ratio >= {WIDTH_RATIO_WIDE}",
            "supports_hypothesis": wide,
        })
    return {
        "hypothesis": "H1_detection_too_wide",
        "stage_localised": "click_detection",
        "records": rows,
        "n_supporting": support,
        "n_evaluated": len(rows),
        "verdict": ("supported" if support else
                    "not_supported_by_evidence"),
    }


def evaluate_boundary_interpolation(evidence: list[dict[str, Any]]
                                    ) -> dict[str, Any]:
    """H2: repair-induced edge steps and window energy loss."""
    records = [e for e in evidence
               if e.get("hypothesis") == "H2_boundary_interpolation"]
    rows, support = [], 0
    for rec in records:
        faulty = (abs(rec["left_edge_jump_delta"]) >= EDGE_JUMP_DELTA_FAULT
                  or abs(rec["right_edge_jump_delta"])
                  >= EDGE_JUMP_DELTA_FAULT
                  or rec["window_energy_change_db"] <= WINDOW_ENERGY_LOSS_DB)
        support += int(faulty)
        rows.append({
            "cand_id": rec["cand_id"],
            "op_id": rec["op_id"],
            "algorithm": rec["algorithm"],
            "window_ms": rec["window_ms"],
            "left_edge_jump_delta": rec["left_edge_jump_delta"],
            "right_edge_jump_delta": rec["right_edge_jump_delta"],
            "window_energy_change_db": rec["window_energy_change_db"],
            "outside_window_identical":
                rec["outside_window_samples_identical"],
            "rule": (f"fault if either edge-jump delta >= "
                     f"{EDGE_JUMP_DELTA_FAULT} or energy change <= "
                     f"{WINDOW_ENERGY_LOSS_DB} dB"),
            "supports_hypothesis": faulty,
        })
    return {
        "hypothesis": "H2_boundary_interpolation",
        "stage_localised": "click_repair",
        "records": rows,
        "n_supporting": support,
        "n_evaluated": len(rows),
        "verdict": ("supported" if support else
                    "not_supported_by_evidence"),
    }


def evaluate_filter_mismatch(evidence: list[dict[str, Any]]
                             ) -> dict[str, Any]:
    """H3: measured notch response vs. requested parameters."""
    records = [e for e in evidence
               if e.get("hypothesis") == "H3_filter_parameter_mismatch"]
    rows, support = [], 0
    for rec in records:
        req = rec["requested"]
        overshoot = (rec["measured_effective_width_hz"]
                     >= EFFECTIVE_BAND_OVERSHOOT * req["bandwidth_hz"])
        side_loss = (rec["worst_analytic_side_gain_db"]
                     <= SIDE_BAND_ANALYTIC_LIMIT_DB)
        voice_loss = (rec["observed_voice_band_300_3000_change_db"]
                      <= VOICE_BAND_OBSERVED_LIMIT_DB)
        faulty = overshoot or side_loss or voice_loss
        support += int(faulty)
        rows.append({
            "cand_id": rec["cand_id"],
            "op_id": rec["op_id"],
            "requested_centre_hz": req["centre_hz"],
            "requested_bandwidth_hz": req["bandwidth_hz"],
            "measured_minus3db_band_hz": rec["measured_minus3db_band_hz"],
            "measured_effective_width_hz":
                rec["measured_effective_width_hz"],
            "target_suppression_db":
                rec["target_interference"]["suppression_db"],
            "worst_analytic_side_gain_db":
                rec["worst_analytic_side_gain_db"],
            "observed_adjacent_band_change_db":
                rec["observed_adjacent_10hz_band_change_db"],
            "observed_voice_band_change_db":
                rec["observed_voice_band_300_3000_change_db"],
            "anomalies": rec["anomalies"],
            "rule": (f"fault if effective width >= "
                     f"{EFFECTIVE_BAND_OVERSHOOT}x requested, worst analytic "
                     f"side gain <= {SIDE_BAND_ANALYTIC_LIMIT_DB} dB or "
                     f"observed voice band <= "
                     f"{VOICE_BAND_OBSERVED_LIMIT_DB} dB"),
            "supports_hypothesis": faulty,
        })
    return {
        "hypothesis": "H3_filter_parameter_mismatch",
        "stage_localised": "hum_notch",
        "records": rows,
        "n_supporting": support,
        "n_evaluated": len(rows),
        "verdict": ("supported" if support else
                    "not_supported_by_evidence"),
    }


def validate_content_classification(evidence: list[dict[str, Any]],
                                    annotations: dict[str, Any]
                                    ) -> dict[str, Any]:
    """Check the consonant/vowel classifier against the controlled truth.

    Windows centred on an injected pop are transient-dominated by construction
    and are expected to classify as consonant_transient regardless of the
    underlying region, so they are reported separately rather than counted as
    vowel misclassifications.
    """
    truth = annotations["onset_segments"] + annotations["vowel_segments"]
    pop_centres = [int(p["centre_ms"] / 1000.0
                       * annotations["sample_rate"])
                   for p in annotations.get("pop_sites", [])]
    rows = []
    correct = total = 0
    pop_window_rows = []
    for rec in (e for e in evidence
                if e.get("hypothesis") == "H1_detection_too_wide"):
        m = rec["mask_to_window"]
        ann = _matching_segment(truth, m["candidate_window"][0],
                                m["candidate_window"][1])
        if ann is None:
            continue
        expected = ("consonant_transient"
                    if ann["kind"] in ("consonant_burst", "vowel_onset")
                    else "sustained_voiced")
        got = rec["input_content_classification"]["label"]
        centre = (m["candidate_window"][0] + m["candidate_window"][1]) // 2
        on_pop = any(abs(centre - pc) < 0.01 * annotations["sample_rate"]
                     for pc in pop_centres)
        row = {"cand_id": rec["cand_id"],
               "ground_truth": ann["kind"],
               "expected_label": expected,
               "classified_as": got,
               "window_contains_injected_pop": on_pop}
        if on_pop:
            row["transient_classification_expected"] = (
                got == "consonant_transient")
            pop_window_rows.append(row)
            continue
        row["correct"] = got == expected
        correct += int(row["correct"])
        total += 1
        rows.append(row)
    return {"non_pop_windows": rows,
            "pop_windows_transient_classification": pop_window_rows,
            "correct": correct, "total": total,
            "accuracy": round(correct / total, 3) if total else None}


def run_controlled_diagnosis(project_dir: str, *,
                             sample_rate: int = 16000,
                             budget: DiagnosticBudget | None = None,
                             wide_threshold_sigma: float = 7.5,
                             narrow_threshold_sigma: float = 8.0,
                             mismatched_bandwidth_hz: float = 12.0,
                             ) -> dict[str, Any]:
    """Full controlled scenario: build, damage, repair and collect evidence.

    Two deliberate parameter probes are included so the evidence can actually
    discriminate the hypotheses without pre-judging one:

      * a low-sigma click detection pass demonstrates H1 (over-wide windows);
      * an over-wide notch probe demonstrates H3 (effective-band overshoot).
    Neither probe is written into the project's repair graph.
    """
    budget = budget or DiagnosticBudget()
    recording = build_controlled_recording(sample_rate=sample_rate)
    damaged, pop_sites = inject_controlled_pops(recording)
    damaged, hum_components = inject_controlled_hum(damaged, recording)

    import os
    from .audioio import write_wav
    os.makedirs(project_dir, exist_ok=True)
    src = os.path.join(project_dir, "controlled_input.wav")
    write_wav(src, damaged, sample_rate)
    proj = Project.create(src, os.path.join(project_dir, "project"))

    # Standard, conservative detection + expert acceptance on the graph.
    proj.run_click_detection(threshold_sigma=narrow_threshold_sigma)
    for c in list(proj.click_candidates):
        if c.repairable:
            proj.accept_click(c.cand_id)
    proj.run_hum_detection(line_frequencies=(50.0, 60.0),
                           snr_threshold_db=12.0, bandwidth_hz=2.0)
    true_lines = {round(h["frequency_hz"], 1) for h in hum_components}
    for c in list(proj.hum_candidates):
        if round(c.frequency, 1) in true_lines:
            proj.accept_hum(c.cand_id)
        else:
            proj.exclude_hum(
                c.cand_id,
                reason="controlled clip: off-grid speech harmonic, keep")

    collector = DiagnosticCollector(
        sample_rate=sample_rate, source_sha256=proj.audio.source_sha256,
        budget=budget)
    rendered, stages = diagnostic_replay(
        proj, collector,
        detect_kwargs_click={"threshold_sigma": narrow_threshold_sigma},
        detect_kwargs_hum={"line_frequencies": (50.0, 60.0),
                           "snr_threshold_db": 12.0, "bandwidth_hz": 2.0})
    baseline_render = proj.render()
    assert np.array_equal(rendered, baseline_render), (
        "diagnostic replay must not alter the audio path")

    # Probe A: deliberately permissive detection to expose H1 evidence.
    wide_collector = DiagnosticCollector(
        sample_rate=sample_rate, source_sha256=proj.audio.source_sha256,
        budget=budget)
    from .diagnostics import diagnose_click_detection
    diagnose_click_detection(
        damaged, sample_rate, wide_collector,
        threshold_sigma=wide_threshold_sigma)
    wide_h1 = evaluate_detection_width(
        wide_collector.evidence, recording.annotations())

    # Probe B: an over-wide notch on a true line, kept out of the project,
    # exposing H3 effective-band overshoot and adjacent-band collateral.
    probe_collector = DiagnosticCollector(
        sample_rate=sample_rate, source_sha256=proj.audio.source_sha256,
        budget=budget)
    fc = hum_components[0]["frequency_hz"]
    probe_op = {
        "op_id": "probe-wide-notch", "op": "hum_notch",
        "cand_id": "probe",
        "params": {"frequency_hz": fc,
                   "bandwidth_hz": mismatched_bandwidth_hz,
                   "quality_scale": 1.0}}
    res = notch_hum(damaged, sample_rate, frequency=fc,
                    bandwidth_hz=mismatched_bandwidth_hz)
    from .diagnostics import diagnose_notch
    diagnose_notch(damaged, res.samples, sample_rate, probe_op,
                   probe_collector)
    probe_h3 = evaluate_filter_mismatch(probe_collector.evidence)

    # Probe C: same pops repaired with spline vs AR to vary H2 boundary
    # evidence under identical windows (still outside the saved graph).
    from .diagnostics import diagnose_click_interpolation
    spline_collector = DiagnosticCollector(
        sample_rate=sample_rate, source_sha256=proj.audio.source_sha256,
        budget=budget)
    y = np.array(damaged, dtype=np.float32, copy=True)
    for i, op in enumerate([o for o in proj.operations
                            if o["op"] == "click_repair"], start=1):
        before = y
        res = repair_click(y, op["start"], op["end"], sample_rate,
                           algorithm="spline",
                           max_gap_ms=op.get("max_gap_ms", 10.0))
        y = res.samples
        diagnose_click_interpolation(
            before, y, op["start"], op["end"], sample_rate,
            {"op_id": f"probe-spline-{i:04d}",
             "cand_id": op["cand_id"], "algorithm": "spline"},
            spline_collector)
    probe_h2 = evaluate_boundary_interpolation(spline_collector.evidence)

    evidence = collector.evidence
    report = {
        "scenario": "controlled_collateral_diagnosis",
        "source_sha256": proj.audio.source_sha256,
        "sample_rate": sample_rate,
        "annotations": recording.annotations(),
        "budget": collector.to_report()["budget"],
        "sampling": collector.to_report()["sampling"],
        "replay": {
            "stages": stages,
            "render_matches_project_render": True,
            "n_operations": len(proj.operations),
        },
        "hypotheses": {
            "H1_detection_too_wide": evaluate_detection_width(
                evidence, recording.annotations()),
            "H2_boundary_interpolation":
                evaluate_boundary_interpolation(evidence),
            "H3_filter_parameter_mismatch":
                evaluate_filter_mismatch(evidence),
        },
        "parameter_probes": {
            "permissive_detection_sigma": wide_threshold_sigma,
            "permissive_detection_H1": wide_h1,
            "wide_notch_bandwidth_hz": mismatched_bandwidth_hz,
            "wide_notch_H3": probe_h3,
            "spline_boundary_H2": probe_h2,
        },
        "classifier_validation": validate_content_classification(
            evidence, recording.annotations()),
        "evidence_chain": collector.to_report(),
    }
    return report
