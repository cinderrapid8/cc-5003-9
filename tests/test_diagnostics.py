"""Tests for the collateral-damage diagnostics.

The brief requires diagnostics that:
  * can be switched on/off and never alter the rendered audio;
  * bind candidates, masks, filter state and input segments together;
  * keep the three damage hypotheses (detection width, boundary
    interpolation, filter mismatch) independently evidenced, with no root
    cause assumed;
  * separate consonant transients from sustained vowels on a controlled
    recording with known ground truth;
  * record target interference and effective band per stage;
  * bound their sampling and logging cost via an explicit budget;
  * expose a CLI command whose replay matches the same evidence on rerun.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from oralrestore.audioio import write_wav
from oralrestore.controlled import run_controlled_diagnosis
from oralrestore.detectors import (
    detect_clicks,
    detect_clicks_with_trace,
    detect_hum_with_trace,
)
from oralrestore.diagnostics import (
    DiagnosticBudget,
    DiagnosticCollector,
    classify_segment,
    diagnose_click_detection,
    diagnose_click_interpolation,
    diagnose_notch,
    diagnostic_replay,
    segment_hash,
)
from oralrestore.project import Project
from oralrestore.repair import notch_hum, repair_click
from oralrestore.synthetic import (
    build_controlled_recording,
    inject_controlled_hum,
    inject_controlled_pops,
    inject_pops,
    synthesize_clean,
)

SR = 16000


@pytest.fixture()
def controlled():
    rec = build_controlled_recording(sample_rate=SR)
    damaged, _ = inject_controlled_pops(rec)
    damaged, _ = inject_controlled_hum(damaged, rec)
    return rec, damaged


@pytest.fixture()
def project(tmp_path, controlled):
    _, damaged = controlled
    src = tmp_path / "controlled.wav"
    write_wav(str(src), damaged, SR)
    proj = Project.create(str(src), str(tmp_path / "proj"))
    proj.run_click_detection(threshold_sigma=8.0)
    for c in list(proj.click_candidates):
        if c.repairable:
            proj.accept_click(c.cand_id)
    proj.run_hum_detection(line_frequencies=(50.0, 60.0),
                           snr_threshold_db=12.0, bandwidth_hz=2.0)
    for c in list(proj.hum_candidates):
        if abs(c.frequency - 120.0) <= 1.5:
            proj.exclude_hum(c.cand_id, reason="speech harmonic")
        else:
            proj.accept_hum(c.cand_id)
    return proj


# ---------------------------------------------------------------------------
# Controlled recording ground truth
# ---------------------------------------------------------------------------


class TestControlledRecording:
    def test_annotated_onsets_vowels_and_bursts(self):
        rec = build_controlled_recording(sample_rate=SR)
        ann = rec.annotations()
        assert len(ann["vowel_segments"]) == 3
        assert len(ann["onset_segments"]) == 6  # 3 vowel onsets + 3 bursts
        assert len(rec.pop_sites) == 0 and len(rec.hum_components) == 0
        kinds = {o["kind"] for o in ann["onset_segments"]}
        assert kinds == {"vowel_onset", "consonant_burst"}
        # Every vowel shares the stable f0, off the 50/60 Hz harmonic grid.
        assert all(v["f0_hz"] == 110.0 for v in ann["vowel_segments"])
        grid = {50 * h for h in range(1, 5)} | {60 * h for h in range(1, 4)}
        assert 110.0 not in grid

    def test_clean_carrier_candidates_only_at_consonant_bursts(self):
        rec = build_controlled_recording(sample_rate=SR)
        cands = detect_clicks(rec.samples, SR)
        assert cands, "consonant bursts must still surface as candidates"
        bursts = [o for o in rec.onset_segments
                  if o["kind"] == "consonant_burst"]
        # Every candidate sits on an annotated consonant burst; none fire on
        # the sustained vowel phrases themselves.
        for cand in cands:
            centre = (cand.start + cand.end) / 2
            assert any(abs(centre - (b["start"] + b["end"]) / 2)
                       < 0.015 * SR for b in bursts)
        # And no candidate falls inside a vowel phrase.
        for cand in cands:
            assert not any(v["start"] < cand.end and cand.start < v["end"]
                           for v in rec.vowel_segments)

    def test_injected_pops_and_hum_are_annotated(self, controlled):
        rec, damaged = controlled
        assert len(rec.pop_sites) == 3
        assert {round(h["frequency_hz"]) for h in rec.hum_components} == {
            50, 100, 150}
        # All three pops still surface as candidates.
        cands = detect_clicks(damaged, SR)
        for site in rec.pop_sites:
            ps = int(site["centre_ms"] / 1000.0 * SR)
            assert any(abs((c.start + c.end) / 2 - ps) < 0.01 * SR
                       for c in cands)


# ---------------------------------------------------------------------------
# Mask / candidate / input-segment linkage
# ---------------------------------------------------------------------------


class TestMaskLinkage:
    def test_trace_links_threshold_span_to_padded_window(self, controlled):
        _, damaged = controlled
        cands, trace = detect_clicks_with_trace(damaged, SR)
        assert trace["stage"] == "click_detection"
        assert trace["n_mask_regions"] == len(cands) == len(trace["regions"])
        for region, cand in zip(trace["regions"], cands):
            # The threshold span is contained in the padded candidate window.
            assert region["threshold_span"][0] >= region["candidate_window"][0]
            assert region["threshold_span"][1] <= region["candidate_window"][1]
            assert region["n_threshold_samples"] <= region["n_window_samples"]
            assert region["cand_id"] == cand.cand_id

    def test_trace_function_matches_plain_detector(self, controlled):
        _, damaged = controlled
        plain = detect_clicks(damaged, SR)
        traced, _ = detect_clicks_with_trace(damaged, SR)
        assert [c.to_dict() for c in plain] == [c.to_dict() for c in traced]

    def test_hum_trace_records_grid_bins_and_threshold_decisions(
            self, controlled):
        _, damaged = controlled
        cands, trace = detect_hum_with_trace(
            damaged, SR, line_frequencies=(50.0, 60.0))
        assert trace["stage"] == "hum_detection"
        assert trace["n_grid_searches"] == len(trace["grid_searches"])
        passing = [g for g in trace["grid_searches"] if g["passes_threshold"]]
        assert len(passing) == len(cands)
        for g in trace["grid_searches"]:
            assert "bin_snr_db" in g and "selected_bin_hz" in g

    def test_segment_refs_carry_range_and_content_hash(self, controlled):
        _, damaged = controlled
        budget = DiagnosticBudget()
        col = DiagnosticCollector(SR, "src-hash", budget)
        diagnose_click_detection(damaged, SR, col)
        assert col.segments, "at least one input segment must be referenced"
        for ref in col.segments:
            assert ref["end"] > ref["start"]
            assert len(ref["content_sha256"]) == 64
            # The hash really binds to that segment's samples.
            assert ref["content_sha256"] == segment_hash(
                damaged, ref["start"], ref["end"])

    def test_every_evidence_record_is_stage_localised(self, project):
        col = DiagnosticCollector(SR, project.audio.source_sha256)
        diagnostic_replay(
            project, col,
            detect_kwargs_click={"threshold_sigma": 8.0},
            detect_kwargs_hum={"line_frequencies": (50.0, 60.0)})
        stages = {e["stage"] for e in col.evidence}
        assert {"click_detection", "hum_detection", "click_repair",
                "hum_notch"}.issubset(stages)
        # Processing-stage evidence always references an input segment.
        for e in col.evidence:
            if e["stage"] in ("click_repair", "hum_notch"):
                assert e.get("segment_ref") or e.get("segment_ref_before")


# ---------------------------------------------------------------------------
# Consonant transient vs sustained vowel
# ---------------------------------------------------------------------------


class TestContentClassification:
    def test_consonant_burst_classified_transient(self, controlled):
        rec, _ = controlled
        burst = next(o for o in rec.onset_segments
                     if o["kind"] == "consonant_burst")
        result = classify_segment(rec.samples, burst["start"], burst["end"],
                                  SR)
        assert result["label"] == "consonant_transient"
        assert result["rules_fired"]

    def test_sustained_vowel_interior_classified_voiced(self):
        rec = build_controlled_recording(sample_rate=SR)
        v = rec.vowel_segments[0]
        # Skip the annotated attack; take the stable middle of the phrase.
        mid0 = v["start"] + int(0.3 * SR)
        mid1 = v["end"] - int(0.1 * SR)
        result = classify_segment(rec.samples, mid0, mid1, SR)
        assert result["label"] == "sustained_voiced"

    def test_short_segment_is_undetermined_not_a_guess(self):
        rec = build_controlled_recording(sample_rate=SR)
        result = classify_segment(rec.samples, 100, 103, SR)
        assert result["label"] == "undetermined"


# ---------------------------------------------------------------------------
# Audio path is untouched by diagnostics
# ---------------------------------------------------------------------------


class TestDiagnosticsDoNotAlterAudio:
    def test_replay_buffer_equals_project_render(self, project):
        col = DiagnosticCollector(SR, project.audio.source_sha256)
        rendered, _ = diagnostic_replay(
            project, col,
            detect_kwargs_click={"threshold_sigma": 8.0},
            detect_kwargs_hum={"line_frequencies": (50.0, 60.0)})
        assert np.array_equal(rendered, project.render())

    def test_detection_diagnostics_do_not_mutate_input(self, controlled):
        _, damaged = controlled
        before = damaged.copy()
        col = DiagnosticCollector(SR, "h")
        diagnose_click_detection(damaged, SR, col)
        assert np.array_equal(damaged, before)

    def test_rendered_wav_byte_identical_with_and_without_diagnostics(
            self, project, tmp_path):
        ordinary = tmp_path / "ordinary.wav"
        project.export_render(str(ordinary))
        col = DiagnosticCollector(SR, project.audio.source_sha256)
        rendered, _ = diagnostic_replay(
            project, col,
            detect_kwargs_click={"threshold_sigma": 8.0},
            detect_kwargs_hum={"line_frequencies": (50.0, 60.0)})
        diag = tmp_path / "diag.wav"
        write_wav(str(diag), rendered, SR)
        h1 = hashlib.sha256(open(ordinary, "rb").read()).hexdigest()
        h2 = hashlib.sha256(open(diag, "rb").read()).hexdigest()
        assert h1 == h2


# ---------------------------------------------------------------------------
# H2 / H3 measured evidence
# ---------------------------------------------------------------------------


class TestStageEvidence:
    def test_h2_records_edge_deltas_and_energy_change(self, controlled):
        _, damaged = controlled
        c = detect_clicks(damaged, SR)[0]
        col = DiagnosticCollector(SR, "h")
        before = damaged.copy()
        res = repair_click(damaged, c.start, c.end, SR, algorithm="ar")
        diagnose_click_interpolation(
            before, res.samples, c.start, c.end, SR,
            {"op_id": "op-1", "cand_id": c.cand_id, "algorithm": "ar"}, col)
        ev = next(e for e in col.evidence
                  if e["hypothesis"] == "H2_boundary_interpolation")
        for key in ("left_edge_jump_delta", "right_edge_jump_delta",
                    "window_energy_change_db",
                    "outside_window_samples_identical"):
            assert key in ev
        assert ev["outside_window_samples_identical"] is True

    def test_h3_records_target_interference_and_effective_band(
            self, controlled):
        _, damaged = controlled
        col = DiagnosticCollector(SR, "h")
        op = {"op_id": "op-n", "cand_id": "hum-x",
              "params": {"frequency_hz": 50.0, "bandwidth_hz": 2.0,
                         "quality_scale": 1.0}}
        res = notch_hum(damaged, SR, frequency=50.0, bandwidth_hz=2.0)
        diagnose_notch(damaged, res.samples, SR, op, col)
        ev = next(e for e in col.evidence
                  if e["hypothesis"] == "H3_filter_parameter_mismatch")
        # The notch suppresses the recorded 50 Hz target substantially.
        assert ev["target_interference"]["suppression_db"] > 20.0
        # A correct 2 Hz notch has a narrow measured effective band and only
        # mild analytic attenuation immediately beside it.
        assert ev["measured_effective_width_hz"] <= 2.0 * 2.0
        assert ev["worst_analytic_side_gain_db"] > -3.0
        assert ev["filter_state"]["application"] == "filtfilt_zero_phase"

    def test_h3_flags_an_over_wide_notch_as_anomaly(self, controlled):
        _, damaged = controlled
        col = DiagnosticCollector(SR, "h")
        op = {"op_id": "op-wide", "cand_id": "probe",
              "params": {"frequency_hz": 50.0, "bandwidth_hz": 12.0,
                         "quality_scale": 1.0}}
        res = notch_hum(damaged, SR, frequency=50.0, bandwidth_hz=12.0)
        diagnose_notch(damaged, res.samples, SR, op, col)
        ev = next(e for e in col.evidence
                  if e["hypothesis"] == "H3_filter_parameter_mismatch")
        assert ev["anomalies"], "the mismatched notch must localise an anomaly"
        assert ev["measured_effective_width_hz"] > 2.0 * 2.0


# ---------------------------------------------------------------------------
# Bounded sampling / logging budget
# ---------------------------------------------------------------------------


class TestBudget:
    def test_segment_and_evidence_caps_are_enforced(self, controlled):
        _, damaged = controlled
        budget = DiagnosticBudget(max_segments=3, max_evidence=5,
                                  max_window_ms=20.0)
        col = DiagnosticCollector(SR, "h", budget)
        diagnose_click_detection(
            damaged, SR, col, threshold_sigma=7.5)
        assert len(col.segments) <= 3
        assert len(col.evidence) <= 5
        assert col.skipped, "budget breaches must be logged, not silent"
        assert all("budget" in s["reason"] for s in col.skipped)

    def test_window_crop_is_bounded_by_max_window_ms(self):
        clean = synthesize_clean(sample_rate=SR)
        budget = DiagnosticBudget(max_window_ms=10.0)
        col = DiagnosticCollector(SR, "h", budget)
        ref = col.take_segment(clean, 0, SR, role="large", cand_id="x")
        assert ref is not None
        assert ref["n_samples"] <= int(SR * 10.0 / 1000.0) + 1
        assert ref["capped_to_budget"] is True

    def test_budget_does_not_grow_with_recording_length(self):
        # A longer clip must not produce more stored segments for the same
        # bounded number of candidates/operations.
        budget = DiagnosticBudget(max_segments=10, max_evidence=20)
        counts = []
        for duration in (3.0, 6.0):
            sr = SR
            n = int(duration * sr)
            t = np.arange(n) / sr
            x = (0.5 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
            x, _ = inject_pops(
                x, sr, centres_ms=[500.0, 1000.0],
                widths_ms=[1.0, 1.5], amplitudes=[0.7, 0.7])
            col = DiagnosticCollector(sr, "h", budget)
            diagnose_click_detection(x, sr, col)
            counts.append((len(col.segments), len(col.evidence)))
        assert counts[0] == counts[1]


# ---------------------------------------------------------------------------
# End-to-end controlled scenario and CLI command
# ---------------------------------------------------------------------------


class TestControlledScenario:
    def test_three_hypotheses_evaluated_independently(self, tmp_path):
        report = run_controlled_diagnosis(str(tmp_path / "diag"))
        hyps = report["hypotheses"]
        assert set(hyps) == {
            "H1_detection_too_wide",
            "H2_boundary_interpolation",
            "H3_filter_parameter_mismatch"}
        for h in hyps.values():
            assert h["verdict"] in (
                "supported", "not_supported_by_evidence")
            assert "stage_localised" in h and h["n_evaluated"] >= 0

    def test_correct_graph_notch_cleared_but_wide_probe_flagged(
            self, tmp_path):
        report = run_controlled_diagnosis(str(tmp_path / "diag"))
        main_h3 = report["hypotheses"]["H3_filter_parameter_mismatch"]
        assert main_h3["n_supporting"] == 0
        probe = report["parameter_probes"]["wide_notch_H3"]
        assert probe["n_supporting"] >= 1

    def test_permissive_detection_produces_more_h1_evidence(self, tmp_path):
        report = run_controlled_diagnosis(str(tmp_path / "diag"))
        wide = report["parameter_probes"]["permissive_detection_H1"]
        main = report["hypotheses"]["H1_detection_too_wide"]
        assert wide["n_evaluated"] > main["n_evaluated"]
        assert wide["n_supporting"] >= 1

    def test_classifier_validation_against_ground_truth(self, tmp_path):
        report = run_controlled_diagnosis(str(tmp_path / "diag"))
        cv = report["classifier_validation"]
        assert cv["total"] > 0 and cv["accuracy"] == 1.0
        # Pop windows are reported separately and read as transients.
        assert len(cv["pop_windows_transient_classification"]) == 3
        assert all(r["transient_classification_expected"]
                   for r in cv["pop_windows_transient_classification"])

    def test_report_is_deterministic_across_runs(self, tmp_path):
        r1 = run_controlled_diagnosis(str(tmp_path / "a"))
        r2 = run_controlled_diagnosis(str(tmp_path / "b"))
        # Drop run-local paths/hashes before comparing the measured content.
        for r in (r1, r2):
            r.pop("source_sha256", None)
            r.pop("evidence_chain", None)
        assert json.dumps(r1, sort_keys=True, default=float) == \
            json.dumps(r2, sort_keys=True, default=float)

    def test_cli_diagnose_command_writes_report_and_matching_render(
            self, tmp_path):
        from oralrestore.cli import main
        out_dir = tmp_path / "cli"
        rc = main(["diagnose", str(out_dir), "--controlled"])
        assert rc == 0
        report_path = out_dir / "diagnostic_report.json"
        assert report_path.exists()
        report = json.load(open(report_path))
        assert report["replay"]["render_matches_project_render"] is True
        # Reopening the saved project and re-rendering stays reproducible.
        proj = Project.open(str(out_dir / "project"))
        again = proj.render()
        first = proj.render()
        assert np.array_equal(again, first)

    def test_cli_disabled_mode_collects_no_evidence(self, project, tmp_path,
                                                    monkeypatch):
        from oralrestore.cli import main
        out = tmp_path / "off.wav"
        rc = main(["diagnose", project.path, "--disabled",
                   "--output", str(out)])
        assert rc == 0
        assert out.exists()
        assert not (tmp_path / "diagnostic_report.json").exists()
