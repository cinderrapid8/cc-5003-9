"""Command-line interface for the restoration workbench.

Subcommands:
  init        create a project from a source WAV (source stays untouched)
  detect      run click and/or hum candidate detection
  candidates  list candidates with decisions and repairability
  accept      accept a candidate (queue a bounded repair / notch)
  exclude     exclude a candidate (e.g. a real plosive or consonant)
  ops         show the ordered repair graph
  reorder     change operation order (matters for overlapping windows)
  undo        roll back the last accepted operation
  render      replay the graph from the original and export a WAV
  demo        build fixtures, run full scenarios, write reports + audio
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from .audioio import sha256_file
from .project import Project
from .diagnostics import (
    DiagnosticBudget,
    DiagnosticCollector,
    diagnostic_replay,
)
from .controlled import run_controlled_diagnosis
from .synthetic import (
    add_legitimate_transients,
    build_fixture_bundle,
    evaluate,
    inject_hum,
    inject_pops,
    make_long_defect,
    synthesize_clean,
    write_report,
)
from .audioio import write_wav


def _print_candidates(proj: Project) -> None:
    print(f"{'id':<12}{'kind':<6}{'at_ms':>10}{'width_ms':>10}"
          f"{'score':>8}  {'status':<10}repairable / note")
    for c in proj.click_candidates:
        print(f"{c.cand_id:<12}{'click':<6}"
              f"{1000 * c.peak_sample / proj.audio.sample_rate:>10.2f}"
              f"{1000 * c.width / proj.audio.sample_rate:>10.2f}"
              f"{c.score:>8.2f}  {c.decision:<10}"
              f"{c.repairable if c.repairable else 'NO: ' + str(c.reject_reason)}")
    for c in proj.hum_candidates:
        print(f"{c.cand_id:<12}{'hum':<6}{c.frequency:>10.2f}"
              f"{'--':>10}{c.snr_db:>8.2f}  {c.decision:<10}"
              f"notch bw={c.bandwidth} Hz")


def cmd_init(args: argparse.Namespace) -> None:
    proj = Project.create(args.source, args.project)
    print(f"created project {proj.path}")
    print(f"source: {proj.audio.source_path}")
    print(f"sha256: {proj.audio.source_sha256}")
    print(f"duration: {proj.audio.duration:.2f}s @ {proj.audio.sample_rate} Hz")


def cmd_detect(args: argparse.Namespace) -> None:
    proj = Project.open(args.project)
    if args.what in ("click", "both"):
        added = proj.run_click_detection(
            method=args.method,
            threshold_sigma=args.threshold,
            half_win=args.half_win,
            max_gap_ms=args.max_gap_ms,
        )
        print(f"click detection: {len(added)} new candidates "
              f"({len(proj.click_candidates)} total)")
    if args.what in ("hum", "both"):
        added = proj.run_hum_detection(
            line_frequencies=tuple(args.line_freq),
            snr_threshold_db=args.hum_snr,
            bandwidth_hz=args.bandwidth,
        )
        print(f"hum detection: {len(added)} new candidates "
              f"({len(proj.hum_candidates)} total)")
    if args.list:
        _print_candidates(proj)


def cmd_candidates(args: argparse.Namespace) -> None:
    _print_candidates(Project.open(args.project))


def cmd_accept(args: argparse.Namespace) -> None:
    proj = Project.open(args.project)
    if args.cand_id.startswith("click-"):
        op = proj.accept_click(args.cand_id, algorithm=args.algorithm,
                               max_gap_ms=args.max_gap_ms)
    else:
        op = proj.accept_hum(args.cand_id, bandwidth_hz=args.bandwidth)
    print("queued operation:")
    print(json.dumps(op, indent=2, ensure_ascii=False))


def cmd_exclude(args: argparse.Namespace) -> None:
    proj = Project.open(args.project)
    if args.cand_id.startswith("click-"):
        proj.exclude_click(args.cand_id, reason=args.reason)
    else:
        proj.exclude_hum(args.cand_id, reason=args.reason)
    print(f"excluded {args.cand_id}; any queued op for it was withdrawn")


def cmd_ops(args: argparse.Namespace) -> None:
    proj = Project.open(args.project)
    rows = proj.repair_list()
    if not rows:
        print("(no operations yet)")
    for row in rows:
        print(f"#{row['order']} {row['op_id']} {row['type']} "
              f"({row['algorithm']}) window={row['window_ms']} ms "
              f"params={json.dumps(row['params'], ensure_ascii=False)}")
    overlaps = proj.overlapping_operations()
    if overlaps:
        print("overlapping windows (applied in the order above):")
        for a, b in overlaps:
            print(f"  {a} <-> {b}")


def cmd_reorder(args: argparse.Namespace) -> None:
    proj = Project.open(args.project)
    proj.reorder_operation(args.op_id, args.direction)
    print("new order:")
    cmd_ops(args)


def cmd_undo(args: argparse.Namespace) -> None:
    proj = Project.open(args.project)
    op = proj.undo_last()
    print("rolled back:" if op else "nothing to roll back", op or "")


def cmd_render(args: argparse.Namespace) -> None:
    proj = Project.open(args.project)
    out = proj.export_render(args.output)
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"rendered {out}")
    print(f"output sha256: {h}")
    print(f"source file untouched: {sha256_file(proj.audio.source_path) == proj.audio.source_sha256}")


# ---------------------------------------------------------------------------
# Demo / evaluation
# ---------------------------------------------------------------------------


def _auto_accept_repairable(proj: Project, algorithm: str,
                            max_gap_ms: float) -> tuple[int, int]:
    """Simulate the expert: accept repairable pops, leave long gaps alone."""
    accepted = excluded = 0
    for c in list(proj.click_candidates):
        if c.decision != "pending":
            continue
        if c.repairable:
            proj.accept_click(c.cand_id, algorithm=algorithm,
                              max_gap_ms=max_gap_ms)
            accepted += 1
        else:
            proj.exclude_click(c.cand_id,
                               reason="long damage: not reliably repairable")
            excluded += 1
    return accepted, excluded


def _demo_scenario_pops(out_dir: str, sr: int, max_gap_ms: float) -> dict:
    clean = synthesize_clean(sample_rate=sr)
    damaged, meta = inject_pops(
        clean, sr,
        centres_ms=[600.0, 1500.0, 2750.0, 3600.0, 5100.0],
        widths_ms=[1.0, 2.0, 1.5, 3.0, 0.8],
        amplitudes=[0.6, 0.8, 0.5, 0.9, 0.7],
    )
    src = os.path.join(out_dir, "scenario_pops_damaged.wav")
    write_wav(src, damaged, sr)
    proj = Project.create(src, os.path.join(out_dir, "project_pops"))
    proj.run_click_detection(method="ar", threshold_sigma=8.0,
                             max_gap_ms=max_gap_ms)
    accepted, excluded = _auto_accept_repairable(
        proj, algorithm="ar", max_gap_ms=max_gap_ms)
    render_path = proj.export_render()
    report = evaluate(proj, clean, meta.centres_ms, hum=None)
    report["scenario"] = "pops_only"
    report["expert_actions"] = {"accepted": accepted, "excluded": excluded}
    report["rendered_wav"] = render_path
    return report


def _demo_scenario_hum(out_dir: str, sr: int) -> dict:
    clean = synthesize_clean(sample_rate=sr)
    damaged, hum = inject_hum(clean, sr, fundamental_hz=50.0)
    src = os.path.join(out_dir, "scenario_hum_damaged.wav")
    write_wav(src, damaged, sr)
    proj = Project.create(src, os.path.join(out_dir, "project_hum"))
    proj.run_hum_detection(line_frequencies=(50.0, 60.0),
                           snr_threshold_db=12.0, bandwidth_hz=2.0)
    # Expert adjudication: accept peaks on the known 50 Hz harmonic grid
    # (50/100/150 Hz); exclude a 120 Hz candidate that is really a speech
    # harmonic landing on the 60 Hz grid.  This is exactly why hum removal is
    # candidate-based rather than an automatic sweep.
    true_line = {50.0, 100.0, 150.0}
    accepted, excluded = [], []
    for c in list(proj.hum_candidates):
        if c.decision != "pending":
            continue
        if any(abs(c.frequency - f) <= 1.5 for f in true_line):
            proj.accept_hum(c.cand_id)
            accepted.append(c.cand_id)
        else:
            proj.exclude_hum(
                c.cand_id,
                reason="speech harmonic, not mains: preserve useful content")
            excluded.append(c.cand_id)
    render_path = proj.export_render()
    report = evaluate(proj, clean, [], hum=hum)
    report["scenario"] = "hum_only"
    report["expert_actions"] = {
        "accepted": accepted, "excluded": excluded,
        "note": "120 Hz speech-harmonic candidate excluded; only true "
                "50/100/150 Hz line components notched",
    }
    report["rendered_wav"] = render_path
    return report


def _demo_scenario_long(out_dir: str, sr: int, max_gap_ms: float) -> dict:
    clean = synthesize_clean(sample_rate=sr)
    damaged = make_long_defect(clean, sr, centre_ms=3000.0,
                               length_ms=60.0, amplitude=0.9)
    src = os.path.join(out_dir, "scenario_long_damaged.wav")
    write_wav(src, damaged, sr)
    proj = Project.create(src, os.path.join(out_dir, "project_long"))
    proj.run_click_detection(method="ar", threshold_sigma=8.0,
                             max_gap_ms=max_gap_ms)
    accepted, excluded = _auto_accept_repairable(
        proj, algorithm="ar", max_gap_ms=max_gap_ms)
    # No render of interpolated long gaps; export only if something was done.
    render_path = proj.export_render() if proj.operations else None
    report = evaluate(proj, clean, [3000.0], hum=None)
    report["scenario"] = "long_defect_refused"
    report["expert_actions"] = {"accepted": accepted, "excluded": excluded}
    report["rendered_wav"] = render_path
    return report


def _demo_false_alarm(out_dir: str, sr: int, max_gap_ms: float) -> dict:
    """Real plosive/tap transients are candidates only; expert excludes them."""
    clean = synthesize_clean(sample_rate=sr)
    # No defect pops at all: every alarm here is a potential false positive.
    signal = add_legitimate_transients(
        clean, sr, centres_ms=[900.0, 2400.0, 4700.0])
    src = os.path.join(out_dir, "scenario_falsealarm_damaged.wav")
    write_wav(src, signal, sr)
    proj = Project.create(src, os.path.join(out_dir, "project_falsealarm"))
    cands = proj.run_click_detection(method="ar", threshold_sigma=8.0,
                                     max_gap_ms=max_gap_ms)
    # Expert recognises legitimate consonants/taps and excludes every one.
    excluded_ids = []
    for c in list(proj.click_candidates):
        if c.decision == "pending":
            proj.exclude_click(
                c.cand_id,
                reason="legitimate plosive/percussion transient: keep")
            excluded_ids.append(c.cand_id)
    render_path = proj.export_render()
    rendered = proj.render()
    # With all candidates excluded, the render must equal the input exactly.
    identity = bool(np.array_equal(rendered, proj.audio.samples))
    report = evaluate(proj, clean, [], hum=None)
    report["scenario"] = "false_alarm_excluded"
    report["expert_actions"] = {"accepted": 0,
                                "excluded": len(excluded_ids),
                                "excluded_ids": excluded_ids}
    report["render_identical_to_input"] = identity
    report["rendered_wav"] = render_path
    return report


def _demo_reproducibility(out_dir: str) -> dict:
    """Re-open a saved project and prove the render recomputes identically."""
    proj_dir = os.path.join(out_dir, "project_pops")
    proj = Project.open(proj_dir)
    first_path = os.path.join(proj_dir, "rendered.wav")
    first_hash = hashlib.sha256(open(first_path, "rb").read()).hexdigest()
    again = proj.render()
    saved = proj.export_render(os.path.join(out_dir, "reopened_render.wav"))
    second_hash = hashlib.sha256(open(saved, "rb").read()).hexdigest()
    # Determinism is established at the byte level: the two exported WAVs
    # share an identical sha256, so the reopened project replays exactly.
    # The code-level check tolerates only float32*32768 scaling jitter
    # (<=1 LSB), never a substantive sample difference.
    import soundfile as sf
    stored_codes, _ = sf.read(saved, dtype="int16", always_2d=False)
    scaled = np.clip(again * 32768.0, -32768.0, 32767.0)
    replay_codes = np.floor(scaled + 0.5).astype(np.int16)
    lsb_gap = int(np.max(np.abs(
        replay_codes.astype(np.int32) - stored_codes.astype(np.int32))))
    codes_within_one_lsb = lsb_gap <= 1
    byte_identical = first_hash == second_hash
    return {
        "reopened_project": proj.path,
        "first_render_sha256": first_hash,
        "reopened_render_sha256": second_hash,
        "wav_bytes_identical": byte_identical,
        "max_pcm16_code_gap": lsb_gap,
        "codes_within_one_lsb": codes_within_one_lsb,
        "operations_replayed": len(proj.operations),
        "deterministic": byte_identical and codes_within_one_lsb,
    }


# ---------------------------------------------------------------------------
# Collateral-damage diagnostics
# ---------------------------------------------------------------------------


def _json_dump(path: str, payload: dict) -> None:
    def default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)!r}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=default)


def cmd_diagnose(args: argparse.Namespace) -> None:
    """Run switchable, bounded diagnostics without changing the audio path."""
    if not args.enabled:
        # Diagnostics off: prove the switch is inert and write no evidence.
        proj = Project.open(args.project)
        out = proj.export_render(
            args.output or os.path.join(proj.path, "rendered.wav"))
        print("diagnostics disabled; no evidence collected")
        print(f"rendered {out} (identical to a normal render)")
        return

    budget = DiagnosticBudget(
        max_segments=args.max_segments,
        max_evidence=args.max_evidence,
        max_window_ms=args.max_window_ms,
    )
    if args.controlled:
        report = run_controlled_diagnosis(
            args.project, sample_rate=args.sample_rate, budget=budget,
            wide_threshold_sigma=args.wide_sigma,
            narrow_threshold_sigma=args.narrow_sigma,
            mismatched_bandwidth_hz=args.wide_bandwidth)
        out_dir = os.path.abspath(args.project)
        # The controlled render must still equal the project's own render.
        report_path = os.path.join(out_dir,
                                   args.report or "diagnostic_report.json")
        _json_dump(report_path, report)
        print(f"controlled diagnosis report: {report_path}")
        print(json.dumps({
            h: v["verdict"] for h, v in report["hypotheses"].items()},
            indent=2))
        return

    proj = Project.open(args.project)
    collector = DiagnosticCollector(
        sample_rate=proj.audio.sample_rate,
        source_sha256=proj.audio.source_sha256, budget=budget)
    rendered, _stages = diagnostic_replay(
        proj, collector,
        detect_kwargs_click={"threshold_sigma": args.narrow_sigma},
        detect_kwargs_hum={"line_frequencies": (50.0, 60.0),
                           "snr_threshold_db": 12.0,
                           "bandwidth_hz": args.notch_bandwidth})
    # Hard invariant: the diagnostic replay renders the same samples as the
    # ordinary render.  Any divergence is a tool defect, not a finding.
    ordinary = proj.render()
    same = bool(np.array_equal(rendered, ordinary))
    if not same:
        raise RuntimeError(
            "diagnostic replay diverged from Project.render; diagnostics "
            "must never alter the audio path")
    out = proj.export_render(
        args.output or os.path.join(proj.path, "diagnostic_render.wav"))
    report = collector.to_report()
    report["render_path"] = os.path.abspath(out)
    report["render_matches_project_render"] = same
    report_path = os.path.abspath(
        args.report or os.path.join(proj.path, "diagnostic_report.json"))
    _json_dump(report_path, report)
    print(f"diagnostic report: {report_path}")
    print(f"render: {out}")
    print(f"render identical to normal render: {same}")
    print(f"segments recorded: {report['sampling']['segments_recorded']} "
          f"(budget {args.max_segments}); evidence records: "
          f"{report['sampling']['evidence_records']} "
          f"(budget {args.max_evidence})")
    for rec in report["evidence"]:
        stage = rec.get("stage")
        if stage in ("click_detection", "hum_detection", "click_repair",
                     "hum_notch"):
            print(f"  stage={stage} hypothesis={rec.get('hypothesis', '-')} "
                  f"cand={rec.get('cand_id', '-')}")


def cmd_demo(args: argparse.Namespace) -> None:
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    sr = args.sample_rate
    max_gap_ms = args.max_gap_ms

    fixtures = build_fixture_bundle(os.path.join(out_dir, "fixtures"), sr)
    reports = {
        "fixtures": fixtures,
        "pops": _demo_scenario_pops(out_dir, sr, max_gap_ms),
        "hum": _demo_scenario_hum(out_dir, sr),
        "long_defect": _demo_scenario_long(out_dir, sr, max_gap_ms),
        "false_alarm": _demo_false_alarm(out_dir, sr, max_gap_ms),
        "reproducibility": _demo_reproducibility(out_dir),
        "config": {
            "sample_rate": sr,
            "max_gap_ms": max_gap_ms,
            "click_threshold_sigma": 8.0,
            "click_detector": "ar_residual",
            "hum_bandwidth_hz": 2.0,
            "note": "uniform gain reduction is never used; every operation "
                    "is local interpolation or a narrow recorded notch",
        },
    }
    path = write_report(reports, out_dir)
    print(f"demo complete; report: {path}")
    print(json.dumps({k: v.get("scenario", k) if isinstance(v, dict) else v
                      for k, v in reports.items()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oralrestore",
                                description="Oral-history restoration workbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init")
    sp.add_argument("source")
    sp.add_argument("project")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("detect")
    sp.add_argument("project")
    sp.add_argument("--what", choices=["click", "hum", "both"], default="both")
    sp.add_argument("--method", choices=["median", "ar"], default="ar")
    sp.add_argument("--threshold", type=float, default=8.0)
    sp.add_argument("--half-win", type=int, default=3)
    sp.add_argument("--max-gap-ms", type=float, default=10.0)
    sp.add_argument("--line-freq", type=float, nargs="+", default=[50.0, 60.0])
    sp.add_argument("--hum-snr", type=float, default=12.0)
    sp.add_argument("--bandwidth", type=float, default=2.0)
    sp.add_argument("--list", action="store_true")
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("candidates")
    sp.add_argument("project")
    sp.set_defaults(func=cmd_candidates)

    sp = sub.add_parser("accept")
    sp.add_argument("project")
    sp.add_argument("cand_id")
    sp.add_argument("--algorithm", choices=["ar", "spline"], default="ar")
    sp.add_argument("--max-gap-ms", type=float, default=10.0)
    sp.add_argument("--bandwidth", type=float, default=2.0)
    sp.set_defaults(func=cmd_accept)

    sp = sub.add_parser("exclude")
    sp.add_argument("project")
    sp.add_argument("cand_id")
    sp.add_argument("--reason", default="")
    sp.set_defaults(func=cmd_exclude)

    sp = sub.add_parser("ops")
    sp.add_argument("project")
    sp.set_defaults(func=cmd_ops)

    sp = sub.add_parser("reorder")
    sp.add_argument("project")
    sp.add_argument("op_id")
    sp.add_argument("direction", choices=["up", "down"])
    sp.set_defaults(func=cmd_reorder)

    sp = sub.add_parser("undo")
    sp.add_argument("project")
    sp.set_defaults(func=cmd_undo)

    sp = sub.add_parser("render")
    sp.add_argument("project")
    sp.add_argument("--output")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("demo")
    sp.add_argument("--out", default="demo/output")
    sp.add_argument("--sample-rate", type=int, default=16000)
    sp.add_argument("--max-gap-ms", type=float, default=10.0)
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser(
        "diagnose",
        help="bounded collateral-damage diagnostics; audio path unchanged")
    sp.add_argument("project",
                    help="project dir (or output dir when --controlled)")
    sp.add_argument("--enabled", dest="enabled", action="store_true",
                    default=True,
                    help="collect evidence (default)")
    sp.add_argument("--disabled", dest="enabled", action="store_false",
                    help="switch diagnostics off; render only, no evidence")
    sp.add_argument("--controlled", action="store_true",
                    help="build the annotated controlled recording and run "
                         "all three hypothesis probes")
    sp.add_argument("--report", default=None,
                    help="diagnostic report path (project dir by default)")
    sp.add_argument("--output", default=None,
                    help="rendered WAV path (project dir by default)")
    sp.add_argument("--sample-rate", type=int, default=16000)
    sp.add_argument("--max-segments", type=int, default=128)
    sp.add_argument("--max-evidence", type=int, default=512)
    sp.add_argument("--max-window-ms", type=float, default=40.0)
    sp.add_argument("--narrow-sigma", type=float, default=8.0,
                    help="threshold for the conservative detection pass")
    sp.add_argument("--wide-sigma", type=float, default=7.5,
                    help="threshold for the permissive H1 probe")
    sp.add_argument("--wide-bandwidth", type=float, default=12.0,
                    help="notch bandwidth for the mismatched H3 probe (Hz)")
    sp.add_argument("--notch-bandwidth", type=float, default=2.0)
    sp.set_defaults(func=cmd_diagnose)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # surface clean CLI errors
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
