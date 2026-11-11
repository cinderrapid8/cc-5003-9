"""Flask backend for the oral-history restoration workbench.

The web layer is a thin adapter over :mod:`oralrestore.project`: it never
implements detection or repair itself, so the CLI, tests and UI all share one
audited engine.  The source WAV stays read-only; the browser only ever reads
the original or a freshly rendered stream.
"""

from __future__ import annotations

import io
import os
import threading
import uuid

import numpy as np
import soundfile as sf
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask.json.provider import DefaultJSONProvider

from .audioio import load_audio, write_wav
from .project import Project
from .synthetic import build_fixture_bundle

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "web")


class NumpyJSONProvider(DefaultJSONProvider):
    """Teach Flask's encoder about numpy scalars/arrays."""

    @staticmethod
    def default(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return DefaultJSONProvider.default(obj)


class ProjectRegistry:
    """In-memory handle table; each project is itself persisted on disk."""

    def __init__(self) -> None:
        self._projects: dict[str, Project] = {}
        self._lock = threading.Lock()

    def add(self, proj: Project) -> str:
        pid = uuid.uuid4().hex[:12]
        with self._lock:
            self._projects[pid] = proj
        return pid

    def get(self, pid: str) -> Project:
        with self._lock:
            try:
                return self._projects[pid]
            except KeyError:
                raise KeyError(f"no open project with id {pid}")

    def list(self) -> dict[str, str]:
        with self._lock:
            return {pid: p.path for pid, p in self._projects.items()}


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> io.BytesIO:
    buf = io.BytesIO()
    sf.write(buf, np.clip(samples, -1, 1), sample_rate,
             format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


def _waveform_peaks(samples: np.ndarray, n_bins: int = 1400) -> dict:
    """Down-sample to min/max envelopes so the browser stays responsive."""
    n = len(samples)
    bin_size = max(1, n // n_bins)
    mins, maxs, centres = [], [], []
    for i in range(0, n, bin_size):
        chunk = samples[i:i + bin_size]
        if len(chunk) == 0:
            continue
        mins.append(float(np.min(chunk)))
        maxs.append(float(np.max(chunk)))
        centres.append((i + len(chunk) / 2.0))
    return {"min": mins, "max": maxs, "centres": centres,
            "n_samples": n, "bin_size": bin_size}


def _spectrum(samples: np.ndarray, sample_rate: int,
              n_points: int = 1024) -> dict:
    n = len(samples)
    # Welch-style averaged periodogram keeps the plot stable.
    seg = min(n, sample_rate)  # 1-second segments, 1 Hz bins
    win = np.hanning(seg)
    halves = [samples[i:i + seg] for i in range(0, n - seg + 1, seg)] or [
        samples[:seg]]
    acc = np.zeros(seg // 2 + 1)
    for h in halves:
        if len(h) < seg:
            h = np.pad(h, (0, seg - len(h)))
        acc += np.abs(np.fft.rfft(h * win)) ** 2
    acc /= max(1, len(halves))
    freqs = np.fft.rfftfreq(seg, 1.0 / sample_rate)
    # Decimate for the browser while retaining low-frequency detail.
    step = max(1, len(freqs) // n_points)
    return {
        "freq": freqs[::step].tolist(),
        "power_db": (10 * np.log10(acc[::step] + 1e-20)).tolist(),
    }


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.json = NumpyJSONProvider(app)
    registry = ProjectRegistry()
    app.config["REGISTRY"] = registry

    def pid() -> str:
        value = request.args.get("project_id") or (
            request.json or {}).get("project_id")
        if not value:
            raise ValueError("project_id is required")
        return value

    @app.errorhandler(KeyError)
    def _not_found(err: KeyError):
        return jsonify({"error": str(err)}), 404

    @app.errorhandler(ValueError)
    def _bad_request(err: ValueError):
        return jsonify({"error": str(err)}), 400

    @app.errorhandler(RuntimeError)
    def _conflict(err: RuntimeError):
        return jsonify({"error": str(err)}), 409

    # ---- static UI -------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/static/<path:fname>")
    def static_files(fname: str):
        return send_from_directory(WEB_DIR, fname)

    @app.get("/api/projects")
    def list_projects():
        return jsonify(registry.list())

    @app.post("/api/project/create")
    def create_project():
        body = request.json or {}
        src = body["source_path"]
        proj_dir = body["project_dir"]
        proj = Project.create(src, proj_dir)
        return jsonify({"project_id": registry.add(proj),
                        "path": proj.path,
                        "sample_rate": proj.audio.sample_rate,
                        "n_samples": len(proj.audio.samples),
                        "duration": proj.audio.duration})

    @app.post("/api/project/open")
    def open_project():
        body = request.json or {}
        proj = Project.open(body["project_dir"])
        return jsonify({"project_id": registry.add(proj),
                        "path": proj.path,
                        "sample_rate": proj.audio.sample_rate,
                        "n_samples": len(proj.audio.samples),
                        "duration": proj.audio.duration})

    @app.post("/api/demo/load")
    def load_demo():
        """Build the synthetic fixture bundle and open the pops+hum project."""
        body = request.json or {}
        out_dir = os.path.abspath(body["out_dir"])
        fixtures = build_fixture_bundle(
            os.path.join(out_dir, "fixtures"),
            sample_rate=int(body.get("sample_rate", 16000)))
        proj = Project.create(fixtures["both"],
                              os.path.join(out_dir, "web_project"))
        return jsonify({"project_id": registry.add(proj),
                        "fixtures": fixtures})

    # ---- detection -------------------------------------------------------
    @app.post("/api/detect/click")
    def detect_click():
        proj = registry.get(pid())
        body = request.json or {}
        kwargs = {k: body[k] for k in
                  ("method", "threshold_sigma", "half_win", "ar_order",
                   "max_gap_ms", "window_pad_ms") if k in body}
        added = proj.run_click_detection(**kwargs)
        return jsonify({"added": len(added),
                        "total": len(proj.click_candidates)})

    @app.post("/api/detect/hum")
    def detect_hum_endpoint():
        proj = registry.get(pid())
        body = request.json or {}
        kwargs = {}
        if "line_frequencies" in body:
            kwargs["line_frequencies"] = tuple(body["line_frequencies"])
        for k in ("snr_threshold_db", "bandwidth_hz", "max_harmonic"):
            if k in body:
                kwargs[k] = body[k]
        added = proj.run_hum_detection(**kwargs)
        return jsonify({"added": len(added),
                        "total": len(proj.hum_candidates)})

    @app.get("/api/candidates")
    def candidates():
        proj = registry.get(pid())
        sr = proj.audio.sample_rate
        clicks = []
        for c in proj.click_candidates:
            d = c.to_dict()
            d["start_ms"] = round(1000 * c.start / sr, 2)
            d["end_ms"] = round(1000 * c.end / sr, 2)
            clicks.append(d)
        return jsonify({"click": clicks,
                        "hum": [c.to_dict() for c in proj.hum_candidates]})

    # ---- expert decisions ------------------------------------------------
    @app.post("/api/click/accept")
    def accept_click():
        proj = registry.get(pid())
        body = request.json or {}
        op = proj.accept_click(
            body["cand_id"],
            algorithm=body.get("algorithm", "ar"),
            max_gap_ms=body.get("max_gap_ms", 10.0),
            params=body.get("params"))
        return jsonify(op)

    @app.post("/api/click/exclude")
    def exclude_click():
        proj = registry.get(pid())
        body = request.json or {}
        proj.exclude_click(body["cand_id"], body.get("reason", ""))
        return jsonify({"status": "excluded"})

    @app.post("/api/hum/accept")
    def accept_hum():
        proj = registry.get(pid())
        body = request.json or {}
        op = proj.accept_hum(
            body["cand_id"],
            bandwidth_hz=body.get("bandwidth_hz"),
            quality_scale=body.get("quality_scale", 1.0))
        return jsonify(op)

    @app.post("/api/hum/exclude")
    def exclude_hum():
        proj = registry.get(pid())
        body = request.json or {}
        proj.exclude_hum(body["cand_id"], body.get("reason", ""))
        return jsonify({"status": "excluded"})

    # ---- repair graph ----------------------------------------------------
    @app.get("/api/operations")
    def operations():
        proj = registry.get(pid())
        return jsonify({"operations": proj.repair_list(),
                        "overlaps": proj.overlapping_operations()})

    @app.post("/api/operations/reorder")
    def reorder():
        proj = registry.get(pid())
        body = request.json or {}
        proj.reorder_operation(body["op_id"], body["direction"])
        return jsonify({"operations": proj.repair_list()})

    @app.post("/api/undo")
    def undo():
        proj = registry.get(pid())
        op = proj.undo_last()
        return jsonify({"rolled_back": op})

    # ---- visualisation & audio ------------------------------------------
    @app.get("/api/waveform")
    def waveform():
        proj = registry.get(pid())
        kind = request.args.get("kind", "original")
        samples = (proj.render() if kind == "rendered"
                   else proj.render_original())
        return jsonify(_waveform_peaks(samples))

    @app.get("/api/spectrum")
    def spectrum():
        proj = registry.get(pid())
        kind = request.args.get("kind", "original")
        samples = (proj.render() if kind == "rendered"
                   else proj.render_original())
        return jsonify(_spectrum(samples, proj.audio.sample_rate))

    @app.get("/api/audio")
    def audio():
        proj = registry.get(pid())
        kind = request.args.get("kind", "original")
        samples = (proj.render() if kind == "rendered"
                   else proj.render_original())
        return send_file(
            _wav_bytes(samples, proj.audio.sample_rate),
            mimetype="audio/wav",
            download_name=f"{kind}.wav")

    @app.get("/api/state")
    def state():
        proj = registry.get(pid())
        return jsonify({
            "source_path": proj.audio.source_path,
            "source_sha256": proj.audio.source_sha256,
            "sample_rate": proj.audio.sample_rate,
            "duration": proj.audio.duration,
            "n_operations": len(proj.operations),
            "n_click_candidates": len(proj.click_candidates),
            "n_hum_candidates": len(proj.hum_candidates),
            "overlaps": proj.overlapping_operations(),
        })

    return app


def main() -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
