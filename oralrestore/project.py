"""Project state: candidates, decisions and the recorded repair graph.

A project lives in a directory containing:

  project.json     manifest with source hash, candidates, decisions and ops
  source.wav.*     never written by this tool; the original stays where it is

Rendering is deterministic: the output is always recomputed from the original
samples by replaying the accepted operations in their recorded order, so
reverting an operation, excluding a candidate or re-opening the project all
reproduce exactly the same audio.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audioio import AudioData, load_audio, sha256_file, write_wav
from .detectors import ClickCandidate, HumCandidate, detect_clicks, detect_hum
from .repair import RepairResult, notch_hum, repair_click

MANIFEST_NAME = "project.json"
RENDER_NAME = "rendered.wav"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


@dataclass
class Project:
    audio: AudioData
    path: str
    click_candidates: list[ClickCandidate] = field(default_factory=list)
    hum_candidates: list[HumCandidate] = field(default_factory=list)
    # Accepted operations in application order (the repair graph).
    operations: list[dict[str, Any]] = field(default_factory=list)
    next_op_id: int = 1

    # ------------------------------------------------------------------
    # Construction / persistence
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, source_wav: str, project_dir: str) -> "Project":
        os.makedirs(project_dir, exist_ok=True)
        audio = load_audio(source_wav)
        proj = cls(audio=audio, path=os.path.abspath(project_dir))
        proj.save()
        return proj

    @classmethod
    def open(cls, project_dir: str) -> "Project":
        project_dir = os.path.abspath(project_dir)
        manifest = os.path.join(project_dir, MANIFEST_NAME)
        with open(manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        audio = load_audio(data["source_path"])
        if audio.source_sha256 != data["source_sha256"]:
            raise RuntimeError(
                "source file has changed since the project was created "
                f"(expected sha256 {data['source_sha256'][:12]}..., got "
                f"{audio.source_sha256[:12]}...); refusing to re-render "
                "against a different original"
            )
        proj = cls(
            audio=audio,
            path=project_dir,
            operations=list(data.get("operations", [])),
            next_op_id=int(data.get("next_op_id", 1)),
        )
        click_fields = set(ClickCandidate.__dataclass_fields__)
        hum_fields = set(HumCandidate.__dataclass_fields__)
        proj.click_candidates = [
            ClickCandidate(**{k: v for k, v in c.items() if k in click_fields})
            for c in data.get("click_candidates", [])
        ]
        proj.hum_candidates = [
            HumCandidate(**{k: v for k, v in c.items() if k in hum_fields})
            for c in data.get("hum_candidates", [])
        ]
        return proj

    def save(self) -> None:
        data = {
            "format_version": 1,
            "source_path": self.audio.source_path,
            "source_sha256": self.audio.source_sha256,
            "original_dtype": self.audio.original_dtype,
            "sample_rate": self.audio.sample_rate,
            "n_samples": len(self.audio.samples),
            "next_op_id": self.next_op_id,
            "click_candidates": [c.to_dict() for c in self.click_candidates],
            "hum_candidates": [c.to_dict() for c in self.hum_candidates],
            "operations": self.operations,
        }
        tmp = os.path.join(self.path, MANIFEST_NAME + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False,
                      default=_json_default)
        os.replace(tmp, os.path.join(self.path, MANIFEST_NAME))

    # ------------------------------------------------------------------
    # Detection (appends candidates; never touches existing decisions)
    # ------------------------------------------------------------------

    def run_click_detection(self, **kwargs: Any) -> list[ClickCandidate]:
        found = detect_clicks(self.audio.samples, self.audio.sample_rate,
                              **kwargs)
        existing = {(c.start, c.end) for c in self.click_candidates}
        added: list[ClickCandidate] = []
        k = len(self.click_candidates)
        for c in found:
            if (c.start, c.end) in existing:
                continue
            k += 1
            c.cand_id = f"click-{k:04d}"
            self.click_candidates.append(c)
            added.append(c)
        self.save()
        return added

    def run_hum_detection(self, **kwargs: Any) -> list[HumCandidate]:
        found = detect_hum(self.audio.samples, self.audio.sample_rate, **kwargs)
        existing = {round(c.frequency, 1) for c in self.hum_candidates}
        added: list[HumCandidate] = []
        k = len(self.hum_candidates)
        for c in found:
            if round(c.frequency, 1) in existing:
                continue
            k += 1
            c.cand_id = f"hum-{k:04d}"
            self.hum_candidates.append(c)
            added.append(c)
        self.save()
        return added

    # ------------------------------------------------------------------
    # Expert decisions
    # ------------------------------------------------------------------

    def _find_click(self, cand_id: str) -> ClickCandidate:
        for c in self.click_candidates:
            if c.cand_id == cand_id:
                return c
        raise KeyError(f"no click candidate {cand_id}")

    def _find_hum(self, cand_id: str) -> HumCandidate:
        for c in self.hum_candidates:
            if c.cand_id == cand_id:
                return c
        raise KeyError(f"no hum candidate {cand_id}")

    def accept_click(self, cand_id: str, *, algorithm: str = "ar",
                     max_gap_ms: float = 10.0,
                     params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Accept a click candidate and queue its bounded interpolation."""
        cand = self._find_click(cand_id)
        if not cand.repairable:
            raise ValueError(
                f"{cand_id} is marked unrepairable: {cand.reject_reason}"
            )
        cand.decision = "accepted"
        op = {
            "op_id": f"op-{self.next_op_id:04d}",
            "op": "click_repair",
            "cand_id": cand_id,
            "algorithm": algorithm,
            "start": cand.start,
            "end": cand.end,
            "max_gap_ms": max_gap_ms,
            "params": params or {},
        }
        self.next_op_id += 1
        self.operations.append(op)
        self.save()
        return op

    def exclude_click(self, cand_id: str, reason: str = "") -> None:
        cand = self._find_click(cand_id)
        cand.decision = "excluded"
        # Any queued operation derived from this candidate is withdrawn.
        self.operations = [
            o for o in self.operations if o.get("cand_id") != cand_id
        ]
        if reason:
            cand.reject_reason = reason
        self.save()

    def accept_hum(self, cand_id: str, *, bandwidth_hz: float | None = None,
                   quality_scale: float = 1.0) -> dict[str, Any]:
        cand = self._find_hum(cand_id)
        cand.decision = "accepted"
        bw = bandwidth_hz if bandwidth_hz is not None else cand.bandwidth
        op = {
            "op_id": f"op-{self.next_op_id:04d}",
            "op": "hum_notch",
            "cand_id": cand_id,
            "algorithm": "iirnotch_filtfilt",
            "start": 0,
            "end": len(self.audio.samples),
            "max_gap_ms": None,
            "params": {
                "frequency_hz": cand.frequency,
                "bandwidth_hz": bw,
                "quality_scale": quality_scale,
            },
        }
        self.next_op_id += 1
        self.operations.append(op)
        self.save()
        return op

    def exclude_hum(self, cand_id: str, reason: str = "") -> None:
        cand = self._find_hum(cand_id)
        cand.decision = "excluded"
        self.operations = [
            o for o in self.operations if o.get("cand_id") != cand_id
        ]
        self.save()

    # ------------------------------------------------------------------
    # Overlapping operations
    # ------------------------------------------------------------------

    def overlapping_operations(self) -> list[tuple[str, str]]:
        """Pairs of click operations whose windows overlap."""
        clicks = [o for o in self.operations if o["op"] == "click_repair"]
        pairs: list[tuple[str, str]] = []
        for i in range(len(clicks)):
            for j in range(i + 1, len(clicks)):
                a, b = clicks[i], clicks[j]
                if a["start"] < b["end"] and b["start"] < a["end"]:
                    pairs.append((a["op_id"], b["op_id"]))
        return pairs

    def reorder_operation(self, op_id: str, direction: str) -> None:
        """Move an operation one step earlier/later in the repair graph.

        Order matters when windows overlap, so it is explicit and user-driven.
        """
        idx = next((i for i, o in enumerate(self.operations)
                    if o["op_id"] == op_id), None)
        if idx is None:
            raise KeyError(op_id)
        if direction == "up" and idx > 0:
            self.operations[idx - 1], self.operations[idx] = (
                self.operations[idx], self.operations[idx - 1])
        elif direction == "down" and idx < len(self.operations) - 1:
            self.operations[idx + 1], self.operations[idx] = (
                self.operations[idx], self.operations[idx + 1])
        else:
            raise ValueError(f"cannot move {op_id} {direction}")
        self.save()

    def undo_last(self) -> dict[str, Any] | None:
        """Roll back the most recent operation and reset its candidate."""
        if not self.operations:
            return None
        op = self.operations.pop()
        cand_id = op.get("cand_id")
        if cand_id:
            for c in self.click_candidates + self.hum_candidates:
                if c.cand_id == cand_id:
                    c.decision = "pending"
        self.save()
        return op

    # ------------------------------------------------------------------
    # Deterministic rendering from the pristine original
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray:
        """Replay every accepted operation on a fresh copy of the original."""
        y = np.array(self.audio.samples, dtype=np.float32, copy=True)
        sr = self.audio.sample_rate
        for op in self.operations:
            res: RepairResult
            if op["op"] == "click_repair":
                res = repair_click(
                    y, op["start"], op["end"], sr,
                    algorithm=op["algorithm"],
                    max_gap_ms=op.get("max_gap_ms", 10.0),
                    **op.get("params", {}),
                )
            elif op["op"] == "hum_notch":
                p = op["params"]
                res = notch_hum(
                    y, sr,
                    frequency=p["frequency_hz"],
                    bandwidth_hz=p["bandwidth_hz"],
                    quality_scale=p.get("quality_scale", 1.0),
                )
            else:
                raise RuntimeError(f"unknown operation {op['op']}")
            y = res.samples
        return y

    def render_original(self) -> np.ndarray:
        """The untouched original buffer — the A/B comparison baseline."""
        return np.array(self.audio.samples, dtype=np.float32, copy=True)

    def export_render(self, out_path: str | None = None) -> str:
        out_path = out_path or os.path.join(self.path, RENDER_NAME)
        write_wav(out_path, self.render(), self.audio.sample_rate)
        return os.path.abspath(out_path)

    def repair_list(self) -> list[dict[str, Any]]:
        """Human/auditor-facing record of every applied step."""
        rows: list[dict[str, Any]] = []
        for i, op in enumerate(self.operations, start=1):
            row = {
                "order": i,
                "op_id": op["op_id"],
                "type": op["op"],
                "cand_id": op.get("cand_id"),
                "algorithm": op["algorithm"],
                "window_ms": [
                    round(1000.0 * op["start"] / self.audio.sample_rate, 3),
                    round(1000.0 * op["end"] / self.audio.sample_rate, 3),
                ],
                "params": op.get("params", {}),
            }
            rows.append(row)
        return rows
