"""API-level tests for the Flask workbench.

These exercise the same engine the browser drives, proving the web layer
preserves the candidate-only and read-only invariants end to end.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from oralrestore.webapp import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture()
def loaded(client, tmp_path):
    resp = client.post("/api/demo/load", json={
        "out_dir": str(tmp_path / "web"), "sample_rate": 16000})
    assert resp.status_code == 200
    return resp.get_json()["project_id"]


def test_index_and_static_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_full_candidate_workflow(client, loaded):
    pid = loaded
    r = client.post("/api/detect/click",
                    json={"project_id": pid, "threshold_sigma": 8})
    assert r.get_json()["total"] == 5
    r = client.post("/api/detect/hum",
                    json={"project_id": pid, "line_frequencies": [50, 60]})
    assert r.get_json()["total"] >= 3

    cand = client.get(f"/api/candidates?project_id={pid}").get_json()
    # everything starts as a pending candidate, nothing auto-applied
    assert all(c["decision"] == "pending" for c in cand["click"])
    assert all(c["decision"] == "pending" for c in cand["hum"])

    # accept all repairable clicks
    for c in cand["click"]:
        assert c["repairable"]
        client.post("/api/click/accept",
                    json={"project_id": pid, "cand_id": c["cand_id"]})
    # accept true 50 Hz lines, exclude the ambiguous 120 Hz speech harmonic
    for c in cand["hum"]:
        if abs(c["frequency"] - 120) <= 1.5:
            client.post("/api/hum/exclude",
                        json={"project_id": pid, "cand_id": c["cand_id"],
                              "reason": "speech harmonic"})
        else:
            client.post("/api/hum/accept",
                        json={"project_id": pid, "cand_id": c["cand_id"]})

    ops = client.get(f"/api/operations?project_id={pid}").get_json()
    assert len(ops["operations"]) == 8  # 5 clicks + 3 hum notches
    # notch params are fully recorded
    notches = [o for o in ops["operations"] if o["type"] == "hum_notch"]
    assert all(o["params"]["frequency_hz"] in (50.0, 100.0, 150.0)
               for o in notches)
    assert all(o["params"]["bandwidth_hz"] == 2.0 for o in notches)


def test_excluded_candidate_keeps_render_identical(client, tmp_path):
    from oralrestore.synthetic import (add_legitimate_transients,
                                       synthesize_clean)
    from oralrestore.audioio import write_wav
    sr = 16000
    sig = add_legitimate_transients(
        synthesize_clean(sample_rate=sr), sr, centres_ms=[900.0, 2400.0])
    src = tmp_path / "legit.wav"
    write_wav(str(src), sig, sr)
    r = client.post("/api/project/create",
                    json={"source_path": str(src),
                          "project_dir": str(tmp_path / "p")})
    pid = r.get_json()["project_id"]
    client.post("/api/detect/click", json={"project_id": pid})
    cand = client.get(f"/api/candidates?project_id={pid}").get_json()
    assert len(cand["click"]) >= 1
    for c in cand["click"]:
        client.post("/api/click/exclude",
                    json={"project_id": pid, "cand_id": c["cand_id"],
                          "reason": "real plosive"})
    orig = client.get(
        f"/api/audio?project_id={pid}&kind=original").data
    rend = client.get(
        f"/api/audio?project_id={pid}&kind=rendered").data
    assert orig == rend  # no accepted op => byte-identical output


def test_audio_endpoints_return_real_wav(client, loaded):
    pid = loaded
    for kind in ("original", "rendered"):
        resp = client.get(f"/api/audio?project_id={pid}&kind={kind}")
        assert resp.status_code == 200
        assert resp.mimetype == "audio/wav"
        y, sr = sf.read(io.BytesIO(resp.data))
        assert sr == 16000 and len(y) == 96000


def test_undo_rolls_back(client, loaded):
    pid = loaded
    client.post("/api/detect/click", json={"project_id": pid})
    cand = client.get(f"/api/candidates?project_id={pid}").get_json()
    cid = cand["click"][0]["cand_id"]
    client.post("/api/click/accept",
                json={"project_id": pid, "cand_id": cid})
    before = len(client.get(
        f"/api/operations?project_id={pid}").get_json()["operations"])
    client.post("/api/undo", json={"project_id": pid})
    after = client.get(
        f"/api/operations?project_id={pid}").get_json()["operations"]
    assert len(after) == before - 1
    cand = client.get(f"/api/candidates?project_id={pid}").get_json()
    assert next(c for c in cand["click"]
                if c["cand_id"] == cid)["decision"] == "pending"


def test_missing_project_is_404(client):
    resp = client.get("/api/candidates?project_id=doesnotexist")
    assert resp.status_code == 404


def test_visualization_payloads(client, loaded):
    pid = loaded
    wf = client.get(
        f"/api/waveform?project_id={pid}&kind=original").get_json()
    assert len(wf["min"]) == len(wf["max"]) == len(wf["centres"])
    sp = client.get(
        f"/api/spectrum?project_id={pid}&kind=rendered").get_json()
    assert len(sp["freq"]) == len(sp["power_db"])
    assert max(sp["freq"]) > 1000
