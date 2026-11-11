/* Oral-history restoration workbench frontend.
 * Talks only to the Flask API; all detection/repair runs in the Python engine.
 */
"use strict";

const state = { projectId: null, clicks: [], hums: [], ops: [] };

const $ = (id) => document.getElementById(id);

function banner(msg, kind = "info") {
  const el = $("banner");
  el.textContent = msg;
  el.className = `banner show ${kind}`;
}

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: state.projectId, ...(body || {}) }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function apiGet(path) {
  const sep = path.includes("?") ? "&" : "?";
  const res = await fetch(`${path}${sep}project_id=${state.projectId}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function audioUrl(kind) {
  return `/api/audio?project_id=${state.projectId}&kind=${kind}`;
}

// ---------------------------------------------------------------------------
// Project loading
// ---------------------------------------------------------------------------

async function loadDemo() {
  const outDir = window.prompt("演示输出目录", "demo/web_output");
  if (!outDir) return;
  try {
    const data = await api("/api/demo/load", {
      out_dir: outDir, sample_rate: 16000,
    });
    state.projectId = data.project_id;
    await afterLoad();
    banner("已载入合成演示：含 5 个窄爆音与 50 Hz 工频。先运行检测再逐条裁决。");
  } catch (e) { banner(e.message, "warn"); }
}

async function openProject() {
  const dir = $("project-dir").value.trim();
  if (!dir) return banner("请输入项目目录", "warn");
  try {
    const data = await api("/api/project/open", { project_dir: dir });
    state.projectId = data.project_id;
    await afterLoad();
  } catch (e) { banner(e.message, "warn"); }
}

async function afterLoad() {
  const s = await apiGet("/api/state");
  $("src-pill").textContent =
    `${s.sample_rate} Hz · ${s.duration.toFixed(2)} s · sha256 ${s.source_sha256.slice(0, 10)}…`;
  $("audio-orig").src = audioUrl("original");
  await refreshAll();
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

async function detectClicks() {
  try {
    const data = await api("/api/detect/click", {
      method: $("click-method").value,
      threshold_sigma: parseFloat($("click-thr").value),
      max_gap_ms: parseFloat($("max-gap").value),
    });
    banner(`新增 ${data.added} 个爆音候选，共 ${data.total} 个。请逐条接受或排除。`);
    await refreshAll();
  } catch (e) { banner(e.message, "warn"); }
}

async function detectHum() {
  try {
    const freqs = Array.from($("line-freq").selectedOptions).map((o) =>
      parseFloat(o.value));
    const data = await api("/api/detect/hum", {
      line_frequencies: freqs,
      bandwidth_hz: parseFloat($("hum-bw").value),
    });
    banner(`新增 ${data.added} 个工频候选，共 ${data.total} 个。`);
    await refreshAll();
  } catch (e) { banner(e.message, "warn"); }
}

// ---------------------------------------------------------------------------
// Expert decisions
// ---------------------------------------------------------------------------

async function acceptClick(id, algorithm) {
  try {
    await api("/api/click/accept", {
      cand_id: id, algorithm,
      max_gap_ms: parseFloat($("max-gap").value),
    });
    await refreshAll();
  } catch (e) { banner(e.message, "warn"); }
}

async function excludeClick(id) {
  const reason = window.prompt("排除原因（如：真实唇齿音/敲击，需保留）", "legitimate transient");
  if (reason === null) return;
  await api("/api/click/exclude", { cand_id: id, reason });
  await refreshAll();
}

async function acceptHum(id) {
  await api("/api/hum/accept", {
    cand_id: id, bandwidth_hz: parseFloat($("hum-bw").value),
  });
  await refreshAll();
}

async function excludeHum(id) {
  const reason = window.prompt("排除原因（如：落在工频网格上的语音谐波）", "speech harmonic");
  if (reason === null) return;
  await api("/api/hum/exclude", { cand_id: id, reason });
  await refreshAll();
}

async function undo() {
  const r = await api("/api/undo", {});
  banner(r.rolled_back ? `已撤销 ${r.rolled_back.op_id}，候选回到待处理。`
                       : "没有可撤销的操作。");
  await refreshAll();
}

async function reorder(opId, direction) {
  await api("/api/operations/reorder", { op_id: opId, direction });
  await refreshAll();
}

// ---------------------------------------------------------------------------
// Rendering / refresh
// ---------------------------------------------------------------------------

async function refreshRender() {
  // Cache-bust so the <audio> element reloads the freshly rendered WAV.
  const url = `${audioUrl("rendered")}&t=${Date.now()}`;
  $("audio-rend").src = url;
  await drawSpectrum("rendered", $("spec-rend"));
}

async function refreshAll() {
  const [cand, ops] = await Promise.all([
    apiGet("/api/candidates"), apiGet("/api/operations"),
  ]);
  state.clicks = cand.click;
  state.hums = cand.hum;
  state.ops = ops.operations;
  renderClickTable();
  renderHumTable();
  renderOps(ops.overlaps);
  await Promise.all([
    drawWaveform(),
    drawSpectrum("original", $("spec-orig")),
    drawSpectrum("rendered", $("spec-rend")),
  ]);
  $("audio-rend").src = `${audioUrl("rendered")}&t=${Date.now()}`;
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

function tag(status) {
  return `<span class="tag ${status}">${status}</span>`;
}

function renderClickTable() {
  $("click-count").textContent = state.clicks.length;
  $("click-rows").innerHTML = state.clicks.map((c) => `
    <tr>
      <td>${c.cand_id}</td>
      <td class="num">${c.start_ms}</td>
      <td class="num">${(c.end_ms - c.start_ms).toFixed(2)}</td>
      <td class="num">${c.score_sigma ?? c.score}</td>
      <td>${c.repairable
        ? '<span class="tag yes">是</span>'
        : '<span class="tag no">否</span>'}</td>
      <td>${tag(c.decision)}</td>
      <td class="row">
        <button class="good" ${c.decision === "accepted" || !c.repairable
          ? "disabled" : ""}
          onclick="acceptClick('${c.cand_id}','ar')">接受(AR)</button>
        <button class="good" ${c.decision === "accepted" || !c.repairable
          ? "disabled" : ""}
          onclick="acceptClick('${c.cand_id}','spline')">接受(样条)</button>
        <button class="bad" ${c.decision === "excluded" ? "disabled" : ""}
          onclick="excludeClick('${c.cand_id}')">排除</button>
      </td>
    </tr>`).join("");
}

function renderHumTable() {
  $("hum-count").textContent = state.hums.length;
  $("hum-rows").innerHTML = state.hums.map((c) => `
    <tr>
      <td>${c.cand_id}</td>
      <td class="num">${c.frequency}</td>
      <td class="num">${c.bandwidth}</td>
      <td class="num">${c.snr_db}</td>
      <td>${tag(c.decision)}</td>
      <td class="row">
        <button class="good" ${c.decision === "accepted" ? "disabled" : ""}
          onclick="acceptHum('${c.cand_id}')">接受陷波</button>
        <button class="bad" ${c.decision === "excluded" ? "disabled" : ""}
          onclick="excludeHum('${c.cand_id}')">排除</button>
      </td>
    </tr>`).join("");
}

function renderOps(overlaps) {
  $("op-rows").innerHTML = state.ops.length ? state.ops.map((o, i) => `
    <tr>
      <td class="num">${o.order}</td>
      <td>${o.op_id}</td>
      <td>${o.type}</td>
      <td>${o.algorithm}</td>
      <td class="num">[${o.window_ms[0]}, ${o.window_ms[1]}]</td>
      <td><code>${JSON.stringify(o.params)}</code></td>
      <td class="row">
        <button ${i === 0 ? "disabled" : ""}
          onclick="reorder('${o.op_id}','up')">↑</button>
        <button ${i === state.ops.length - 1 ? "disabled" : ""}
          onclick="reorder('${o.op_id}','down')">↓</button>
      </td>
    </tr>`).join("")
    : '<tr><td colspan="7" class="muted">尚无已接受的修复操作</td></tr>';
  $("overlap-note").textContent = overlaps.length
    ? `检测到重叠窗口：${overlaps.map((p) => p.join(" ↔ ")).join("；")}。按上方顺序依次作用。`
    : "";
}

// ---------------------------------------------------------------------------
// Canvas drawing
// ---------------------------------------------------------------------------

async function drawWaveform() {
  const cv = $("wave");
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth || 900;
  cv.width = cssW * dpr; cv.height = 220 * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = cssW, H = 220;
  const data = await apiGet("/api/waveform?kind=original");
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#5f7fa0";
  ctx.lineWidth = 1;
  const n = data.n_samples;
  const x = (sample) => (sample / n) * W;
  ctx.beginPath();
  for (let i = 0; i < data.centres.length; i++) {
    const cx = x(data.centres[i]);
    ctx.moveTo(cx, H / 2 + data.min[i] * H / 2);
    ctx.lineTo(cx, H / 2 + data.max[i] * H / 2);
  }
  ctx.stroke();
  // centre line
  ctx.strokeStyle = "#33404f";
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
  // candidate boxes
  const colors = {
    pending: "#4f9cf9", accepted: "#34c98e",
    excluded: "#ef6a6a",
  };
  for (const c of state.clicks) {
    const x0 = x(c.start), x1 = x(c.end);
    ctx.fillStyle = c.repairable
      ? (colors[c.decision] || "#4f9cf9") + "55"
      : "#6b6b6b55";
    ctx.fillRect(x0, 4, Math.max(2, x1 - x0), H - 8);
  }
}

async function drawSpectrum(kind, cv) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth || 460;
  cv.width = cssW * dpr; cv.height = 180 * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = cssW, H = 180;
  const data = await apiGet(`/api/spectrum?kind=${kind}`);
  ctx.clearRect(0, 0, W, H);
  const f = data.freq, p = data.power_db;
  const fMax = Math.min(4000, f[f.length - 1]); // focus on speech band
  const pMin = -120, pMax = -20;
  const X = (ff) => (ff / fMax) * W;
  const Y = (db) => H - ((db - pMin) / (pMax - pMin)) * H;
  ctx.strokeStyle = kind === "rendered" ? "#34c98e" : "#4f9cf9";
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < f.length; i++) {
    if (f[i] > fMax) break;
    const xx = X(f[i]), yy = Y(Math.max(pMin, Math.min(pMax, p[i])));
    if (!started) { ctx.moveTo(xx, yy); started = true; }
    else ctx.lineTo(xx, yy);
  }
  ctx.stroke();
  ctx.fillStyle = "#8fa0b3";
  ctx.font = "11px sans-serif";
  ctx.fillText(`${kind === "rendered" ? "修复后" : "原始"} 频谱 (0–4 kHz)`, 8, 16);
}

// expose handlers used by inline onclick attributes
window.acceptClick = acceptClick;
window.excludeClick = excludeClick;
window.acceptHum = acceptHum;
window.excludeHum = excludeHum;
window.reorder = reorder;

$("btn-demo").onclick = loadDemo;
$("btn-open").onclick = openProject;
$("btn-detect-click").onclick = detectClicks;
$("btn-detect-hum").onclick = detectHum;
$("btn-undo").onclick = undo;
$("btn-render").onclick = refreshRender;
window.addEventListener("resize", () => {
  if (state.projectId) { drawWaveform(); drawSpectrum("original", $("spec-orig"));
    drawSpectrum("rendered", $("spec-rend")); }
});
