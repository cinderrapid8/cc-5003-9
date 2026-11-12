# 口述历史录音修复台（Oral-History Restoration Workbench）

面向**合成的窄脉冲爆音**与**稳定工频干扰**的半自动修复工具。设计原则：

- **原件只读**：源 WAV 永不被写入或覆盖；每次渲染都从原始样本按修复图重新计算。
- **候选而非自动删除**：检测只产出候选。短促辅音、真实敲击同样会产生尖峰，
  必须由专家逐条**接受**或**排除**。
- **爆音修复有界**：插补只作用于专家认可的短窗口（默认 ≤ 10 ms）；更长的缺损
  被标记为「无法可靠修复」，拒绝猜测。
- **工频滤除留痕**：每个陷波记录中心频率与带宽（默认 2 Hz 窄带、零相位），
  避免把附近有用成分一并当作噪声。
- **全程可回退、可复算**：撤销、重开项目都能得到逐字节一致的渲染；重叠区间有
  明确的应用顺序，可手动调整。
- **禁止统一降音量**：没有任何全局增益衰减作为「降噪」手段，只有局部插补与窄带陷波。

## 目录结构

```
oralrestore/
  audioio.py      只读音频 I/O、sha256 校验、WAV 导出
  detectors.py    爆音候选（AR/中值残差）与工频候选检测（含掩码/频谱轨迹）
  repair.py       AR 双向插补、三次样条插补、IIR 陷波（含参数留痕）
  project.py      项目清单、专家决策、修复图、确定性渲染、回退
  diagnostics.py  误伤诊断：证据采集、片段分类、三假设度量、有预算的重放
  controlled.py   受控录音场景编排与逐假设证据判定
  synthetic.py    干净基准、干扰注入、受控录音、量化评估指标
  webapp.py       Flask REST 后端（与 CLI 共用同一引擎）
  cli.py          命令行入口（含 diagnose 诊断命令）
web/              单页界面（波形/频谱、候选裁决、修复顺序、前后试听）
tests/            57 个 pytest（引擎不变量 + Web API + 诊断证据链）
demo/             演示输出（真实 WAV、项目清单、评估报告）
```

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 一键演示与评估

```bash
.venv/bin/python -m oralrestore.cli demo --out demo/output
```

产出（均为真实文件）：

- `fixtures/`：干净基准、仅爆音、仅工频、爆音+工频、长缺损、合法瞬态六个 WAV；
- `project_pops/`、`project_hum/`、`project_long/`、`project_falsealarm/`：
  各自的 `project.json` 修复清单与 `rendered.wav`；
- `evaluation_report.json`：检测命中/误报、窗外保持、噪声残留、信号保真、
  重叠顺序、重开复算结果。

### 评估场景与预期

| 场景 | 检测 | 专家动作 | 关键指标 |
|---|---|---|---|
| 仅爆音 | 5/5 命中、干净信号 0 误报 | 接受 5 个短窗 AR 插补 | 修复窗外逐样本差 = 0 |
| 仅工频 | 50/100/150 Hz 命中，120 Hz 语音谐波列为歧义候选 | 接受 3 条真线、排除 120 Hz | 陷波带外 SNR ≈ 69 dB，语音频带变化 0 dB，工频抑制 35–56 dB |
| 长缺损（60 ms） | 标记为不可修复 | 拒绝插补 | 渲染与输入一致，不猜测 |
| 合法瞬态（辅音/敲击） | 可能报为候选 | 全部排除 | 渲染与输入逐样本相同 |
| 重开项目 | — | — | WAV sha256 一致、PCM16 码差 ≤ 1 LSB |

## 命令行工作流

```bash
PY=.venv/bin/python
$PY -m oralrestore.cli init  input.wav projects/job1
$PY -m oralrestore.cli detect projects/job1 --what both --list
$PY -m oralrestore.cli candidates projects/job1
$PY -m oralrestore.cli accept  projects/job1 click-0001 --algorithm ar
$PY -m oralrestore.cli accept  projects/job1 hum-0001 --bandwidth 2
$PY -m oralrestore.cli exclude projects/job1 click-0002 --reason "real plosive"
$PY -m oralrestore.cli ops projects/job1
$PY -m oralrestore.cli reorder projects/job1 op-0003 up
$PY -m oralrestore.cli undo projects/job1
$PY -m oralrestore.cli render projects/job1 --output out/restored.wav
```

`project.json` 记录每个候选的检测器、参数、专家决策，以及每个操作的算法、
窗口（毫秒）与完整参数；重开时先校验源文件 sha256，再按操作顺序重算。

## 误伤声音诊断（语音细节变弱排查）

修复后语音细节变弱可能来自三个彼此独立的环节，诊断**不预设根因**，而是让
每条假设各自举证、并把异常定位到具体处理阶段：

| 假设 | 定位阶段 | 证据 |
|---|---|---|
| H1 检测过宽 | `click_detection` | 阈值掩码跨距 vs. 填充后候选窗口的比率、窗口内输入内容分类 |
| H2 边界插补 | `click_repair` | 修复前后边界跳变增量（非绝对值）、窗口能量变化、窗外零改动 |
| H3 滤波参数失配 | `hum_notch` | 解析实测 -3 dB 有效带宽、目标干扰抑制量、邻接有用频带增益、滤波器系数状态 |

每条证据都关联四要素：**候选、处理阶段、输入片段（样本区间 + 内容 sha256，
不保留音频副本）、测量值**。输入片段先按固定规则分类为辅音瞬态
（`consonant_transient`）或持续元音（`sustained_voiced`），分类所用特征与
触发规则随证据一并记录。

### 受控录音

`diagnose --controlled` 生成一段带标注的受控录音：三个共用稳定 110 Hz 谐波
栈的浊音短语（每个带清晰起音）、停顿处三个辅音突发，并在已知位置注入爆音和
50/100/150 Hz 工频。因为内容真值已知，分类器结论可被校验，三个假设也能在
相同素材上被分别举证。另含三个**只用于诊断、不写入修复图**的参数探测：

- 放宽检测阈值（默认 σ=7.5）暴露 H1 的过宽窗口；
- 12 Hz 过宽陷波暴露 H3 的有效带宽越界与邻带损伤；
- 同窗口改用样条插补，与 AR 的 H2 边界证据对照。

### 开关、预算与声音不变性

```bash
PY=.venv/bin/python
# 对现有项目做诊断（渲染与普通 render 逐字节一致，报告另存）
$PY -m oralrestore.cli diagnose projects/job1
# 关闭诊断：只渲染、不采集任何证据
$PY -m oralrestore.cli diagnose projects/job1 --disabled
# 受控录音全流程（报告写入输出目录的 diagnostic_report.json）
$PY -m oralrestore.cli diagnose demo/diag --controlled
# 显式限制采样与日志开销
$PY -m oralrestore.cli diagnose projects/job1 \
    --max-segments 64 --max-evidence 256 --max-window-ms 20
```

- 诊断是引擎旁的观察者，复用未改动的检测/修复函数；报告中的
  `render_matches_project_render` 与诊断 WAV 的 sha256 共同证明开关诊断
  不改变声音（测试以逐字节相等断言）。
- 采样受 `DiagnosticBudget` 硬约束：片段数、证据条数、单段分析时长、频谱
  网格数均有上限；超限项进入报告的 `sampling.skipped` 而不是静默截断，开销
  不随录音长度增长。
- 报告按源文件 sha256 绑定输入；口述修复复跑同一项目时，重放顺序、参数与
  证据一一对应，结果可重复。

## 网页界面

```bash
.venv/bin/python -m oralrestore.webapp     # http://127.0.0.1:5000
```

- 「载入合成演示」一键加载爆音+工频项目；
- 波形上以颜色区分待处理/已接受/已排除/不可修复候选；
- 每个爆音可选 AR 或样条插补，每个工频可选陷波带宽，或排除；
- 修复图显示应用顺序与重叠提示，可上移/下移、撤销；
- 左右两张频谱与两个音频播放器分别对应**原始（只读）**与**当前修复渲染**，
  支持前后 A/B 试听。

## 测试

```bash
.venv/bin/python -m pytest tests/ -q
```

覆盖：源文件哈希稳定、篡改拒绝、候选不自动删除、窗外零改动、长缺损拒绝、
陷波参数留痕与语音保持、撤销、重叠顺序、重开复算、禁止全局衰减、全部 Web
API 行为，以及诊断证据链（掩码-窗口-片段绑定、辅音/元音分类、三假设独立
判定、有效带宽与目标干扰定位、预算限流、诊断开关下渲染逐字节一致、CLI
复跑对应相同证据）。
