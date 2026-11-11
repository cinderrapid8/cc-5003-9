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
  detectors.py    爆音候选（AR/中值残差）与工频候选检测
  repair.py       AR 双向插补、三次样条插补、IIR 陷波（含参数留痕）
  project.py      项目清单、专家决策、修复图、确定性渲染、回退
  synthetic.py    干净基准、干扰注入、量化评估指标
  webapp.py       Flask REST 后端（与 CLI 共用同一引擎）
  cli.py          命令行入口
web/              单页界面（波形/频谱、候选裁决、修复顺序、前后试听）
tests/            30 个 pytest（引擎不变量 + Web API）
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
陷波参数留痕与语音保持、撤销、重叠顺序、重开复算、禁止全局衰减，以及全部
Web API 行为。
