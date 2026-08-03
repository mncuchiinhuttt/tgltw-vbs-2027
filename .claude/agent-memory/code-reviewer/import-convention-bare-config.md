---
name: import-convention-bare-config
description: Why every models/*.py file does `from config import X` (bare) instead of a dotted import, and how to verify it actually resolves before flagging it as broken
metadata:
  type: project
---

Every file under `models/` (embedding.py, object_detector.py, asr.py, siglip_embedder.py,
etc.) imports its config constants with a bare `from config import ...` — never
`from preprocessing.config import ...` or `from inference_code.config import ...`.
This is intentional and consistent, not a bug, but it is fragile and easy to
mis-flag in review.

**Why it works:** there is no root-level `config.py`. Instead there are two
sibling configs: `preprocessing/config.py` and `inference-code/config.py`.
`models/*.py`'s bare `import config` only resolves when the *directory
containing config.py* is on `sys.path`. That happens two ways in this repo:
1. Running `python preprocessing/main.py` (or `inference-code/main.py`) as a
   script — Python auto-prepends the script's own directory to `sys.path[0]`.
2. `cd preprocessing && python main.py` — cwd equivalent of the above.

Both entry points (`preprocessing/main.py`, `inference-code/main.py`) *also*
explicitly `sys.path.append(parent-of-parent)` (project root) so that dotted
imports like `from models.asr import WhisperASR` and
`from preprocessing.config import ...` (used by files *inside*
`preprocessing/` itself, e.g. `preprocessing/video/ocr.py`,
`preprocessing/audio/asr_segment_filter.py`) also resolve.

**How to apply:** before flagging `from config import` in a `models/*.py`
file as a broken/inconsistent import, don't just run
`python -c "from models.asr import WhisperASR"` from the repo root — that
fails (`ModuleNotFoundError: No module named 'config'`) because sys.path[0]
is cwd (repo root), which has no `config.py`. It only succeeds if cwd is
`preprocessing/` or `inference-code/`, or if invoked via one of the real
entry-point scripts. This tripped up a "manual pre-verification" claim during
the faster-whisper ASR migration review (2026-08-04) — the claimed working
import command needs the right cwd to actually be true; verify with the
correct invocation rather than trusting the claim at face value.
