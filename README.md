# Cube Bench

> A reproducible Rubik’s Cube benchmark suite for probing multimodal LLMs across perception, grounding, and closed-loop control.

---

## Install & Quick Start

```bash
# (Recommended) Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1) Clone & install (editable dev mode)
git clone <repo-url> cube-bench
cd cube-bench
pip install -e .

# 2) Precompute IDA* / optimal-distance graphs
#    Note: this can take hours to compute..
cube-bench --build

# 3) CLI help (see available commands/flags)
cube-bench --help

# 4) Check the install
python -c "import cube_bench as cb; print('cube_bench version:', getattr(cb, '__version__', 'unknown'))"
