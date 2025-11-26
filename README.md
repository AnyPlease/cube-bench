# Cube Bench

> A reproducible Rubik’s Cube benchmark suite for probing multimodal LLMs across perception, grounding, and closed-loop control.

---

## Install & Quick Start

```bash
# 1) Clone the repository
git clone <repo-url> cube-bench
cd cube-bench

# 2) Create & activate a virtual environment (Highly Recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install pinned dependencies
#    CRITICAL: Run this *before* installing the package to prevent version conflicts.
pip install -r requirements.txt

# 4) Install the package in editable dev mode
pip install -e .

# 5) Precompute IDA* / optimal-distance graphs
#    Note: this can take ~8 hours to compute.
cube-bench --build

# 6) Check the install
python -c "import cube_bench as cb; print('cube_bench version:', getattr(cb, '__version__', 'unknown'))"