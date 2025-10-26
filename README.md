# Cube Bench

> A reproducible Rubik’s-Cube benchmark suite for probing multimodal LLMs (MLLMs) across perception, grounding, and closed-loop control.

---

## TL;DR

```bash
# 1) Clone & install (editable dev mode recommended)
git clone <your-repo-url> cube-bench
cd cube-bench
pip install -e .

# 2) Build IDA* heuristic/optimal-distance graphs (large, slow)
cube-bench --build
