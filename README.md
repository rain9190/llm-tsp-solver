# LLM as an End-to-End TSP Solver

Teaching small language models to solve the **Travelling Salesman Problem** end-to-end.

The model reads a list of city coordinates as text and directly creates a tour, with no external solver at inference time. It is trained by **supervised fine-tuning
on an OR-Tools teacher**, then improved with **reward-based fine-tuning** (RAFT),
and the effect of is studied across 0.5B, 1.5B parameter models and compared to a 7B parameter zero-shot baseline.

## TL;DR

- Fine-tuned **Qwen2.5-0.5B** and **Qwen2.5-1.5B** with LoRA + 4-bit quantization on Colab T4 GPU.
- A **0.5B** model learns the format but systematically drops cities on larger instances (0% feasibility at 14+ cities) — a capacity limit, confirmed because best-of-N sampling doesn't fix it.
- Scaling to **1.5B** raises feasibility on large instances from **0% → 97%**, then reward fine-tuning perfects it to **100%**.
- The fine-tuned **1.5B beats a zero-shot 7B general model** (~100% vs ~43% feasibility) despite being ~4.7× smaller — demonstrating that task-specific fine-tuning beats raw scale here.
- Inference-time **best-of-8** lowers the optimality gap to ~92%.

## The problem

The **Travelling Salesman Problem (TSP)**: given *n* cities, find the shortest closed tour
visiting each exactly once. It is a NP-hard combinatorial optimization problem — with *n*
cities there are (*n*−1)!/2 tours. The question this project explores is whether a general language model can be taught to *directly output* a good tour as text, end-to-end.

## Method

1. **Data**: generated random euclidean instances and solved each near-optimally with **Google OR-Tools**. Each instance along with its soluion becomes the training pair.
2. **SFT**: LoRA-fine-tuned Qwen2.5-0.5B to imitate the teacher's tours (`notebooks/01_sft.ipynb`).
3. **Scaling**: repeated on Qwen2.5-1.5B to test whether increasing the number of parameters fix the feasibility failure found at the 0.5B parameter model.
4. **Reward fine-tuning**: sampled N tours per instance, kept the shortest *feasible* one (verifiable reward = tour length), and fine-tuned on those self-generated best tours (`notebooks/02_reward.ipynb`). This is **rejection-sampling fine-tuning (RAFT)**.
5. **Evaluation**: calculated feasibility rate and optimality gap on the test instances, with the fine tuned models (`notebooks/03_eval.ipynb`), plus a zero-shot 7B baseline (`notebooks/04_baseline_7b.ipynb`).

## Results

All models are evaluated on the **same held-out test set** (200 Euclidean TSP instances,
10–20 cities, seeds disjoint from training). Two metrics:

- **Feasibility rate**: fraction of outputs that are valid tours (every city exactly once).
- **Optimality gap**: over feasible outputs, mean % by which the tour exceeds the OR-Tools optimum.

Results are split by instance size (10–13 vs 14–20 cities), which was the key diagnostic.
Reference: optimal = 0% gap; nearest-neighbour heuristic ≈ 15% gap.

### Main comparison (single-shot)

| Model | Training | 10–13 feasibility | optimality gap | 14–20 feasibility | optimality gap |
|:-------:|:----------:|:---:|:---:|:---:|:---:|
| Qwen2.5-7B | **none** | 44.2% | 65.3% |43.1% | 109.6% |
| Qwen2.5-0.5B | SFT | 85.7% | 98.9% | **0.0%** | - |
| Qwen2.5-1.5B | SFT | 100% | 101.1% |96.7% | 147.6% |

| Qwen2.5-1.5B | SFT + RAFT | **100% (overall)** | **91.8% (overall)** |

### Key findings

1. **A 7B general model, zero-shot, is unreliable**: valid tours occur less than half the time (~43–44%), regardless of instance size. It understands TSP but doesn't satisfy the output format/constraint without fine-tuning.
2. **Task-specific fine-tuning beats raw scale**: the fine-tuned 1.5B (~100% feasibility) outperforms the 4.7×-larger zero-shot 7B.
3. **Feasibility is capacity-bound at small scale**: the 0.5B does not produce valid tours at all for 14+ cities. Scaling to 1.5B resolves it (0% -> 97%).
4. **Reward fine-tuning improves the results**: creates valid tours 100% of the time and improves the optimlaity gap from ~100-148% to 91.8% (26.18% improvement)

## Repository structure

```
llm-tsp-solver/
├── src/
│   ├── tsp_env.py        # instance generator + OR-Tools teacher
│   ├── serialize.py      # text format, parser, feasibility verifier
│   ├── make_dataset.py   # builds train/test JSONL
│   └── evaluate.py       # feasibility + optimality-gap scoring
├── notebooks/
│   ├── 01_sft.ipynb           # supervised fine-tuning (0.5B / 1.5B)
│   ├── 02_reward.ipynb        # rejection-sampling reward fine-tuning
│   ├── 03_eval.ipynb          # evaluation + best-of-N
│   └── 04_baseline_7b.ipynb   # zero-shot 7B baseline
├── app/
│   └── tsp_chatbot.py    # Gradio demo
└── docs/
    ├── EXPERIMENT_LOG.md # full experiment record
    └── study_document.pdf
```

## Reproduce

```bash
# 1. Generating data (Local setup)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src && python make_dataset.py        # writes data/train.jsonl, data/test.jsonl

# 2. Training & evaluation (Colab, T4 GPU)
#    Run the notebooks in order: 01_sft -> 02_reward -> 03_eval.
#    Trained LoRA adapters are saved to Google Drive (not committed — see below).
```

Datasets and trained adapters are **not** committed (model weights are build artifacts, not source). The data is regenerated by `make_dataset.py`; the adapters by running the notebooks.

## Tools

Python, PyTorch, Hugging Face `transformers` / `trl` / `peft`, `bitsandbytes` (4-bit), Google OR-Tools, NumPy, Matplotlib, Gradio. Models: Qwen2.5-Instruct (0.5B, 1.5B, 7B).

## Acknowledgement

Method based on *Large Language Models as End-to-end Combinatorial Optimization Solvers* (LLMCoSolver, NeurIPS 2025). This is an independent student-scale reproduction.
