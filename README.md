# Semantic Equivalence, Structural Fragility

**Quantifying Prompt Entropy Degradation in Large Language Models**

> Research project for the IEEE AI Track. This repository contains the experimental pipeline used to quantify *prompt brittleness* by measuring Shannon entropy over output logits, rather than relying on binary accuracy benchmarks.

---

## Abstract

Prompt engineering is largely treated as an empirical heuristic. This project tests whether minor, semantic-preserving syntactic perturbations (whitespace, Markdown wrapping, JSON restructuring) induce measurable statistical chaos in the output logit distribution of open-source LLMs — prior to any observable drop in task accuracy. We hypothesize the existence of a **Critical Entropy Threshold** that predicts logical failure before the model finishes generating a response.

- **Dataset:** [GSM8k](https://github.com/openai/grade-school-math) (multi-step logical/mathematical reasoning)
- **Models:** Llama-3-8B, Gemma-2 (via Hugging Face Transformers)
- **Core metric:** Shannon entropy (\(H\)) over softmax-normalized logits

---

## Repository Structure

```
prompt-entropy-fragility/
├── notebooks/
│   └── 01_logit_extraction.ipynb   # Main experiment notebook
├── src/
│   ├── perturbations.py            # Whitespace / Markdown / JSON perturbations
│   ├── entropy.py                  # Shannon entropy calculation
│   └── model_runner.py             # Model loading + logit extraction
├── data/
│   └── gsm8k/                      # Dataset (downloaded, not committed)
├── results/
│   └── figures/                    # Output plots for Deliverable 3
├── requirements.txt
└── README.md
```

---

## Team

| Role | Focus Area |
|---|---|
| Lead / Coordination | Information Theory & Entropy |
| Member 2 | Robustness & Formatting |
| Member 3 | Output Mechanics & Logits |
| Member 4 | Benchmarks & Datasets (GSM8k) |

> Team name (internal): *Charlie y los Kirks*

---

## Methodology Overview

1. **Perturbation:** Each GSM8k prompt \(P_{base}\) is transformed into three semantically-equivalent variants:
   - \(P_{space}\): random double spaces / trailing newlines
   - \(P_{mark}\): Markdown/XML wrapping
   - \(P_{struct}\): JSON-like schema restructuring

2. **Logit extraction:** For each variant, the model generates the first \(k\) tokens. Raw logits (pre-softmax) are captured at each step.

3. **Entropy calculation:**

   \[
   p(x_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} \quad (1)
   \]

   \[
   H(X) = -\sum_{i=1}^{N} p(x_i) \log_2 p(x_i) \quad (2)
   \]

4. **Threshold validation:** \(\Delta H = H_{perturbed} - H_{base}\) is correlated against binary correctness to locate the point where entropy escalation predicts functional collapse.

---

## Setup

```bash
git clone https://github.com/<your-username>/prompt-entropy-fragility.git
cd prompt-entropy-fragility
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook notebooks/01_logit_extraction.ipynb
```

Model access: some models (e.g. Llama-3) require a Hugging Face account and license acceptance. Run `huggingface-cli login` before executing the notebook.

---

## Status

- [x] Literature review (20 references)
- [x] Methodology design
- [ ] Perturbation pipeline (`src/perturbations.py`)
- [ ] Logit extraction pipeline (`src/model_runner.py`)
- [ ] Entropy analysis + threshold validation
- [ ] Figures and statistical results (Deliverable 3)

---

## License

MIT — for academic and research use.

## Citation

```bibtex
@inproceedings{charlieandloskirks2026,
  title={Semantic Equivalence, Structural Fragility: Quantifying Prompt Entropy Degradation in Large Language Models},
  author={TBD},
  booktitle={IEEE},
  year={2026}
}
```
