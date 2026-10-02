"""
quick_experiment.py

Minimal REAL run of the full pipeline (perturb -> generate -> logits ->
entropy) on distilgpt2 with 5 hardcoded basic-math questions, so we have
genuine numbers to report ahead of the deadline. Not the paper's actual
experiment (that uses Gemma-2 + GSM8k, see notebooks/01_logit_extraction.ipynb) --
this is a small, fast, real-data stand-in.

distilgpt2 is not instruction-tuned and is weak at arithmetic, so `correct`
may be False across most rows -- that's an honest result, not a bug. The
entropy signal is what this script is meant to produce, not accuracy.
"""

import re
import sys

sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from entropy import delta_entropy, shannon_entropy
from model_runner import load_model
from perturbations import perturb_json, perturb_markdown, perturb_whitespace

MODEL_NAME = "distilgpt2"
MAX_NEW_TOKENS = 12  # long enough to catch a numeric answer in the text

QUESTIONS = [
    {"id": 1, "question": "What is 2 + 2?", "answer": "4"},
    {
        "id": 2,
        "question": "If John has 5 apples and buys 3 more, how many apples does he have?",
        "answer": "8",
    },
    {"id": 3, "question": "What is 10 minus 4?", "answer": "6"},
    {
        "id": 4,
        "question": "If a train travels 60 miles in 2 hours, what is its speed in miles per hour?",
        "answer": "30",
    },
    {"id": 5, "question": "What is 3 times 5?", "answer": "15"},
]


def generate_and_score(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Runs one generation pass; returns (first_token_logits, generated_text)."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    first_token_logits = outputs.scores[0][0].detach().cpu().numpy()
    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return first_token_logits, generated_text


def is_correct(generated_text, expected_answer):
    return re.search(rf"\b{re.escape(expected_answer)}\b", generated_text) is not None


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load_model(MODEL_NAME)
    print("Model loaded.\n")

    rows = []
    for q in QUESTIONS:
        base_logits, base_text = generate_and_score(model, tokenizer, q["question"])
        h_base = shannon_entropy(base_logits)

        # Called directly (not apply_all_perturbations) so we can pass a
        # fixed seed per problem for reproducibility.
        variants = {
            "whitespace": perturb_whitespace(q["question"], seed=42 + q["id"]),
            "markdown": perturb_markdown(q["question"], seed=42 + q["id"]),
            "json": perturb_json(q["question"]),
        }

        for ptype, ptext in variants.items():
            p_logits, p_text = generate_and_score(model, tokenizer, ptext)
            h_pert = shannon_entropy(p_logits)
            d_h = delta_entropy(h_base, h_pert)
            rows.append(
                {
                    "problem_id": q["id"],
                    "perturbation_type": ptype,
                    "H_base": h_base,
                    "H_perturbed": h_pert,
                    "delta_H": d_h,
                    "correct": is_correct(p_text, q["answer"]),
                }
            )

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(df.to_string(index=False))

    csv_path = "results/quick_experiment_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="delta_H", by="perturbation_type", ax=ax)
    ax.set_title(f"delta_H by perturbation type ({MODEL_NAME}, n={len(QUESTIONS)} problems)")
    ax.set_ylabel("delta_H (bits)")
    plt.suptitle("")
    fig.tight_layout()

    out_path = "results/figures/entropy_boxplot.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved boxplot to {out_path}")

    return df


if __name__ == "__main__":
    main()
