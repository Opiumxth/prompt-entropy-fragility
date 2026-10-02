"""
model_runner.py

Handles loading Gemma-2 via Hugging Face Transformers and extracting
raw output logits for the first k generated tokens.

Model note: Gemma-2 checkpoints (e.g. "google/gemma-2-2b",
"google/gemma-2-9b") are gated on the Hugging Face Hub -- you still need
a HF account, to accept Google's usage license on the model page, and
to run `huggingface-cli login` (or set the HF_TOKEN env var) before
`from_pretrained` will succeed. It avoids Meta's separate approval-review
process for Llama-3, but it is not "ungated" in the sense of anonymous
access.

CPU note: this pipeline works on CPU, but two steps are the bottleneck
for a model in the 2B-9B range:
  1. `from_pretrained` in load_model() -- downloading + materializing
     billions of float32 params into RAM (9B params * 4 bytes ~= 36GB).
  2. `model.generate` in extract_logits() -- each forward pass is
     dominated by CPU matmuls with no batching/kernel fusion benefits
     from cuda, so k tokens can take from several seconds to minutes
     per prompt depending on model size and CPU.
For quick CPU-only smoke tests, prefer a small checkpoint such as
"google/gemma-2-2b"; reserve "google/gemma-2-9b" for GPU runs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: str = None):
    """
    Loads a causal LLM and its tokenizer.

    Args:
        model_name: Hugging Face model identifier.
        device: Target device ("cuda" or "cpu").
                  Auto-detected if None.

    Returns:
        (model, tokenizer) tuple.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Some causal LM tokenizers (e.g. GPT-2 style) ship without a pad
    # token, which makes generate() raise/warn. Gemma-2's tokenizer
    # already defines one, but we fall back to eos_token defensively
    # so this function stays generic across model families.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SLOW ON CPU: this line downloads (first run) and loads the full
    # parameter set into memory. For gemma-2-9b in float32 that is
    # ~36GB of RAM -- likely to be slow or to fail on a laptop.
    # low_cpu_mem_usage streams weights in instead of doubling peak
    # RAM usage during load.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()

    return model, tokenizer


def extract_logits(
    model,
    tokenizer,
    prompt: str,
    k: int = 5,
    device: str = None
):
    """
    Generates up to k tokens and extracts the raw logits
    for each generated token.

    Args:
        model: Loaded causal language model.
        tokenizer: Corresponding tokenizer.
        prompt: Input prompt.
        k: Number of generated tokens to analyze.
        device: Device where the model is located.

    Returns:
        List of numpy arrays containing the logits for each
        generation step.
    """

    if device is None:
        device = next(model.parameters()).device

    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    # SLOW ON CPU: each of the k forward passes below runs at CPU
    # matmul speed (no tensor-core acceleration). Expect this to
    # dominate wall-clock time for the whole pipeline on CPU-only
    # machines, especially at larger k or with gemma-2-9b.
    # do_sample=False (greedy) keeps generation deterministic, and
    # since no other logits processors are active, outputs.scores are
    # the model's raw pre-softmax logits at each step -- exactly what
    # shannon_entropy() in entropy.py expects.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=k,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # outputs.scores contains one tensor per generated token.
    # Each tensor has shape:
    # [batch_size, vocabulary_size]
    logits = [
        score[0].detach().cpu().numpy()
        for score in outputs.scores
    ]

    return logits
