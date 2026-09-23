"""
model_runner.py

Handles loading open-source LLMs (Llama-3-8B, Gemma-2) via Hugging Face
Transformers and extracting raw output logits for the first k generated
tokens.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: str = None):
    """
    Loads a causal LLM and its tokenizer.

    Args:
        model_name: Hugging Face model identifier
            (e.g. "meta-llama/Meta-Llama-3-8B", "google/gemma-2-9b").
        device: Target device ("cuda" or "cpu"). Auto-detected if None.

    Returns:
        (model, tokenizer) tuple.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


def extract_logits(model, tokenizer, prompt: str, k: int = 5, device: str = "cpu"):
    """
    Runs inference on the given prompt and extracts the raw logits
    for the first k generated tokens.

    Args:
        model: Loaded Hugging Face causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Input prompt string.
        k: Number of initial tokens to extract logits for.
        device: Device the model is on.

    Returns:
        List of numpy arrays, one per generated token, each containing
        the raw logit vector over the vocabulary.

    Implementation note: use model.generate(..., output_scores=True,
    return_dict_in_generate=True) to capture per-step logits before
    sampling/argmax selection.
    """
    # TODO: implement generation loop with output_scores=True
    raise NotImplementedError
