"""
model_runner.py

Handles loading Gemma-2 9B via Hugging Face Transformers
and extracting raw output logits for the first k generated tokens.
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
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

    # Generate k tokens and keep the scores
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

