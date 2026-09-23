"""
perturbations.py

Implements the three syntactic, non-semantic perturbation strategies
applied to GSM8k prompts:
    1. Whitespace perturbation
    2. Markdown/XML wrapping perturbation
    3. JSON-like structural perturbation

Each function must preserve strict semantic equivalence with the
original prompt -- only surface-level formatting changes.
"""

import random


def perturb_whitespace(prompt: str, seed: int = None) -> str:
    """
    Inserts random double spaces and/or trailing newline characters
    into the prompt without altering token content.

    Args:
        prompt: The original GSM8k question text.
        seed: Optional random seed for reproducibility.

    Returns:
        The perturbed prompt string.
    """
    # TODO: implement whitespace injection logic
    raise NotImplementedError


def perturb_markdown(prompt: str) -> str:
    """
    Wraps the instruction in Markdown/XML-style formatting
    (e.g. headers, bold text, or <question> tags) without changing
    the lexical content.

    Args:
        prompt: The original GSM8k question text.

    Returns:
        The perturbed prompt string.
    """
    # TODO: implement markdown wrapping logic
    raise NotImplementedError


def perturb_json(prompt: str) -> str:
    """
    Reshapes the prompt into a JSON-like schema
    (e.g. {"task": "...", "question": "..."}) while preserving
    the original semantic content.

    Args:
        prompt: The original GSM8k question text.

    Returns:
        The perturbed prompt string (JSON-formatted).
    """
    # TODO: implement JSON restructuring logic
    raise NotImplementedError


def apply_all_perturbations(prompt: str) -> dict:
    """
    Applies all three perturbation strategies to a single prompt
    and returns a dictionary mapping perturbation type to result.

    Args:
        prompt: The original GSM8k question text.

    Returns:
        dict with keys: "base", "whitespace", "markdown", "json"
    """
    return {
        "base": prompt,
        "whitespace": perturb_whitespace(prompt),
        "markdown": perturb_markdown(prompt),
        "json": perturb_json(prompt),
    }
