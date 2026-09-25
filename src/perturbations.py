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

import json
import random


def perturb_whitespace(prompt: str, seed: int = None) -> str:
    """
    Inserts random double spaces and/or a trailing newline character
    into the prompt without altering token content.

    Args:
        prompt: The original GSM8k question text.
        seed: Optional random seed for reproducibility.

    Returns:
        The perturbed prompt string.
    """
    if seed is not None:
        random.seed(seed)

    words = prompt.split(" ")

    # Insert an extra space after ~40% of the word boundaries.
    perturbed_words = []
    for i, word in enumerate(words):
        perturbed_words.append(word)
        if i < len(words) - 1 and random.random() < 0.4:
            perturbed_words.append("")  # "".join with " " creates a double space

    perturbed = " ".join(perturbed_words)

    # Add a trailing newline about half the time.
    if random.random() < 0.5:
        perturbed += "\n"

    return perturbed


def perturb_markdown(prompt: str, style: str = None, seed: int = None) -> str:
    """
    Wraps the instruction in Markdown/XML-style formatting
    (e.g. headers, bold text, or <question> tags) without changing
    the lexical content.

    Args:
        prompt: The original GSM8k question text.
        style: One of "bold", "header", "xml". If None, one is chosen
            at random (optionally using `seed` for reproducibility).
        seed: Optional random seed for reproducibility.

    Returns:
        The perturbed prompt string.
    """
    styles = {
        "bold": lambda p: f"**Question:** {p}",
        "header": lambda p: f"## Question\n{p}",
        "xml": lambda p: f"<question>{p}</question>",
    }

    if style is None:
        if seed is not None:
            random.seed(seed)
        style = random.choice(list(styles.keys()))

    if style not in styles:
        raise ValueError(f"Unknown style '{style}'. Choose from {list(styles.keys())}")

    return styles[style](prompt)


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
    payload = {
        "task": "math_problem",
        "question": prompt,
    }
    return json.dumps(payload)


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


if __name__ == "__main__":
    test_prompt = "If John has 5 apples and buys 3 more, how many does he have?"

    print("BASE:")
    print(repr(test_prompt))
    print()

    print("WHITESPACE:")
    print(repr(perturb_whitespace(test_prompt, seed=42)))
    print()

    print("MARKDOWN (bold):")
    print(repr(perturb_markdown(test_prompt, style="bold")))
    print()

    print("MARKDOWN (header):")
    print(repr(perturb_markdown(test_prompt, style="header")))
    print()

    print("MARKDOWN (xml):")
    print(repr(perturb_markdown(test_prompt, style="xml")))
    print()

    print("JSON:")
    print(repr(perturb_json(test_prompt)))
    print()

    print("ALL TOGETHER:")
    print(apply_all_perturbations(test_prompt))