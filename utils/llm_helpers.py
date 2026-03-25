"""utils/llm_helpers.py — Shared LLM helpers for AI-Data-Science-Sprints."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


def get_client() -> OpenAI:
    """Return an authenticated OpenAI client (reads OPENAI_API_KEY env var)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY: export OPENAI_API_KEY='your-key'")
    return OpenAI(api_key=api_key)


def chat(
    prompt: str,
    system: str = "You are a helpful data science assistant.",
    model: str = "gpt-4-turbo",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Send a prompt and return the text response.

    Examples
    --------
    >>> response = chat("Explain the bias-variance tradeoff.")
    >>> print(response)
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def get_json(
    prompt: str,
    system: str = "You are a helpful assistant that always responds with valid JSON.",
    model: str = "gpt-4-turbo",
) -> dict[str, Any]:
    """Send a prompt and parse the response as JSON."""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed a list of texts using the OpenAI embeddings API."""
    client = get_client()
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def generate_python(task: str, model: str = "gpt-4") -> str:
    """Generate clean Python code from a natural-language task description."""
    system = (
        "Write clean, type-annotated, PEP-8-compliant Python code with NumPy-style "
        "docstrings. Return only the code block, no markdown fences."
    )
    return chat(task, system=system, model=model)


def generate_tests(code: str, model: str = "gpt-4") -> str:
    """Generate pytest unit tests for the given Python code."""
    prompt = f"Write comprehensive pytest tests with edge cases for:\n```python\n{code}\n```\nReturn only the test code."
    return chat(prompt, system="You are a Python testing expert.", model=model)


def describe_image(image_url: str, question: str = "Describe this image in detail.") -> str:
    """Describe or analyse an image using GPT-4 Vision."""
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
    )
    return response.choices[0].message.content


def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    model: str = "dall-e-3",
) -> str:
    """Generate an image with DALL-E and return its URL."""
    client = get_client()
    response = client.images.generate(model=model, prompt=prompt, size=size, quality=quality, n=1)
    return response.data[0].url
