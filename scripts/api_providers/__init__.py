"""Chat-API providers behind one interface for scripts/eval_api.py.

Each provider module wraps a single vendor SDK and exposes a ``Provider``
class with:

    KEY_ENV            env var(s) the SDK reads its API key from, in priority order
    EFFORTS            accepted --reasoning-effort values (None: pass anything through)
    await check(model)                      free lookup that the key can see `model`
    await complete(prompt, model=..., reasoning_effort=..., max_output_tokens=...)
        -> Completion

Vendor errors surface as ``ProviderError`` so the eval loop records them
without knowing which SDK raised them. Modules are imported only when their
provider is used, so an SDK that is not needed never has to import cleanly.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

PROVIDERS = ("openai", "anthropic", "gemini")


@dataclass
class Completion:
    text: str
    # "completed", "incomplete" (hit the output cap), or "refusal".
    status: str
    # The vendor's own finish signal, kept verbatim for debugging.
    stop_reason: str | None
    input_tokens: int | None
    # Billed output, reasoning included, so costs compare across providers.
    output_tokens: int | None
    # The reasoning share of output_tokens, where the vendor reports it.
    reasoning_tokens: int | None


class ProviderError(Exception):
    """A request that failed after the SDK's own retries."""


def detect_provider(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "gemini"
    return "openai"


def load_provider(name: str):
    module = importlib.import_module(f"{__name__}.{name}_provider")
    return module.Provider()
