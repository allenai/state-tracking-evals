"""Anthropic Messages API provider.

Requests stream (and collect the final message) because state-bench prompts
run to hundreds of thousands of tokens and --max-output-tokens can be large,
either of which can outlast a non-streaming HTTP request.

Server-side refusal fallbacks are deliberately not enabled: they would re-run
a declined request on a different model and score that model under this one's
name. A refusal is recorded as such instead.
"""

from __future__ import annotations

import anthropic

from . import Completion, ProviderError


class Provider:
    KEY_ENV = ("ANTHROPIC_API_KEY",)
    EFFORTS = ("low", "medium", "high", "xhigh", "max")

    def __init__(self) -> None:
        self.client = anthropic.AsyncAnthropic(max_retries=8, timeout=900)

    async def check(self, model: str) -> None:
        try:
            await self.client.models.retrieve(model)
        except anthropic.APIError as error:
            raise ProviderError(_describe(error)) from error
        except TypeError as error:
            # Raised before any request when no credential source resolves.
            raise ProviderError(f"no Anthropic credentials found: {error}") from error

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        reasoning_effort: str | None,
        max_output_tokens: int | None,
    ) -> Completion:
        request = {
            "model": model,
            # The Messages API requires an output cap.
            "max_tokens": max_output_tokens or 16_384,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Without an effort the model's own thinking default applies, as for
        # the other providers; with one, adaptive thinking runs at that depth.
        if reasoning_effort:
            request["thinking"] = {"type": "adaptive"}
            request["output_config"] = {"effort": reasoning_effort}
        try:
            async with self.client.messages.stream(**request) as stream:
                message = await stream.get_final_message()
        except anthropic.APIError as error:
            raise ProviderError(_describe(error)) from error

        if message.stop_reason == "max_tokens":
            status = "incomplete"
        elif message.stop_reason == "refusal":
            status = "refusal"
        else:
            status = "completed"
        usage = message.usage
        input_tokens = (
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0)
        )
        return Completion(
            text="".join(
                block.text for block in message.content if block.type == "text"
            ),
            status=status,
            stop_reason=message.stop_reason,
            input_tokens=input_tokens,
            # Thinking is billed as output and already counted here; the API
            # does not report it separately.
            output_tokens=usage.output_tokens,
            reasoning_tokens=None,
        )


def _describe(error: anthropic.APIError) -> str:
    # The SDK has already retried these; say so, since a rerun may succeed.
    if isinstance(
        error,
        anthropic.RateLimitError
        | anthropic.InternalServerError
        | anthropic.APIConnectionError,
    ):
        return f"{type(error).__name__} after retries: {error}"
    return f"{type(error).__name__}: {error}"
