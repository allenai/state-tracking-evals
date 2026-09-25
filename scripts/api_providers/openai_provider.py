"""OpenAI Responses API provider."""

from __future__ import annotations

import openai

from . import Completion, ProviderError


class Provider:
    KEY_ENV = ("OPENAI_API_KEY",)
    # Reasoning efforts differ across OpenAI models, so the API validates them.
    EFFORTS = None

    def __init__(self) -> None:
        self.client = openai.AsyncOpenAI(max_retries=8, timeout=900)

    async def check(self, model: str) -> None:
        try:
            await self.client.models.retrieve(model)
        except openai.APIError as error:
            raise ProviderError(f"{type(error).__name__}: {error}") from error

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        reasoning_effort: str | None,
        max_output_tokens: int | None,
    ) -> Completion:
        request = {"model": model, "input": prompt}
        if reasoning_effort:
            request["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens:
            request["max_output_tokens"] = max_output_tokens
        try:
            response = await self.client.responses.create(**request)
        except openai.APIError as error:
            raise ProviderError(f"{type(error).__name__}: {error}") from error

        reason = getattr(response.incomplete_details, "reason", None)
        if response.status == "incomplete" and reason == "content_filter":
            status = "refusal"
        elif response.status == "incomplete":
            status = "incomplete"
        else:
            status = "completed"
        usage = response.usage
        details = getattr(usage, "output_tokens_details", None) if usage else None
        return Completion(
            text=response.output_text or "",
            status=status,
            stop_reason=reason or response.status,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
        )
