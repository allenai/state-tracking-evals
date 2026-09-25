"""Google Gemini API provider (google-genai SDK).

The client reads GOOGLE_API_KEY, falling back to GEMINI_API_KEY.
"""

from __future__ import annotations

from google import genai
from google.genai import errors, types

from . import Completion, ProviderError

# Finish reasons meaning the model or a safety filter declined to answer.
_REFUSAL_REASONS = {
    types.FinishReason.SAFETY,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.SPII,
    types.FinishReason.RECITATION,
}


class Provider:
    KEY_ENV = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    EFFORTS = ("minimal", "low", "medium", "high")

    def __init__(self) -> None:
        self.client = genai.Client(
            http_options=types.HttpOptions(
                timeout=900_000,  # milliseconds
                retry_options=types.HttpRetryOptions(attempts=9),
            )
        )

    async def check(self, model: str) -> None:
        try:
            await self.client.aio.models.get(model=model)
        except errors.APIError as error:
            raise ProviderError(f"{type(error).__name__}: {error}") from error

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        reasoning_effort: str | None,
        max_output_tokens: int | None,
    ) -> Completion:
        config = types.GenerateContentConfig(max_output_tokens=max_output_tokens)
        if reasoning_effort:
            config.thinking_config = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(reasoning_effort.upper())
            )
        try:
            response = await self.client.aio.models.generate_content(
                model=model, contents=prompt, config=config
            )
        except errors.APIError as error:
            raise ProviderError(f"{type(error).__name__}: {error}") from error

        candidate = response.candidates[0] if response.candidates else None
        finish = candidate.finish_reason if candidate else None
        if candidate is None or finish in _REFUSAL_REASONS:
            # No candidate at all means the prompt itself was blocked.
            status = "refusal"
        elif finish == types.FinishReason.MAX_TOKENS:
            status = "incomplete"
        else:
            status = "completed"
        parts = (
            (candidate.content.parts or []) if candidate and candidate.content else []
        )
        # Thought summaries come back as parts too; only the answer is scored.
        text = "".join(part.text for part in parts if part.text and not part.thought)
        usage = response.usage_metadata
        answer_tokens = (usage.candidates_token_count or 0) if usage else None
        thought_tokens = (usage.thoughts_token_count or 0) if usage else None
        blocked = getattr(response.prompt_feedback, "block_reason", None)
        return Completion(
            text=text,
            status=status,
            stop_reason=(finish or blocked).value if (finish or blocked) else None,
            input_tokens=usage.prompt_token_count if usage else None,
            # Thinking is billed as output, so it is folded in here to match
            # the other providers' output_tokens.
            output_tokens=(answer_tokens + thought_tokens if usage else None),
            reasoning_tokens=thought_tokens,
        )
