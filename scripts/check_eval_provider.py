"""Check configured provider without logging credentials or provider error content."""

from adaptive_rag.core.config import get_settings
from adaptive_rag.generation.openai_chat import OpenAIChatModel


def main() -> int:
    settings = get_settings()
    if settings.llm_api_key is None or settings.llm_model is None:
        print("Missing configured provider")
        return 1
    try:
        model = OpenAIChatModel(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout_seconds=20,
            max_retries=0,
            temperature=0,
            max_tokens=32,
        )
        model.complete([{"role": "user", "content": "Reply with OK only."}])
    except Exception as error:
        print(f"Provider check failed: {type(error).__name__}")
        cause = error.__cause__
        if cause is not None:
            print(f"Cause: {type(cause).__name__}; status={getattr(cause, 'status_code', None)}")
        return 1
    print("Configured provider responded successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
