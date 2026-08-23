from openai import OpenAI

from app.config import (
    GROQ_BASE_URL,
    GROQ_MODEL_NAME,
    LLM_PROVIDER,
    XAI_BASE_URL,
    XAI_MODEL_NAME,
    get_groq_api_key,
    get_xai_api_key,
)


def get_llm_client(
    provider: str | None = None,
    model_name: str | None = None,
) -> tuple[OpenAI, str]:
    """Return the configured LLM client and model name."""
    selected_provider = provider or LLM_PROVIDER

    if selected_provider == "xai":
        client = OpenAI(
            api_key=get_xai_api_key(),
            base_url=XAI_BASE_URL,
        )
        return client, model_name or XAI_MODEL_NAME

    if selected_provider == "groq":
        client = OpenAI(
            api_key=get_groq_api_key(),
            base_url=GROQ_BASE_URL,
        )
        return client, model_name or GROQ_MODEL_NAME

    raise ValueError(
        f"Unsupported LLM provider: {selected_provider}. "
        "Currently supported providers: xai, groq"
    )