import os


def get_provider() -> str:
    """
    Returns the configured LLM provider.

    Supported:
    - groq
    - gemini
    """

    return os.environ.get(
        "LLM_PROVIDER",
        "groq",
    ).lower()
