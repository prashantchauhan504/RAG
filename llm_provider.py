from langchain_google_genai import ChatGoogleGenerativeAI

import config


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Build a chat model client that works across different Google API setups.
    We try user-provided GOOGLE_LLM_MODEL first, then common fallbacks.
    """
    candidates = [
        config.GOOGLE_LLM_MODEL,
        "models/gemini-2.5-flash",
        "gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "gemini-2.0-flash",
        "models/gemini-flash-latest",
        "gemini-flash-latest",
        "models/gemini-1.5-flash",
        "gemini-1.5-flash",
    ]

    seen = set()
    ordered_candidates = []
    for model in candidates:
        if model and model not in seen:
            ordered_candidates.append(model)
            seen.add(model)

    last_error = None
    for model_name in ordered_candidates:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.2,
            )
            llm.invoke("health check")
            print(f"Using chat model: {model_name}")
            return llm
        except Exception as e:  # keep simple for beginner project
            last_error = e
            continue

    raise RuntimeError(
        "Could not find a working Google chat model for this API key.\n"
        "Set GOOGLE_LLM_MODEL in .env to a model your account supports.\n"
        f"Last error: {last_error}"
    )

