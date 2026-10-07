from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config


def get_embedder() -> GoogleGenerativeAIEmbeddings:
    """
    Build an embedding client that works across different Google API setups.
    We try user-provided EMBEDDING_MODEL first, then a few common fallbacks.
    """
    candidates = [
        config.EMBEDDING_MODEL,
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        "models/text-embedding-004",
        "text-embedding-004",
        "models/embedding-001",
        "embedding-001",
    ]

    # remove duplicates while preserving order
    seen = set()
    ordered_candidates = []
    for model in candidates:
        if model and model not in seen:
            ordered_candidates.append(model)
            seen.add(model)

    last_error = None
    for model_name in ordered_candidates:
        try:
            embedder = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=config.GOOGLE_API_KEY,
            )
            # quick probe call - confirms model works with this API key/account
            embedder.embed_query("health check")
            print(f"Using embedding model: {model_name}")
            return embedder
        except Exception as e:  # keep simple for beginner project
            last_error = e
            continue

    raise RuntimeError(
        "Could not find a working Google embedding model for this API key.\n"
        "Set EMBEDDING_MODEL in .env to a model your account supports.\n"
        f"Last error: {last_error}"
    )

