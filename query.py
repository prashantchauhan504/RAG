"""
Phase B — Query pipeline

Run:
    python query.py "your question here"

What this script does:
  1. Embed the question (Google embeddings)
  2. Search Pinecone for top-k similar chunks
  3. Build a prompt with retrieved chunks
  4. Ask Google LLM for final answer
  5. Print answer + sources
"""

import sys

from pinecone import Pinecone

import config
from embedding_provider import get_embedder
from llm_provider import get_llm


def retrieve(question: str, top_k: int) -> list[dict]:
    """
    Embed the question and query Pinecone.
    Returns list of matches with metadata.
    """
    embedder = get_embedder()
    question_vector = embedder.embed_query(question)

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX)

    result = index.query(
        vector=question_vector,
        top_k=top_k,
        include_metadata=True,
    )
    return result.get("matches", [])


def build_context(matches: list[dict]) -> str:
    """
    Convert retrieved chunks into a plain text context block for the LLM.
    """
    context_parts = []
    for i, m in enumerate(matches, start=1):
        md = m.get("metadata", {}) or {}
        source = md.get("source", "unknown-source")
        page = md.get("page", "unknown-page")
        text = md.get("text", "")

        context_parts.append(
            f"[Chunk {i}] source={source} page={page}\n{text}\n"
        )
    return "\n".join(context_parts)


def answer_with_llm(question: str, context: str) -> str:
    """
    Ask Google LLM to answer using only retrieved context.
    """
    llm = get_llm()

    prompt = f"""
You are a helpful RAG assistant.
Answer only from the provided context.
If the answer is not in context, say: "I could not find that in the indexed documents."

Question:
{question}

Context:
{context}
"""
    response = llm.invoke(prompt)
    return response.content


def print_sources(matches: list[dict]) -> None:
    print("\nSources:")
    if not matches:
        print("- No matches found in Pinecone.")
        return

    for i, m in enumerate(matches, start=1):
        md = m.get("metadata", {}) or {}
        source = md.get("source", "unknown-source")
        page = md.get("page", "unknown-page")
        score = m.get("score", 0.0)
        print(f"- [{i}] source={source}, page={page}, score={score:.4f}")


def ask(question: str) -> dict:
    print(f"\nQuestion: {question}")
    print(f"Retrieving top {config.TOP_K} chunks from Pinecone...")
    matches = retrieve(question, top_k=config.TOP_K)

    if not matches:
        print(
            "\nNo chunks found in Pinecone.\n"
            "Run ingestion first with a text-rich PDF."
        )
        return {
            "question": question,
            "answer": "I could not find that in the indexed documents.",
            "sources": [],
            "status": "no_matches",
        }

    context = build_context(matches)
    answer = answer_with_llm(question, context)

    print("\nAnswer:\n")
    print(answer)
    print_sources(matches)
    sources = []
    for m in matches:
        md = m.get("metadata", {}) or {}
        sources.append(
            {
                # "source": md.get("source", "unknown-source"),
                "page": md.get("page", "unknown-page"),
                "score": float(m.get("score", 0.0)),
            }
        )
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "status": "ok",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)

    ask(sys.argv[1])

