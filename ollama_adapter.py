# inngest does not support ollama by default
# so we need a OllamaAdapter implementation

import os
from ollama import AsyncClient
from dotenv import load_dotenv

load_dotenv()


class OllamaAdapter:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3.5")
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY", "ollama")
        self.temperature = temperature

        self.client = AsyncClient()

    async def generate(
        self, question: str, context: list[str], sources: list[str]
    ) -> str:
        context_block = "\n\n".join(f"- {ctx}" for ctx in context).strip()
        if not context_block:
            context_block = "No relevant context found."
        source_block = (
            "\n".join(f"- {src}" for src in sources).strip()
            if sources
            else "No sources found."
        )
        response = await self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise RAG assistant. "
                        "Use only the provided context. "
                        "If the answer is not supported by the context, say you don't know. "
                        "Do not invent facts or sources."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Use the following context to answer the question.\n\n"
                        f"Context:\n{context_block}\n\n"
                        f"Known sources: {source_block}\n\n"
                        f"Question: {question}\n\n"
                        "Answer concisely. "
                        "If possible, cite the relevant source names exactly as provided."
                    ),
                },
            ],
        )
        answer = response.message.content
        return (answer or "").strip()
