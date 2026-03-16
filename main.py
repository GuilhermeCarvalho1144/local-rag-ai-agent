import os
import uuid
import datetime
import logging
import inngest
import inngest.fast_api
from fastapi import FastAPI
from dotenv import load_dotenv
from vector_db import QdrantStorage
from ollama_adapter import OllamaAdapter
from data_loader import load_and_chunk_pdf, embed_texts
from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult,
    RAGSearchResult,
    RAGQueryResult,
)

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="local_rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

llm = OllamaAdapter(
    model="qwen3.5",
    base_url=os.getenv("OLLAMA_BASE_URL"),
    api_key=os.getenv("OLLAMA_API_KEY"),
    temperature=0.2,
)


@inngest_client.create_function(
    fn_id="local_rag_app/ingest_pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def ingest_pdf(ctx: inngest.Context):

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, sources_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        sources_id = chunks_and_src.sources_id

        vectors = embed_texts(chunks)
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sources_id}_{i}"))
            for i in range(len(chunks))
        ]
        payloads = [
            {"source": sources_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]
        QdrantStorage().upsert(vectors=vectors, ids=ids, payloads=payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load-and-chunk-pdf", lambda: _load(ctx), output_type=RAGChunkAndSrc
    )
    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="local_rag_app/search_pdf",
    trigger=inngest.TriggerEvent(event="rag/search_pdf"),
)
async def search_pdf(ctx: inngest.Context):
    def _search(question: str, top_k: int) -> RAGSearchResult:
        query_vector = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vector, top_k)
        return RAGSearchResult(
            context=found["contexts"], sources=found["sources"]
        )

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    async def _generate() -> RAGSearchResult:
        answer = await llm.generate(question, found.context, found.sources)
        return RAGQueryResult(
            answer=answer,
            sources=found.sources,
            num_contexts=len(found.context),
        )

    result = await ctx.step.run(
        "generate-answer", lambda: _generate(), output_type=RAGQueryResult
    )
    return result.model_dump()


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [ingest_pdf, search_pdf])
