import ollama
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

MODEL_NAME = "qwen3-embedding"


splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [
        doc.text
        for doc in docs
        if getattr(doc, "text", None)  # no image-only txt
    ]
    chunks = []

    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks


def embed_texts(texts: list[str]):
    response = ollama.embed(model=MODEL_NAME, input=texts, dimensions=3072)
    return response["embeddings"]
