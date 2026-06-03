from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Read PDF, split sentences, embed, and store in vector DB
EMBED_DIM = 384

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = embedder.encode(texts, convert_to_numpy=False, show_progress_bar=False)
    normalized = []
    for vec in embeddings:
        if hasattr(vec, "tolist"):
            normalized.append(vec.tolist())
        else:
            normalized.append(list(vec))
    return normalized