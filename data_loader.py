from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import re

load_dotenv()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Read PDF, split sentences, embed, and store in vector DB
EMBED_DIM = 384

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def clean_text(text: str) -> str:
    """Clean research paper text by removing metadata and artifacts"""
    # Remove arXiv metadata lines
    text = re.sub(r'arXiv:\d+\.\d+.*\[.*\]\s*\d+.*', '', text)
    # Remove "Published at" conference lines
    text = re.sub(r'Published at.*\d{4}', '', text)
    # Remove page numbers (single/double digits at line boundaries)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    # Remove figure/table references and captions (Figure 1:, Table 1:, etc.)
    text = re.sub(r'(Figure|Table|Fig\.|Tbl\.)\s*\d+[:\-].*?(?=\n\n|\n(?=[A-Z])|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove image/placeholder markers
    text = re.sub(r'\[.*?(?:image|figure|picture|photo|diagram|chart|plot|graph).*?\]', '', text, flags=re.IGNORECASE)
    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        # Clean text first
        cleaned = clean_text(t)
        if cleaned:  # Only process non-empty text
            chunks.extend(splitter.split_text(cleaned))
    
    # Filter out very short chunks (noise)
    chunks = [c for c in chunks if len(c.strip()) > 50]
    
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