import fitz  # PyMuPDF
import os
import textwrap

from llama_index import Document, VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores import ChromaVectorStore
from chromadb.config import Settings
import chromadb
DATA_DIR = "Documents"
OUTPUT_DIR = "outputs"
VECTOR_STORE_DIR = "./vector_store"
MAX_CHUNK_SIZE = 500  # chars per chunk

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_pdf_text_with_headings(pdf_path):
    """Extract text with heading detection from PDF."""
    doc = fitz.open(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    pages_data = []

    for page_num, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        current_heading = None
        page_chunks = []

        for block in page_dict["blocks"]:
            for line in block.get("lines", []):
                text_line = " ".join(span["text"] for span in line["spans"]).strip()
                if not text_line:
                    continue

                # Detect heading: short line + larger font
                avg_font_size = sum(span["size"] for span in line["spans"]) / len(line["spans"])
                if len(text_line.split()) <= 7 and avg_font_size > 12:
                    current_heading = text_line  # update heading
                    continue

                # Append line with current heading
                page_chunks.append({
                    "file_name": pdf_name,
                    "page_number": page_num,
                    "heading": current_heading,
                    "text": text_line
                })

        # Combine lines into one page entry
        if page_chunks:
            pages_data.extend(page_chunks)

    return pages_data


def chunk_text(text, max_size=MAX_CHUNK_SIZE):
    """Chunk text with some overlap for context."""
    chunks = textwrap.wrap(text, max_size, break_long_words=False, replace_whitespace=False)
    return chunks


def create_documents(pages_data):
    """Create LlamaIndex Documents with metadata including headings."""
    documents = []
    current_page = pages_data[0]["page_number"] if pages_data else None
    buffer = ""

    for entry in pages_data:
        buffer += " " + entry["text"]

        if len(buffer) >= MAX_CHUNK_SIZE:
            # Create a chunk
            metadata = {
                "file_name": entry["file_name"],
                "page_number": entry["page_number"],
                "heading": entry["heading"]
            }
            documents.append(Document(text=buffer.strip(), extra_info=metadata))
            buffer = ""

    # Add any remaining buffer
    if buffer.strip():
        last_entry = pages_data[-1]
        metadata = {
            "file_name": last_entry["file_name"],
            "page_number": last_entry["page_number"],
            "heading": last_entry["heading"]
        }
        documents.append(Document(text=buffer.strip(), extra_info=metadata))

    return documents
def build_index(documents):
    """Build and persist vector index using simple vector store."""
    embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)

    # Use simple vector store (no external dependencies)
    index = VectorStoreIndex.from_documents(
        documents, 
        service_context=service_context
    )
    
    # Persist the index
    index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
    print(f"Index persisted to {VECTOR_STORE_DIR}")
    
    return index




if __name__ == "__main__":
    all_pages = []

    for file in os.listdir(DATA_DIR):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, file)
            print(f"Processing: {file}")
            pages_data = extract_pdf_text_with_headings(pdf_path)
            all_pages.extend(pages_data)

    print(f"Total text lines extracted: {len(all_pages)}")

    documents = create_documents(all_pages)
    print(f"Created {len(documents)} chunks/documents with headings.")

    index = build_index(documents)
    print("Index built and saved successfully.")
