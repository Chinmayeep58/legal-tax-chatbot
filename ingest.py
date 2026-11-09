
import os
import re
import json
import fitz
import faiss
import numpy as np

from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = "data"
VECTOR_DIR = "vectorstore"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

os.makedirs(VECTOR_DIR, exist_ok=True)


# ==========================================
# 1. FIND ALL PDF DOCUMENTS
# ==========================================

def find_pdf_files():

    pdf_files = list(Path(DATA_DIR).rglob("*.pdf"))

    # Also include PDFs with uppercase extensions
    pdf_files += [
        p for p in Path(DATA_DIR).rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".pdf"
        and p not in pdf_files
    ]

    return pdf_files


# ==========================================
# 2. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdf(pdf_path):

    extracted_pages = []

    try:

        document = fitz.open(pdf_path)

        for page_number, page in enumerate(document):

            text = page.get_text("text")

            if text and text.strip():

                extracted_pages.append({

                    "text": text,

                    "page": page_number + 1,

                    "source": Path(pdf_path).name,

                    "path": str(pdf_path),

                    "category": Path(pdf_path).parent.name

                })

        document.close()

    except Exception as error:

        print(f"Error processing {pdf_path}: {error}")

    return extracted_pages


# ==========================================
# 3. CLEAN EXTRACTED TEXT
# ==========================================

def clean_text(text):

    # Remove null characters
    text = text.replace("\x00", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove excessive special characters
    text = re.sub(r"[^\w\s.,;:()/%\-&₹]", "", text)

    return text.strip()


# ==========================================
# 4. CREATE OVERLAPPING CHUNKS
# ==========================================

def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        # Prefer ending at a word boundary
        if end < len(text):

            space_position = text.rfind(
                " ", start, end
            )

            if space_position > start:
                end = space_position

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ==========================================
# 5. PROCESS ALL DOCUMENTS
# ==========================================

def process_documents():

    all_chunks = []

    pdf_files = find_pdf_files()

    print(f"\nFound {len(pdf_files)} PDF documents.\n")

    if not pdf_files:

        print("No PDF documents found in data folder.")

        return []

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}")

        pages = extract_text_from_pdf(pdf_path)

        if not pages:

            print("WARNING: No extractable text found.")
            print("This PDF may be scanned or image-based.")
            continue

        total_chunks = 0

        for page_data in pages:

            cleaned_text = clean_text(
                page_data["text"]
            )

            if not cleaned_text:
                continue

            page_chunks = create_chunks(cleaned_text)

            for chunk_number, chunk in enumerate(page_chunks):

                all_chunks.append({

                    "text": chunk,

                    "source": page_data["source"],

                    "page": page_data["page"],

                    "path": page_data["path"],

                    "category": page_data["category"],

                    "chunk_id": chunk_number

                })

                total_chunks += 1

        print(f"Pages extracted: {len(pages)}")
        print(f"Chunks generated: {total_chunks}\n")

    return all_chunks


# ==========================================
# 6. GENERATE EMBEDDINGS
# ==========================================

def generate_embeddings(chunks):

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Generating embeddings...\n")

    embeddings = model.encode(

        texts,

        batch_size=32,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True

    )

    return embeddings.astype("float32")


# ==========================================
# 7. SAVE FAISS DATABASE
# ==========================================

def save_vector_database(chunks, embeddings):

    dimension = embeddings.shape[1]

    # Inner Product = Cosine Similarity
    # because vectors are normalized

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # Save FAISS index
    index_path = os.path.join(
        VECTOR_DIR,
        "index.faiss"
    )

    faiss.write_index(index, index_path)

    # Save metadata
    metadata_path = os.path.join(
        VECTOR_DIR,
        "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nVector database saved successfully!")

    print("Total vectors:", index.ntotal)

    print("Embedding dimension:", dimension)

    print("Index location:", index_path)

    print("Metadata location:", metadata_path)


# ==========================================
# 8. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("LEGAL AND TAX RAG - DOCUMENT INGESTION")
    print("=" * 50)

    chunks = process_documents()

    if not chunks:

        print("\nNo chunks generated. Check your PDFs.")

        exit(1)

    print("\nTotal chunks generated:", len(chunks))

    embeddings = generate_embeddings(chunks)

    save_vector_database(chunks, embeddings)

    print("\n" + "=" * 50)

    print("DOCUMENT INGESTION COMPLETED!")

    print("=" * 50)
