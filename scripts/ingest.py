import asyncio
import argparse
import os
import sys
from datetime import date
from pathlib import Path

# Add backend directory to path so imports work correctly
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import AsyncSessionLocal
from app.services import rag as rag_service

async def main():
    parser = argparse.ArgumentParser(description="Ingest medical guidelines PDF or TXT into the RAG vector store.")
    parser.add_argument("--file", required=True, help="Path to the guideline file (PDF or TXT)")
    parser.add_argument("--title", required=True, help="Title of the guideline document")
    parser.add_argument("--version", default=None, help="Version identifier of the guideline")
    parser.add_argument("--author", default=None, help="Author/Organization publishing the guideline")
    parser.add_argument("--pub-date", default=None, help="Publication date in YYYY-MM-DD format")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Reading file: {file_path.name}...")
    file_bytes = file_path.read_bytes()
    
    # 1. Parse text from file
    pages_data = []
    if file_path.suffix.lower() == ".pdf":
        print("Extracting pages from PDF...")
        pages_data = rag_service.extract_pdf_text(file_bytes)
    elif file_path.suffix.lower() == ".txt":
        print("Reading raw text file...")
        text_content = file_bytes.decode("utf-8", errors="ignore")
        pages_data = [{"text": text_content, "page": 1}]
    else:
        print("Error: Unsupported file format. Only PDF and TXT files are supported.", file=sys.stderr)
        sys.exit(1)
        
    if not pages_data or not any(p["text"] for p in pages_data):
        print("Error: No readable text found in file.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Chunk text
    print("Segmenting text into chunks...")
    chunks = rag_service.chunk_text(pages_data)
    print(f"Created {len(chunks)} chunks.")
    
    # 3. Generate embeddings
    print("Generating vector embeddings via OpenAI API...")
    chunk_texts = [c["content"] for c in chunks]
    embeddings = await rag_service.get_embeddings(chunk_texts)
    
    # 4. Parse publication date
    parsed_date = None
    if args.pub_date:
        try:
            parsed_date = date.fromisoformat(args.pub_date)
        except ValueError:
            print("Error: Invalid date format. Please use YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)
            
    # 5. Persist in database
    print("Storing in database...")
    async with AsyncSessionLocal() as session:
        try:
            guideline = await rag_service.store_guideline(
                db=session,
                title=args.title,
                version=args.version,
                author=args.author,
                publication_date=parsed_date,
                chunks=chunks,
                embeddings=embeddings
            )
            print("\nIngestion Completed Successfully!")
            print(f"Guideline ID: {guideline.id}")
            print(f"Title:        {guideline.title}")
            print(f"Total Chunks: {len(chunks)}")
        except Exception as e:
            print(f"Error persisting guideline in database: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
