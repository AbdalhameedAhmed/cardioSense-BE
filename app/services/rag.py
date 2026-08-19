import asyncio
import io
import logging
import re
from datetime import date
from typing import List, Dict, Any, Optional
import pypdf
import tiktoken
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.models.guideline import Guideline, GuidelineChunk

logger = logging.getLogger(__name__)

def extract_pdf_text(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file.
    Returns: List of dicts with {"text": str, "page": int}
    """
    pages_data = []
    pdf_file = io.BytesIO(file_bytes)
    reader = pypdf.PdfReader(pdf_file)
    
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Postgres TEXT/VARCHAR columns reject NUL bytes, which some PDFs'
        # embedded fonts/glyph mappings cause pypdf to emit.
        text = text.replace("\x00", "")
        pages_data.append({
            "text": text.strip(),
            "page": idx + 1
        })
    
    return pages_data

def chunk_text(pages_data: List[Dict[str, Any]], chunk_size_tokens: int = 250, chunk_overlap_tokens: int = 50) -> List[Dict[str, Any]]:
    """
    Chunks extracted page text using a token-based sliding window.
    Tracks which page(s) each chunk belongs to.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []
    
    # Flatten all text with page tracking
    token_stream = []
    for page in pages_data:
        tokens = encoding.encode(page["text"])
        for token in tokens:
            token_stream.append((token, page["page"]))
            
    if not token_stream:
        return []
        
    step = chunk_size_tokens - chunk_overlap_tokens
    if step <= 0:
        step = chunk_size_tokens
        
    idx = 0
    while idx < len(token_stream):
        chunk_tokens_with_pages = token_stream[idx : idx + chunk_size_tokens]
        
        # Decode the tokens back to text
        chunk_tokens = [t[0] for t in chunk_tokens_with_pages]
        chunk_text_content = encoding.decode(chunk_tokens)
        
        # Determine page range for this chunk
        chunk_pages = sorted(list(set(t[1] for t in chunk_tokens_with_pages)))
        page_start = chunk_pages[0] if chunk_pages else 1
        page_end = chunk_pages[-1] if chunk_pages else 1
        
        chunks.append({
            "content": chunk_text_content.strip(),
            "metadata": {
                "page_start": page_start,
                "page_end": page_end
            }
        })
        
        idx += step
        # If we reached the end but have very few tokens remaining, break to prevent tiny chunks
        if idx >= len(token_stream):
            break
            
    return chunks

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings using Gemini or OpenAI API.
    If no API key is configured, returns a mock zero-vector.
    """
    if not texts:
        return []

    # 1. Check Gemini
    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-gemini-api-key":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings_service = GoogleGenerativeAIEmbeddings(
                google_api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_EMBEDDING_MODEL,
                output_dimensionality=768
            )
            # The free tier caps embedding requests per minute, so large documents
            # (many chunks) must be sent in small batches with backoff on 429s
            # rather than one call for the whole text list.
            batch_size = 90
            max_retries = 5
            results: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                for attempt in range(max_retries):
                    try:
                        results.extend(await embeddings_service.aembed_documents(batch))
                        break
                    except Exception as e:
                        if "RESOURCE_EXHAUSTED" not in str(e) or attempt == max_retries - 1:
                            raise
                        delay_match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", str(e))
                        delay = int(delay_match.group(1)) + 1 if delay_match else 60
                        logger.warning(f"Gemini embedding quota hit, retrying batch in {delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
            return results
        except Exception as e:
            # Do NOT fall back to zero-vectors here: a zero-vector embedding
            # silently poisons the vector store (it inserts successfully and
            # looks fine via the API, but can never be meaningfully retrieved
            # again). Callers (e.g. the ingest endpoint) already handle this
            # exception and surface a real error instead.
            logger.error(f"Error fetching embeddings from Gemini: {e}")
            raise

    # 2. Check OpenAI
    is_openai_mock = (
        not settings.OPENAI_API_KEY or 
        settings.OPENAI_API_KEY == "your-openai-api-key" or 
        settings.OPENAI_API_KEY.startswith("mock")
    )
    if not is_openai_mock:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            response = await client.embeddings.create(
                input=texts,
                model=settings.EMBEDDING_MODEL
            )
            return [r.embedding for r in response.data]
        except Exception as e:
            # Same reasoning as the Gemini branch above: fail loudly rather
            # than silently writing unusable zero-vector embeddings.
            logger.error(f"Error fetching embeddings from OpenAI: {e}")
            raise
        
    # 3. Mock Fallback
    dim = 768 if (settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-gemini-api-key") else 1536
    logger.warning(f"No AI API key configured. Using mock zero-vector embeddings ({dim} dim).")
    return [[0.0] * dim for _ in texts]

async def store_guideline(
    db: AsyncSession,
    title: str,
    version: Optional[str],
    author: Optional[str],
    publication_date: Optional[date],
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> Guideline:
    """
    Persists guideline metadata and guideline chunks in the database.
    """
    # 1. Create guideline metadata
    guideline = Guideline(
        title=title,
        version=version,
        author=author,
        publication_date=publication_date
    )
    db.add(guideline)
    await db.flush()  # Populates guideline.id
    
    # 2. Add guideline chunks
    for chunk, embedding in zip(chunks, embeddings):
        db_chunk = GuidelineChunk(
            guideline_id=guideline.id,
            content=chunk["content"],
            metadata_json=chunk["metadata"],
            embedding=embedding
        )
        db.add(db_chunk)
        
    await db.commit()
    await db.refresh(guideline)
    return guideline
