from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import date
from typing import List, Optional
from app.core.database import get_db
from app.schemas.guideline import GuidelineResponse
from app.models.guideline import Guideline
from app.services import rag as rag_service

router = APIRouter()

@router.post("/ingest", response_model=GuidelineResponse, status_code=status.HTTP_201_CREATED)
async def ingest_guideline(
    title: str = Form(...),
    version: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    publication_date: Optional[date] = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Ingests a clinical guideline PDF or text file.
    Extracts text, chunks it, generates vector embeddings, and stores them.
    """
    # 1. Read file bytes
    file_bytes = await file.read()
    
    # 2. Extract pages depending on file type
    filename_lower = file.filename.lower()
    pages_data = []
    
    if filename_lower.endswith(".pdf"):
        try:
            pages_data = rag_service.extract_pdf_text(file_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse PDF file: {str(e)}"
            )
    elif filename_lower.endswith(".txt"):
        text_content = file_bytes.decode("utf-8", errors="ignore").replace("\x00", "")
        pages_data = [{"text": text_content, "page": 1}]
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only PDF and TXT files are allowed."
        )
        
    if not pages_data or not any(p["text"] for p in pages_data):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No readable text found in the uploaded file."
        )

    # 3. Chunk the extracted text
    try:
        chunks = rag_service.chunk_text(pages_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to segment/chunk text: {str(e)}"
        )

    # 4. Generate embeddings for the chunks
    try:
        chunk_texts = [c["content"] for c in chunks]
        embeddings = await rag_service.get_embeddings(chunk_texts)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}"
        )

    # 5. Store the guideline and its chunks in the database
    try:
        guideline = await rag_service.store_guideline(
            db=db,
            title=title,
            version=version,
            author=author,
            publication_date=publication_date,
            chunks=chunks,
            embeddings=embeddings
        )
        return guideline
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist guideline in vector store: {str(e)}"
        )

@router.get("/", response_model=List[GuidelineResponse])
async def list_guidelines(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    """Lists metadata for all ingested clinical guidelines."""
    try:
        result = await db.execute(
            select(Guideline)
            .order_by(Guideline.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
