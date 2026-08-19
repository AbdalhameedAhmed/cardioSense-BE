from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any
from app.models.guideline import GuidelineChunk, Guideline
from app.services import rag as rag_service

async def search_guideline_chunks(
    db: AsyncSession,
    query_text: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Computes embedding for query_text and queries pgvector for the closest chunks
    using cosine similarity (ascending cosine distance). Each result carries its
    source guideline title, page range, and cosine distance so callers can both
    cite the source precisely and gauge retrieval confidence.
    """
    # 1. Get embedding for the query text
    embeddings = await rag_service.get_embeddings([query_text])
    if not embeddings:
        return []

    query_vector = embeddings[0]

    # 2. Search database using pgvector's cosine distance operator
    # pgvector provides .cosine_distance() on Vector columns in SQLAlchemy
    distance_col = GuidelineChunk.embedding.cosine_distance(query_vector).label("distance")
    query = (
        select(GuidelineChunk, Guideline.title, distance_col)
        .join(Guideline, GuidelineChunk.guideline_id == Guideline.id)
        .order_by(distance_col)
        .limit(limit)
    )

    result = await db.execute(query)
    rows = result.all()

    # 3. Format result
    results = []
    for chunk, guideline_title, distance in rows:
        metadata = chunk.metadata_json or {}
        results.append({
            "id": chunk.id,
            "guideline_id": chunk.guideline_id,
            "guideline_title": guideline_title,
            "content": chunk.content,
            "metadata": metadata,
            "page_start": metadata.get("page_start"),
            "page_end": metadata.get("page_end"),
            "distance": float(distance),
        })

    return results
