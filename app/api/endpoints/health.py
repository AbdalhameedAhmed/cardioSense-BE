from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

router = APIRouter()

@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint that verifies the FastAPI service,
    database connection, and availability of the pgvector extension.
    """
    db_status = "unhealthy"
    pgvector_status = "unavailable"
    db_error = None

    try:
        # Test database connection
        result = await db.execute(text("SELECT 1;"))
        if result.scalar() == 1:
            db_status = "healthy"
            
            # Test pgvector extension existence
            vector_result = await db.execute(text(
                "SELECT extname FROM pg_extension WHERE extname = 'vector';"
            ))
            if vector_result.scalar() == "vector":
                pgvector_status = "available"
            else:
                pgvector_status = "not_installed"
    except Exception as e:
        db_error = str(e)

    status_code = 200
    if db_status == "unhealthy" or pgvector_status == "not_installed":
        # We still return 200 for diagnosis but mark the health state
        pass

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": {
            "status": db_status,
            "pgvector": pgvector_status,
            "error": db_error
        }
    }
