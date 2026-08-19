import os
from pathlib import Path
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

logger = logging.getLogger("cardiocompass.database")
logging.basicConfig(level=logging.INFO)

# Setup async engine for runtime endpoints
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True
)

AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Setup sync engine for database initialization/health check queries
sync_engine = create_engine(
    settings.DATABASE_URL_SYNC,
    echo=False,
    pool_pre_ping=True
)

async def get_db():
    """Dependency to get async database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

def init_db():
    """Initializes the database by executing schema.sql if tables do not exist."""
    schema_path = Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        logger.warning(f"schema.sql not found at {schema_path}. Skipping automatic initialization.")
        return

    # Determine active embedding vector dimension
    dim = 768 if (settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-gemini-api-key") else 1536
    logger.info(f"Initializing database schema (Active Embedding Vector Dimension: {dim})...")
    
    try:
        with sync_engine.connect() as conn:
            # Check if one of the core tables already exists to prevent duplicate execution errors
            table_check = conn.execute(text(
                "SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'patients');"
            )).scalar()

            if not table_check:
                logger.info("Core tables not found. Executing schema.sql...")
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema_sql = f.read()
                
                # Replace embedding dimension in schema text before execution
                schema_sql = schema_sql.replace("embedding VECTOR(1536)", f"embedding VECTOR({dim})")
                
                # Execute the SQL schema block
                conn.execute(text(schema_sql))
                conn.commit()
                logger.info("Database schema initialized successfully.")
            else:
                logger.info("Database tables already exist. Ensuring schema is up to date...")
                # Run migrations to make guidelines table match SQLAlchemy models
                try:
                    conn.execute(text("ALTER TABLE guidelines ADD COLUMN IF NOT EXISTS author VARCHAR(255);"))
                    conn.execute(text("ALTER TABLE guidelines ALTER COLUMN organization DROP NOT NULL;"))
                    conn.execute(text("ALTER TABLE guidelines ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;"))
                    conn.commit()
                    logger.info("Ensured guidelines table structure is up to date.")
                except Exception as migration_err:
                    logger.warning(f"Could not automatically run guidelines migration: {migration_err}")
                    conn.rollback()

                # Run alter table statement in case the user switched between OpenAI and Gemini
                try:
                    conn.execute(text(f"ALTER TABLE guideline_chunks ALTER COLUMN embedding TYPE VECTOR({dim});"))
                    conn.commit()
                    logger.info(f"Ensured column 'guideline_chunks.embedding' is type VECTOR({dim}).")
                except Exception as alter_err:
                    logger.warning(f"Could not automatically alter embedding column size: {alter_err}. If you have existing data, you may need to clear the tables.")
                    conn.rollback()
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        # We don't fail hard here in case local DB setup is pending by user,
        # but we print a clear log
