import asyncio
import sys
from pathlib import Path
from sqlalchemy import text

# Add backend directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import AsyncSessionLocal

async def main():
    async with AsyncSessionLocal() as session:
        try:
            # 1. Query Guidelines Count
            gl_res = await session.execute(text("SELECT count(*), string_agg(title, ', ') FROM guidelines;"))
            gl_count, gl_titles = gl_res.fetchone()
            
            # 2. Query Guideline Chunks Count
            chunk_res = await session.execute(text("SELECT count(*), count(embedding) FROM guideline_chunks;"))
            chunk_count, embedding_count = chunk_res.fetchone()
            
            print("==================================================")
            print("         DATABASE GUIDELINE & VECTOR STATUS       ")
            print("==================================================")
            print(f"Total Guidelines:      {gl_count}")
            print(f"Guideline Titles:      {gl_titles or 'None'}")
            print(f"Total Chunks:          {chunk_count}")
            print(f"Populated Embeddings:  {embedding_count} / {chunk_count}")
            print("==================================================")
            
            if chunk_count > 0:
                print("\nSample Chunk Preview:")
                sample_res = await session.execute(text(
                    "SELECT id, substring(content, 1, 120) AS preview FROM guideline_chunks LIMIT 3;"
                ))
                for row in sample_res.fetchall():
                    print(f"- [ID: {row[0]}] {row[1]}...")
            
        except Exception as e:
            print(f"Error querying database: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
