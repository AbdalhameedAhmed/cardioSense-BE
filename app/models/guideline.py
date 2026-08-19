import uuid
from sqlalchemy import Column, String, Date, DateTime, ForeignKey, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.models.base import Base

class Guideline(Base):
    __tablename__ = "guidelines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    version = Column(String, nullable=True)
    author = Column(String, nullable=True)
    publication_date = Column(Date, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    chunks = relationship("GuidelineChunk", back_populates="guideline", cascade="all, delete-orphan")

class GuidelineChunk(Base):
    __tablename__ = "guideline_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guideline_id = Column(UUID(as_uuid=True), ForeignKey("guidelines.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    # Avoid shadowing SQLAlchemy's Base.metadata by naming the attribute metadata_json but mapping to column name "metadata"
    metadata_json = Column("metadata", JSON, default=dict, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    guideline = relationship("Guideline", back_populates="chunks")
