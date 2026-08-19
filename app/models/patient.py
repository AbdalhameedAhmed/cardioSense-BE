import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, Numeric, ForeignKey, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    age = Column(Integer, nullable=True)
    sex = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship to cases
    cases = relationship("PatientCase", back_populates="patient", cascade="all, delete-orphan")

class PatientCase(Base):
    __tablename__ = "patient_cases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(50), default="active")
    
    systolic_bp = Column(Numeric(5, 2), nullable=True)
    diastolic_bp = Column(Numeric(5, 2), nullable=True)
    smoking = Column(Boolean, nullable=True)
    diabetes = Column(Boolean, nullable=True)
    kidney_disease = Column(Boolean, nullable=True)
    previous_cvd = Column(Boolean, nullable=True)
    
    total_cholesterol = Column(Numeric(5, 2), nullable=True)
    hdl = Column(Numeric(5, 2), nullable=True)
    
    symptoms = Column(JSON, default=list, nullable=False)
    medications = Column(JSON, default=list, nullable=False)
    additional_data = Column(JSON, default=dict, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="cases")
