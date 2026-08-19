from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

# ====================
# Patient Schemas
# ====================
class PatientBase(BaseModel):
    age: Optional[int] = Field(None, ge=1, le=120, description="Age of the patient (1-120)")
    sex: Optional[str] = Field(None, description="Sex of the patient ('male', 'female', 'other')")

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            normalized = v.strip().lower()
            if normalized not in ["male", "female", "other"]:
                raise ValueError("Sex must be 'male', 'female', or 'other'")
            return normalized
        return v

class PatientCreate(PatientBase):
    pass

class PatientResponse(PatientBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime


# ====================
# Case Schemas
# ====================
class CaseBase(BaseModel):
    status: str = Field("active", description="Status of the patient case (e.g., 'active', 'archived')")
    systolic_bp: Optional[float] = Field(None, ge=50, le=250, description="Systolic blood pressure in mmHg (50-250)")
    diastolic_bp: Optional[float] = Field(None, ge=30, le=150, description="Diastolic blood pressure in mmHg (30-150)")
    smoking: Optional[bool] = Field(None, description="Current smoking status")
    diabetes: Optional[bool] = Field(None, description="Diabetes status")
    kidney_disease: Optional[bool] = Field(None, description="Chronic kidney disease status")
    previous_cvd: Optional[bool] = Field(None, description="History of previous cardiovascular disease")
    total_cholesterol: Optional[float] = Field(None, ge=50, le=500, description="Total cholesterol level in mg/dL (50-500)")
    hdl: Optional[float] = Field(None, ge=10, le=150, description="HDL cholesterol level in mg/dL (10-150)")
    symptoms: List[str] = Field(default_factory=list, description="List of symptoms reported by the patient")
    medications: List[str] = Field(default_factory=list, description="List of current medications")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Flexible field for extra clinical information")

    @field_validator("diastolic_bp")
    @classmethod
    def validate_bp_relationship(cls, v: Optional[float], info: Any) -> Optional[float]:
        # Validate that diastolic is less than systolic if both are present
        systolic = info.data.get("systolic_bp")
        if v is not None and systolic is not None:
            if v >= systolic:
                raise ValueError("Diastolic BP must be less than Systolic BP")
        return v

class CaseCreateRequest(CaseBase):
    # This unified schema allows intake flow to pass either patient_id OR demographics to create one
    patient_id: Optional[UUID] = Field(None, description="UUID of an existing patient. If omitted, age and sex must be provided.")
    age: Optional[int] = Field(None, ge=1, le=120, description="Required if patient_id is not provided.")
    sex: Optional[str] = Field(None, description="Required if patient_id is not provided ('male', 'female', 'other')")

    @field_validator("sex")
    @classmethod
    def validate_intake_sex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            normalized = v.strip().lower()
            if normalized not in ["male", "female", "other"]:
                raise ValueError("Sex must be 'male', 'female', or 'other'")
            return normalized
        return v

class CaseUpdate(BaseModel):
    status: Optional[str] = None
    systolic_bp: Optional[float] = Field(None, ge=50, le=250)
    diastolic_bp: Optional[float] = Field(None, ge=30, le=150)
    smoking: Optional[bool] = None
    diabetes: Optional[bool] = None
    kidney_disease: Optional[bool] = None
    previous_cvd: Optional[bool] = None
    total_cholesterol: Optional[float] = Field(None, ge=50, le=500)
    hdl: Optional[float] = Field(None, ge=10, le=150)
    symptoms: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    additional_data: Optional[Dict[str, Any]] = None

class CaseResponse(CaseBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    patient_id: UUID
    created_at: datetime
    updated_at: datetime
    patient: Optional[PatientResponse] = None
