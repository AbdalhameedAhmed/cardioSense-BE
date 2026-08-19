from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from uuid import UUID
from typing import List, Optional
from app.models.patient import Patient, PatientCase
from app.schemas.patient import CaseCreateRequest, CaseUpdate

async def create_patient_case(db: AsyncSession, request: CaseCreateRequest) -> PatientCase:
    """
    Creates a new patient case. If patient_id is provided, links to an existing patient.
    Otherwise, creates a new patient using provided age and sex demographics.
    """
    # 1. Resolve or create patient
    if request.patient_id:
        result = await db.execute(select(Patient).where(Patient.id == request.patient_id))
        patient = result.scalar_one_or_none()
        if not patient:
            raise ValueError(f"Patient with ID {request.patient_id} not found.")
    else:
        # Create a new patient
        if request.age is None or request.sex is None:
            raise ValueError("Demographics (age and sex) are required to create a new patient.")
        
        patient = Patient(
            age=request.age,
            sex=request.sex
        )
        db.add(patient)
        await db.flush()  # Populates patient.id

    # 2. Create the case linked to the patient
    db_case = PatientCase(
        patient_id=patient.id,
        status=request.status,
        systolic_bp=request.systolic_bp,
        diastolic_bp=request.diastolic_bp,
        smoking=request.smoking,
        diabetes=request.diabetes,
        kidney_disease=request.kidney_disease,
        previous_cvd=request.previous_cvd,
        total_cholesterol=request.total_cholesterol,
        hdl=request.hdl,
        symptoms=request.symptoms,
        medications=request.medications,
        additional_data=request.additional_data
    )
    db.add(db_case)
    await db.commit()
    
    # 3. Refresh and load relationships
    # We do a fresh select to ensure eager loading of the patient relationship
    refreshed_result = await db.execute(
        select(PatientCase)
        .where(PatientCase.id == db_case.id)
        .options(selectinload(PatientCase.patient))
    )
    return refreshed_result.scalar_one()

async def get_case_by_id(db: AsyncSession, case_id: UUID) -> Optional[PatientCase]:
    """Retrieves a single patient case by UUID including patient demographics."""
    result = await db.execute(
        select(PatientCase)
        .where(PatientCase.id == case_id)
        .options(selectinload(PatientCase.patient))
    )
    return result.scalar_one_or_none()

async def list_cases(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[PatientCase]:
    """Retrieves list of all patient cases sorted by creation date (newest first)."""
    result = await db.execute(
        select(PatientCase)
        .options(selectinload(PatientCase.patient))
        .order_by(PatientCase.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())

async def update_case(db: AsyncSession, case_id: UUID, update_data: CaseUpdate) -> Optional[PatientCase]:
    """Updates clinical fields of a patient case."""
    result = await db.execute(
        select(PatientCase)
        .where(PatientCase.id == case_id)
        .options(selectinload(PatientCase.patient))
    )
    db_case = result.scalar_one_or_none()
    if not db_case:
        return None

    # Update fields that were provided
    update_dict = update_data.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(db_case, key, value)

    await db.commit()
    await db.refresh(db_case)
    return db_case
