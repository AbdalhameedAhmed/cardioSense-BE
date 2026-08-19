from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from typing import List
from app.core.database import get_db
from app.schemas.patient import CaseCreateRequest, CaseResponse, CaseUpdate
from app.services import patient as patient_service

router = APIRouter()

@router.post("/", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(request: CaseCreateRequest, db: AsyncSession = Depends(get_db)):
    """Create a new patient case (and optionally a new patient)."""
    try:
        new_case = await patient_service.create_patient_case(db, request)
        return new_case
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@router.get("/", response_model=List[CaseResponse])
async def list_cases(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    """List patient cases (newest first)."""
    try:
        cases = await patient_service.list_cases(db, skip=skip, limit=limit)
        return cases
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(case_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get detailed information for a specific patient case."""
    db_case = await patient_service.get_case_by_id(db, case_id)
    if not db_case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case with ID {case_id} not found."
        )
    return db_case

@router.put("/{case_id}", response_model=CaseResponse)
async def update_case(case_id: UUID, update_data: CaseUpdate, db: AsyncSession = Depends(get_db)):
    """Update clinical parameters of an existing case."""
    updated = await patient_service.update_case(db, case_id, update_data)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case with ID {case_id} not found."
        )
    return updated
