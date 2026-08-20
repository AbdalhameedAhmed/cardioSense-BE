import re
import logging
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import List

from app.core.database import get_db, AsyncSessionLocal
from app.models.agent import AgentSession, AgentMessage
from app.models.patient import PatientCase
from app.schemas.agent import SessionResponse, SessionCreateRequest, MessageSendRequest
from app.services.graph import graph, extract_text
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)
router = APIRouter()

def extract_parameters_from_text(text: str) -> dict:
    """
    Utility function to extract clinical parameters from chat conversations.
    Supports basic conversational clinical updates.
    """
    updates = {}
    text_lower = text.lower()
    
    # Blood pressure: SBP/DBP e.g., 140/90 or 120 / 80
    bp_match = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', text_lower)
    if bp_match:
        updates["systolic_bp"] = int(bp_match.group(1))
        updates["diastolic_bp"] = int(bp_match.group(2))
        
    # Age: e.g. "age is 52" or "52 years old"
    age_match = re.search(r'\b(?:age|years?|yo)\b\s*(?:is|of)?\s*(\d{1,3})', text_lower)
    if age_match:
        updates["age"] = int(age_match.group(1))
    elif not bp_match:
        # Fallback to single number if it's a short text
        num_match = re.search(r'\b(\d{1,3})\b', text_lower)
        if num_match and len(text_lower) < 20:
            updates["age"] = int(num_match.group(1))
            
    # Smoking status
    if "smoke" in text_lower or "smok" in text_lower:
        if any(neg in text_lower for neg in ["no", "never", "not", "false", "non", "stop"]):
            updates["smoking"] = False
        else:
            updates["smoking"] = True
            
    # Diabetes status
    if "diabet" in text_lower:
        if any(neg in text_lower for neg in ["no", "not", "false", "non"]):
            updates["diabetes"] = False
        else:
            updates["diabetes"] = True
            
    # Kidney Disease status
    if "kidney" in text_lower or "ckd" in text_lower:
        if any(neg in text_lower for neg in ["no", "not", "false", "non"]):
            updates["kidney_disease"] = False
        else:
            updates["kidney_disease"] = True
            
    # Previous CVD history
    if "cvd" in text_lower or "cardio" in text_lower or "stroke" in text_lower or "heart attack" in text_lower:
        if any(neg in text_lower for neg in ["no", "not", "false", "never"]):
            updates["previous_cvd"] = False
        else:
            updates["previous_cvd"] = True
            
    # Lipids: cholesterol e.g. "cholesterol is 200" or "cholesterol: 180"
    chol_match = re.search(r'\b(?:cholesterol|total chol|tc)\b\s*(?:is|:)?\s*(\d{2,3})', text_lower)
    if chol_match:
        updates["total_cholesterol"] = float(chol_match.group(1))
        
    # HDL: e.g. "hdl is 50" or "hdl: 45"
    hdl_match = re.search(r'\b(?:hdl)\b\s*(?:is|:)?\s*(\d{2,3})', text_lower)
    if hdl_match:
        updates["hdl"] = float(hdl_match.group(1))
        
    return updates


@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(payload: SessionCreateRequest):
    """
    Starts a clinical assistant session for a patient case.

    Deliberately does NOT hold one DB connection open for the whole request:
    graph.ainvoke() below can take anywhere from a few seconds to over a
    minute (Gemini embedding/LLM calls, including rate-limit backoff), and
    Supabase's connection pooler kills connections left idle-in-transaction
    across a wait that long. So DB access is split into two short-lived
    sessions bracketing the slow graph call, with no connection held open
    during it.
    """
    try:
        # Phase A: quick read + row creation, connection closed immediately after.
        async with AsyncSessionLocal() as db:
            try:
                case_result = await db.execute(
                    select(PatientCase)
                    .where(PatientCase.id == payload.case_id)
                    .options(selectinload(PatientCase.patient))
                )
                case = case_result.scalar_one_or_none()
                if not case:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Patient case with ID {payload.case_id} not found."
                    )

                session = AgentSession(case_id=payload.case_id, state={})
                db.add(session)
                await db.commit()
                await db.refresh(session)
                session_id = session.id

                case_dict = {
                    "age": case.patient.age,
                    "sex": case.patient.sex,
                    "systolic_bp": case.systolic_bp,
                    "diastolic_bp": case.diastolic_bp,
                    "smoking": case.smoking,
                    "diabetes": case.diabetes,
                    "kidney_disease": case.kidney_disease,
                    "previous_cvd": case.previous_cvd,
                    "total_cholesterol": case.total_cholesterol,
                    "hdl": case.hdl,
                    "symptoms": case.symptoms,
                    "medications": case.medications
                }
            except HTTPException:
                raise
            except Exception:
                await db.rollback()
                raise

        initial_state = {
            "messages": [],
            "case_id": str(payload.case_id),
            "case_data": case_dict,
            "missing_fields": [],
            "rag_context": [],
            "citations": [],
            "evidence_sufficient": False,
            "retrieval_confidence": 0,
            "risk_category": None,
            "recommendations": [],
            "evaluation_complete": False
        }

        # Phase B: run the graph (embeddings + LLM calls) with no DB connection held open.
        try:
            output_state = await graph.ainvoke(
                initial_state,
                config={"configurable": {"db_factory": AsyncSessionLocal}}
            )
        except Exception as e:
            logger.error(f"Error invoking graph on session startup: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize assistant graph: {str(e)}"
            )

        ai_message_text = "Hello, how can I help you today?"
        if output_state.get("messages"):
            # extract_text() is a defensive last line here: graph nodes should
            # already normalize their own output, but AgentMessage.content is a
            # VARCHAR column, so any list-shaped content that slips through
            # would otherwise fail at the DB layer instead of at the source.
            ai_message_text = extract_text(output_state["messages"][-1].content)

        # Phase C: fresh short-lived connection for the final write.
        async with AsyncSessionLocal() as db:
            try:
                ai_msg = AgentMessage(session_id=session_id, role="ai", content=ai_message_text)
                db.add(ai_msg)

                session_res = await db.execute(select(AgentSession).where(AgentSession.id == session_id))
                session = session_res.scalar_one()
                session.state = {
                    "missing_fields": output_state.get("missing_fields", []),
                    "rag_context": output_state.get("rag_context", []),
                    "citations": output_state.get("citations", []),
                    "evidence_sufficient": output_state.get("evidence_sufficient", False),
                    "retrieval_confidence": output_state.get("retrieval_confidence", 0),
                    "risk_category": output_state.get("risk_category"),
                    "recommendations": output_state.get("recommendations", []),
                    "evaluation_complete": output_state.get("evaluation_complete", False)
                }
                await db.commit()

                stmt = (
                    select(AgentSession)
                    .where(AgentSession.id == session_id)
                    .options(selectinload(AgentSession.messages))
                )
                res = await db.execute(stmt)
                return res.scalar_one()
            except HTTPException:
                raise
            except Exception:
                await db.rollback()
                raise
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{session_id}/message", response_model=SessionResponse)
async def send_message(session_id: UUID, payload: MessageSendRequest):
    """
    Sends a message to the session, updates patient case with parsed info,
    and runs the agent.

    Same split-session structure as create_session above, and for the same
    reason: no DB connection is held open across graph.ainvoke()'s slow
    embedding/LLM calls, since Supabase's pooler kills idle-in-transaction
    connections that wait that long.
    """
    try:
        # Phase A: quick reads/writes needed before invoking the graph.
        async with AsyncSessionLocal() as db:
            try:
                stmt = (
                    select(AgentSession)
                    .where(AgentSession.id == session_id)
                    .options(selectinload(AgentSession.messages))
                )
                result = await db.execute(stmt)
                session = result.scalar_one_or_none()
                if not session:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Agent session with ID {session_id} not found."
                    )

                user_msg = AgentMessage(session_id=session_id, role="human", content=payload.content)
                db.add(user_msg)
                await db.flush()

                case_stmt = (
                    select(PatientCase)
                    .where(PatientCase.id == session.case_id)
                    .options(selectinload(PatientCase.patient))
                )
                case_res = await db.execute(case_stmt)
                case = case_res.scalar_one_or_none()

                if case:
                    extracted = extract_parameters_from_text(payload.content)
                    if extracted:
                        logger.info(f"Extracted clinical parameters from conversation: {extracted}")
                        for key, val in extracted.items():
                            setattr(case, key, val)
                        db.add(case)
                        await db.flush()

                langchain_messages = []
                all_messages_stmt = select(AgentMessage).where(AgentMessage.session_id == session_id).order_by(AgentMessage.created_at.asc())
                all_messages_res = await db.execute(all_messages_stmt)
                db_messages = all_messages_res.scalars().all()

                for msg in db_messages:
                    if msg.role == "human":
                        langchain_messages.append(HumanMessage(content=msg.content))
                    else:
                        langchain_messages.append(AIMessage(content=msg.content))

                case_dict = {}
                if case:
                    case_dict = {
                        "age": case.patient.age,
                        "sex": case.patient.sex,
                        "systolic_bp": case.systolic_bp,
                        "diastolic_bp": case.diastolic_bp,
                        "smoking": case.smoking,
                        "diabetes": case.diabetes,
                        "kidney_disease": case.kidney_disease,
                        "previous_cvd": case.previous_cvd,
                        "total_cholesterol": case.total_cholesterol,
                        "hdl": case.hdl,
                        "symptoms": case.symptoms,
                        "medications": case.medications
                    }

                case_id = session.case_id
                session_prior_state = dict(session.state or {})

                await db.commit()
            except HTTPException:
                raise
            except Exception:
                await db.rollback()
                raise

        input_state = {
            "messages": langchain_messages,
            "case_id": str(case_id),
            "case_data": case_dict,
            "missing_fields": session_prior_state.get("missing_fields", []),
            "rag_context": session_prior_state.get("rag_context", []),
            "citations": session_prior_state.get("citations", []),
            "evidence_sufficient": session_prior_state.get("evidence_sufficient", False),
            "retrieval_confidence": session_prior_state.get("retrieval_confidence", 0),
            "risk_category": session_prior_state.get("risk_category"),
            "recommendations": session_prior_state.get("recommendations", []),
            "evaluation_complete": session_prior_state.get("evaluation_complete", False)
        }

        # Phase B: run the graph (embeddings + LLM calls) with no DB connection held open.
        try:
            output_state = await graph.ainvoke(
                input_state,
                config={"configurable": {"db_factory": AsyncSessionLocal}}
            )
        except Exception as e:
            logger.error(f"Error running assistant graph on message send: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to execute assistant graph: {str(e)}"
            )

        ai_message_text = "I received your message, thank you."
        if output_state.get("messages"):
            ai_message_text = extract_text(output_state["messages"][-1].content)

        # Phase C: fresh short-lived connection for the final write.
        async with AsyncSessionLocal() as db:
            try:
                ai_msg = AgentMessage(session_id=session_id, role="ai", content=ai_message_text)
                db.add(ai_msg)

                session_res = await db.execute(select(AgentSession).where(AgentSession.id == session_id))
                session = session_res.scalar_one()
                session.state = {
                    "missing_fields": output_state.get("missing_fields", []),
                    "rag_context": output_state.get("rag_context", []),
                    "citations": output_state.get("citations", []),
                    "evidence_sufficient": output_state.get("evidence_sufficient", False),
                    "retrieval_confidence": output_state.get("retrieval_confidence", 0),
                    "risk_category": output_state.get("risk_category"),
                    "recommendations": output_state.get("recommendations", []),
                    "evaluation_complete": output_state.get("evaluation_complete", False)
                }
                await db.commit()

                # populate_existing() forces a fresh load of `messages` rather than
                # reusing this (fresh) session's identity map — harmless here since
                # this is a brand-new session with nothing cached yet, but kept for
                # consistency with the reasoning that bit us before.
                reload_stmt = (
                    select(AgentSession)
                    .where(AgentSession.id == session_id)
                    .options(selectinload(AgentSession.messages))
                    .execution_options(populate_existing=True)
                )
                res = await db.execute(reload_stmt)
                return res.scalar_one()
            except HTTPException:
                raise
            except Exception:
                await db.rollback()
                raise
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: UUID, db: AsyncSession = Depends(get_db)):
    """Retrieves session details and full chat history."""
    try:
        stmt = (
            select(AgentSession)
            .where(AgentSession.id == session_id)
            .options(selectinload(AgentSession.messages))
        )
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent session with ID {session_id} not found."
            )
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/by-case/{case_id}", response_model=SessionResponse)
async def get_session_by_case(case_id: UUID, db: AsyncSession = Depends(get_db)):
    try:
        stmt = (
            select(AgentSession)
            .where(AgentSession.case_id == case_id)
            .options(selectinload(AgentSession.messages))
            .order_by(AgentSession.created_at.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No agent session found for case {case_id}."
            )
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
