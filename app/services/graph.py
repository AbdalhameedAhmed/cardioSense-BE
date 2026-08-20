import json
import logging
from typing import Dict, Any, List, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from app.core.config import settings
from app.services.vector_search import search_guideline_chunks

logger = logging.getLogger(__name__)

def extract_text(content: Any) -> str:
    """
    Normalizes a langchain message's `.content` to a plain string.

    Some models (e.g. newer Gemini models with "thought signatures") return
    content as a list of parts (dicts with 'type'/'text'/'extras' keys) rather
    than a plain string. AgentMessage.content is a VARCHAR column, so passing
    that list through unflattened fails at the DB layer with an asyncpg
    DataError. This must be applied to every LLM response before it's stored
    or returned to the API.
    """
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content

# Cosine distance above this is treated as "not actually relevant" rather than
# just a weaker match. Calibrated from scripts/eval_retrieval.py against this
# store's embeddings (gemini-embedding-001, 768-dim): true topical matches
# scored 0.19-0.25, deliberately out-of-domain queries scored 0.31-0.37 — the
# midpoint of that gap is used here. Re-run the eval script and adjust this
# if the embedding model, chunking, or corpus changes.
RELEVANCE_DISTANCE_THRESHOLD = 0.28

# Confidence-score calibration anchors, taken from the SAME eval_retrieval.py
# run as the threshold above: the closest true-positive distance we observed
# (-> 100% confidence) and the farthest confirmed true-negative/out-of-domain
# distance (-> 0% confidence). This is a linear interpolation between two real
# measured points, not an arbitrary formula — re-run the eval script and
# update these anchors if the embedding model, chunking, or corpus changes.
CONFIDENCE_DISTANCE_FLOOR = 0.19
CONFIDENCE_DISTANCE_CEIL = 0.37

def distance_to_confidence(distance: float) -> int:
    """Maps a cosine distance to a 0-100 retrieval confidence score."""
    span = CONFIDENCE_DISTANCE_CEIL - CONFIDENCE_DISTANCE_FLOOR
    normalized = (CONFIDENCE_DISTANCE_CEIL - distance) / span
    return round(max(0.0, min(1.0, normalized)) * 100)

# ====================
# State Definition
# ====================
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    case_id: str
    case_data: Dict[str, Any]
    missing_fields: List[str]
    rag_context: List[str]
    citations: List[Dict[str, Any]]
    evidence_sufficient: bool
    retrieval_confidence: int
    risk_category: Optional[str]
    recommendations: List[str]
    evaluation_complete: bool

# ====================
# Node 1: Check Missing Fields
# ====================
def check_missing_fields(state: AgentState) -> Dict[str, Any]:
    """Inspects case_data to discover missing clinical variables."""
    case = state.get("case_data", {})
    missing = []
    
    # Check demographics
    if case.get("age") is None:
        missing.append("age")
    if case.get("sex") is None:
        missing.append("sex")
        
    # Check vitals & history
    if case.get("systolic_bp") is None:
        missing.append("systolic_bp")
    if case.get("diastolic_bp") is None:
        missing.append("diastolic_bp")
    if case.get("smoking") is None:
        missing.append("smoking")
    if case.get("diabetes") is None:
        missing.append("diabetes")
    if case.get("kidney_disease") is None:
        missing.append("kidney_disease")
    if case.get("previous_cvd") is None:
        missing.append("previous_cvd")
    if case.get("total_cholesterol") is None:
        missing.append("total_cholesterol")
    if case.get("hdl") is None:
        missing.append("hdl")

    return {
        "missing_fields": missing,
        "evaluation_complete": len(missing) == 0
    }

# ====================
# Conditional Router Edge
# ====================
def route_next_node(state: AgentState) -> str:
    """Decides whether to ask more clarifying questions or proceed to RAG + Evaluation."""
    # If the user's last message contains "evaluate" or "assess", force evaluation
    last_message = ""
    if state["messages"]:
        last_message = state["messages"][-1].content.lower()
        
    if "evaluate" in last_message or "assess" in last_message or "calculate" in last_message:
        return "rag"
        
    if state.get("evaluation_complete") or not state.get("missing_fields"):
        return "rag"
    else:
        return "interview"

# ====================
# Node 2: Interview Questions
# ====================
async def interview_node(state: AgentState) -> Dict[str, Any]:
    """Generates a conversational question asking for one or more missing variables."""
    missing = state.get("missing_fields", [])
    
    # Readable names for fields
    field_labels = {
        "age": "age",
        "sex": "biological sex",
        "systolic_bp": "systolic blood pressure",
        "diastolic_bp": "diastolic blood pressure",
        "smoking": "smoking status",
        "diabetes": "diabetes status",
        "kidney_disease": "kidney disease history",
        "previous_cvd": "prior history of cardiovascular disease",
        "total_cholesterol": "total cholesterol level",
        "hdl": "HDL cholesterol level"
    }
    
    missing_readable = [field_labels.get(f, f) for f in missing]
    
    # 1. Fallback Rule-Based (or if no API key is set)
    has_gemini = settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-gemini-api-key"
    has_openai = settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your-openai-api-key" and not settings.OPENAI_API_KEY.startswith("mock")
    is_mock = not (has_gemini or has_openai)
    
    if is_mock:
        fields_str = ", ".join(missing_readable[:3])
        if len(missing_readable) > 3:
            fields_str += f" (and {len(missing_readable) - 3} other values)"
            
        content = f"To conduct a thorough cardiovascular risk evaluation, I require a bit more clinical context. Could you please provide the patient's **{fields_str}**?"
        return {"messages": [AIMessage(content=content)]}

    # 2. LLM Node execution
    try:
        if has_gemini:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_LLM_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.4
            )
        else:
            llm = ChatOpenAI(
                model=settings.LLM_MODEL, 
                openai_api_key=settings.OPENAI_API_KEY, 
                temperature=0.4
            )
        
        system_prompt = (
            "You are CardioCompass, a polite clinical decision support assistant.\n"
            "Your goal is to gather missing clinical metrics required for cardiovascular risk scoring.\n"
            "The following metrics are currently missing: " + ", ".join(missing_readable) + ".\n"
            "Formulate a warm, professional clinical question asking for the next 1 or 2 most important missing metrics.\n"
            "Do NOT dump the entire checklist at once. Keep it conversational and brief, as if speaking to a doctor."
        )
        
        # Build chat input
        messages = [
            AIMessage(content=system_prompt)
        ] + list(state["messages"])
        
        response = await llm.ainvoke(messages)
        return {"messages": [AIMessage(content=extract_text(response.content))]}

    except Exception as e:
        logger.error(f"Error calling LLM in interview node: {e}")
        # Graceful fallback in case of rate limits / network dropouts
        content = f"Could you please specify the patient's **{missing_readable[0]}**?"
        return {"messages": [AIMessage(content=content)]}

# ====================
# Node 3: RAG Retrieval
# ====================
from langchain_core.runnables import RunnableConfig

async def rag_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Retrieves relevant guideline documents using pgvector cosine similarity."""
    # A session FACTORY, not a live session: this node opens its own short-lived
    # connection for the vector query rather than reusing one held open across
    # the request's slow embedding/LLM calls, which caused Supabase's pooler to
    # kill idle-in-transaction connections before the caller's final commit.
    db_factory = config.get("configurable", {}).get("db_factory")
    if not db_factory:
        logger.error("No database session factory provided in graph config.")
        return {"rag_context": [], "citations": [], "evidence_sufficient": False, "retrieval_confidence": 0}

    case = state.get("case_data", {})
    sys_bp = case.get("systolic_bp")
    dia_bp = case.get("diastolic_bp")
    symptoms = case.get("symptoms", [])

    # Construct a search query representing the patient profile
    query = "cardiovascular risk "
    if sys_bp or dia_bp:
        query += f"hypertension blood pressure {sys_bp}/{dia_bp} mmHg "
    if case.get("diabetes"):
        query += "diabetes "
    if symptoms:
        query += " symptoms " + " ".join(symptoms)

    try:
        async with db_factory() as db:
            chunks = await search_guideline_chunks(db, query, limit=3)
        context = [c["content"] for c in chunks]
        citations = [
            {
                "guideline_title": c["guideline_title"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "distance": c["distance"],
                "confidence": distance_to_confidence(c["distance"]),
            }
            for c in chunks
        ]
        # Evidence only counts as "sufficient" if at least one retrieved chunk
        # is actually close to the query, not merely the least-bad of a bad batch.
        # The overall retrieval_confidence uses that same best (lowest) distance.
        best_distance = min((c["distance"] for c in chunks), default=None)
        evidence_sufficient = best_distance is not None and best_distance <= RELEVANCE_DISTANCE_THRESHOLD
        retrieval_confidence = distance_to_confidence(best_distance) if best_distance is not None else 0
        return {
            "rag_context": context,
            "citations": citations,
            "evidence_sufficient": evidence_sufficient,
            "retrieval_confidence": retrieval_confidence,
        }
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return {"rag_context": [], "citations": [], "evidence_sufficient": False, "retrieval_confidence": 0}

# ====================
# Node 4: Risk Evaluation
# ====================
def format_citations(citations: List[Dict[str, Any]]) -> str:
    """
    Renders a deterministic, code-generated source list from retrieved chunk
    metadata. This is never written by the LLM, so it cannot hallucinate a
    source that wasn't actually retrieved — it's the ground-truth citation
    trail for whatever the model's prose claims.
    """
    if not citations:
        return ""
    lines = ["\n**Sources (WHO guideline excerpts retrieved for this assessment):**"]
    for i, c in enumerate(citations, start=1):
        title = c.get("guideline_title") or "Unknown guideline"
        page_start = c.get("page_start")
        page_end = c.get("page_end")
        if page_start and page_end:
            pages = f"p. {page_start}" if page_start == page_end else f"pp. {page_start}–{page_end}"
        else:
            pages = "page unknown"
        confidence = c.get("confidence")
        confidence_str = f" — {confidence}% match confidence" if confidence is not None else ""
        lines.append(f"- [{i}] *{title}*, {pages}{confidence_str}")
    return "\n".join(lines)


async def evaluate_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes clinical metrics and RAG context to output risk categories and recommendations."""
    case = state.get("case_data", {})
    context = state.get("rag_context", [])
    citations = state.get("citations", [])
    evidence_sufficient = state.get("evidence_sufficient", False)
    retrieval_confidence = state.get("retrieval_confidence", 0)

    # Demographics
    age = case.get("age", 50)
    sex = case.get("sex", "unknown")
    sys_bp = case.get("systolic_bp", 120)
    dia_bp = case.get("diastolic_bp", 80)

    # 0. Clinical safety gate: refuse to fabricate a risk score when retrieval
    # didn't actually find guideline evidence relevant to this case, rather
    # than let the LLM confidently invent one from general knowledge.
    if not evidence_sufficient:
        content = (
            "### Cardiovascular Risk Assessment\n\n"
            "**Risk Category:** Insufficient Evidence\n\n"
            f"**Retrieval Confidence:** {retrieval_confidence}% (below the threshold required to proceed)\n\n"
            "I could not retrieve guideline content that confidently matches this patient's profile, "
            "so I'm not going to generate a risk category or treatment recommendation from general "
            "knowledge alone.\n\n"
            "**Recommended next step:** Please have a licensed clinician review this case directly, "
            "or provide more specific clinical details (exact blood pressure readings, relevant symptoms, "
            "comorbidities) so a more targeted guideline search can be attempted."
        )
        return {
            "messages": [AIMessage(content=content)],
            "risk_category": "Insufficient Evidence",
            "recommendations": ["Refer to a licensed clinician for direct evaluation."],
            "evaluation_complete": True
        }

    # 1. Fallback Rule-Based (or if no API key is set)
    has_gemini = settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-gemini-api-key"
    has_openai = settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your-openai-api-key" and not settings.OPENAI_API_KEY.startswith("mock")
    is_mock = not (has_gemini or has_openai)
    
    if is_mock:
        # Simple clinical rules as a placeholder
        risk = "Low"
        recs = ["Recommend regular cardiovascular checkups."]
        
        # Check hypertension
        if sys_bp >= 140 or dia_bp >= 90:
            risk = "High"
            recs = [
                "Classified as Stage 2 Hypertension based on SBP >= 140 or DBP >= 90 mmHg.",
                "Recommend pharmacological therapy (e.g., ACE inhibitors, ARBs, or CCBs) as per ESC/AHA guidelines.",
                "Advise low sodium diet, regular physical exercise, and SBP monitoring."
            ]
        elif sys_bp >= 130 or dia_bp >= 80:
            risk = "Moderate"
            recs = [
                "Classified as Stage 1 Hypertension / Elevated BP.",
                "Recommend lifestyle modifications (low sodium, weight loss, exercise).",
                "Re-evaluate blood pressure in 3-6 months."
            ]
            
        if case.get("diabetes") or case.get("previous_cvd"):
            risk = "Very High"
            recs.insert(0, "Patient has high-risk co-morbidities (Diabetes/CVD). Target blood pressure should be < 130/80 mmHg.")
            
        content = (
            f"### Cardiovascular Risk Assessment Report\n\n"
            f"**Risk Category:** {risk} Risk\n\n"
            f"**Retrieval Confidence:** {retrieval_confidence}%\n\n"
            f"**Clinical Recommendations:**\n"
            + "\n".join([f"- {r}" for r in recs]) + "\n\n"
            f"*(Note: Simulated evaluation run due to local development mode).* "
            + format_citations(citations)
        )
        
        return {
            "messages": [AIMessage(content=content)],
            "risk_category": risk,
            "recommendations": recs,
            "evaluation_complete": True
        }

    # 2. LLM Node execution
    try:
        if has_gemini:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_LLM_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.2
            )
        else:
            llm = ChatOpenAI(
                model=settings.LLM_MODEL, 
                openai_api_key=settings.OPENAI_API_KEY, 
                temperature=0.2
            )
        
        # Structured response prompt
        prompt = (
            "You are CardioCompass, a cardiology decision support engine.\n"
            "Analyze the following patient case and guideline context to formulate a cardiovascular risk category and recommendations.\n\n"
            f"Patient Profile:\n"
            f"- Age: {age}\n"
            f"- Sex: {sex}\n"
            f"- Blood Pressure: {sys_bp}/{dia_bp} mmHg\n"
            f"- Smoker: {case.get('smoking')}\n"
            f"- Diabetes: {case.get('diabetes')}\n"
            f"- Kidney Disease: {case.get('kidney_disease')}\n"
            f"- Prior CVD: {case.get('previous_cvd')}\n"
            f"- Cholesterol: {case.get('total_cholesterol')} mg/dL (HDL: {case.get('hdl')})\n"
            f"- Symptoms: {', '.join(case.get('symptoms', []))}\n"
            f"- Medications: {', '.join(case.get('medications', []))}\n\n"
            f"Guideline Excerpts (numbered — a verified source list for these will be appended "
            f"automatically after your response, so cite them inline as [1], [2], etc. exactly as numbered here):\n"
            + "\n---\n".join(f"[{i+1}] {c}" for i, c in enumerate(context)) + "\n\n"
            "Your output must be formatted as JSON with keys:\n"
            "- 'risk_category': string (e.g. 'Low', 'Moderate', 'High', 'Very High')\n"
            "- 'recommendations': array of strings (actionable clinical guidance)\n"
            "- 'summary': string (a professional summary explaining the rating, citing excerpts inline as [1]/[2]/etc. "
            "— do not state facts that aren't supported by the numbered excerpts above)\n"
            "Return ONLY raw JSON, do not wrap in markdown codeblocks."
        )
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # Attempt parsing the JSON response robustly
        content_str = extract_text(response.content).strip()
        if content_str.startswith("```"):
            lines = content_str.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content_str = "\n".join(lines).strip()

        data = json.loads(content_str)
        summary = data.get("summary", "Evaluation complete.")
        risk_category = data.get("risk_category", "Unknown")
        recs = data.get("recommendations", [])
        
        formatted_message = (
            f"### Cardiovascular Risk Assessment Report\n\n"
            f"**Risk Category:** {risk_category} Risk\n\n"
            f"**Retrieval Confidence:** {retrieval_confidence}%\n\n"
            f"{summary}\n\n"
            f"**Clinical Recommendations:**\n"
            + "\n".join([f"- {r}" for r in recs])
            + "\n" + format_citations(citations)
        )
        
        return {
            "messages": [AIMessage(content=formatted_message)],
            "risk_category": risk_category,
            "recommendations": recs,
            "evaluation_complete": True
        }
        
    except Exception as e:
        logger.error(f"Error in LLM evaluation node: {e}")
        fallback_content = (
            "Risk evaluation completed. However, a processing error occurred when rendering recommendations. "
            "Recommend manual guideline review." + format_citations(citations)
        )
        return {
            "messages": [AIMessage(content=fallback_content)],
            "risk_category": "Unknown",
            "recommendations": [],
            "evaluation_complete": True
        }

# ====================
# Compile the Graph
# ====================
workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("check_missing", check_missing_fields)
workflow.add_node("interview", interview_node)
workflow.add_node("rag", rag_node)
workflow.add_node("evaluate", evaluate_node)

# Set entry point
workflow.add_edge(START, "check_missing")

# Add conditional edges
workflow.add_conditional_edges(
    "check_missing",
    route_next_node,
    {
        "interview": "interview",
        "rag": "rag"
    }
)

workflow.add_edge("interview", END)
workflow.add_edge("rag", "evaluate")
workflow.add_edge("evaluate", END)

# Final graph instance
graph = workflow.compile()
