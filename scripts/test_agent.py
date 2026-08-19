import asyncio
import sys
from pathlib import Path
from uuid import UUID

# Add backend directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import AsyncSessionLocal
from app.models.patient import Patient, PatientCase
from app.services.graph import graph
from app.models.agent import AgentSession, AgentMessage

async def main():
    print("==================================================")
    print("      TESTING CARDIOSENSE RAG & AGENT WORKFLOW    ")
    print("==================================================")

    async with AsyncSessionLocal() as session:
        # 1. Create a Patient
        patient = Patient(
            age=62,
            sex="female"
        )
        session.add(patient)
        await session.flush()
        
        # 2. Create a Case representing Stage 2 Hypertension (matches WHO guideline thresholds)
        # SBP: 155, DBP: 95, Smoker: Yes, Diabetes: Yes
        case = PatientCase(
            patient_id=patient.id,
            status="active",
            systolic_bp=155.0,
            diastolic_bp=95.0,
            smoking=True,
            diabetes=True,
            kidney_disease=False,
            previous_cvd=False,
            total_cholesterol=210.0,
            hdl=45.0,
            symptoms=["occasional headache", "mild fatigue"],
            medications=[]
        )
        session.add(case)
        await session.flush()
        
        # 3. Create an Agent Session
        agent_session = AgentSession(
            case_id=case.id,
            state={}
        )
        session.add(agent_session)
        await session.flush()
        
        await session.commit()
        
        print(f"Created Patient ID:  {patient.id}")
        print(f"Created Case ID:     {case.id}")
        print(f"Created Session ID:  {agent_session.id}")
        print("\nPatient Clinical Profile:")
        print("  - Age: 62 | Sex: Female")
        print("  - BP: 155/95 mmHg (Stage 2 Hypertension)")
        print("  - Smoker: Yes | Diabetes: Yes")
        print("  - Cholesterol: 210 mg/dL | HDL: 45 mg/dL")
        print("==================================================")

        # 4. Invoke the LangGraph State Machine
        print("\nInvoking LangGraph State Machine...")
        
        # Build initial state
        initial_state = {
            "messages": [],
            "case_id": str(case.id),
            "case_data": {
                "age": patient.age,
                "sex": patient.sex,
                "systolic_bp": float(case.systolic_bp),
                "diastolic_bp": float(case.diastolic_bp),
                "smoking": case.smoking,
                "diabetes": case.diabetes,
                "kidney_disease": case.kidney_disease,
                "previous_cvd": case.previous_cvd,
                "total_cholesterol": float(case.total_cholesterol) if case.total_cholesterol else None,
                "hdl": float(case.hdl) if case.hdl else None,
                "symptoms": case.symptoms or [],
                "medications": case.medications or []
            },
            "missing_fields": [],
            "rag_context": [],
            "risk_category": None,
            "recommendations": [],
            "evaluation_complete": False
        }
        
        # Compile graph and invoke config
        config = {"configurable": {"db": session}}
        
        try:
            result = await graph.ainvoke(initial_state, config=config)
            
            print("\n==================================================")
            print("                 EVALUATION RESULT                ")
            print("==================================================")
            print(f"Risk Category:  {result.get('risk_category')}")
            print(f"Evaluation Complete: {result.get('evaluation_complete')}")
            
            print("\nRAG Retrieved Context (WHO Guidelines):")
            for idx, ctx in enumerate(result.get("rag_context", [])):
                print(f"\n[{idx+1}] {ctx[:250]}...")
                
            print("\nAgent Final Response (Gemini 1.5 Flash):")
            if result.get("messages"):
                last_msg = result["messages"][-1]
                print(last_msg.content)
            print("==================================================")
            
        except Exception as e:
            print(f"\nExecution error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
