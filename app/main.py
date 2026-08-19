from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import init_db
from app.api.endpoints import health, cases, guidelines, agent

# Create FastAPI app
app = FastAPI(
    title="CardioCompass API",
    description="Medical Risk & Guideline Assistant backend service",
    version="0.1.0"
)

# Configure CORS
# In production, specify the actual origins of the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to initialize the database
@app.on_event("startup")
def on_startup():
    init_db()

# Include routers
app.include_router(health.router, prefix="/api", tags=["System"])
app.include_router(cases.router, prefix="/api/cases", tags=["Cases"])
app.include_router(guidelines.router, prefix="/api/guidelines", tags=["Guidelines"])
app.include_router(agent.router, prefix="/api/agent/sessions", tags=["Agent"])

@app.get("/")
def read_root():
    return {
        "name": "CardioCompass API",
        "description": "Production-quality clinical decision support assistant MVP",
        "status": "online"
    }
