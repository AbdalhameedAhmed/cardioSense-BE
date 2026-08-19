-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Patients table
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    age INT,
    sex VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Patient Cases table
CREATE TABLE IF NOT EXISTS patient_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'active',
    systolic_bp NUMERIC(5,2),
    diastolic_bp NUMERIC(5,2),
    smoking BOOLEAN,
    diabetes BOOLEAN,
    kidney_disease BOOLEAN,
    previous_cvd BOOLEAN,
    total_cholesterol NUMERIC(5,2),
    hdl NUMERIC(5,2),
    symptoms JSONB DEFAULT '[]'::jsonb,
    medications JSONB DEFAULT '[]'::jsonb,
    additional_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Sessions table
CREATE TABLE IF NOT EXISTS agent_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES patient_cases(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'interviewing',
    current_node VARCHAR(100),
    state JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Messages table
CREATE TABLE IF NOT EXISTS agent_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES agent_sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Guidelines table
CREATE TABLE IF NOT EXISTS guidelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    organization VARCHAR(100),
    author VARCHAR(255),
    version VARCHAR(50),
    publication_date DATE,
    source_url TEXT,
    license TEXT,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Guideline Chunks table
CREATE TABLE IF NOT EXISTS guideline_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    guideline_id UUID REFERENCES guidelines(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding VECTOR(1536), -- 1536-dimensional vector for OpenAI/standard embeddings
    page INT,
    section VARCHAR(255),
    topic VARCHAR(255),
    condition VARCHAR(255),
    population VARCHAR(255),
    recommendation_type VARCHAR(100),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Citations table
CREATE TABLE IF NOT EXISTS citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES patient_cases(id) ON DELETE CASCADE,
    guideline_chunk_id UUID REFERENCES guideline_chunks(id) ON DELETE CASCADE,
    recommendation_id UUID, -- For optional integration if recommendations are separately saved
    source VARCHAR(255),
    title TEXT,
    section VARCHAR(255),
    page INT,
    source_url TEXT,
    relevance_score NUMERIC(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_patients_created_at ON patients(created_at);
CREATE INDEX IF NOT EXISTS idx_patient_cases_patient_id ON patient_cases(patient_id);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_case_id ON agent_sessions(case_id);
CREATE INDEX IF NOT EXISTS idx_agent_messages_session_id ON agent_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_guideline_chunks_guideline_id ON guideline_chunks(guideline_id);
CREATE INDEX IF NOT EXISTS idx_citations_case_id ON citations(case_id);

-- Vector HNSW index for similarity search (optional, pgvector specific)
-- CREATE INDEX ON guideline_chunks USING hnsw (embedding vector_cosine_ops);
