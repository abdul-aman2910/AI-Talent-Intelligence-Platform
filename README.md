# AI Talent Intelligence Platform

An AI-powered recruitment intelligence platform that transforms resumes into recruiter-ready insights using semantic search, Retrieval-Augmented Generation (RAG), structured LLM outputs, function calling, and an AI recruiter agent.

## Overview

The AI Talent Intelligence Platform analyzes a candidate's resume against a job description and produces evidence-grounded recruitment insights.

Instead of relying only on keyword matching, the platform uses semantic embeddings and FAISS vector search to retrieve relevant resume evidence. Google Gemini then reasons over that evidence to generate structured candidate analysis, job-description alignment, and personalized interview questions.

The system is designed around an important principle:

> If something is not evidenced in the resume, the system should not invent it or automatically conclude that the candidate lacks it.

## What the Platform Does

- Extracts text from uploaded PDF resumes
- Splits resume text into overlapping chunks
- Generates semantic embeddings using Sentence Transformers
- Stores and searches embeddings using FAISS
- Retrieves relevant resume evidence for recruiter questions
- Uses RAG to generate evidence-grounded answers
- Produces structured responses using Pydantic
- Analyzes candidate technical strengths
- Analyzes resume evidence against a job description
- Identifies requirements that are not evidenced in the resume
- Generates candidate-specific technical interview questions
- Uses Gemini function/tool calling to dynamically search resume information
- Orchestrates the workflow through an AI recruiter agent
- Presents results through a FastAPI web application

## Architecture

```text
                     Resume PDF
                         |
                         v
                  PDF Text Parser
                    PDFPlumber
                         |
                         v
                   Text Chunking
                 Overlapping Chunks
                         |
                         v
              Sentence Transformer
                    Embeddings
                         |
                         v
                  FAISS Vector
                     Search
                         |
                  Relevant Evidence
                         |
             +-----------+-----------+
             |                       |
             v                       v
            RAG                Recruiter Agent
             |                       |
             v                       v
       Gemini Model <--------> Tool Calling
             |
             v
       Pydantic Structured
            Output
             |
             v
    Recruiter Intelligence
          Dashboard
```

## Core AI Pipeline

```text
Embeddings
    ↓
FAISS Vector Search
    ↓
RAG
    ↓
Gemini
    ↓
Pydantic Structured Output
    ↓
Tool Calling
    ↓
Recruiter Agent
```

## Technology Stack

### Programming & Backend

- Python
- FastAPI
- Jinja2

### Document Processing

- PDFPlumber
- Text chunking

### Embeddings & Retrieval

- Sentence Transformers
- `all-MiniLM-L6-v2`
- 384-dimensional embeddings
- FAISS
- `IndexFlatL2`
- Semantic search

### Generative AI

- Google Gemini API
- Retrieval-Augmented Generation (RAG)
- Structured LLM outputs
- Pydantic
- Function / tool calling
- AI agents

### Testing

- PyTest

## Main Components

### `src/resume_parser.py`

Extracts text from uploaded PDF resumes using PDFPlumber.

### `src/chunker.py`

Splits extracted resume text into overlapping chunks while avoiding unnecessary word splitting.

### `src/embeddings.py`

Generates semantic embeddings using:

```text
all-MiniLM-L6-v2
```

The model produces 384-dimensional vectors for each text chunk.

### `src/vector_store.py`

Provides a FAISS vector store using:

```text
IndexFlatL2
```

It stores resume embeddings and performs nearest-neighbor searches.

### `src/retriever.py`

Connects the embedding model and FAISS vector store.

Given a natural-language query, it:

1. Generates an embedding for the query
2. Searches the FAISS index
3. Retrieves the most relevant resume chunks
4. Returns the retrieved evidence and distances

### `src/rag.py`

Implements Retrieval-Augmented Generation.

The system retrieves relevant resume evidence first and then sends that evidence to Gemini.

The model is instructed to:

- Use only the supplied resume evidence
- Avoid inventing candidate information
- Provide supporting evidence
- Report insufficient evidence when appropriate

### `src/schemas.py`

Defines Pydantic schemas for structured AI responses, including:

- RAG responses
- Candidate analysis
- JD analysis
- Interview questions

### `src/tools.py`

Contains application-side tools that operate on the semantic resume retrieval system.

The main tool is:

```text
search_resume()
```

It retrieves relevant resume evidence based on a semantic query.

### `src/agent.py`

Implements the AI recruiter agent.

The agent can use the resume search tool when candidate-specific information is required.

The tool-calling workflow is:

```text
LLM decides
     ↓
Application executes tool
     ↓
Tool returns resume evidence
     ↓
LLM interprets evidence
     ↓
Final recruiter response
```

### `src/jd_analyzer.py`

Analyzes the candidate's resume evidence against a supplied job description.

It identifies:

- Matching strengths
- Requirements not evidenced in the resume
- Supporting evidence
- A recruiter-friendly summary

### `src/interview_generator.py`

Generates candidate-specific technical interview questions using both:

- Job description
- Relevant resume evidence

The questions are designed to validate the candidate's actual technical understanding.

### `main.py`

FastAPI application entry point.

The endpoint:

```text
POST /analyze
```

accepts:

- Resume PDF
- Job description

and orchestrates the complete analysis workflow.

## Evidence-Grounded Reasoning

A key design principle is distinguishing between **absence of evidence** and **evidence of absence**.

For example, if a job description requires Kafka but the resume does not mention Kafka, the system should not state:

```text
Candidate does not know Kafka.
```

Instead, it should say:

```text
Kafka is not evidenced in the resume and requires validation.
```

This makes the system more suitable for recruiter assistance because the AI is expected to reason from retrieved evidence rather than invent candidate history.

## Candidate Analysis

The recruiter agent analyzes broad technical categories such as:

- Programming and data engineering
- Cloud and big data
- Machine learning and AI
- Backend development
- DevOps and infrastructure
- Projects and technical implementation

It also identifies areas that may require further investigation.

## Job Description Analysis

The platform compares relevant resume evidence against a supplied job description.

The analysis separates:

```text
Matching strengths
```

from:

```text
Requirements not evidenced in the resume
```

This prevents the system from treating a missing resume keyword as proof that a candidate cannot perform a particular task.

## Interview Question Generation

The platform generates interview questions based on both candidate evidence and job requirements.

Questions can focus on:

- Technologies explicitly mentioned in the resume
- Candidate projects
- Architecture decisions
- Implementation details
- Technical concepts relevant to the role
- JD requirements that require validation

Each generated question contains:

```text
question
topic
reason
```

## Project Structure

```text
AI-Talent-Intelligence-Platform/
│
├── .env
├── .env.example
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── interview_generator.py
│   ├── jd_analyzer.py
│   ├── rag.py
│   ├── resume_parser.py
│   ├── retriever.py
│   ├── schemas.py
│   ├── tools.py
│   └── vector_store.py
│
├── tests/
│   ├── test_agent.py
│   ├── test_agent_tool.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_gemini.py
│   ├── test_interview_generator.py
│   ├── test_jd_analyzer.py
│   ├── test_parser.py
│   ├── test_rag.py
│   ├── test_resume_chunks.py
│   ├── test_resume_embeddings.py
│   ├── test_retriever.py
│   ├── test_structured_analysis.py
│   ├── test_tools.py
│   ├── test_tool_calling.py
│   ├── test_tool_schema.py
│   └── test_vector_store.py
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
└── uploads/
```

## Setup

### 1. Clone the repository

```bash
git clone <https://github.com/abdul-aman2910/AI-Talent-Intelligence-Platform>
cd AI-Talent-Intelligence-Platform
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

A safe template is available as:

```text
.env.example
```

Never commit the real `.env` file to GitHub.

## Run Locally

Start the FastAPI application:

```powershell
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

Upload a resume, enter a job description, and run the analysis.

## Testing

Run the complete test suite:

```powershell
pytest
```

Individual components can also be tested:

```powershell
pytest tests/test_embeddings.py
pytest tests/test_retriever.py
pytest tests/test_rag.py
pytest tests/test_tool_calling.py
pytest tests/test_agent.py
```

## Security

The project uses environment variables for API credentials.

Do not commit:

```text
.env
```

Do not place real API keys inside source code.

Uploaded resumes are stored in the runtime `uploads/` directory and are excluded from Git through `.gitignore`.

Personal resumes or other documents containing private contact information should not be committed to a public repository.

## Deployment

The application can be deployed as a FastAPI web service on a compatible hosting platform.

For deployment, configure:

```text
GOOGLE_API_KEY
```

as a server-side environment variable or secret.

The application should be started with:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Current Capabilities

The platform currently demonstrates:

- Semantic resume retrieval
- FAISS vector search
- RAG
- Evidence-grounded Gemini responses
- Structured Pydantic outputs
- Resume search tool calling
- AI recruiter agent orchestration
- Job-description-aware analysis
- Candidate-specific interview question generation
- FastAPI integration
- Recruiter-focused web dashboard

## Future Improvements

Potential future enhancements include:

- Persistent vector databases
- Multi-resume candidate search
- Candidate comparison workflows
- Retrieval evaluation metrics
- RAG evaluation datasets
- Hallucination evaluation
- Recruiter feedback loops
- Authentication and authorization
- Persistent candidate storage
- Production monitoring
- Background processing for large resume batches
- More advanced agent tools
- Production deployment

## Disclaimer

This project is intended for educational, portfolio, and demonstration purposes.

AI-generated recruitment insights should be treated as decision-support information and reviewed by a human recruiter. The system should not be used as the sole basis for employment decisions.

## Author

Abdul Aman

Built as a portfolio project demonstrating practical applications of:

- Big Data
- Python
- Generative AI
- RAG
- Vector Search
- Semantic Retrieval
- Function Calling
- AI Agents
- FastAPI
