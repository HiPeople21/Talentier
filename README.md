# TalentScope — LinkedIn Candidate Discovery

An AI-powered recruiter tool that finds real LinkedIn candidates based on technical skills, experience level, and location filters. Built with **React TSX** + **FastAPI** + **LangGraph** + **Ollama**.

## Architecture

```
React TSX (Vite) → FastAPI → LangGraph Agent → Ollama (llama3.2)
                                    ↓
                              DuckDuckGo Search
                            (real LinkedIn profiles)
```

The LangGraph agent runs a multi-step pipeline:
1. **plan_search** (LLM) — Plans the optimal search query
2. **search_linkedin** (DuckDuckGo) — Fetches real LinkedIn profiles
3. **enrich** (LLM) — Cleans and enriches candidate data
4. **evaluate** (LLM) — Scores and ranks candidates by relevance
5. **decide** — If results are poor, loops back to refine the search

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Ollama** with GPU support (recommended)

## Setup

### 1. Install Ollama (with GPU support)

```bash
# Install native Ollama (includes CUDA/GPU support)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.2

# Start the server
ollama serve
```

> **Note:** If you previously installed Ollama via snap, remove it first with `sudo snap remove ollama` — the snap version doesn't support GPU acceleration.

### 2. Backend

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend

```bash
cd frontend
npm install
```

## Running

Open **three terminals**:

```bash
# Terminal 1 — Ollama
ollama serve

# Terminal 2 — Backend
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000

# Terminal 3 — Frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

## Usage

1. Select **technical skills** (Python, React, AWS, etc.)
2. Choose an **experience level** (Junior → Principal)
3. Pick a **location** (or leave as "Any")
4. Click **Search Candidates**
5. View real LinkedIn profiles, click cards for details, open profiles on LinkedIn

## Tech Stack

| Layer | Tech |
|-------|------|
| Frontend | React, TypeScript, Vite |
| Backend | FastAPI, Python |
| AI Agent | LangGraph, LangChain |
| LLM | Ollama (llama3.2) |
| Search | DuckDuckGo HTML |
| Styling | Vanilla CSS (dark theme, glassmorphism) |