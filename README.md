# Agentic AI & RAG HR Automation Demo

This project demonstrates how Agentic AI and Retrieval-Augmented Generation (RAG) can revolutionize HR by automating routine tasks and providing intelligent, context-aware responses. It features:

- **Agentic AI**: Automating PTO requests (Moveworks-style)
- **Agentic RAG**: Retrieving employee policies for dynamic queries (Ema's Assistant-style)
- **PDF Policy Documents**: Realistic policy ingestion from PDF files
- **Simple React Frontend**: For interactive demo

---

## Project Structure

```
hragent/
    main.py                # FastAPI backend with agent logic
    requirements.txt       # Backend dependencies
    policies/              # Sample policy PDF documents

```

---

## Prerequisites
- Python 3.9+
- Node.js (v16+ recommended)
- An OpenAI API key (for LLM and embeddings)

---

## Backend Setup (FastAPI + LangChain)

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Add your OpenAI API key:**
   - The backend will prompt for your key on first run, or set it as an environment variable:
     ```bash
     export OPENAI_API_KEY=sk-...
     ```

3. **Run the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   ```
   - The API will be available at [http://localhost:8000](http://localhost:8000)

4. **Policy Documents:**
   - Place your HR policy PDFs in `policies/`. The backend will automatically ingest and use them for RAG.

---

## Example Employees
- John Doe (id: 1)
- Jane Smith (id: 2)
- Tony Stark (id: 3)

---

## Extending
- Add more policy PDFs to `policies` for richer RAG.
- Expand agent logic in `main.py` for more HR workflows.

---

## Credits
- Built with FastAPI, LangChain, OpenAI, FAISS, React.
- Inspired by Moveworks and Ema's Assistant.
