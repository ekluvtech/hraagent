import os
import getpass
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from datetime import datetime
from PyPDF2 import PdfReader

# --- FastAPI app ---
app = FastAPI()

# --- API Models ---
class PTORequest(BaseModel):
    employee_id: int
    days: int

class PolicyQuery(BaseModel):
    query: str
    employee_id: Optional[int] = None

# --- Setup API keys (in practice, set these securely) ---
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# --- Initialize LLM and RAG resources ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# --- Simulated HR Database (SQLite in-memory) ---
conn = sqlite3.connect(':memory:', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    pto_balance INTEGER,
                    tenure_months INTEGER
                  )''')
cursor.execute('''CREATE TABLE pto_requests (
                    id INTEGER PRIMARY KEY,
                    employee_id INTEGER,
                    days INTEGER,
                    request_date TEXT,
                    status TEXT
                  )''')
cursor.execute("INSERT INTO employees (id, name, pto_balance, tenure_months) VALUES (1, 'John Doe', 15, 12)")
cursor.execute("INSERT INTO employees (id, name, pto_balance, tenure_months) VALUES (2, 'Jane Smith', 5, 3)")
cursor.execute("INSERT INTO employees (id, name, pto_balance, tenure_months) VALUES (3, 'Tony Stark', 6, 8)")
conn.commit()

# --- Load Policy Documents from PDFs ---
def load_policy_pdfs(policy_dir="backend/policies"):
    docs = []
    for fname in os.listdir(policy_dir):
        if fname.endswith(".pdf"):
            path = os.path.join(policy_dir, fname)
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs

# --- Simulated Policy Documents for RAG ---
policies = load_policy_pdfs()
splits = text_splitter.split_documents(policies)
vector_store = FAISS.from_documents(splits, embeddings)

# --- Define Tools for the Agent ---
@tool
def automate_pto_request(employee_id: int, days: int) -> str:
    """Automate PTO requests: Check balance, process request, update database."""
    cursor.execute("SELECT pto_balance FROM employees WHERE id = ?", (employee_id,))
    result = cursor.fetchone()
    if not result:
        return "Employee not found."
    balance = result[0]
    if balance >= days:
        request_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO pto_requests (employee_id, days, request_date, status) VALUES (?, ?, ?, ?)",
                       (employee_id, days, request_date, "Approved"))
        cursor.execute("UPDATE employees SET pto_balance = pto_balance - ? WHERE id = ?",
                       (days, employee_id))
        conn.commit()
        return f"PTO request for {days} days approved on {request_date}. New balance: {balance - days}"
    else:
        return f"Insufficient PTO balance. Current balance: {balance}"

@tool
def retrieve_policy(query: str, employee_id: int = None) -> str:
    """Retrieve and reason over policies using RAG."""
    docs = vector_store.similarity_search(query, k=2)
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    if employee_id:
        cursor.execute("SELECT tenure_months FROM employees WHERE id = ?", (employee_id,))
        tenure = cursor.fetchone()
        if tenure:
            tenure = tenure[0]
            prompt = ChatPromptTemplate.from_template(
                "Based on the policy: {context}\n"
                "And employee tenure: {tenure} months.\n"
                "Answer the query: {query}\n"
                "Include eligibility reasoning."
            )
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"context": docs_content, "tenure": tenure, "query": query})
            return response
    return f"Retrieved policies: {docs_content}"

tools = [automate_pto_request, retrieve_policy]
memory = MemorySaver()
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

def hr_workflow(employee_id: int, query: str):
    config = {"configurable": {"thread_id": "hr_thread"}}
    input_message = {"messages": [{"role": "user", "content": f"Employee ID: {employee_id}. Query: {query}"}]}
    response = ""
    for step in agent_executor.stream(input_message, config, stream_mode="values"):
        if step["messages"][-1].content:
            response += step["messages"][-1].content + "\n"
    return response.strip()

# --- FastAPI Endpoints ---
@app.post("/agentic-ai/pto-request")
def api_pto_request(req: PTORequest):
    result = automate_pto_request(req.employee_id, req.days)
    return {"result": result}

@app.post("/agentic-rag/policy-query")
def api_policy_query(req: PolicyQuery):
    # If both PTO and policy in query, use agent
    if req.employee_id and ("pto" in req.query.lower() or "policy" in req.query.lower()):
        result = hr_workflow(req.employee_id, req.query)
        return {"result": result}
    # Otherwise, just RAG
    result = retrieve_policy(req.query, req.employee_id)
    return {"result": result}
