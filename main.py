import os
#from fastapi import FastAPI, Request
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from pydantic import BaseModel
from typing import Optional
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
import logging
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama

# Configure logging
logging.basicConfig(level=logging.INFO)

class NumpyFloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
        "supports_credentials": True
    }
})
app.json_encoder = NumpyFloatEncoder


# --- FastAPI app ---
#app = FastAPI()

# --- API Models ---
class PTORequest(BaseModel):
    employee_id: int
    days: int

class PolicyQuery(BaseModel):
    query: str
    employee_id: Optional[int] = None

# --- Initialize LLM and RAG resources (Ollama) ---
# Ensure Ollama is running locally and the models are pulled: `ollama pull llama3.2`, `ollama pull nomic-embed-text`
llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
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

# Insert sample PTO requests
cursor.execute("INSERT INTO pto_requests (id, employee_id, days, request_date, status) VALUES (1, 1, 2, '2024-08-15 09:00:00', 'Approved')")
cursor.execute("INSERT INTO pto_requests (id, employee_id, days, request_date, status) VALUES (2, 2, 1, '2024-08-20 14:30:00', 'Approved')")
cursor.execute("INSERT INTO pto_requests (id, employee_id, days, request_date, status) VALUES (3, 3, 3, '2024-08-22 11:15:00', 'Pending')")
cursor.execute("INSERT INTO pto_requests (id, employee_id, days, request_date, status) VALUES (4, 1, 1, '2024-08-25 16:45:00', 'Approved')")
cursor.execute("INSERT INTO pto_requests (id, employee_id, days, request_date, status) VALUES (5, 2, 2, '2024-08-26 10:20:00', 'Rejected')")

conn.commit()

# --- Load Policy Documents from PDFs ---
def load_policy_pdfs(policy_dir="policies"):
    docs = []
    if not os.path.isdir(policy_dir):
        return docs
    for fname in os.listdir(policy_dir):
        if fname.endswith(".pdf"):
            path = os.path.join(policy_dir, fname)
            print(f"Loading policy document: {fname}")
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
    logging.error(f"Processing PTO request for employee_id: {employee_id}, days: {days}")
    cursor.execute("SELECT pto_balance FROM employees WHERE id = ?", (employee_id,))
    result = cursor.fetchone()
    if not result:
        return "Employee not found."
    balance = result[0]
    logging.info(f"Employee {employee_id} has {balance} PTO days. Requesting {days} days.")
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
def retrieve_policy(query: str, employee_id: Optional[int] = None) -> str:
    """Retrieve and reason over policies using RAG."""
    logging.error(f"Retrieving policy for query: {query} and employee_id: {employee_id}")
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
            logging.error(f"Generated response: {response}")
            return response
    return f"Retrieved policies: {docs_content}"

tools = [automate_pto_request, retrieve_policy]
memory = MemorySaver()
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

def hr_workflow(employee_id: int, query: str):
    config = {"configurable": {"thread_id": "hr_thread"}}
    input_message = {"messages": [{"role": "user", "content": f"Employee ID: {employee_id}. Query: {query}"}]}
    logging.info(f"Input message: {input_message}")
    response = ""
    for step in agent_executor.stream(input_message, config, stream_mode="values"):
        if step["messages"][-1].content:
            response += step["messages"][-1].content + "\n"
    return response.strip()

# --- FastAPI Endpoints ---
@app.route('/agentic-ai/pto-request', methods=['POST'])
#@app.post("/agentic-ai/pto-request")
def api_pto_request():
    try:
        data = request.get_json() 
        logging.info("Received PTO request data: %s", data)
        employee_id = int(data.get('employee_id'))  # Ensure integer
        days = int(data.get('days'))  # Ensure integer
        logging.info("Parsed employee_id: %d, days: %d", employee_id, days)
        if employee_id is None or days is None:
            return jsonify({"error": "Missing employee_id or days"}), 400
        result = automate_pto_request.invoke({"employee_id": employee_id, "days": days})
        return {"result": result}
    except (ValueError, TypeError):
        return jsonify({"error": "employee_id and days must be integers"}), 400
   

@app.route("/agentic-rag/policy-query",methods=['POST'])
def api_policy_query():
    try:
        data = request.get_json() 
        logging.info("Received policy query data: %s", data)
        employee_id = data.get('employee_id')  # Get as-is, don't convert yet
        query = data.get('query')
        logging.info("Parsed employee_id: %s, query: %s", employee_id, query)
        
        # Convert employee_id to int only if it exists
        if employee_id is not None:
            try:
                employee_id = int(employee_id)
            except (ValueError, TypeError):
                return jsonify({"error": "employee_id must be a valid integer"}), 400
        
        if employee_id and ("pto" in query.lower() or "policy" in query.lower()):
            logging.info("Processing PTO request")
            result = hr_workflow(employee_id, query)
            return {"result": result}
        # Otherwise, just RAG
        result = retrieve_policy.invoke({"query": query, "employee_id": employee_id})
        return {"result": result}
    except Exception as e:
        logging.error("Error parsing input data", exc_info=True)
        return jsonify({"error": f"Error processing request: {str(e)}"}), 400
    

if __name__ == '__main__':
  
    app.run(debug=True, host='0.0.0.0', port=8000) 