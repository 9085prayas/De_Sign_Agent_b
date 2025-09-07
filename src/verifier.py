import os
import io
import json
import logging
from PIL import Image
import pytesseract
from docx import Document
from PyPDF2 import PdfReader
import google.generativeai as genai
from cachetools import cached, TTLCache
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_clauses_to_check() -> list[str]:
    """Loads the list of clauses from the JSON config file."""
    try:
        with open("clauses.json", 'r') as f:
            clauses = json.load(f)
            logging.info(f"Successfully loaded {len(clauses)} clauses to check from clauses.json")
            return clauses
    except FileNotFoundError:
        logging.warning("clauses.json not found. Falling back to default list.")
        return [
            "Indemnification", "Limitation of Liability", "Intellectual Property Rights",
            "Confidentiality", "Termination for Cause", "Governing Law & Jurisdiction"
        ]

CLAUSES_TO_CHECK = load_clauses_to_check()

cache = TTLCache(maxsize=100, ttl=3600)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "company-playbook"
EMBEDDING_MODEL = "models/text-embedding-004"

try:
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    logging.info("Successfully connected to Pinecone index.")
except Exception as e:
    logging.warning(f"Could not connect to Pinecone. RAG features will be disabled. Error: {e}")
    pinecone_index = None

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "".join(page.extract_text() for page in pdf_reader.pages)
        logging.info("Successfully extracted text from PDF.")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_bytes: bytes) -> str:
    try:
        document = Document(io.BytesIO(docx_bytes))
        text = "\n".join([para.text for para in document.paragraphs])
        logging.info("Successfully extracted text from DOCX.")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        logging.info("Successfully extracted text from image using OCR.")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from image with OCR: {e}")
        return ""

def retrieve_playbook_context(query: str, n_results: int = 3) -> str:
    if not pinecone_index:
        return "No playbook context available."
    try:
        query_embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )['embedding']

        results = pinecone_index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True
        )
        
        context = "\n---\n".join([match['metadata']['text'] for match in results['matches']])
        logging.info(f"Retrieved context for query: '{query[:50]}...'")
        return context
    except Exception as e:
        logging.error(f"Error retrieving from Pinecone: {e}")
        return "Failed to retrieve playbook context."

def generate_rag_llm_prompt(contract_text: str, playbook_context: str, clause_name: str) -> str:
    prompt = f"""
    You are an expert AI paralegal specializing in contract risk analysis. Your primary goal is to ensure compliance with our company's legal playbook.

    **Instructions:**
    1.  **Analyze the Clause**: Read the provided clause text from the contract.
    2.  **Consult the Playbook**: Review the relevant sections from our company's legal playbook provided below.
    3.  **Compare and Assess Risk**: Compare the contract's clause against the playbook's guidance. Assign a risk level ('Low', 'Medium', 'High').
        - 'Low' risk means it aligns perfectly with our playbook.
        - 'Medium' risk means it deviates slightly but is acceptable.
        - 'High' risk means it contradicts our playbook, is ambiguous, or is missing entirely.
    4.  **Justify**: Clearly explain *why* the clause has the assigned risk level, referencing specific points from the playbook.
    5.  **Output Format**: Respond ONLY with a valid JSON object with the following structure:
        {{
          "clause_name": "{clause_name}",
          "is_present": boolean,
          "confidence_score": float,
          "risk_level": "Low | Medium | High",
          "justification": "Your analysis comparing the clause to the playbook.",
          "cited_text": "The most relevant quote from the contract if present, otherwise an empty string."
        }}

    ---
    **COMPANY PLAYBOOK CONTEXT for '{clause_name}':**
    ---
    {playbook_context}
    ---
    **CONTRACT TEXT:**
    ---
    {contract_text}
    """
    return prompt

@cached(cache)
async def analyze_contract_text(contract_text: str, api_key: str):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        all_analyses = []

        for clause_name in CLAUSES_TO_CHECK:
            logging.info(f"Analyzing clause: {clause_name}")
            
            playbook_context = retrieve_playbook_context(f"Company policy for {clause_name} clause")
            prompt = generate_rag_llm_prompt(contract_text, playbook_context, clause_name)
            
            response = await model.generate_content_async(prompt)
            response_text = response.text.strip().replace("```json", "").replace("```", "")
            
            try:
                parsed_clause = json.loads(response_text)
                all_analyses.append(parsed_clause)
            except json.JSONDecodeError:
                logging.warning(f"Could not decode JSON for clause '{clause_name}'. Skipping.")
                all_analyses.append({
                    "clause_name": clause_name,
                    "is_present": False,
                    "confidence_score": 0.5,
                    "risk_level": "Medium",
                    "justification": "AI failed to produce a valid analysis for this clause.",
                    "cited_text": ""
                })

        return {"analysis": all_analyses}

    except Exception as e:
        logging.error(f"An unexpected error occurred during RAG LLM verification: {e}")
        raise

async def generate_clause_suggestion(clause_name: str, api_key: str, risky_text: str = "") -> str:
    try:
        genai.configure(api_key=api_key)
        prompt_action = "is missing from a contract. Please draft a standard, fair, and legally sound version of this clause."
        if risky_text:
            prompt_action = f"is one-sided or risky. Please rewrite it to be more balanced and fair. Here is the original text:\n---{risky_text}\n---"

        prompt = f"You are an expert AI contract lawyer. The following clause, '{clause_name}', {prompt_action}"
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        logging.error(f"An unexpected error occurred during clause suggestion: {e}")
        return "Error: Could not generate suggestion."

async def generate_plain_english_summary(contract_text: str, api_key: str) -> str:
    try:
        genai.configure(api_key=api_key)
        prompt = f"""
        You are an expert at translating complex legal documents into simple, plain English. 
        Analyze the following contract and provide a concise summary (2-3 short paragraphs) that a non-lawyer can easily understand. 
        Focus on the key obligations for each party and the most significant risks.
        ---
        CONTRACT TEXT:
        {contract_text}
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        logging.error(f"An unexpected error occurred during summary generation: {e}")
        return "Error: Could not generate summary."

async def verify_contract_clauses(file_bytes: bytes, content_type: str, api_key: str):
    text = ""
    if content_type == "application/pdf":
        text = extract_text_from_pdf(file_bytes)
    elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file_bytes)
    elif content_type in ["image/jpeg", "image/png"]:
        text = extract_text_from_image(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {content_type}")

    if not text:
        return {"error": "Could not extract any text from the uploaded file."}

    return await analyze_contract_text(text, api_key)