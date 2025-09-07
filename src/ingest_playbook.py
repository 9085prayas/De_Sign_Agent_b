# ingest_playbook.py (Updated to extract clauses)

import os
import time
import glob
import json
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not all([GEMINI_API_KEY, PINECONE_API_KEY]):
    raise ValueError("One or more required environment variables are not set.")

genai.configure(api_key=GEMINI_API_KEY)

PLAYBOOK_DIRECTORY = "playbooks/" 
INDEX_NAME = "company-playbook"
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768
CLAUSES_CONFIG_FILE = "clauses.json"

def extract_clause_titles_from_text(full_text: str) -> list[str]:
    """Uses an LLM to identify and extract clause titles from playbook text."""
    try:
        logging.info("Using LLM to extract clause titles from playbook text...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Analyze the following legal playbook text. Your task is to identify and list all the specific legal clause titles it discusses.
        Examples include "Indemnification", "Limitation of Liability", "Intellectual Property Rights", etc.
        Do not invent clauses. Only extract titles that are explicitly discussed in the text.

        Respond ONLY with a valid JSON array of strings. For example:
        ["Indemnification", "Confidentiality", "Governing Law & Jurisdiction"]

        ---
        PLAYBOOK TEXT:
        {full_text}
        """
        
        response = model.generate_content(prompt)
        # Clean up the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        clause_titles = json.loads(cleaned_response)
        
        if isinstance(clause_titles, list):
            logging.info(f"Successfully extracted {len(clause_titles)} clause titles.")
            return clause_titles
        return []
    except Exception as e:
        logging.error(f"Failed to extract clause titles with LLM: {e}")
        return []

def main():
    logging.info("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_existed = INDEX_NAME in pc.list_indexes().names()

    if not index_existed:
        logging.info(f"Creating new Pinecone index: '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
        logging.info("Index created successfully.")
    
    index = pc.Index(INDEX_NAME)

    if index_existed:
        logging.info(f"Attempting to clear all existing vectors from index '{INDEX_NAME}'...")
        try:
            index.delete(delete_all=True)
            logging.info("Index cleared successfully.")
        except NotFoundException:
            logging.info("Index was already empty. No vectors to delete.")
    
    playbook_files = glob.glob(os.path.join(PLAYBOOK_DIRECTORY, "*.pdf"))

    if not playbook_files:
        logging.error(f"No PDF files found in the '{PLAYBOOK_DIRECTORY}' directory.")
        return

    logging.info(f"Found {len(playbook_files)} playbook(s) to process.")
    
    all_clause_titles = set()

    for playbook_file in playbook_files:
        try:
            logging.info(f"--- Processing file: {playbook_file} ---")
            
            loader = PyPDFLoader(playbook_file)
            documents = loader.load()
            
            # Combine all text from the document for clause extraction
            full_document_text = "\n".join([doc.page_content for doc in documents])
            extracted_titles = extract_clause_titles_from_text(full_document_text)
            all_clause_titles.update(extracted_titles)

            # Continue with chunking and upserting
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            chunked_texts = [doc.page_content for doc in docs]
            
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=chunked_texts,
                task_type="retrieval_document"
            )
            embeddings = result['embedding']

            vectors_to_upsert = []
            file_basename = os.path.basename(playbook_file)
            for i, (text, embedding) in enumerate(zip(chunked_texts, embeddings)):
                vector_id = f"{file_basename}_chunk_{i}"
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {"text": text, "source": file_basename}
                })

            for i in range(0, len(vectors_to_upsert), 100):
                batch = vectors_to_upsert[i:i+100]
                index.upsert(vectors=batch)

        except Exception as e:
            logging.error(f"Failed to process {playbook_file}. Error: {e}")

    # Save the consolidated list of unique clause titles
    if all_clause_titles:
        sorted_clauses = sorted(list(all_clause_titles))
        with open(CLAUSES_CONFIG_FILE, 'w') as f:
            json.dump(sorted_clauses, f, indent=2)
        logging.info(f"Successfully saved {len(sorted_clauses)} unique clause titles to {CLAUSES_CONFIG_FILE}.")

    logging.info(f"✅ Ingestion complete. Index '{INDEX_NAME}' now has {index.describe_index_stats()['total_vector_count']} vectors.")

if __name__ == "__main__":
    main()