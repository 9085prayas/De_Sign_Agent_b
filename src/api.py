import os
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional

from .security import require_scope
from .verifier import (
    verify_contract_clauses,
    generate_clause_suggestion,
    generate_plain_english_summary
)

router = APIRouter()

ALLOWED_CONTENT_TYPES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/jpeg",
    "image/png"
]

class SummarizeRequest(BaseModel):
    contract_text: str

class SuggestionRequest(BaseModel):
    clause_name: str
    risky_text: Optional[str] = ""

def get_gemini_api_key():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    return api_key

@router.post(
    "/verify",
    summary="Analyze a contract document for high-risk clauses",
    dependencies=[Depends(require_scope("contract.verify:clauses"))]
)
async def verify_document(
    file: UploadFile = File(...),
    gemini_api_key: str = Depends(get_gemini_api_key)
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    try:
        file_bytes = await file.read()
        verification_result = await verify_contract_clauses(
            file_bytes=file_bytes,
            content_type=file.content_type,
            api_key=gemini_api_key
        )
        
        if "error" in verification_result:
            raise HTTPException(status_code=400, detail=verification_result["error"])
            
        return verification_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")


@router.post(
    "/summarize",
    summary="Generate a plain English summary of a contract",
    dependencies=[Depends(require_scope("contract.verify:clauses"))]
)
async def summarize_contract(
    request: SummarizeRequest,
    gemini_api_key: str = Depends(get_gemini_api_key)
):
    summary = await generate_plain_english_summary(request.contract_text, gemini_api_key)
    return {"summary": summary}


@router.post(
    "/suggest-clause",
    summary="Get an AI-generated suggestion for a clause",
    dependencies=[Depends(require_scope("contract.verify:clauses"))]
)
async def suggest_clause_fix(
    request: SuggestionRequest,
    gemini_api_key: str = Depends(get_gemini_api_key)
):
    suggestion = await generate_clause_suggestion(
        clause_name=request.clause_name,
        risky_text=request.risky_text,
        api_key=gemini_api_key
    )
    return {"suggestion": suggestion}