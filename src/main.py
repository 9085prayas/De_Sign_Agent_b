from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="De-Sign AI Contract Co-Pilot Backend",
    description="A backend for contract analysis, summarization, and Q&A using the Gemini API.",
    version="1.0.1",
)

origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1", tags=["Contract Analysis"])

@app.get("/", tags=["Health"])
def read_root():
    return {"status": "De-Sign Backend is running"}