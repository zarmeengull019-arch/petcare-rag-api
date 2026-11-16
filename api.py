
import pickle
import re
from pathlib import Path
from collections import Counter

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "rag_database.pkl"

app = FastAPI(
    title="Cat Disease RAG API",
    description="Local veterinary knowledge retrieval for PetCare AI",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if not DATABASE_PATH.exists():
    raise RuntimeError(f"Database not found: {DATABASE_PATH}")

with DATABASE_PATH.open("rb") as file:
    data = pickle.load(file)

texts = data["texts"]
metadata = data["metadata"]

if not texts:
    raise RuntimeError("Knowledge database is empty.")

print(f"Loaded {len(texts)} knowledge chunks.")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    disease: str | None = Field(default=None, max_length=100)


STOP_WORDS = {
    "the", "is", "are", "a", "an", "and", "or", "to", "of", "in",
    "on", "for", "my", "me", "it", "what", "how", "can", "does",
    "do", "with", "about", "please", "tell", "cats", "cat"
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [word for word in words if word not in STOP_WORDS]


def retrieve_chunks(question, disease=None, limit=3):
    search_text = f"{disease or ''} {question}"
    query_words = set(tokenize(search_text))

    if not query_words:
        return []

    scored = []

    for index, chunk in enumerate(texts):
        chunk_words = tokenize(chunk)
        word_counts = Counter(chunk_words)

        overlap = sum(
            min(word_counts[word], 3)
            for word in query_words
        )

        # Give a small boost when the disease/topic appears in a chunk.
        disease_boost = 0
        if disease:
            disease_words = set(tokenize(disease))
            disease_boost = 3 * len(disease_words & set(chunk_words))

        score = overlap + disease_boost

        if score > 0:
            scored.append((score, index))

    scored.sort(reverse=True)
    return scored[:limit]


def make_answer(question, disease=None):
    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    matches = retrieve_chunks(question, disease)

    if not matches:
        return {
            "answer": (
                "I could not find a matching topic in the local knowledge "
                "base. Try mentioning the disease or symptom, such as "
                "ringworm, itching, hair loss, or fleas. A veterinarian "
                "should assess a cat with persistent or worsening symptoms."
            ),
            "sources": [],
            "mode": "keyword_retrieval"
        }

    selected_texts = []
    sources = []

    for score, index in matches:
        selected_texts.append(texts[index])

        source = str(
            metadata[index].get("source", "Unknown source")
        )
        source = source.replace("\\", "/").split("/")[-1]

        if source not in sources:
            sources.append(source)

    # Return retrieved veterinary text rather than inventing an AI answer.
    answer = (
        "I found the following information in the local cat-health "
        "knowledge base. This is reference information, not a diagnosis:\n\n"
        + "\n\n---\n\n".join(selected_texts)
        + "\n\nIf your cat has severe, persistent, or worsening symptoms, "
        "please consult a veterinarian."
    )

    return {
        "answer": answer,
        "sources": sources,
        "mode": "keyword_retrieval"
    }


@app.get("/")
def home():
    return {
        "message": "PetCare AI local knowledge API is running.",
        "status": "ok"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "knowledge_chunks": len(texts),
        "mode": "keyword_retrieval"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = make_answer(request.question, request.disease)

    return {
        "question": request.question,
        "disease": request.disease,
        **result
    }


@app.post("/voice-ask")
async def voice_question(
    file: UploadFile = File(...),
    disease: str | None = Form(default=None),
):
    # Voice transcription still requires a speech-recognition service.
    await file.close()

    raise HTTPException(
        status_code=503,
        detail=(
            "Voice recognition is not enabled in local demo mode. "
            "Please use the text question endpoint."
        )
    )
