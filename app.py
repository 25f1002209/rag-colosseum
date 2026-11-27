
import json
import re
import numpy as np

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# 1. INITIALIZE APPLICATION
# --------------------------------------------------

app = FastAPI(title="RAG Colosseum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

# --------------------------------------------------
# 2. LOAD KNOWLEDGE BASE
# --------------------------------------------------

DATA_PATH = Path("data/dataset.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)


def load_passages(data):

    if isinstance(data, list):
        items = data
    else:
        items = data.get(
            "passages",
            data.get("documents", data.get("data", []))
        )

    passages = []

    for i, item in enumerate(items):

        if isinstance(item, str):
            text = item
            index = i

        elif isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("content")
                or item.get("passage")
            )

            index = item.get("id", item.get("index", i))

        else:
            continue

        if text:
            passages.append({
                "id": int(index),
                "text": str(text)
            })

    passages.sort(key=lambda x: x["id"])

    if len(passages) != 50:
        raise ValueError(
            f"Expected 50 passages, loaded {len(passages)}. "
            "Check dataset.json format."
        )

    return passages


passages = load_passages(raw_data)

texts = [p["text"] for p in passages]
passage_ids = [p["id"] for p in passages]

# --------------------------------------------------
# 3. GENERATE PASSAGE EMBEDDINGS
# --------------------------------------------------

passage_embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    convert_to_numpy=True
)

# --------------------------------------------------
# 4. SPLIT PASSAGES INTO SENTENCES
# --------------------------------------------------

sentence_records = []

for passage in passages:

    sentences = re.split(
        r'(?<=[.!?])\s+',
        passage["text"]
    )

    for sentence in sentences:

        sentence = sentence.strip()

        if sentence:
            sentence_records.append({
                "passage_id": passage["id"],
                "text": sentence
            })


sentence_texts = [
    s["text"] for s in sentence_records
]

sentence_embeddings = model.encode(
    sentence_texts,
    normalize_embeddings=True,
    convert_to_numpy=True
)

# --------------------------------------------------
# 5. REQUEST FORMAT
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


# --------------------------------------------------
# 6. RETRIEVAL
# --------------------------------------------------

def retrieve(question, top_k):

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    similarities = passage_embeddings @ query_embedding

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    results = []

    for idx in ranked_indices[:top_k]:

        results.append({
            "id": passage_ids[idx],
            "text": texts[idx],
            "score": float(similarities[idx])
        })

    return results, query_embedding


# --------------------------------------------------
# 7. ANSWER EXTRACTION
# --------------------------------------------------

def extract_answer(question, query_embedding):

    similarities = (
        sentence_embeddings @ query_embedding
    )

    ranked = np.argsort(similarities)[::-1]

    best_score = float(similarities[ranked[0]])

    # Abstain if no sufficiently relevant sentence exists
    if best_score < 0.30:
        return None, best_score, []

    best = sentence_records[ranked[0]]

    selected = [best]

    # Allow a second sentence for cross-passage synthesis
    for idx in ranked[1:]:

        score = float(similarities[idx])

        candidate = sentence_records[idx]

        if (
            score >= max(0.30, best_score - 0.07)
            and candidate["passage_id"] != best["passage_id"]
            and len(selected) < 2
        ):
            selected.append(candidate)

        if len(selected) == 2:
            break

    answer = " ".join(
        item["text"] for item in selected
    )

    return answer, best_score, selected


# --------------------------------------------------
# 8. MAIN QUERY ENDPOINT
# --------------------------------------------------

@app.post("/query")
def query_api(request: QueryRequest):

    question = request.question.strip()

    if not question:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    top_k = max(1, min(request.top_k, 10))

    results, query_embedding = retrieve(
        question,
        top_k
    )

    answer, answer_score, selected = extract_answer(
        question,
        query_embedding
    )

    # Abstention threshold
    if answer is None or answer_score < 0.30:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.20
        }

    # Keep only passages that support the extracted answer
    source_ids = list(dict.fromkeys(
        item["passage_id"] for item in selected
    ))

    # Include other retrieved passages as supporting context
    for item in results:
        if item["id"] not in source_ids:
            source_ids.append(item["id"])

    source_ids = source_ids[:top_k]

    score_lookup = {
        item["id"]: item["score"]
        for item in results
    }

    source_ids.sort(
        key=lambda x: score_lookup.get(x, 0),
        reverse=True
    )

    scores = [
        round(
            max(0.0, min(1.0, score_lookup[sid])),
            4
        )
        for sid in source_ids
    ]

    confidence = min(
        0.99,
        max(0.0, answer_score)
    )

    return {
        "answer": answer,
        "sources": source_ids,
        "scores": scores,
        "confidence": round(confidence, 4)
    }


# --------------------------------------------------
# 9. HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "running",
        "passages_loaded": len(passages)
    }


@app.get("/health")
def health():

    return {"status": "healthy"}
