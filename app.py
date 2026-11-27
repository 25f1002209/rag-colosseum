
import json
import re
import threading
import numpy as np

from pathlib import Path
from functools import lru_cache
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding


# ==================================================
# 1. APPLICATION
# ==================================================

app = FastAPI(title="RAG Colosseum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = TextEmbedding(
    model_name=MODEL_NAME,
    threads=1
)

model_lock = threading.Lock()


# ==================================================
# 2. LOAD KNOWLEDGE BASE
# ==================================================

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
            f"Expected 50 passages, got {len(passages)}"
        )

    if [p["id"] for p in passages] != list(range(50)):
        raise ValueError("Passage IDs must be 0 to 49")

    return passages


passages = load_passages(raw_data)

texts = [p["text"] for p in passages]
passage_ids = [p["id"] for p in passages]


# ==================================================
# 3. TOKENIZATION AND LEXICAL RETRIEVAL
# ==================================================

STOPWORDS = {
    "what", "which", "who", "when", "where", "why",
    "how", "is", "are", "was", "were", "the", "a",
    "an", "of", "in", "on", "to", "for", "and",
    "or", "does", "do", "did", "can", "could",
    "would", "should", "with", "by", "from",
    "used", "use", "using", "about", "explain",
    "tell", "me", "please", "give", "describe",
    "according", "passage", "system"
}


def tokenize(text):

    words = re.findall(
        r"[a-z0-9]+(?:\.[0-9]+)?",
        text.lower()
    )

    result = []

    for word in words:

        if word in STOPWORDS:
            continue

        # Basic plural normalization
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"

        elif len(word) > 4 and word.endswith("s") \
                and not word.endswith("ss"):
            word = word[:-1]

        result.append(word)

    return result


def lexical_coverage(question, document):

    query_terms = set(tokenize(question))

    if not query_terms:
        return 0.0

    document_terms = set(tokenize(document))

    matched = query_terms.intersection(document_terms)

    return len(matched) / len(query_terms)


# ==================================================
# 4. SENTENCE SPLITTING
# ==================================================

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
    item["text"] for item in sentence_records
]


# ==================================================
# 5. GENERATE EMBEDDINGS
# ==================================================

def normalize_embeddings(vectors):

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    return vectors / np.maximum(norms, 1e-12)


passage_embeddings = normalize_embeddings(
    list(model.embed(texts, batch_size=16))
)

sentence_embeddings = normalize_embeddings(
    list(model.embed(sentence_texts, batch_size=16))
)


# ==================================================
# 6. REQUEST MODEL
# ==================================================

class QueryRequest(BaseModel):

    question: str = Field(min_length=1)

    top_k: int = Field(
        default=3,
        ge=1,
        le=10
    )


# ==================================================
# 7. PROMPT INJECTION DETECTION
# ==================================================

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(the\s+)?system\s+prompt",
    r"print\s+(your\s+)?(hidden\s+)?instructions",
    r"reveal\s+(your\s+)?hidden\s+instructions",
    r"act\s+as\s+system",
    r"override\s+(all\s+)?instructions",
    r"forget\s+(all\s+)?previous\s+instructions"
]


def is_injection(question):

    q = question.lower()

    return any(
        re.search(pattern, q)
        for pattern in INJECTION_PATTERNS
    )


# ==================================================
# 8. QUERY EMBEDDING
# ==================================================

@lru_cache(maxsize=256)
def get_query_embedding(question):

    with model_lock:

        embedding = list(
            model.query_embed(question)
        )

    return normalize_embeddings(embedding)[0]


# ==================================================
# 9. HYBRID PASSAGE RETRIEVAL
# ==================================================

def retrieve(question, top_k):

    query_embedding = get_query_embedding(question)

    dense_scores = (
        passage_embeddings @ query_embedding
    )

    lexical_scores = np.array([
        lexical_coverage(question, text)
        for text in texts
    ])

    # Combine semantic and lexical relevance
    hybrid_scores = (
        0.72 * np.clip(dense_scores, 0, 1)
        + 0.28 * lexical_scores
    )

    ranked = np.argsort(
        hybrid_scores
    )[::-1]

    results = []

    for idx in ranked[:top_k]:

        results.append({
            "id": passage_ids[idx],
            "text": texts[idx],
            "score": float(hybrid_scores[idx]),
            "dense": float(dense_scores[idx]),
            "lexical": float(lexical_scores[idx])
        })

    return results, query_embedding


# ==================================================
# 10. SENTENCE-LEVEL ANSWER EXTRACTION
# ==================================================

def extract_answer(question, query_embedding, results):

    eligible_ids = {
        item["id"] for item in results
    }

    eligible_indices = [
        i for i, item in enumerate(sentence_records)
        if item["passage_id"] in eligible_ids
    ]

    if not eligible_indices:
        return None, 0.0, []

    dense_scores = (
        sentence_embeddings @ query_embedding
    )

    candidates = []

    for idx in eligible_indices:

        item = sentence_records[idx]

        dense = float(dense_scores[idx])

        lexical = lexical_coverage(
            question,
            item["text"]
        )

        score = (
            0.72 * max(0.0, dense)
            + 0.28 * lexical
        )

        candidates.append({
            "passage_id": item["passage_id"],
            "text": item["text"],
            "score": score,
            "dense": dense,
            "lexical": lexical
        })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = candidates[0]

    # Reject unrelated and weakly supported questions
    if (
        best["dense"] < 0.32
        or (
            best["dense"] < 0.48
            and best["lexical"] < 0.25
        )
    ):
        return None, best["score"], []

    selected = [best]

    # Cross-passage synthesis
    compare_words = {
        "compare", "difference", "differences",
        "versus", "between", "both", "contrast"
    }

    query_words = set(tokenize(question))

    is_comparison = bool(
        query_words.intersection(compare_words)
    )

    for candidate in candidates[1:]:

        if len(selected) >= 2:
            break

        if candidate["passage_id"] in {
            item["passage_id"] for item in selected
        }:
            continue

        if candidate["dense"] < 0.38:
            continue

        if candidate["lexical"] < 0.20:
            continue

        # Avoid unrelated second sentences
        if candidate["score"] < best["score"] - 0.12:
            continue

        selected.append(candidate)

    # Keep the extracted factual sentences intact
    answer = " ".join(
        item["text"] for item in selected
    )

    return answer, best["score"], selected


# ==================================================
# 11. RESPONSE FORMAT
# ==================================================

def abstain():

    return {
        "answer": None,
        "sources": [],
        "scores": [],
        "confidence": 0.20
    }


# ==================================================
# 12. MAIN API ENDPOINT
# ==================================================

@app.post("/query")
def query_api(request: QueryRequest):

    question = request.question.strip()

    if not question:
        return abstain()

    # Reject direct instruction override attempts
    if is_injection(question):
        return abstain()

    top_k = request.top_k

    # Retrieve relevant passages
    results, query_embedding = retrieve(
        question,
        top_k
    )

    # Extract factual answer only from retrieved passages
    answer, answer_score, selected = extract_answer(
        question,
        query_embedding,
        results
    )

    if answer is None:
        return abstain()

    # Supporting source passages
    selected_ids = {
        item["passage_id"] for item in selected
    }

    source_results = [
        item for item in results
        if item["id"] in selected_ids
    ]

    # Add other retrieved passages if space is available
    for item in results:

        if len(source_results) >= top_k:
            break

        if item["id"] not in {
            x["id"] for x in source_results
        }:
            source_results.append(item)

    source_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    sources = [
        item["id"] for item in source_results
    ]

    scores = [
        round(
            max(0.0, min(1.0, item["score"])),
            4
        )
        for item in source_results
    ]

    confidence = min(
        0.99,
        max(0.61, answer_score)
    )

    return {
        "answer": answer,
        "sources": sources,
        "scores": scores,
        "confidence": round(confidence, 4)
    }


# ==================================================
# 13. HEALTH CHECK
# ==================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "passages_loaded": len(passages)
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "passages_loaded": len(passages)
    }
