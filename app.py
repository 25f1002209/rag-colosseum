
import json
import re
import math
from pathlib import Path
from collections import Counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional


# --------------------------------------------------
# 1. Load the knowledge base
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"

with open(KB_PATH, "r", encoding="utf-8") as file:
    passages = json.load(file)

if not isinstance(passages, list) or not passages:
    raise ValueError("Knowledge base must be a non-empty JSON list.")

if not all(isinstance(p, str) for p in passages):
    raise ValueError("Every passage must be a string.")

print(f"Loaded {len(passages)} passages.")


# --------------------------------------------------
# 2. Tokenization
# --------------------------------------------------

def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())


# Remove common words that do not help retrieval
STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an",
    "and", "or", "of", "to", "in", "on", "for",
    "with", "by", "what", "which", "who", "when",
    "where", "how", "does", "do", "did", "can",
    "could", "would", "should", "about", "from",
    "that", "this", "it", "its", "their", "they",
    "be", "as", "at", "than", "explain", "describe",
    "tell", "me", "please", "give", "show"
}


def useful_tokens(text):
    return [
        token for token in tokenize(text)
        if token not in STOPWORDS
    ]


# --------------------------------------------------
# 3. Build a lightweight BM25 index
# --------------------------------------------------

N = len(passages)

tokenized_passages = [
    tokenize(passage) for passage in passages
]

document_lengths = [
    len(tokens) for tokens in tokenized_passages
]

average_length = (
    sum(document_lengths) / N if N else 0
)

# Document frequency for every token
document_frequency = Counter()

for tokens in tokenized_passages:
    document_frequency.update(set(tokens))


def calculate_bm25(query):
    query_tokens = useful_tokens(query)

    if not query_tokens:
        return [0.0] * N

    k1 = 1.5
    b = 0.75

    scores = []

    for doc_index, doc_tokens in enumerate(tokenized_passages):

        term_counts = Counter(doc_tokens)
        doc_length = document_lengths[doc_index]

        score = 0.0

        for term in query_tokens:

            frequency = term_counts.get(term, 0)

            if frequency == 0:
                continue

            df = document_frequency.get(term, 0)

            # Positive inverse document frequency
            idf = math.log(
                1 + (N - df + 0.5) / (df + 0.5)
            )

            denominator = frequency + k1 * (
                1 - b + b * doc_length / max(average_length, 1)
            )

            score += idf * (
                frequency * (k1 + 1) / denominator
            )

        scores.append(score)

    return scores


# --------------------------------------------------
# 4. Retrieve relevant passages
# --------------------------------------------------

def retrieve(question, top_k):

    raw_scores = calculate_bm25(question)

    ranked_indices = sorted(
        range(N),
        key=lambda i: raw_scores[i],
        reverse=True
    )

    selected = ranked_indices[:top_k]

    results = []

    for index in selected:

        raw_score = raw_scores[index]

        # Convert BM25 into a bounded score
        normalized_score = raw_score / (raw_score + 3.0)

        results.append({
            "index": index,
            "raw_score": raw_score,
            "score": normalized_score
        })

    return results


# --------------------------------------------------
# 5. Extract answer sentences from retrieved passages
# --------------------------------------------------

def extract_answer(question, retrieved):

    query_words = set(useful_tokens(question))

    if not query_words:
        return None, []

    candidates = []

    for item in retrieved:

        passage_index = item["index"]
        passage = passages[passage_index]

        sentences = re.split(
            r"(?<=[.!?])\s+",
            passage.strip()
        )

        for sentence in sentences:

            sentence_words = set(useful_tokens(sentence))

            if not sentence_words:
                continue

            overlap = query_words.intersection(sentence_words)

            if not overlap:
                continue

            coverage = len(overlap) / len(query_words)

            # Score sentence relevance
            sentence_score = (
                0.75 * coverage +
                0.25 * item["score"]
            )

            candidates.append({
                "sentence": sentence.strip(),
                "source": passage_index,
                "coverage": coverage,
                "score": sentence_score
            })

    if not candidates:
        return None, []

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    selected = []
    seen = set()

    for candidate in candidates:

        sentence = candidate["sentence"]

        if sentence.lower() in seen:
            continue

        # Require reasonable overlap with the question
        if candidate["coverage"] < 0.45:
            continue

        selected.append(candidate)
        seen.add(sentence.lower())

        if len(selected) >= 3:
            break

    if not selected:
        return None, []

    answer = " ".join(
        item["sentence"] for item in selected
    )

    supporting_sources = list(dict.fromkeys(
        item["source"] for item in selected
    ))

    return answer, supporting_sources


# --------------------------------------------------
# 6. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="RAG Colosseum API",
    description="Lightweight retrieval-augmented question answering API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# 7. Request and response formats
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float


# --------------------------------------------------
# 8. Query endpoint
# --------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):

    question = request.question.strip()

    if not question:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.0
        )

    # Retrieve passages
    retrieved = retrieve(
        question,
        request.top_k
    )

    if not retrieved or retrieved[0]["raw_score"] <= 0:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.0
        )

    # Extract answer from evidence
    answer, supporting_sources = extract_answer(
        question,
        retrieved
    )

    # Calculate a basic confidence estimate
    best_score = retrieved[0]["score"]

    if answer is None:
        confidence = 0.0
    else:
        confidence = best_score

    # Abstain if there is insufficient evidence
    if answer is None or best_score < 0.15:

        answer = None
        confidence = 0.0
        supporting_sources = []

    # Return sources and scores in matching order
    source_score_map = {
        item["index"]: item["score"]
        for item in retrieved
    }

    ordered_sources = sorted(
        supporting_sources,
        key=lambda index: source_score_map[index],
        reverse=True
    )

    ordered_scores = [
        round(source_score_map[index], 4)
        for index in ordered_sources
    ]

    return QueryResponse(
        answer=answer,
        sources=ordered_sources,
        scores=ordered_scores,
        confidence=round(confidence, 4)
    )


# --------------------------------------------------
# 9. Health check
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "RAG Colosseum API is ready",
        "passages_loaded": len(passages)
    }