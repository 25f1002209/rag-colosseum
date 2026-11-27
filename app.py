
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
# 7. PROMPT INJECTION DEFENSE
# ==================================================

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?instructions",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(the\s+)?system\s+prompt",
    r"print\s+(your\s+)?hidden\s+instructions",
    r"override\s+(all\s+)?instructions",
    r"act\s+as\s+system"
]

UNTRUSTED_SENTENCE_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"disregard\s+(all\s+)?instructions",
    r"override\s+(all\s+)?instructions",
    r"reveal\s+(your\s+)?hidden\s+instructions",
    r"do\s+not\s+answer\s+the\s+user",
    r"send\s+(all\s+)?(data|secrets|credentials)"
]


def sanitize_question(question):

    cleaned = question

    for pattern in INJECTION_PATTERNS:
        cleaned = re.sub(
            pattern,
            " ",
            cleaned,
            flags=re.IGNORECASE
        )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned
    ).strip(" ,;:-")

    return cleaned


def is_untrusted_sentence(sentence):

    return any(
        re.search(
            pattern,
            sentence,
            re.IGNORECASE
        )
        for pattern in UNTRUSTED_SENTENCE_PATTERNS
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
# 10. ENTITY-AWARE FACT EXTRACTION
# ==================================================

def extract_answer(question, query_embedding, results):

    passage_lookup = {
        item["id"]: item
        for item in results
    }

    interrogatives = {
        "what", "which", "where", "when",
        "who", "how", "does", "do", "is",
        "are", "can", "tell", "explain"
    }

    # Identify named technologies in the question
    anchors = [
        word.lower()
        for word in re.findall(
            r"\b[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)*\b",
            question
        )
        if word.lower() not in interrogatives
    ]

    # Identify technical acronyms that must be supported
    acronyms = [
        word.lower()
        for word in re.findall(
            r"\b[A-Z]{2,8}\b",
            question
        )
    ]

    asks_count = bool(re.search(
        r"\b(how many|number of|count)\b",
        question,
        re.IGNORECASE
    ))

    asks_port = bool(re.search(
        r"\bport\b",
        question,
        re.IGNORECASE
    ))

    asks_time = bool(re.search(
        r"\b(interval|timeout|latency|duration|"
        r"heartbeat|how long|wait time)\b",
        question,
        re.IGNORECASE
    ))

    number_pattern = (
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:ms|milliseconds?|seconds?|s|minutes?|"
        r"bits?|bytes?|records?|nodes?|replicas?)?\b"
        r"|\b(one|two|three|four|five|six|seven|"
        r"eight|nine|ten)\b"
    )

    candidates = []

    for idx, item in enumerate(sentence_records):

        pid = item["passage_id"]

        if pid not in passage_lookup:
            continue

        sentence = item["text"]
        parent = passage_lookup[pid]["text"]

        # Never use recognized injected instructions as evidence
        if is_untrusted_sentence(sentence):
            continue

        dense = float(
            sentence_embeddings[idx] @ query_embedding
        )

        sentence_lex = lexical_coverage(
            question,
            sentence
        )

        parent_lex = lexical_coverage(
            question,
            parent
        )

        # Match named entities against the parent passage
        anchor_score = 0.0

        if anchors:
            matched = sum(
                anchor in parent.lower()
                for anchor in anchors
            )

            anchor_score = matched / len(anchors)

        # Required acronyms must exist in the supporting passage
        acronym_valid = all(
            acronym in parent.lower()
            for acronym in acronyms
        )

        if not acronym_valid:
            continue

        # Combined sentence and passage relevance
        score = (
            0.25 * max(0.0, dense)
            + 0.25 * sentence_lex
            + 0.25 * parent_lex
            + 0.25 * anchor_score
        )

        has_number = bool(
            re.search(number_pattern, sentence, re.IGNORECASE)
        )

        # Prioritize numerical answers when requested
        if asks_count:
            score += 0.18 if has_number else -0.18

        # Prioritize measurements and time values
        if asks_time:
            score += 0.12 if has_number else -0.12

        # A port answer must actually contain a port and number
        if asks_port:

            valid_port = (
                "port" in sentence.lower()
                and has_number
            )

            score += 0.18 if valid_port else -0.25

        candidates.append({
            "passage_id": pid,
            "text": sentence,
            "score": score,
            "dense": dense,
            "parent_lex": parent_lex,
            "anchor_score": anchor_score
        })

    if not candidates:
        return None, 0.0, []

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = candidates[0]

    # Abstention checks
    if best["dense"] < 0.30:
        return None, best["score"], []

    if (
        best["parent_lex"] < 0.15
        and best["anchor_score"] < 0.5
    ):
        return None, best["score"], []

    if asks_count and not re.search(
        number_pattern,
        best["text"],
        re.IGNORECASE
    ):
        return None, best["score"], []

    if asks_port and not (
        "port" in best["text"].lower()
        and re.search(number_pattern, best["text"], re.IGNORECASE)
    ):
        return None, best["score"], []

    # Simple questions should get one direct factual sentence
    selected = [best]

    # Multiple sentences only when the question requires synthesis
    multi_intent = bool(re.search(
        r"\b(compare|contrast|difference|differences|"
        r"versus|vs|both)\b",
        question,
        re.IGNORECASE
    )) or bool(re.search(
        r"\band\b",
        question,
        re.IGNORECASE
    ))

    if multi_intent:

        for candidate in candidates[1:]:

            if len(selected) >= min(3, len(results)):
                break

            if candidate["score"] < best["score"] - 0.18:
                continue

            if candidate["dense"] < 0.30:
                continue

            if candidate["passage_id"] == best["passage_id"]:
                # Allow multiple sentences from the same passage
                selected.append(candidate)
                continue

            if candidate["parent_lex"] < 0.15:
                continue

            selected.append(candidate)

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

def abstain():

    return {
        "answer": None,
        "sources": [],
        "scores": [],
        "confidence": 0.20
    }


@app.post("/query")
def query_api(request: QueryRequest):

    question = sanitize_question(
        request.question.strip()
    )

    if not question or not tokenize(question):
        return abstain()

    top_k = max(1, min(request.top_k, 10))

    # Retrieve passages
    results, query_embedding = retrieve(
        question,
        top_k
    )

    # Extract answer from supporting sentences
    answer, answer_score, selected = extract_answer(
        question,
        query_embedding,
        results
    )

    if answer is None or not selected:
        return abstain()

    # Include only passages supporting the selected answer
    selected_ids = list(dict.fromkeys(
        item["passage_id"] for item in selected
    ))

    score_lookup = {
        item["id"]: item["score"]
        for item in results
    }

    selected_ids.sort(
        key=lambda pid: score_lookup.get(pid, 0),
        reverse=True
    )

    selected_ids = selected_ids[:top_k]

    sources = selected_ids

    scores = [
        round(
            max(
                0.0,
                min(1.0, score_lookup[pid])
            ),
            4
        )
        for pid in sources
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
