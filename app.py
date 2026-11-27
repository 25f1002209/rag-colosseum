import json
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding


# ============================================================
# 1. APPLICATION AND MODEL
# ============================================================

app = FastAPI(title="RAG Colosseum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_LOCK = threading.Lock()

model = TextEmbedding(
    model_name=MODEL_NAME,
    threads=1,
)


# ============================================================
# 2. LOAD THE 50-PASSAGE KNOWLEDGE BASE
# ============================================================

DATA_PATH = Path("data/dataset.json")


def load_passages(data):
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = (
            data.get("passages")
            or data.get("documents")
            or data.get("data")
            or []
        )
    else:
        items = []

    loaded = []

    for i, item in enumerate(items):
        if isinstance(item, str):
            text = item
            passage_id = i
        elif isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("content")
                or item.get("passage")
            )
            passage_id = item.get("id", item.get("index", i))
        else:
            continue

        if text:
            loaded.append({
                "id": int(passage_id),
                "text": str(text).strip(),
            })

    loaded.sort(key=lambda x: x["id"])

    if len(loaded) != 50:
        raise ValueError(
            f"Expected 50 passages, loaded {len(loaded)}. "
            "Check data/dataset.json."
        )

    if [p["id"] for p in loaded] != list(range(50)):
        raise ValueError("Passage IDs must be exactly 0 through 49.")

    return loaded


with DATA_PATH.open("r", encoding="utf-8") as f:
    passages = load_passages(json.load(f))

PASSAGE_TEXTS = [p["text"] for p in passages]
PASSAGE_IDS = [p["id"] for p in passages]
PASSAGE_BY_ID = {p["id"]: p["text"] for p in passages}


# ============================================================
# 3. TOKENIZATION
# ============================================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "what", "which", "who", "when", "where", "why", "how",
    "does", "do", "did", "can", "could", "would", "should", "will",
    "may", "might", "must", "of", "in", "on", "at", "to", "for",
    "from", "by", "with", "as", "into", "about", "and", "or",
    "it", "its", "this", "that", "these", "those", "me", "please",
    "tell", "explain", "describe", "give", "according", "passage",
    "system", "use", "used", "using",
}


def normalize_token(token):
    token = token.lower()

    # Small deterministic normalization suitable for this fixed corpus.
    if token == "decoupling":
        return "decouple"
    if token == "decoupled":
        return "decouple"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]

    return token


def tokenize(text):
    raw = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [
        normalize_token(t)
        for t in raw
        if normalize_token(t) not in STOPWORDS
    ]


def token_coverage(query_tokens, document):
    if not query_tokens:
        return 0.0

    doc_tokens = set(tokenize(document))
    matched = sum(1 for token in query_tokens if token in doc_tokens)
    return matched / len(query_tokens)


def extract_anchors(question):
    """Identify capitalized technical names such as Raft, Redis, AdamW, QUIC."""
    words = re.findall(
        r"\b[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)*\b",
        question,
    )

    ignored = {
        "What", "Which", "Who", "When", "Where", "Why", "How",
        "Does", "Do", "Did", "Can", "Could", "Would", "Should",
        "Is", "Are", "The",
    }

    anchors = []
    for word in words:
        if word not in ignored:
            anchors.extend(tokenize(word))

    # Preserve order and remove duplicates.
    return list(dict.fromkeys(anchors))


def anchor_coverage(anchors, document):
    if not anchors:
        return 0.0

    doc_tokens = set(tokenize(document))
    return sum(1 for a in anchors if a in doc_tokens) / len(anchors)


# ============================================================
# 4. SPLIT PASSAGES INTO SENTENCES
# ============================================================

sentence_records = []

for passage in passages:
    pieces = re.split(r"(?<=[.!?])\s+", passage["text"])

    for sentence in pieces:
        sentence = sentence.strip()
        if sentence:
            sentence_records.append({
                "passage_id": passage["id"],
                "text": sentence,
            })

SENTENCE_TEXTS = [x["text"] for x in sentence_records]


# ============================================================
# 5. EMBEDDINGS
# ============================================================

def normalize_embeddings(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


PASSAGE_EMBEDDINGS = normalize_embeddings(
    list(model.embed(PASSAGE_TEXTS, batch_size=16))
)

SENTENCE_EMBEDDINGS = normalize_embeddings(
    list(model.embed(SENTENCE_TEXTS, batch_size=16))
)


@lru_cache(maxsize=512)
def query_embedding_for(question):
    # FastEmbed's model object is protected because the free instance
    # has limited CPU and memory resources.
    with MODEL_LOCK:
        vector = list(model.query_embed(question))

    return normalize_embeddings(vector)[0]


# ============================================================
# 6. REQUEST SCHEMA
# ============================================================

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


# ============================================================
# 7. PROMPT-INJECTION HANDLING
# ============================================================

INJECTION_PATTERNS = [
    r"\bignore\b.{0,50}\b(previous|prior|earlier|all)\b.{0,30}\b(instructions?|rules?|directions?)\b",
    r"\bdisregard\b.{0,50}\b(instructions?|rules?|directions?)\b",
    r"\bforget\b.{0,40}\b(previous|prior|earlier|all)\b.{0,30}\b(instructions?|rules?)\b",
    r"\b(reveal|show|print|repeat| disclose)\b.{0,40}\b(system prompt|hidden instructions|developer message)\b",
    r"\b(override|bypass)\b.{0,40}\b(safety|security|instructions?|rules?)\b",
    r"\byou are now\b.{0,40}\b(system|admin|developer|unrestricted)\b",
    r"\bact as\b.{0,30}\b(system|developer|administrator)\b",
    r"\b(jailbreak|prompt injection)\b",
]

INJECTION_RE = re.compile(
    "|".join(f"(?:{p})" for p in INJECTION_PATTERNS),
    flags=re.IGNORECASE,
)

UNTRUSTED_CONTENT_RE = re.compile(
    r"\b(ignore|disregard|override|reveal|exfiltrate|"
    r"hidden instructions|system prompt|developer message|"
    r"send credentials|send secrets)\b",
    flags=re.IGNORECASE,
)


def sanitize_question(question):
    """
    Remove recognizable instruction-override clauses.
    The system is extractive and never executes retrieved text as instructions.
    """
    cleaned = INJECTION_RE.sub(" ", question)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned


def contains_injection(question):
    return bool(INJECTION_RE.search(question))


def is_untrusted_content(sentence):
    return bool(UNTRUSTED_CONTENT_RE.search(sentence))


# ============================================================
# 8. HYBRID PASSAGE RETRIEVAL
# ============================================================

def retrieve(question, top_k):
    q_tokens = tokenize(question)
    anchors = extract_anchors(question)
    q_vector = query_embedding_for(question)

    dense = PASSAGE_EMBEDDINGS @ q_vector

    lexical = np.array(
        [token_coverage(q_tokens, text) for text in PASSAGE_TEXTS],
        dtype=np.float32,
    )

    entity = np.array(
        [anchor_coverage(anchors, text) for text in PASSAGE_TEXTS],
        dtype=np.float32,
    )

    # Dense retrieval handles paraphrases; lexical/entity matching
    # distinguishes deliberately similar technical passages.
    scores = (
        0.38 * np.clip(dense, 0.0, 1.0)
        + 0.42 * lexical
        + 0.20 * entity
    )

    order = np.argsort(scores)[::-1]

    results = []
    for idx in order[:top_k]:
        results.append({
            "id": PASSAGE_IDS[idx],
            "text": PASSAGE_TEXTS[idx],
            "score": float(scores[idx]),
            "dense": float(dense[idx]),
            "lexical": float(lexical[idx]),
            "entity": float(entity[idx]),
        })

    return results, q_vector


# ============================================================
# 9. QUESTION INTENT AND FACT CHECKS
# ============================================================

def question_intent(question):
    q = question.lower()

    return {
        "port": bool(re.search(r"\bport\b", q)),
        "count": bool(re.search(
            r"\b(how many|number of|count|quorum)\b", q
        )),
        "time": bool(re.search(
            r"\b(interval|timeout|heartbeat|latency|duration|"
            r"how long|wait time|rto|lease term)\b", q
        )),
        "comparison": bool(re.search(
            r"\b(compare|comparison|contrast|difference|differences|"
            r"versus|vs\.?|both)\b", q
        )),
        "multi": bool(re.search(r"\band\b", q)),
    }


NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|seconds?|sec|s|"
    r"minutes?|bits?|bytes?|records?|nodes?|replicas?)?\b"
    r"|\b(one|two|three|four|five|six|seven|eight|nine|ten)\b",
    flags=re.IGNORECASE,
)


def sentence_has_requested_fact(sentence, intent, q_tokens):
    s = sentence.lower()

    if intent["port"]:
        if "port" not in s or not NUMBER_RE.search(sentence):
            return False

    if intent["count"]:
        count_terms = {
            "acceptor", "node", "replica", "participant",
            "quorum", "member", "server", "layer", "block",
            "sample", "retry", "round",
        }
        asks_count_about = bool(set(q_tokens) & count_terms)
        if asks_count_about and not NUMBER_RE.search(sentence):
            return False

    if intent["time"]:
        if not NUMBER_RE.search(sentence):
            return False

    return True


# ============================================================
# 10. FACTUAL SENTENCE RANKING AND ANSWER EXTRACTION
# ============================================================

def extract_answer(question, q_vector, retrieved):
    q_tokens = tokenize(question)
    anchors = extract_anchors(question)
    intent = question_intent(question)

    allowed_ids = {x["id"] for x in retrieved}
    passage_scores = {x["id"]: x["score"] for x in retrieved}

    candidates = []

    for idx, record in enumerate(sentence_records):
        pid = record["passage_id"]

        if pid not in allowed_ids:
            continue

        sentence = record["text"]
        parent = PASSAGE_BY_ID[pid]

        # Do not treat suspicious retrieved text as instructions or evidence.
        if is_untrusted_content(sentence):
            continue

        sent_dense = float(SENTENCE_EMBEDDINGS[idx] @ q_vector)
        sent_lex = token_coverage(q_tokens, sentence)
        parent_lex = token_coverage(q_tokens, parent)
        entity_score = anchor_coverage(anchors, parent)

        score = (
            0.18 * max(0.0, sent_dense)
            + 0.32 * sent_lex
            + 0.25 * parent_lex
            + 0.25 * entity_score
        )

        if not sentence_has_requested_fact(sentence, intent, q_tokens):
            score -= 0.35

        candidates.append({
            "passage_id": pid,
            "text": sentence,
            "score": score,
            "dense": sent_dense,
            "sentence_lexical": sent_lex,
            "parent_lexical": parent_lex,
            "entity": entity_score,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    if not candidates:
        return None, 0.0, []

    # Keep only candidates that contain the requested fact type.
    valid = [
        c for c in candidates
        if sentence_has_requested_fact(c["text"], intent, q_tokens)
    ]

    if not valid:
        return None, candidates[0]["score"], []

    best = valid[0]

    # Avoid answering off-topic or weakly grounded questions.
    if best["dense"] < 0.28:
        return None, best["score"], []

    if best["sentence_lexical"] == 0 and best["entity"] == 0:
        return None, best["score"], []

    selected = [best]

    # Synthesize only when the question explicitly asks for a comparison,
    # multiple facts, or contains a clear conjunction.
    needs_multiple = intent["comparison"] or intent["multi"]

    if needs_multiple:
        used = {best["passage_id"]}

        for candidate in valid[1:]:
            if len(selected) >= 3:
                break

            if candidate["score"] < best["score"] - 0.20:
                continue

            # Prefer a different passage for cross-boundary synthesis,
            # but allow a second sentence from the same passage if needed.
            if candidate["passage_id"] in used:
                continue

            if candidate["dense"] < 0.28:
                continue

            selected.append(candidate)
            used.add(candidate["passage_id"])

    answer = " ".join(item["text"] for item in selected)
    return answer, best["score"], selected


# ============================================================
# 11. RESPONSE HELPERS
# ============================================================

def abstain():
    return {
        "answer": None,
        "sources": [],
        "scores": [],
        "confidence": 0.20,
    }


# ============================================================
# 12. MAIN ENDPOINT
# ============================================================

@app.post("/query")
def query_api(request: QueryRequest):
    original_question = request.question.strip()

    if not original_question:
        return abstain()

    had_injection = contains_injection(original_question)
    question = sanitize_question(original_question)

    # Pure injection / no remaining factual query: abstain.
    if not question or not tokenize(question):
        return abstain()

    top_k = max(1, min(request.top_k, 10))

    retrieved, q_vector = retrieve(question, top_k)

    answer, answer_score, selected = extract_answer(
        question,
        q_vector,
        retrieved,
    )

    if answer is None or not selected:
        return abstain()

    # Cite only passages that contributed selected answer sentences.
    source_ids = list(dict.fromkeys(
        item["passage_id"] for item in selected
    ))

    score_lookup = {x["id"]: x["score"] for x in retrieved}

    source_ids.sort(
        key=lambda pid: score_lookup.get(pid, 0.0),
        reverse=True,
    )

    source_ids = source_ids[:top_k]

    scores = [
        round(
            max(0.0, min(1.0, score_lookup[pid])),
            4,
        )
        for pid in source_ids
    ]

    # This is a bounded ranking score, not a calibrated probability.
    confidence = round(
        max(0.61, min(0.99, float(answer_score))),
        4,
    )

    return {
        "answer": answer,
        "sources": source_ids,
        "scores": scores,
        "confidence": confidence,
    }


# ============================================================
# 13. HEALTH AND ROOT ROUTES
# ============================================================

@app.get("/")
@app.head("/")
def home():
    return {
        "status": "running",
        "passages_loaded": len(passages),
    }


@app.get("/health")
@app.head("/health")
def health():
    return {
        "status": "healthy",
        "passages_loaded": len(passages),
    }
