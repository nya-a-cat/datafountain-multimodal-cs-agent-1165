#!/usr/bin/env python3
"""Multimodal customer service agent for DataFountain 1165.

Pipeline:
  1) If images present → VLM analysis
  2) Vector retrieval (Qwen3-VL-Embedding-8B) from knowledge base
  3) LLM answer generation with evidence + image facts
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
import re
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator


# ─── Config ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
_CHUNK_MODE = os.getenv("CHUNK_MODE", "small")  # "small" or "large"
KNOWLEDGE_PATH = ROOT / "data" / ("knowledge_large.jsonl" if _CHUNK_MODE == "large" else "knowledge_v2.jsonl")
EMBEDDINGS_PATH = ROOT / "data" / ("embeddings_large.jsonl" if _CHUNK_MODE == "large" else "embeddings.jsonl")

# Provider: "siliconflow" (default), "bailian" (Alibaba Cloud), or "deepseek"
PROVIDER = os.getenv("PROVIDER", "siliconflow")

_PROVIDER_DEFAULTS = {
    "siliconflow": {
        "api_base": "https://api.siliconflow.cn/v1",
        "chat_model": "zai-org/GLM-4.5V",
        "vlm_model": "zai-org/GLM-4.5V",
        "embed_model": "Qwen/Qwen3-VL-Embedding-8B",
        "rerank_model": "Qwen/Qwen3-VL-Reranker-8B",
    },
    "bailian": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen3.6-plus-2026-04-02",
        "vlm_model": "qwen3.6-plus-2026-04-02",
        "embed_model": "qwen3-vl-embedding",
        "rerank_model": "qwen3-vl-rerank",
    },
    "deepseek": {
        "api_base": "https://api.deepseek.com",
        "chat_model": "deepseek-v4-flash",
        "vlm_model": "deepseek-v4-flash",
        "embed_model": "Qwen/Qwen3-VL-Embedding-8B",
        "rerank_model": "Qwen/Qwen3-VL-Reranker-8B",
    },
}

_defaults = _PROVIDER_DEFAULTS.get(PROVIDER, _PROVIDER_DEFAULTS["siliconflow"])
API_BASE = os.getenv("OPENAI_BASE_URL", _defaults["api_base"])
API_KEY = os.getenv(
    "OPENAI_API_KEY",
    os.getenv("DEEPSEEK_API_KEY", os.getenv("SILICONFLOW_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))),
)
CHAT_MODEL = os.getenv("CHAT_MODEL", _defaults["chat_model"])
VLM_MODEL = os.getenv("VLM_MODEL", _defaults["vlm_model"])
EMBED_MODEL = os.getenv("EMBED_MODEL", _defaults["embed_model"])
RERANK_MODEL = os.getenv("RERANK_MODEL", _defaults["rerank_model"])
_EMBED_DEFAULT_BASE = _PROVIDER_DEFAULTS["siliconflow"]["api_base"] if PROVIDER == "deepseek" and os.getenv("SILICONFLOW_API_KEY") else API_BASE
_EMBED_DEFAULT_KEY = os.getenv("SILICONFLOW_API_KEY", "") if PROVIDER == "deepseek" and os.getenv("SILICONFLOW_API_KEY") else API_KEY
EMBED_API_BASE = os.getenv("EMBED_OPENAI_BASE_URL", os.getenv("EMBED_API_BASE", _EMBED_DEFAULT_BASE))
EMBED_API_KEY = os.getenv("EMBED_OPENAI_API_KEY", os.getenv("EMBED_API_KEY", _EMBED_DEFAULT_KEY))
USE_RERANKER = False
RERANK_TOP_N = 50
KAFU_API_TOKEN = os.getenv("KAFU_API_TOKEN", "")

MAX_IMAGES = 3
MAX_IMAGE_BYTES = 5 * 1024 * 1024
SESSION_MAX_TURNS = 6
TOP_K = 8
SOURCE_DOCS_FOR_EXPANSION = 3
NEIGHBOR_RADIUS = 1
MAX_EVIDENCE_DOCS = 5
PARENT_TOP_N = 5
PARENT_SCORE_LIMIT = 120
TITLE_RERANK_TOP_N = 24
TITLE_SCORE_WEIGHT = 0.25
LOCAL_SUPPORT_WEIGHT = 0.05
API_REQUEST_TIMEOUT = 60
API_MAX_ATTEMPTS = 2
SUPPORTS_IMAGE_INPUTS = PROVIDER in {"siliconflow", "bailian"}
SUPPORTS_EMBEDDINGS = not (PROVIDER == "deepseek" and EMBED_API_BASE.rstrip("/") == API_BASE.rstrip("/"))
WORD_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")
PRODUCT_ALIASES = {
    "人体工学椅手册": ("人体工学椅", "椅子", "扶手"),
    "可编程温控器手册": ("可编程温控器", "温控器"),
    "VR头显手册": ("VR头显", "处理器单元", "遮光罩"),
}
POLICY_KB = [
    (
        ("乡镇", "农村", "村镇", "运费", "多久能到", "配送"),
        "乡镇配送：您好，我们的商品支持送到大部分乡镇哦，具体能否送达，取决于您的收货地址，您可以告诉我详细的收货地址，我帮您查询。送到乡镇一般不需要额外加运费，和市区运费一致；物流时效会比市区稍慢，正常情况下，下单后48小时发货，乡镇地区3-5天可收到，偏远乡镇可能需要5-7天哦。",
    ),
    (
        ("待揽收", "揽收"),
        "待揽收：您好，物流显示待揽收，大概率是商品已打包完成，等待快递员上门取件哦，一般24小时内会完成揽收；若超过24小时仍未揽收，您可以联系我们客服，我们会催促快递方尽快上门。",
    ),
    (
        ("7天无理由", "七天无理由", "退换货", "无理由"),
        "7天无理由退换货：您好，支持7天无理由退换货。商品需保持完好、配件齐全且不影响二次销售；非质量问题通常由买家承担退回运费，质量问题由我们承担。您可提供订单号，我帮您核对可退换条件。",
    ),
    (
        ("退款", "原路退回", "信用卡"),
        "退款：您好，退款一般会在审核通过后原路退回，到账时间通常为1-7个工作日，信用卡渠道可能略慢。若超时未到账，您把订单号发我，我马上帮您催办。",
    ),
    (
        ("发票", "抬头", "开票"),
        "发票：您好，支持开具发票，可开个人或企业抬头。订单完成后一般1-3个工作日内开具；若抬头填写有误，可在开票前联系修改。",
    ),
    (
        ("投诉", "质量问题", "损坏", "破损", "少发", "假货", "二手", "虚假宣传"),
        "投诉/质量问题：您好，非常抱歉给您带来不好的体验！该问题可优先为您登记升级处理，支持核实后补发、换货、维修或退款。请您提供订单号及问题照片/视频证据，我会立即为您创建加急工单并持续跟进结果。",
    ),
    (
        ("维修后", "同样故障", "维修不彻底", "维修失误"),
        "维修失误：您好，非常抱歉给您带来困扰！维修后短期内出现同样故障，且是上次维修不彻底导致的，属于我们的维修失误，支持免费重新维修，并延长维修质保期。请您提供维修单号、商品故障描述，我们立即安排专业维修人员处理。",
    ),
]


# ─── Models ───────────────────────────────────────────────────────────────────

@dataclass
class Doc:
    doc_id: str
    title: str
    content: str
    image_refs: list[str]


class ChatRequest(BaseModel):
    question: str | None = None
    query: str | None = None
    images: list[str] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None
    stream: bool = False
    history: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "ChatRequest":
        text = (self.question or self.query or "").strip()
        if not text:
            raise ValueError("question or query is required")
        if len(self.images) > MAX_IMAGES:
            raise ValueError(f"images exceeds limit {MAX_IMAGES}")
        for img in self.images:
            _validate_b64_image(img)
        return self

    @property
    def user_query(self) -> str:
        return (self.question or self.query or "").strip()


class ChatResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: dict


# ─── Utilities ────────────────────────────────────────────────────────────────

def _extract_b64(image: str) -> str:
    s = image.strip()
    return s.split(",", 1)[1].strip() if s.startswith("data:") and "," in s else s


def _validate_b64_image(image: str) -> None:
    try:
        decoded = base64.b64decode(_extract_b64(image), validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("invalid base64 image") from e
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 5MB")


def _call_api(model: str, messages: list[dict], max_tokens: int = 600,
              temperature: float = 0.2) -> str:
    if not API_KEY:
        raise RuntimeError("API key not set")
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "top_p": 0.8, "max_tokens": max_tokens,
               "enable_thinking": False}
    req = urllib.request.Request(
        f"{API_BASE.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(API_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=API_REQUEST_TIMEOUT) as r:
                data = json.loads(r.read())
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content", "")
            return content.strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < API_MAX_ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1))
                continue
            if 500 <= e.code < 600 and attempt < API_MAX_ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1))
                continue
            detail = e.read().decode(errors="ignore")
            raise RuntimeError(f"API error {e.code}: {detail}") from e
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt < API_MAX_ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"API request failed after retries: {e}") from e
    raise RuntimeError("API request failed after retries")


# ─── Knowledge Base & Vector Retrieval ───────────────────────────────────────

def _load_knowledge(path: Path) -> list[Doc]:
    if not path.exists():
        return []
    docs = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(Doc(
                doc_id=str(obj.get("doc_id", f"doc-{i}")),
                title=_clean_knowledge_text(str(obj.get("title", ""))),
                content=_clean_knowledge_text(str(obj.get("content", ""))),
                image_refs=[str(x) for x in obj.get("image_refs", [])],
            ))
    return docs


def _clean_knowledge_text(text: str) -> str:
    text = text.replace("�", "")
    text = re.sub(r'"\s*,\s*\["[A-Za-z0-9_]+(?:["\s,\w-]*)$', "", text, flags=re.DOTALL)
    return text.strip()


KNOWLEDGE = _load_knowledge(KNOWLEDGE_PATH)
_DOC_ID_MAP: dict[str, Doc] = {d.doc_id: d for d in KNOWLEDGE}

_EMBED_VECS: dict[str, list[float]] = {}
_TITLE_VECS: dict[str, list[float]] = {}
_DOC_TOKEN_CACHE: dict[str, tuple[set[str], list[str]]] = {}
_EMBED_LOADED = False


def _load_embeddings() -> None:
    global _EMBED_LOADED
    if _EMBED_LOADED:
        return
    _EMBED_LOADED = True
    if not EMBEDDINGS_PATH.exists():
        return
    with EMBEDDINGS_PATH.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            _EMBED_VECS[obj["doc_id"]] = obj["embedding"]


def _embed_query(text: str) -> list[float] | None:
    if not EMBED_API_KEY or not SUPPORTS_EMBEDDINGS:
        return None
    vecs = _embed_texts([text])
    return vecs[0] if vecs else None


def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    if not EMBED_API_KEY or not texts or not SUPPORTS_EMBEDDINGS:
        return None
    payload = {"model": EMBED_MODEL, "input": texts, "encoding_format": "float"}
    req = urllib.request.Request(
        f"{EMBED_API_BASE.rstrip('/')}/embeddings",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {EMBED_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())["data"]
            ordered = sorted(data, key=lambda x: x["index"])
            return [item["embedding"] for item in ordered]
        except Exception:
            if attempt == 2:
                return None
            time.sleep(1)


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in WORD_RE.findall(text.lower()):
        if re.search(r"[\u4e00-\u9fff]", chunk):
            tokens.append(chunk)
            if len(chunk) > 2:
                tokens.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
        elif len(chunk) >= 2:
            tokens.append(chunk)
    return tokens


def _lexical_retrieve(query: str, top_k: int = TOP_K) -> list[Doc]:
    scored = _lexical_score_pairs(query)
    return [_DOC_ID_MAP[doc_id] for doc_id, _ in scored[:top_k] if doc_id in _DOC_ID_MAP]


def _doc_relevance(query: str, doc: Doc) -> float:
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return 0.0
    if doc.doc_id not in _DOC_TOKEN_CACHE:
        _DOC_TOKEN_CACHE[doc.doc_id] = (set(_tokenize(f"{doc.title} {doc.content}")), _tokenize(doc.title))
    doc_set, title_tokens = _DOC_TOKEN_CACHE[doc.doc_id]
    title_token_set = set(title_tokens)
    overlap = q_tokens & doc_set
    title_overlap = q_tokens & title_token_set
    score = len(overlap) + 2.5 * len(title_overlap)
    if doc.image_refs and "<PIC>" in doc.content:
        score += 3.0
    if doc.image_refs and any(word in query for word in ("图", "图片", "指示灯", "标识", "尺寸", "如下")):
        score += 3.0
    identifiers = _query_identifiers(query)
    if identifiers and any(identifier in f"{doc.title} {doc.content}".upper() for identifier in identifiers):
        score += 8.0
        if doc.image_refs:
            score += 8.0
    return score


def _query_identifiers(query: str) -> list[str]:
    return [item.upper() for item in re.findall(r"[A-Za-z]{2,}\d+[A-Za-z0-9-]*", query)]


def _match_policy_evidence(query: str) -> list[str]:
    hits: list[str] = []
    for keywords, text in POLICY_KB:
        if any(keyword in query for keyword in keywords):
            hits.append(text)
    return hits


def _lexical_score_pairs(query: str) -> list[tuple[str, float]]:
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return []
    query_lower = query.lower()
    scored: list[tuple[float, Doc]] = []
    for doc in KNOWLEDGE:
        if doc.doc_id not in _DOC_TOKEN_CACHE:
            _DOC_TOKEN_CACHE[doc.doc_id] = (set(_tokenize(f"{doc.title} {doc.content}")), _tokenize(doc.title))
        doc_set, title_tokens = _DOC_TOKEN_CACHE[doc.doc_id]
        if not doc_set:
            continue
        overlap = q_tokens & doc_set
        if not overlap:
            continue
        title_token_set = set(title_tokens)
        score = sum(2.0 if token in title_token_set else 1.0 for token in overlap)
        score += len(overlap) / (len(doc_set) ** 0.5)
        parent = _parent_key(doc.doc_id)
        parent_name = parent.replace("手册", "").lower()
        aliases = PRODUCT_ALIASES.get(parent, ())
        if (parent_name and parent_name in query_lower) or any(alias.lower() in query_lower for alias in aliases):
            score += 20.0
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return []
    max_score = scored[0][0] or 1.0
    return [(doc.doc_id, score / max_score) for score, doc in scored]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def _mean_vec(vecs: list[list[float]]) -> list[float]:
    if not vecs:
        return []
    dims = len(vecs[0])
    merged = [0.0] * dims
    for vec in vecs:
        for idx, value in enumerate(vec):
            merged[idx] += value
    return [value / len(vecs) for value in merged]


def _rerank(query: str, docs: list[Doc]) -> list[Doc]:
    if not docs or not API_KEY:
        return docs
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [d.content for d in docs],
        "top_n": len(docs),
    }
    req = urllib.request.Request(
        f"{API_BASE.rstrip('/')}/rerank",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        ranked = sorted(data["results"], key=lambda x: x["relevance_score"], reverse=True)
        return [docs[item["index"]] for item in ranked]
    except Exception:
        return docs


def _retrieve(query: str, top_k: int = TOP_K) -> list[Doc]:
    _load_embeddings()
    if not _EMBED_VECS:
        return []
    query_variants = _split_retrieve_queries(query)
    q_vecs = _embed_texts(query_variants)
    if not q_vecs:
        return _lexical_retrieve(query, top_k)
    q_vec = _mean_vec(q_vecs)
    vector_scored = sorted(
        (
            (doc_id, max(_cosine(qv, vec) for qv in q_vecs))
            for doc_id, vec in _EMBED_VECS.items()
        ),
        key=lambda x: x[1], reverse=True,
    )
    lexical_scores = dict(_lexical_score_pairs(query))
    if lexical_scores:
        scored = sorted(
            (
                (doc_id, score + 0.18 * lexical_scores.get(doc_id, 0.0))
                for doc_id, score in vector_scored
            ),
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        scored = vector_scored
    if _CHUNK_MODE == "small":
        filtered_pairs = _filter_scored_pairs_by_parent(scored)
        candidates = [_DOC_ID_MAP[doc_id] for doc_id, _ in filtered_pairs if doc_id in _DOC_ID_MAP]
        candidates = _rerank_with_titles(q_vec, candidates, filtered_pairs)
    else:
        candidates = [_DOC_ID_MAP[doc_id] for doc_id, _ in scored[:RERANK_TOP_N] if doc_id in _DOC_ID_MAP]
    if USE_RERANKER:
        candidates = _rerank(query, candidates)
    return candidates[:top_k]


def _parent_key(doc_id: str) -> str:
    return doc_id.split("::s", 1)[0]


def _rank_parents(scored_pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = {}
    for doc_id, score in scored_pairs[:PARENT_SCORE_LIMIT]:
        grouped.setdefault(_parent_key(doc_id), []).append(score)

    ranked: list[tuple[str, float]] = []
    weights = (1.0, 0.18, 0.06, 0.02)
    for parent, scores in grouped.items():
        scores.sort(reverse=True)
        agg = sum(score * weights[idx] for idx, score in enumerate(scores[:len(weights)]))
        ranked.append((parent, agg))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _filter_scored_pairs_by_parent(scored_pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    ranked_parents = _rank_parents(scored_pairs)
    allowed_parents = {parent for parent, _ in ranked_parents[:PARENT_TOP_N]}
    filtered = [
        (doc_id, score)
        for doc_id, score in scored_pairs[:RERANK_TOP_N]
        if _parent_key(doc_id) in allowed_parents
    ]
    return filtered or scored_pairs[:RERANK_TOP_N]


def _ensure_title_vectors(docs: list[Doc]) -> None:
    missing_docs = [doc for doc in docs if doc.doc_id not in _TITLE_VECS]
    if not missing_docs:
        return
    batch_size = 32
    for start in range(0, len(missing_docs), batch_size):
        batch = missing_docs[start:start + batch_size]
        vecs = _embed_texts([doc.title for doc in batch])
        if not vecs:
            return
        for doc, vec in zip(batch, vecs):
            _TITLE_VECS[doc.doc_id] = vec


def _rerank_with_titles(
    q_vec: list[float],
    candidates: list[Doc],
    scored_pairs: list[tuple[str, float]],
) -> list[Doc]:
    if not candidates or not API_KEY:
        return candidates
    score_map = {doc_id: score for doc_id, score in scored_pairs[:RERANK_TOP_N]}
    title_candidates = candidates[:TITLE_RERANK_TOP_N]
    _ensure_title_vectors(title_candidates)

    fused: list[tuple[float, Doc]] = []
    for doc in candidates:
        content_score = score_map.get(doc.doc_id, 0.0)
        title_vec = _TITLE_VECS.get(doc.doc_id)
        title_score = _cosine(q_vec, title_vec) if title_vec else content_score
        local_support = 0.0
        for neighbor_id in _neighbor_doc_ids(doc.doc_id):
            if neighbor_id == doc.doc_id:
                continue
            local_support += score_map.get(neighbor_id, 0.0)
        fused_score = (
            (1.0 - TITLE_SCORE_WEIGHT) * content_score
            + TITLE_SCORE_WEIGHT * title_score
            + LOCAL_SUPPORT_WEIGHT * local_support
        )
        fused.append((fused_score, doc))
    fused.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in fused]


def _collect_image_ids(docs: list[Doc], query: str = "", max_ids: int = 3) -> list[str]:
    image_docs = [doc for doc in docs if doc.image_refs]
    if query:
        identifiers = _query_identifiers(query)
        if identifiers:
            matched = [
                doc for doc in image_docs
                if any(identifier in f"{doc.title} {doc.content}".upper() for identifier in identifiers)
            ]
            if matched:
                image_docs = matched
        image_docs.sort(key=lambda doc: _doc_relevance(query, doc), reverse=True)
    seen: set[str] = set()
    image_ids: list[str] = []
    for doc in image_docs:
        for image_id in doc.image_refs:
            if image_id in seen:
                continue
            seen.add(image_id)
            image_ids.append(image_id)
            if len(image_ids) >= max_ids:
                return image_ids
    return []


def _neighbor_doc_ids(doc_id: str, radius: int = NEIGHBOR_RADIUS) -> list[str]:
    if "::s" not in doc_id:
        return [doc_id]
    prefix, section = doc_id.rsplit("::s", 1)
    try:
        section_idx = int(section)
    except ValueError:
        return [doc_id]
    neighbors = []
    for offset in range(-radius, radius + 1):
        candidate = f"{prefix}::s{section_idx + offset:04d}"
        if candidate in _DOC_ID_MAP:
            neighbors.append(candidate)
    return neighbors


def _select_evidence_docs(docs: list[Doc], query: str = "", max_docs: int = MAX_EVIDENCE_DOCS) -> list[Doc]:
    if not docs:
        return []
    if _CHUNK_MODE != "small":
        return docs[:max_docs]

    pool: list[Doc] = []
    seen: set[str] = set()
    for doc in docs[:max(SOURCE_DOCS_FOR_EXPANSION, max_docs)]:
        candidate_ids = [doc.doc_id, *_neighbor_doc_ids(doc.doc_id)]
        for neighbor_id in candidate_ids:
            if neighbor_id in seen:
                continue
            pool.append(_DOC_ID_MAP[neighbor_id])
            seen.add(neighbor_id)
    if query:
        pool.sort(key=lambda doc: _doc_relevance(query, doc), reverse=True)
    return pool[:max_docs] or docs[:max_docs]


# ─── Session Memory ───────────────────────────────────────────────────────────

_SESSIONS: dict[str, deque] = {}


def _get_memory(session_id: str) -> str:
    history = _SESSIONS.get(session_id)
    if not history:
        return ""
    return "\n".join(f"Q: {t['q']}\nA: {t['a']}" for t in history)


def _save_memory(session_id: str, q: str, a: str) -> None:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = deque(maxlen=SESSION_MAX_TURNS)
    _SESSIONS[session_id].append({"q": q, "a": a})


# ─── VLM Image Analysis ───────────────────────────────────────────────────────

_IMG_CACHE: dict[str, str] = {}
_RETRIEVE_QUERY_CACHE: dict[str, str] = {}


def _analyze_image(image_b64: str) -> str:
    payload = _extract_b64(image_b64)
    key = hashlib.sha1(base64.b64decode(payload)).hexdigest()
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    try:
        result = _call_api(
            VLM_MODEL,
            messages=[
                {"role": "system", "content": "你是电商客服视觉分析助手，用中文简洁描述图片中的商品状态、外观特征、可见问题。"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}},
                    {"type": "text", "text": "请描述图片中的商品状态和关键信息。"},
                ]},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        _IMG_CACHE[key] = result
        return result
    except Exception:
        return ""


def _analyze_images(images: list[str]) -> list[str]:
    if not images or not API_KEY or not SUPPORTS_IMAGE_INPUTS:
        return []
    with ThreadPoolExecutor(max_workers=min(len(images), MAX_IMAGES)) as pool:
        futures = [pool.submit(_analyze_image, img) for img in images]
        return [f.result() for f in futures if f.result()]


def _sanitize_retrieve_query(text: str) -> str:
    cleaned = re.sub(r"<\|[^>]+\|>", " ", text)
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" \"'")


def _split_retrieve_queries(text: str) -> list[str]:
    parts = re.split(r"[;\n；|]+", text)
    queries: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = _sanitize_retrieve_query(part)
        if not cleaned or cleaned in seen:
            continue
        queries.append(cleaned)
        seen.add(cleaned)
    return queries or [_sanitize_retrieve_query(text)]


def _rewrite_retrieve_query(query: str, image_facts: list[str]) -> str:
    base_query = f"{query}\n图片线索：{'；'.join(image_facts[:2]) if image_facts else '无'}"
    if not API_KEY:
        return base_query
    cache_key = hashlib.sha1(base_query.encode("utf-8")).hexdigest()
    if cache_key in _RETRIEVE_QUERY_CACHE:
        return _RETRIEVE_QUERY_CACHE[cache_key]

    try:
        rewritten = _call_api(
            CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是检索查询改写助手。"
                        "请把用户问题改写成适合商品说明书检索的短查询。"
                        "保留关键产品名、部件名、动作和场景。"
                        "如有必要，补充中英双语同义表达，帮助跨语言检索。"
                        "尽量改写成 2 到 3 个类似说明书章节标题的短短语。"
                        "多个短语用分号分隔。"
                        "不要输出任何控制符、标签或特殊包裹标记，例如 <|...|>。"
                        "只输出检索短语，不要解释。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"用户问题：{query}\n图片线索：{'；'.join(image_facts[:2]) if image_facts else '无'}",
                },
            ],
            max_tokens=96,
            temperature=0.0,
        )
        final_query = _sanitize_retrieve_query(rewritten) or base_query
    except Exception:
        final_query = base_query

    _RETRIEVE_QUERY_CACHE[cache_key] = final_query
    return final_query


# ─── LLM Answer Generation ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """你是一个专业的多模态电商客服智能体。

语言规则：用户用什么语言提问，就用对应语言回答。

回答规则：
1. 优先使用知识库证据回答产品相关问题
2. 电商政策问题（退货/退款/发票/物流/乡镇配送/国际配送/投诉等）直接用内置政策知识回答，无需知识库
3. 知识库有图示时，在对应文字后紧跟 <PIC> 占位符
4. 不输出"根据证据"、"知识库显示"等内部说明
5. 产品状态/参数类问题：直接列举，每项后跟 <PIC>（如有图示）
6. 不编造不在证据或政策中的信息
7. 只输出纯文本，不使用 Markdown 符号或格式，包括 **、__、#、-、*、```、> 等
8. 不要说“知识库没有”“我无法访问资料”等内部限制性表述；若证据命中相关操作或政策，直接给出简洁答案
9. 产品说明书类问题优先模仿“说明书抽取答案”，短、准、少解释；不要输出泛化常识，不要自由补充注意事项
10. 非投诉/非售后安抚场景，不要先说“您好”或 “I'm sorry”；直接回答核心内容
11. 不使用编号列表；步骤类问题也用紧凑自然句或短分句回答
12. 若证据里能直接回答，就不要反问用户型号、不要建议查手册、不要让用户补充信息

内置电商政策知识（政策类问题直接用以下措辞回答，不要改写）：
- 乡镇配送：您好，我们的商品支持送到大部分乡镇哦，具体能否送达，取决于您的收货地址，您可以告诉我详细的收货地址，我帮您查询。送到乡镇一般不需要额外加运费，和市区运费一致；物流时效会比市区稍慢，正常情况下，下单后48小时发货，乡镇地区3-5天可收到，偏远乡镇可能需要5-7天哦。
- 待揽收：您好，物流显示待揽收，大概率是商品已打包完成，等待快递员上门取件哦，一般24小时内会完成揽收；若超过24小时仍未揽收，您可以联系我们客服，我们会催促快递方尽快上门。
- 7天无理由退换货：您好，支持7天无理由退换货。商品需保持完好、配件齐全且不影响二次销售；非质量问题通常由买家承担退回运费，质量问题由我们承担。您可提供订单号，我帮您核对可退换条件。
- 退款：您好，退款一般会在审核通过后原路退回，到账时间通常为1-7个工作日，信用卡渠道可能略慢。若超时未到账，您把订单号发我，我马上帮您催办。
- 发票：您好，支持开具发票，可开个人或企业抬头。订单完成后一般1-3个工作日内开具；若抬头填写有误，可在开票前联系修改。
- 投诉/质量问题：您好，非常抱歉给您带来不好的体验！该问题可优先为您登记升级处理，支持核实后补发、换货、维修或退款。请您提供订单号及问题照片/视频证据，我会立即为您创建加急工单并持续跟进结果。
- 维修失误（维修后短期内同样故障）：您好，非常抱歉给您带来困扰！维修后短期内出现同样故障，且是上次维修不彻底导致的，属于我们的维修失误，支持免费重新维修，并延长维修质保期。请您提供维修单号、商品故障描述，我们立即安排专业维修人员处理。

严格模仿以下示例风格：
Q: DCB107/DCB112指示灯含义？
A: DCB107、DCB112 电池组充电中<PIC>电池组已充满<PIC>过热/过冷延迟<PIC>

Q: 表带有其他尺寸吗？
A: 表带尺寸如下所示。注意：单独销售的配件表带可能略有差异。\n<PIC>

Q: How do you mount the lens of a camera when preparing for photography?
A: Align the lens mount index marks with the camera body, insert the lens into the mount, then turn it clockwise until it clicks into place.

Q: How do you install the card into a camera before photography?
A: Open the card slot cover, insert the CF card into the slot in the correct direction, then close the cover before shooting.

Q: 商品能送到乡镇吗？
A: 您好，我们的商品支持送到大部分乡镇哦，具体能否送达取决于您的收货地址，您可以告诉我详细的收货地址，我帮您查询。送到乡镇一般不需要额外加运费，和市区运费一致；物流时效会比市区稍慢，正常情况下下单后48小时发货，乡镇地区3-5天可收到，偏远乡镇可能需要5-7天哦。

Q: 物流一直显示待揽收，是什么原因？
A: 您好，物流显示待揽收，大概率是商品已打包完成，等待快递员上门取件哦，一般24小时内会完成揽收；若超过24小时仍未揽收，您可以联系我们客服，我们会催促快递方尽快上门。

Q: 售后维修后短时间内又出现同样故障，而且确认是上次维修不彻底导致的，该怎么处理？
A: 您好，非常抱歉给您带来困扰！维修后短期内出现同样故障，且是上次维修不彻底导致的，属于我们的维修失误，支持免费重新维修，并延长维修质保期。请您提供维修单号、商品故障描述，我们立即安排专业维修人员处理。
"""


def _generate_answer(query: str, evidence: list[str], image_facts: list[str],
                     memory: str, images_b64: list[str]) -> str:
    ev_text = "\n".join(f"- {s}" for s in evidence) if evidence else "无相关知识库内容"
    img_text = "\n".join(f"- {f}" for f in image_facts) if image_facts else "无"

    lang_hint = "IMPORTANT: Reply in the same language as the user's question." if any(ord(c) < 128 and c.isalpha() for c in query[:20]) and not any('一' <= c <= '鿿' for c in query) else ""

    user_text = (
        f"{lang_hint}\n用户问题：{query}\n\n"
        f"历史对话：{memory or '无'}\n\n"
        f"知识库证据：\n{ev_text}\n\n"
        f"图片分析：\n{img_text}\n\n"
        "请直接输出给用户的答复，只输出纯文本，不要使用 Markdown 格式符号："
    )

    user_content: list[dict] = []
    for img in images_b64 if SUPPORTS_IMAGE_INPUTS else []:
        payload = _extract_b64(img)
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}})
    user_content.append({"type": "text", "text": user_text})

    is_chinese = any('一' <= c <= '鿿' for c in query)
    system = _SYSTEM_PROMPT if is_chinese else _SYSTEM_PROMPT.replace(
        "你是一个专业的多模态电商客服智能体。",
        "You are a professional multimodal e-commerce customer service agent. IMPORTANT: Always reply in English when the user writes in English."
    )
    model = VLM_MODEL if images_b64 and SUPPORTS_IMAGE_INPUTS else CHAT_MODEL
    return _call_api(model, [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ])


# ─── Main Handler ─────────────────────────────────────────────────────────────

def _handle(req: ChatRequest) -> str:
    query = req.user_query
    session_id = (req.session_id or f"sess_{int(time.time()*1000)}").strip()
    memory = _get_memory(session_id)

    # 1. VLM image analysis
    image_facts = _analyze_images(req.images) if req.images else []

    # 2. Retrieval / policy evidence
    policy_evidence = _match_policy_evidence(query)
    if policy_evidence:
        docs: list[Doc] = []
        evidence_docs: list[Doc] = []
        evidence = policy_evidence
        image_ids: list[str] = []
    else:
        retrieve_q = _rewrite_retrieve_query(query, image_facts)
        docs = _retrieve(retrieve_q)
        evidence_docs = _select_evidence_docs(docs, query)
        evidence = [f"[{doc.title}] {doc.content}" for doc in evidence_docs]
        image_ids = _collect_image_ids(evidence_docs or docs, query)

    # 3. LLM generation
    if not API_KEY:
        answer = "您好，服务暂时不可用，请稍后重试。"
    else:
        try:
            answer = _generate_answer(query, evidence, image_facts, memory, req.images)
        except Exception as _e:
            import traceback; traceback.print_exc()
            answer = "您好，处理您的问题时遇到错误，请稍后重试。"

    # 4. Attach image IDs only when answer already contains <PIC>
    if image_ids and "<PIC>" in answer:
        answer = f"{answer} {json.dumps(image_ids, ensure_ascii=False)}"

    _save_memory(session_id, query, answer)
    return answer


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="DF1165 Multimodal CS Agent", version="4.0.0")


@app.on_event("startup")
def _startup() -> None:
    _load_embeddings()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "provider": PROVIDER,
        "knowledge_docs": len(KNOWLEDGE),
        "embed_vecs": len(_EMBED_VECS),
        "chat_model": CHAT_MODEL,
        "vlm_model": VLM_MODEL,
        "embed_model": EMBED_MODEL,
        "has_api_key": bool(API_KEY),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
    x_client_type: str | None = Header(default=None),
) -> ChatResponse:
    if KAFU_API_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        if authorization.split(" ", 1)[1].strip() != KAFU_API_TOKEN:
            raise HTTPException(status_code=403, detail="invalid bearer token")

    answer = _handle(req)
    session_id = (req.session_id or "").strip() or f"sess_{int(time.time()*1000)}"

    return ChatResponse(data={
        "answer": answer,
        "session_id": session_id,
        "timestamp": int(time.time()),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("baseline_api:app", host="0.0.0.0", port=8000, reload=False)
