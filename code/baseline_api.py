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
KNOWLEDGE_PATH = ROOT / "data" / "knowledge.jsonl"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings.jsonl"

# Provider: "siliconflow" (default) or "bailian" (Alibaba Cloud)
PROVIDER = os.getenv("PROVIDER", "siliconflow")

_PROVIDER_DEFAULTS = {
    "siliconflow": {
        "api_base": "https://api.siliconflow.cn/v1",
        "chat_model": "Qwen/Qwen3.6-35B-A3B",
        "vlm_model": "Qwen/Qwen3.6-35B-A3B",
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
}

_defaults = _PROVIDER_DEFAULTS.get(PROVIDER, _PROVIDER_DEFAULTS["siliconflow"])
API_BASE = os.getenv("OPENAI_BASE_URL", _defaults["api_base"])
API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("SILICONFLOW_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")))
CHAT_MODEL = os.getenv("CHAT_MODEL", _defaults["chat_model"])
VLM_MODEL = os.getenv("VLM_MODEL", _defaults["vlm_model"])
EMBED_MODEL = os.getenv("EMBED_MODEL", _defaults["embed_model"])
RERANK_MODEL = os.getenv("RERANK_MODEL", _defaults["rerank_model"])
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"
RERANK_TOP_N = 50
KAFU_API_TOKEN = os.getenv("KAFU_API_TOKEN", "")

MAX_IMAGES = 3
MAX_IMAGE_BYTES = 5 * 1024 * 1024
SESSION_MAX_TURNS = 6
TOP_K = 6


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
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content", "")
            return content.strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            detail = e.read().decode(errors="ignore")
            raise RuntimeError(f"API error {e.code}: {detail}") from e
    raise RuntimeError("API rate limit exceeded after retries")


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
                title=str(obj.get("title", "")),
                content=str(obj.get("content", "")),
                image_refs=[str(x) for x in obj.get("image_refs", [])],
            ))
    return docs


KNOWLEDGE = _load_knowledge(KNOWLEDGE_PATH)
_DOC_ID_MAP: dict[str, Doc] = {d.doc_id: d for d in KNOWLEDGE}

_EMBED_VECS: dict[str, list[float]] = {}
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
    if not API_KEY:
        return None
    payload = {"model": EMBED_MODEL, "input": [text], "encoding_format": "float"}
    req = urllib.request.Request(
        f"{API_BASE.rstrip('/')}/embeddings",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())["data"][0]["embedding"]
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


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
    q_vec = _embed_query(query)
    if q_vec is None:
        return []
    scored = sorted(
        ((doc_id, _cosine(q_vec, vec)) for doc_id, vec in _EMBED_VECS.items()),
        key=lambda x: x[1], reverse=True,
    )
    candidates = [_DOC_ID_MAP[doc_id] for doc_id, _ in scored[:RERANK_TOP_N] if doc_id in _DOC_ID_MAP]
    if USE_RERANKER:
        candidates = _rerank(query, candidates)
    return candidates[:top_k]


def _collect_image_ids(docs: list[Doc], max_ids: int = 3) -> list[str]:
    # Only take images from the top-ranked doc to avoid mixing unrelated images
    for doc in docs:
        if doc.image_refs:
            return list(dict.fromkeys(doc.image_refs))[:max_ids]
    return []


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
    if not images or not API_KEY:
        return []
    with ThreadPoolExecutor(max_workers=min(len(images), MAX_IMAGES)) as pool:
        futures = [pool.submit(_analyze_image, img) for img in images]
        return [f.result() for f in futures if f.result()]


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

Q: 商品能送到乡镇吗？
A: 您好，我们的商品支持送到大部分乡镇哦，具体能否送达取决于您的收货地址，您可以告诉我详细的收货地址，我帮您查询。送到乡镇一般不需要额外加运费，和市区运费一致；物流时效会比市区稍慢，正常情况下下单后48小时发货，乡镇地区3-5天可收到，偏远乡镇可能需要5-7天哦。
"""


def _generate_answer(query: str, evidence: list[str], image_facts: list[str],
                     memory: str, images_b64: list[str]) -> str:
    ev_text = "\n".join(f"- {s}" for s in evidence) if evidence else "无相关知识库内容"
    img_text = "\n".join(f"- {f}" for f in image_facts) if image_facts else "无"

    user_text = (
        f"用户问题：{query}\n\n"
        f"历史对话：{memory or '无'}\n\n"
        f"知识库证据：\n{ev_text}\n\n"
        f"图片分析：\n{img_text}\n\n"
        "请直接输出给用户的答复："
    )

    user_content: list[dict] = []
    for img in images_b64:
        payload = _extract_b64(img)
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}})
    user_content.append({"type": "text", "text": user_text})

    model = VLM_MODEL if images_b64 else CHAT_MODEL
    return _call_api(model, [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])


# ─── Main Handler ─────────────────────────────────────────────────────────────

def _handle(req: ChatRequest) -> str:
    query = req.user_query
    session_id = (req.session_id or f"sess_{int(time.time()*1000)}").strip()
    memory = _get_memory(session_id)

    # 1. VLM image analysis
    image_facts = _analyze_images(req.images) if req.images else []

    # 2. Vector retrieval
    retrieve_q = f"{query} {' '.join(image_facts[:2])}" if image_facts else query
    docs = _retrieve(retrieve_q)
    evidence = [f"[{doc.title}] {doc.content}" for doc in docs[:TOP_K]]
    image_ids = _collect_image_ids(docs)

    # 3. LLM generation
    if not API_KEY:
        answer = "您好，服务暂时不可用，请稍后重试。"
    else:
        try:
            answer = _generate_answer(query, evidence, image_facts, memory, req.images)
        except Exception:
            answer = "您好，处理您的问题时遇到错误，请稍后重试。"

    # 4. Attach image IDs
    if image_ids:
        if "<PIC>" not in answer:
            answer = f"{answer} <PIC>"
        answer = f"{answer} {json.dumps(image_ids, ensure_ascii=False)}"

    _save_memory(session_id, query, answer)
    return answer


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="DF1165 Multimodal CS Agent", version="4.0.0")


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
