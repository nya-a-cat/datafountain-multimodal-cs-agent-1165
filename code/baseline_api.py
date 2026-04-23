#!/usr/bin/env python3
"""Multimodal customer service agent for DataFountain 1165.

Pipeline:
  1) If images present → VLM analysis (Qwen3-VL-8B-Instruct)
  2) BM25 retrieval from local knowledge base
  3) LLM answer generation (Qwen3.5-122B-A10B) with evidence + image facts
  4) E-commerce policy questions → template answers (fast, no LLM needed)
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import base64
import binascii
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import urllib.error
import urllib.request

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator


# ─── Config ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = ROOT / "data" / "knowledge.jsonl"

CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen3.5-397B-A17B")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3.5-397B-A17B")
API_BASE = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("SILICONFLOW_API_KEY", ""))
KAFU_API_TOKEN = os.getenv("KAFU_API_TOKEN", "")

MAX_IMAGES = 3
MAX_IMAGE_BYTES = 5 * 1024 * 1024
SESSION_MAX_TURNS = 6
TOP_K = 8

WORD_RE = re.compile(r"[一-鿿]+|[A-Za-z0-9_]+")
EN_STOPWORDS = {"a","an","and","are","as","at","be","by","do","for","from",
                "how","i","if","in","is","it","me","my","of","on","or",
                "should","that","the","this","to","use","what","when","where",
                "while","with","you","your"}


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

def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in WORD_RE.findall(text.lower()):
        if re.search(r"[一-鿿]", chunk):
            tokens.append(chunk)
            if len(chunk) > 2:
                tokens.extend(chunk[i:i+2] for i in range(len(chunk)-1))
        elif len(chunk) >= 2 and chunk not in EN_STOPWORDS:
            tokens.append(chunk)
    return tokens


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


# ─── Knowledge Base & BM25 ────────────────────────────────────────────────────

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


class BM25Index:
    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self.doc_tf: list[Counter] = []
        self.doc_sets: list[set] = []
        self.title_sets: list[set] = []
        self.lengths: list[int] = []
        df: Counter = Counter()
        for doc in docs:
            tokens = _tokenize(f"{doc.title} {doc.content}")
            tf = Counter(tokens)
            self.doc_tf.append(tf)
            self.doc_sets.append(set(tf))
            self.title_sets.append(set(_tokenize(doc.title)))
            self.lengths.append(max(1, len(tokens)))
            df.update(set(tf))
        n = max(1, len(docs))
        self.avg_len = sum(self.lengths) / n
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 5) -> list[tuple[Doc, float]]:
        q_tokens = list(dict.fromkeys(_tokenize(query)))
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        scores: list[tuple[float, int]] = []
        for i, tf in enumerate(self.doc_tf):
            dl = self.lengths[i]
            score = 0.0
            for t in q_tokens:
                f = tf.get(t, 0)
                if f <= 0:
                    continue
                idf = self.idf.get(t, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                score += idf * f * (self.k1 + 1) / max(denom, 1e-9)
            if score > 0:
                title_hit = len(q_set & self.title_sets[i])
                score += 0.5 * title_hit
                scores.append((score, i))
        scores.sort(reverse=True)
        return [(self.docs[i], s) for s, i in scores[:top_k]]


KNOWLEDGE = _load_knowledge(KNOWLEDGE_PATH)
INDEX = BM25Index(KNOWLEDGE)


def _retrieve(query: str, top_k: int = TOP_K) -> list[Doc]:
    return [doc for doc, _ in INDEX.search(query, top_k)]


def _pick_sentences(query: str, docs: list[Doc], limit: int = 5) -> list[str]:
    q_set = set(_tokenize(query))
    candidates: list[tuple[float, str]] = []
    for doc in docs:
        for sent in re.split(r"[。！？；\n]+", doc.content):
            sent = sent.strip()
            if len(sent) < 10:
                continue
            overlap = len(q_set & set(_tokenize(sent)))
            if overlap > 0:
                candidates.append((overlap, sent))
    candidates.sort(reverse=True)
    seen: set[str] = set()
    result = []
    for _, s in candidates:
        if s not in seen:
            seen.add(s)
            result.append(s)
            if len(result) >= limit:
                break
    return result


def _collect_image_ids(docs: list[Doc], max_ids: int = 3) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        for ref in doc.image_refs:
            if ref not in seen:
                seen.add(ref)
                ids.append(ref)
                if len(ids) >= max_ids:
                    return ids
    return ids


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


# ─── E-commerce Policy Templates ─────────────────────────────────────────────

_POLICY_RULES: list[tuple[list[str], str, str]] = [
    (["投诉", "辱骂", "虚假宣传", "假货", "少了一件", "包装破损", "拆封", "二手", "过期", "保质期", "太差了", "不满意", "差评"],
     "complaint",
     "您好，非常抱歉给您带来不好的体验！该问题可优先为您登记升级处理，支持核实后补发、换货、维修或退款。请您提供订单号及问题照片/视频证据，我会立即为您创建加急工单并持续跟进结果。"),
    (["7天无理由", "七天无理由", "退货", "换货"],
     "returns",
     "您好，支持7天无理由退换货。商品需保持完好、配件齐全且不影响二次销售；非质量问题通常由买家承担退回运费，质量问题由我们承担。您可提供订单号，我帮您核对可退换条件。"),
    (["退款"],
     "refund",
     "您好，退款一般会在审核通过后原路退回，到账时间通常为1-7个工作日，信用卡渠道可能略慢。若超时未到账，您把订单号发我，我马上帮您催办。"),
    (["发票", "抬头", "开票"],
     "invoice",
     "您好，支持开具发票，可开个人或企业抬头。订单完成后一般1-3个工作日内开具；若抬头填写有误，可在开票前联系修改。您把订单号和开票信息发我，我帮您立即处理。"),
    (["待揽收", "揽收"],
     "pickup",
     "您好，物流显示待揽收，大概率是商品已打包完成，等待快递员上门取件哦，一般24小时内会完成揽收；若超过24小时仍未揽收，您可以联系我们客服，我们会催促快递方尽快上门。"),
    (["乡镇"],
     "rural",
     "您好，我们的商品支持送到大部分乡镇哦，具体能否送达取决于您的收货地址，您可以告诉我详细的收货地址，我帮您查询。送到乡镇一般不需要额外加运费，和市区运费一致；物流时效会比市区稍慢，正常情况下下单后48小时发货，乡镇地区3-5天可收到，偏远乡镇可能需要5-7天哦。"),
    (["寄到国外", "国际配送", "海外"],
     "international",
     "您好，部分商品支持国际配送，具体国家、运费与时效需按收货地和商品属性核算。请提供国家/地区与商品链接，我帮您确认是否可寄及预计费用。"),
    (["物流", "发货", "多久能到", "运费"],
     "shipping",
     "您好，当前物流与发货时效可帮您实时核查。若您方便提供订单号，我可以立即查询发货节点、预计送达时间，并为您跟进异常状态。"),
    (["维修", "人为损坏", "保修", "质保", "故障"],
     "warranty",
     "您好，售后支持检测、维修与换新评估。人为损坏通常可维修但会产生维修费用，非人为质量问题在保修范围内可按政策免费处理。请提供订单号、故障现象和照片，我帮您判断最优处理方案。"),
]


def _match_policy(query: str) -> str | None:
    q = re.sub(r"\s+", "", query.lower())
    for keywords, _, template in _POLICY_RULES:
        if any(k in q for k in keywords):
            return template
    return None


# ─── LLM Answer Generation ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """你是一个专业的多模态电商客服智能体。

回答规则：
1. 优先使用提供的知识库证据和图片分析结果作答
2. 回答风格参考示例：自然、简洁、礼貌，直接给出结论和操作步骤
3. 如果问题涉及图片，结合图片分析结果回答
4. 如果知识库有相关图示，在合适位置使用 <PIC> 占位
5. 不要编造不在证据中的信息
6. 不要输出"根据证据"、"知识库显示"等内部说明

回答示例风格：
- 产品问题："DCB107、DCB112 电池组充电中<PIC>电池组已充满<PIC>过热/过冷延迟<PIC>"
- 操作问题："表带尺寸如下所示。注意：单独销售的配件表带可能略有差异。\n<PIC>"
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

    # 1. Policy template (fast path, no LLM)
    policy = _match_policy(query)
    if policy:
        _save_memory(session_id, query, policy)
        return policy

    # 2. VLM image analysis
    image_facts = _analyze_images(req.images) if req.images else []

    # 3. BM25 retrieval
    retrieve_q = query
    if image_facts:
        retrieve_q = f"{query} {' '.join(image_facts[:2])}"
    docs = _retrieve(retrieve_q)
    evidence = _pick_sentences(query, docs)
    image_ids = _collect_image_ids(docs)

    # 4. LLM generation
    if not API_KEY:
        answer = "您好，服务暂时不可用，请稍后重试。"
    else:
        try:
            answer = _generate_answer(query, evidence, image_facts, memory, req.images)
        except Exception as e:
            answer = f"您好，处理您的问题时遇到错误，请稍后重试。"

    # 5. Attach image IDs
    if image_ids:
        if "<PIC>" not in answer:
            answer = f"{answer} <PIC>"
        answer = f"{answer} {json.dumps(image_ids, ensure_ascii=False)}"

    _save_memory(session_id, query, answer)
    return answer


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="DF1165 Multimodal CS Agent", version="3.0.0")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "knowledge_docs": len(KNOWLEDGE),
        "chat_model": CHAT_MODEL,
        "vlm_model": VLM_MODEL,
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
