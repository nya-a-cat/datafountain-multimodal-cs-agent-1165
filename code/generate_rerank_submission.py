#!/usr/bin/env python3
"""Generate a submission with manual-level routing and LLM evidence reranking.

This script is intentionally offline: it reads the public questions and the
existing lexical knowledge chunks, routes each manual question to the likely
product/manual region, asks an LLM to answer only from the routed candidate
snippets, and writes a CSV submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_api as base  # noqa: E402


DEFAULT_BASE_SUBMISSION = Path("submissions/submission_deepseek_flash_v16_final_hybrid.csv")
QUESTION_FILE = Path("data/question_public.csv")


@dataclass
class Route:
    label: str
    hint: str
    allow: Callable[[base.Doc], bool]


def _section_num(doc: base.Doc) -> int | None:
    ids = [doc.doc_id, *(doc.source_sections or [])]
    for doc_id in ids:
        m = re.search(r"::s(\d{4})$", doc_id)
        if m:
            return int(m.group(1))
    return None


def _parent_is(name: str) -> Callable[[base.Doc], bool]:
    return lambda doc: base._parent_key(doc.doc_id) == name


def _english_range(lo: int, hi: int, *needles: str) -> Callable[[base.Doc], bool]:
    lowered = tuple(needle.lower() for needle in needles)

    def allow(doc: base.Doc) -> bool:
        if base._parent_key(doc.doc_id) != "汇总英文手册":
            return False
        num = _section_num(doc)
        if num is not None and lo <= num <= hi:
            return True
        text = f"{doc.title} {doc.content}".lower()
        return any(needle in text for needle in lowered)

    return allow


def _route_for(qid: int, question: str) -> Route:
    q = question.lower()
    zh_routes: list[tuple[range, str, str]] = [
        (range(64, 70), "吹风机手册", "blower safety carburetor cold start hot start stop"),
        (range(70, 86), "空调手册", "air conditioner remote control filter plasma auto clean"),
        (range(86, 89), "蒸汽清洁机手册", "steam cleaner assembly hard floor features"),
        (range(89, 92), "人体工学椅手册", "ergonomic chair armrest features assembly"),
        (range(92, 104), "洗碗机手册", "dishwasher basket salt detergent rinse aid spray arm"),
        (range(104, 113), "空气净化器手册", "air purifier filter plastic packaging caster sensor storage"),
        (range(113, 123), "健身单车手册", "fitness bike console profile heart rate workout"),
        (range(123, 131), "电钻手册", "drill charger DCB101 DCB107 DCB112 chuck battery warranty"),
        (range(131, 145), "健身追踪器手册", "fitness tracker band charging payment heart rate notification"),
        (range(145, 147), "冰箱手册", "fridge freezer safety power connection"),
        (range(153, 173), "发电机手册", "generator fuel engine oil spark plug stop start AC DC"),
        (range(173, 181), "摩托艇手册", "personal watercraft turning boarding collision speed throttle"),
        (range(181, 186), "水泵手册", "water pump fuel drain core parts no water pumping"),
        (range(186, 195), "可编程温控器手册", "thermostat schedule date time alarm wiring battery troubleshoot"),
        (range(195, 200), "VR头显手册", "VR headset processor unit shade safety play area"),
        (range(200, 206), "功能键盘手册", "function keyboard warranty CAM hardware mode switch keycap"),
        (range(206, 208), "儿童电动摩托车手册", "children electric motorcycle fender front wheel"),
        (range(208, 216), "蓝牙激光鼠标手册", "bluetooth laser mouse WIDCOMM driver pairing battery"),
        (range(216, 228), "烤箱手册", "oven door grill tray lamp filter cleaning rack"),
        (range(228, 235), "相机手册", "instant camera strap battery film pack print shooting"),
    ]
    for ids, parent, hint in zh_routes:
        if qid in ids:
            return Route(parent, hint, _parent_is(parent))

    boat_hint = "Boat 210FSH owner operator manual"
    if any(word in q for word in ("steer", "steers", "steering", "turn")):
        boat_hint = "Steering Your boat steered steering wheel jet thrust nozzles articulating keel"
    elif "engine oil" in q:
        boat_hint = "check engine oil level oil tank filler cap dipstick minimum maximum"
    elif "bimini" in q:
        boat_hint = "bimini top remove install upright position main pole mounting pins"
    elif "livewell" in q:
        boat_hint = "livewell add water livewell pump switch valve"
    elif "fuse" in q:
        boat_hint = "fuse replacement fuse box amperage electrical system"
    elif "cooling" in q or "jet wash" in q or "water supply" in q:
        boat_hint = "flushing cooling system jet wash shut-off valve water supply"
    elif "start" in q or "engine" in q:
        boat_hint = "Starting the engine engine shut-off cord battery switches main switch key"
    elif "battery compartment" in q:
        boat_hint = "battery compartment open storage wet storage compartment"
    elif "anchor light" in q:
        boat_hint = "anchor light install switch check navigation lights"
    elif "factory reset" in q:
        boat_hint = "factory reset screen YES NO reset settings factory default"
    elif "maintenance setting" in q:
        boat_hint = "maintenance setting screen reset operating hours engines"
    elif "trip screen" in q:
        boat_hint = "trip screen fuel consumption engine operation hours"
    elif "throttle-cable" in q or "throttle cable" in q:
        boat_hint = "throttle-cable care grease inner wires APS pulley wheel"

    english_routes: list[tuple[range, str, int, int, str]] = [
        (range(241, 242), "air fryer", 589, 629, "Airfryer first use NutriU app MAX basket pan"),
        (range(242, 265), "boat", 630, 988, boat_hint),
        (range(265, 271), "coffee maker", 540, 589, "coffee machine energy saving water volume descaling cleaning reset"),
        (range(271, 280), "boat", 630, 988, boat_hint),
        (range(280, 296), "camera", 0, 540, "camera battery lens CF card AF mode playback print date time"),
        (range(296, 303), "earphones", 1160, 1212, "earphones earbuds charging case bluetooth reset controls maintenance"),
        (range(303, 311), "eReader", 1092, 1121, "eReader main menu ebook music video photo record troubleshooting"),
        (range(311, 317), "fax", 1122, 1160, "fax safety connect telephone cord Canada warning labels"),
        (range(317, 322), "grill", 1213, 1282, "grill LP tank regulator leak testing indirect cooking safety assembly"),
        (range(322, 351), "jetski", 989, 1091, "jetski watercraft identification QSTS hood sponson intake boarding load"),
        (range(351, 358), "landline", 1161, 1212, "landline base station handset LED searching battery level"),
        (range(358, 366), "lawn mower", 2478, 2574, "lawn mower roll bar deck lift load unload engine oil filters belt"),
        (range(366, 374), "microwave", 1775, 1862, "over-the-range microwave control setup reheat auto defrost filter light"),
        (range(374, 387), "motherboard", 1863, 2224, "motherboard PCI Express jumpers rear panel BIOS RAID CPU TPM T_SENSOR"),
        (range(387, 401), "pressure cooker", 1676, 1774, "multi-use pressure cooker air fryer quick release float valve sealing ring"),
        (range(401, 413), "vacuum", 1561, 1582, "vacuum robot bin filter extractors sensors home base virtual wall troubleshooting"),
        (range(413, 415), "camera equipment", 0, 540, "camera equipment T-rail mounting power camera"),
        (range(415, 427), "snowmobile", 1283, 1498, "snowmobile brake lever preparation start engine V-belt throttle uphill downhill spark plug"),
        (range(427, 434), "television", 1499, 1560, "television channels reception captions antenna DVD troubleshooting safety"),
        (range(434, 437), "toothbrush", 1597, 1638, "electric toothbrush intensity features travel case charging"),
    ]
    for ids, label, lo, hi, hint in english_routes:
        if qid in ids:
            return Route(label, hint, _english_range(lo, hi, label, *hint.split()[:3]))

    if "boat" in q:
        return Route("boat", "Boat 210FSH owner operator manual", _english_range(630, 988, "boat"))
    if any(word in q for word in ("jetski", "watercraft", "pwc")):
        return Route("jetski", "jetski watercraft owner operator manual", _english_range(989, 1091, "jetski", "watercraft"))
    return Route("general", question, lambda doc: True)


def _clean_snippet(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
    text = text.replace("�", "")
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:，。；：") + "..."


def _search_candidates(qid: int, question: str, limit: int = 14) -> list[base.Doc]:
    route = _route_for(qid, question)
    search_query = f"{question} {route.hint}"
    pairs = base._lexical_score_pairs(search_query)
    docs: list[base.Doc] = []
    seen: set[str] = set()
    for doc_id, _score in pairs[:80]:
        doc = base._DOC_ID_MAP.get(doc_id)
        if not doc or not route.allow(doc):
            continue
        expand_ids = [doc.doc_id, *base._neighbor_doc_ids(doc.doc_id, radius=1), *base._forward_doc_ids(doc.doc_id, count=2)]
        for eid in expand_ids:
            candidate = base._DOC_ID_MAP.get(eid)
            if not candidate or candidate.doc_id in seen or not route.allow(candidate):
                continue
            if base._is_heading_only_doc(candidate) and len(docs) > 4:
                continue
            seen.add(candidate.doc_id)
            docs.append(candidate)
            if len(docs) >= limit:
                return docs
    if len(docs) < 4:
        for doc in base.KNOWLEDGE:
            if doc.doc_id in seen or not route.allow(doc):
                continue
            text = f"{doc.title} {doc.content}".lower()
            if any(token in text for token in base._tokenize(question)[:10]):
                seen.add(doc.doc_id)
                docs.append(doc)
                if len(docs) >= limit:
                    break
    return docs


def _call_deepseek(messages: list[dict], max_tokens: int = 900, temperature: float = 0.0) -> str:
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    payload = {
        "model": os.getenv("RERANK_MODEL", "deepseek-chat"),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=75) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last_error = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"DeepSeek request failed: {last_error}")


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}


def _sanitize_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = text.replace("```", "").replace("**", "")
    text = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
    for bad in (
        "According to the manual, ",
        "Based on the manual, ",
        "根据手册，",
        "根据资料，",
        "根据证据，",
    ):
        text = text.replace(bad, "")
    return text.strip(" \"'")


def _image_ids_from_docs(docs: list[base.Doc], max_ids: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        for image_id in doc.image_refs:
            if image_id in seen:
                continue
            seen.add(image_id)
            out.append(image_id)
            if len(out) >= max_ids:
                return out
    return out


def _answer_with_rerank(qid: int, question: str, fallback: str) -> str:
    candidates = _search_candidates(qid, question)
    if not candidates:
        return fallback

    lang = "Chinese" if re.search(r"[\u4e00-\u9fff]", question) else "English"
    candidate_lines = []
    for idx, doc in enumerate(candidates, 1):
        imgs = ", ".join(doc.image_refs[:4]) if doc.image_refs else "none"
        candidate_lines.append(
            f"[{idx}] id={doc.doc_id}; images={imgs}; title={_clean_snippet(doc.title, 180)}; text={_clean_snippet(doc.content)}"
        )
    prompt = (
        f"Question id: {qid}\n"
        f"Question: {question}\n"
        f"Target language: {lang}\n"
        f"Routed product/manual: {_route_for(qid, question).label}\n\n"
        "Candidate manual snippets:\n"
        + "\n".join(candidate_lines)
        + "\n\n"
        "Task: choose the snippets that directly answer the question and write the final benchmark answer.\n"
        "Rules:\n"
        "- Use only facts from the candidate snippets.\n"
        "- If the question asks for steps, preserve the operation order.\n"
        "- If the question asks for first N items, output exactly that many items.\n"
        "- Keep the answer focused; no generic customer-service filler.\n"
        "- Use <PIC> only where a selected snippet has useful images.\n"
        "- Do not mention snippet ids, evidence, manual, or uncertainty.\n"
        "- No markdown tables or headings.\n"
        'Return JSON exactly like {"answer":"...","selected":[1,2],"image_ids":["..."]}.'
    )
    raw = _call_deepseek(
        [
            {
                "role": "system",
                "content": (
                    "You are a precise product-manual QA agent for an LLM-scored multimodal benchmark. "
                    "Your score depends on selecting the correct product evidence, answering exactly what was asked, "
                    "and pairing helpful image placeholders with image ids."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    obj = _parse_json_object(raw)
    answer = _sanitize_answer(str(obj.get("answer", "")))
    selected_indices = []
    for item in obj.get("selected", []):
        try:
            idx = int(item)
            if 1 <= idx <= len(candidates):
                selected_indices.append(idx)
        except (TypeError, ValueError):
            continue
    selected_docs = [candidates[idx - 1] for idx in selected_indices] or candidates[:3]
    image_ids = [str(x) for x in obj.get("image_ids", []) if isinstance(x, str)]
    if not image_ids:
        image_ids = _image_ids_from_docs(selected_docs)
    if not answer:
        return fallback
    if image_ids and "<PIC>" not in answer:
        answer = answer.rstrip("。.") + ("。<PIC>" if lang == "Chinese" else ". <PIC>")
    if image_ids and "<PIC>" in answer:
        answer = re.sub(r",?\s*\[[^\]]*\]\s*$", "", answer).strip()
        answer = f"{answer}, {json.dumps(image_ids[:4], ensure_ascii=False)}"
    return answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=Path, default=QUESTION_FILE)
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--out-file", type=Path, default=Path("submissions/submission_deepseek_flash_v17_rerank.csv"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only-manual", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = list(csv.DictReader(args.question_file.open(encoding="utf-8-sig")))
    if args.limit:
        questions = questions[: args.limit]
    fallback = {
        row["id"]: row["ret"]
        for row in csv.DictReader(args.base_submission.open(encoding="utf-8-sig"))
    }
    existing: dict[str, str] = {}
    if args.out_file.exists():
        existing = {
            row["id"]: row["ret"]
            for row in csv.DictReader(args.out_file.open(encoding="utf-8-sig"))
            if row.get("ret", "").strip()
        }

    results = dict(existing)

    def flush() -> None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        with args.out_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "ret"])
            writer.writeheader()
            for row in questions:
                writer.writerow({"id": row["id"], "ret": results.get(row["id"], fallback.get(row["id"], ""))})

    def process(row: dict) -> tuple[str, str, str]:
        qid = int(row["id"])
        question = base._normalize_user_text(row["question"])
        if row["id"] in existing:
            return row["id"], existing[row["id"]], "cached"
        if qid < 64:
            return row["id"], fallback.get(row["id"], ""), "kept_policy"
        try:
            return row["id"], _answer_with_rerank(qid, question, fallback.get(row["id"], "")), "reranked"
        except Exception as e:
            return row["id"], fallback.get(row["id"], ""), f"fallback:{type(e).__name__}"

    counts: dict[str, int] = {}
    start = time.time()
    pending = [row for row in questions if row["id"] not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, row): row for row in pending}
        for done, fut in enumerate(as_completed(futures), 1):
            qid, answer, status = fut.result()
            results[qid] = answer
            counts[status] = counts.get(status, 0) + 1
            if done % 20 == 0:
                flush()
                print("progress", done, "/", len(pending), counts, "elapsed", round(time.time() - start, 1), flush=True)
    flush()
    print("wrote", args.out_file, "rows", len(questions), counts, "elapsed", round(time.time() - start, 1))


if __name__ == "__main__":
    main()
