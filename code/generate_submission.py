#!/usr/bin/env python3
"""Generate submission CSV for DataFountain 1165.

Flow:
1) read question_public.csv
2) call /chat endpoint for each question
3) write submission CSV with columns: id, ret
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib import request, error


WORD_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")
MAX_ATTACH_IMAGES = 3
MAX_IMAGE_BYTES = 5 * 1024 * 1024
BAD_ANSWER_MARKERS = (
    "我们要求",
    "我们被问到",
    "我们分析",
    "用户问",
    "知识库证据",
    "根据规则",
    "处理您的问题时遇到错误",
    "系统暂时不可用",
    "I don't have specific",
    "please refer to the user manual",
)


class KnowledgeDoc:
    def __init__(self, title: str, content: str, image_refs: list[str]) -> None:
        self.title = title
        self.content = content
        self.image_refs = image_refs
        self.tokens = set(_tokenize(f"{title} {content}"))


@lru_cache(maxsize=4096)
def _encode_image_b64(path_str: str) -> str:
    return base64.b64encode(Path(path_str).read_bytes()).decode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission CSV from question file")
    parser.add_argument(
        "--question-file",
        type=Path,
        default=Path("data/question_public.csv"),
        help="Path to question_public.csv",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("submissions/submission_siliconflow.csv"),
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000/chat",
        help="Chat API URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry times on transient API/network errors",
    )
    parser.add_argument(
        "--knowledge-file",
        type=Path,
        default=Path("data/knowledge.jsonl"),
        help="Path to knowledge jsonl with image_refs",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/images"),
        help="Directory of extracted image assets",
    )
    parser.add_argument(
        "--attach-images",
        action="store_true",
        help="Attach related images to each question when available",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers for API calls",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Write partial results every N completed answers",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output file and regenerate all answers",
    )
    return parser.parse_args()


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in WORD_RE.findall(text.lower()):
        if not chunk:
            continue
        if re.search(r"[\u4e00-\u9fff]", chunk):
            tokens.append(chunk)
            if len(chunk) > 2:
                tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
        elif len(chunk) >= 2:
            tokens.append(chunk)
    return tokens


def _is_image_related_question(question: str) -> bool:
    q = question.lower()
    keywords = ("图", "图片", "照片", "拍照", "示意图", "看图", "外观", "image", "photo", "pic")
    return any(k in q for k in keywords)


def _load_knowledge(path: Path) -> list[KnowledgeDoc]:
    docs: list[KnowledgeDoc] = []
    if not path.exists():
        return docs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                KnowledgeDoc(
                    title=str(obj.get("title", "")),
                    content=str(obj.get("content", "")),
                    image_refs=[str(x) for x in obj.get("image_refs", [])],
                )
            )
    return docs


def _build_image_index(images_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not images_dir.exists():
        return out
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        out[p.stem.lower()] = p
    return out


def _guess_related_images(
    question: str,
    docs: list[KnowledgeDoc],
    image_index: dict[str, Path],
) -> list[tuple[str, str]]:
    if not docs or not image_index:
        return []
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return []

    scored: list[tuple[int, KnowledgeDoc]] = []
    for doc in docs:
        overlap = len(q_tokens & doc.tokens)
        if overlap > 0 and doc.image_refs:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)

    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, doc in scored[:12]:
        for ref in doc.image_refs:
            key = ref.lower().strip()
            p = image_index.get(key)
            if not p:
                continue
            if key in seen:
                continue
            if p.stat().st_size > MAX_IMAGE_BYTES:
                continue
            b64 = _encode_image_b64(str(p.resolve()))
            chosen.append((ref, b64))
            seen.add(key)
            if len(chosen) >= MAX_ATTACH_IMAGES:
                return chosen
    return chosen


def normalize_answer(text: str) -> str:
    # Keep one-line answers to avoid CSV line breaks in `ret` column.
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def call_chat_api(
    api_url: str,
    qid: str,
    question: str,
    timeout: float,
    retries: int,
    image_ids: list[str],
    images_b64: list[str],
) -> str:
    payload = {
        "session_id": f"sess_{qid}",
        "question": question,
        "image_ids": image_ids,
        "images": images_b64,
        "history": [],
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": f"req-{qid}-{uuid.uuid4().hex[:8]}",
        "X-Client-Type": "submission-generator",
    }
    token = os.getenv("KAFU_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = request.Request(
        api_url,
        data=body,
        headers=headers,
        method="POST",
    )

    attempts = max(0, retries) + 1
    for idx in range(attempts):
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                obj: dict[str, Any] = json.loads(text)
                if isinstance(obj.get("code"), int) and obj.get("code") != 0:
                    if idx < attempts - 1:
                        time.sleep(0.4 * (idx + 1))
                        continue
                    return "您好，服务处理失败，请稍后重试。"

                answer = obj.get("answer")
                if not isinstance(answer, str):
                    data = obj.get("data")
                    if isinstance(data, dict):
                        answer = data.get("answer")

                if isinstance(answer, str) and answer.strip():
                    return answer.strip()
                if idx < attempts - 1:
                    time.sleep(0.3 * (idx + 1))
                    continue
                return "您好，服务返回为空，建议补充更具体的问题描述后重试。"
        except (error.URLError, json.JSONDecodeError, TimeoutError):
            if idx < attempts - 1:
                time.sleep(0.5 * (idx + 1))
                continue
            return "您好，系统暂时不可用，请稍后重试。"
    return "您好，系统暂时不可用，请稍后重试。"


def read_questions(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("id", "")).strip()
            question = str(row.get("question", "")).strip()
            if not qid:
                continue
            rows.append((qid, question))
    return rows


def write_submission(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "ret"])
        for qid, ret in rows:
            writer.writerow([qid, normalize_answer(ret)])


def _is_bad_answer(text: str) -> bool:
    return (not text.strip()) or any(marker in text for marker in BAD_ANSWER_MARKERS)


def read_existing_submission(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("id", "")).strip()
            ret = str(row.get("ret", "")).strip()
            if qid and ret:
                out[qid] = ret
    return out


def _process_one(
    idx: int,
    qid: str,
    qtext: str,
    args: argparse.Namespace,
    docs: list[KnowledgeDoc],
    image_index: dict[str, Path],
) -> tuple[int, str, str]:
    """Worker: returns (original_index, qid, answer)."""
    should_attach = args.attach_images and _is_image_related_question(qtext)
    attachments = _guess_related_images(qtext, docs, image_index) if should_attach else []
    image_ids = [image_id for image_id, _ in attachments]
    images = [image_b64 for _, image_b64 in attachments]
    answer = call_chat_api(args.api_url, qid, qtext, args.timeout, args.retries, image_ids, images)
    return idx, qid, answer


def main() -> None:
    args = parse_args()
    questions = read_questions(args.question_file)
    docs = _load_knowledge(args.knowledge_file) if args.attach_images else []
    image_index = _build_image_index(args.images_dir) if args.attach_images else {}

    total = len(questions)
    results: dict[int, tuple[str, str]] = {}
    existing = {} if args.no_resume else read_existing_submission(args.out_file)
    for i, (qid, _qtext) in enumerate(questions):
        ret = existing.get(qid, "")
        if ret and not _is_bad_answer(ret):
            results[i] = (qid, ret)

    pending = [
        (i, qid, qtext)
        for i, (qid, qtext) in enumerate(questions)
        if i not in results
    ]
    done = len(results)
    if done:
        print(f"Resuming: {done}/{total} good existing answers, {len(pending)} pending")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_one, i, qid, qtext, args, docs, image_index): i
            for i, qid, qtext in pending
        }
        for future in as_completed(futures):
            idx, qid, answer = future.result()
            results[idx] = (qid, answer)
            done += 1
            if args.flush_every > 0 and (done % args.flush_every == 0):
                partial_rows = [results[i] for i in sorted(results)]
                write_submission(args.out_file, partial_rows)
            if done % 20 == 0 or done == total:
                print(f"\r[{done}/{total}] answers received", end="", flush=True)

    print()

    out_rows = [results[i] for i in range(total)]
    write_submission(args.out_file, out_rows)
    print(f"Done. wrote {len(out_rows)} rows to {args.out_file}")


if __name__ == "__main__":
    main()
