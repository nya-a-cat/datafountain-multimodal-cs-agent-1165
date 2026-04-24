#!/usr/bin/env python3
"""Retry only error rows in a submission CSV and merge back safely."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request


API_URL = "http://127.0.0.1:8000/chat"
TIMEOUT = 300
WORKERS = 8
SAVE_EVERY = 1
ERROR_MARKERS = [
    "处理您的问题时遇到错误",
    "系统暂时不可用",
    "服务暂时不可用",
    "服务返回为空",
    "服务处理失败",
]
STOP_REQUESTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry only failed rows in a submission CSV")
    parser.add_argument("--submission-file", type=Path, default=Path("submissions/submission_large_chunk.csv"))
    parser.add_argument("--question-file", type=Path, default=Path("data/question_public.csv"))
    parser.add_argument("--api-url", type=str, default=os.getenv("RETRY_API_URL", API_URL))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("RETRY_TIMEOUT", str(TIMEOUT))))
    parser.add_argument("--workers", type=int, default=int(os.getenv("RETRY_WORKERS", str(WORKERS))))
    parser.add_argument("--save-every", type=int, default=int(os.getenv("RETRY_SAVE_EVERY", str(SAVE_EVERY))))
    return parser.parse_args()


def _normalize_answer(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def _read_questions(path: Path) -> dict[str, str]:
    questions: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = str(row.get("id", "")).strip()
            question = str(row.get("question", "")).strip()
            if qid:
                questions[qid] = question
    return questions


def _read_submission(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def _write_submission(path: Path, rows: list[list[str]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    tmp_path.replace(path)


def _is_error_answer(text: str) -> bool:
    return any(marker in text for marker in ERROR_MARKERS)


def _request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nReceived signal {signum}, will stop after the current save point.", flush=True)


def call_chat(api_url: str, timeout: int, qid: str, question: str) -> str:
    payload = {"session_id": f"sess_{qid}", "question": question, "stream": False}
    body = json.dumps(payload, ensure_ascii=False).encode()
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": f"req-{qid}-{uuid.uuid4().hex[:8]}",
        "X-Client-Type": "retry-errors",
    }
    req = request.Request(api_url, data=body, headers=headers, method="POST")
    attempt = 0
    while not STOP_REQUESTED:
        attempt += 1
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            answer = str(data.get("data", {}).get("answer", "")).strip()
            if answer and not _is_error_answer(answer):
                return answer
            print(f"  [id={qid}] attempt {attempt}: bad answer, retrying...", flush=True)
        except Exception as exc:
            print(f"  [id={qid}] attempt {attempt}: {exc}, retrying...", flush=True)
        time.sleep(min(2 ** min(attempt, 5), 32))
    return ""


def _process_one(api_url: str, timeout: int, i: int, qid: str, question: str) -> tuple[int, str]:
    answer = call_chat(api_url, timeout, qid, question) if question else ""
    return i, answer


def main() -> None:
    args = parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _request_stop)

    rows = _read_submission(args.submission_file)
    questions = _read_questions(args.question_file)
    error_indices = [
        (i, rows[i][0])
        for i in range(1, len(rows))
        if len(rows[i]) > 1 and _is_error_answer(rows[i][1])
    ]
    total = len(error_indices)
    print(f"Retrying {total} error rows with {args.workers} workers...", flush=True)
    if total == 0:
        return

    completed = 0
    updated = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(_process_one, args.api_url, args.timeout, i, qid, questions.get(qid, "")): (i, qid)
                for i, qid in error_indices
            }
            for future in as_completed(futures):
                i, qid = futures[future]
                try:
                    _, answer = future.result()
                except Exception as exc:
                    print(f"  [id={qid}] worker crashed: {exc}", flush=True)
                    answer = ""
                if answer:
                    rows[i][1] = _normalize_answer(answer)
                    updated += 1
                completed += 1
                if completed % max(1, args.save_every) == 0 or completed == total or STOP_REQUESTED:
                    _write_submission(args.submission_file, rows)
                    remaining = sum(
                        1
                        for idx in range(1, len(rows))
                        if len(rows[idx]) > 1 and _is_error_answer(rows[idx][1])
                    )
                    print(f"[{completed}/{total}] saved, updated={updated}, remaining_errors={remaining}", flush=True)
                if STOP_REQUESTED:
                    break
    finally:
        _write_submission(args.submission_file, rows)

    remaining = sum(
        1
        for idx in range(1, len(rows))
        if len(rows[idx]) > 1 and _is_error_answer(rows[idx][1])
    )
    print(f"Done. Updated {args.submission_file}, remaining_errors={remaining}", flush=True)
    if STOP_REQUESTED:
        sys.exit(130)


if __name__ == "__main__":
    main()
