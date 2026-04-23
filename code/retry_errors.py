#!/usr/bin/env python3
"""Retry only error rows in a submission CSV and merge back."""

import csv, json, time, uuid
from pathlib import Path
from urllib import request, error
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://127.0.0.1:8000/chat"
TIMEOUT = 40
RETRIES = 3
WORKERS = 4
ERROR_MARKERS = ["处理您的问题时遇到错误", "系统暂时不可用", "服务返回为空", "服务处理失败"]


def call_chat(qid: str, question: str) -> str:
    payload = {"session_id": f"sess_{qid}", "question": question, "stream": False}
    body = json.dumps(payload, ensure_ascii=False).encode()
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": f"req-{qid}-{uuid.uuid4().hex[:8]}",
    }
    req = request.Request(API_URL, data=body, headers=headers, method="POST")
    for i in range(RETRIES):
        try:
            with request.urlopen(req, timeout=TIMEOUT) as r:
                d = json.loads(r.read())
            ans = d.get("data", {}).get("answer", "")
            if ans and not any(m in ans for m in ERROR_MARKERS):
                return ans
        except Exception:
            pass
        time.sleep(2 ** i)
    return ""


def main() -> None:
    sub_path = Path("submissions/submission_v15.csv")
    q_path = Path("data/question_public.csv")

    # load questions
    questions: dict[str, str] = {}
    with q_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            questions[row["id"]] = row["question"]

    # load submission
    rows: list[list[str]] = []
    with sub_path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))

    error_indices = [(i, rows[i][0]) for i in range(1, len(rows)) if any(m in rows[i][1] for m in ERROR_MARKERS)]
    print(f"Retrying {len(error_indices)} error rows with {WORKERS} workers...")

    def process(i: int, qid: str) -> tuple[int, str]:
        q = questions.get(qid, "")
        ans = call_chat(qid, q) if q else ""
        return i, ans

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process, i, qid): (i, qid) for i, qid in error_indices}
        for future in as_completed(futures):
            i, ans = future.result()
            if ans:
                rows[i][1] = " ".join(ans.replace("\r", " ").replace("\n", " ").split())
            done += 1
            if done % 20 == 0 or done == len(error_indices):
                print(f"\r[{done}/{len(error_indices)}] retried", end="", flush=True)

    print()
    with sub_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Done. Updated {sub_path}")


if __name__ == "__main__":
    main()
