#!/usr/bin/env python3
"""Build embedding index from knowledge.jsonl using BAAI/bge-m3."""

from __future__ import annotations
import json, os, time
import urllib.request
from pathlib import Path

API_BASE = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("SILICONFLOW_API_KEY", ""))
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
BATCH_SIZE = 32
KNOWLEDGE_PATH = Path("data/knowledge.jsonl")
OUT_PATH = Path("data/embeddings.jsonl")


def embed_batch(texts: list[str]) -> list[list[float]]:
    payload = {"model": EMBED_MODEL, "input": texts, "encoding_format": "float"}
    req = urllib.request.Request(
        f"{API_BASE.rstrip('/')}/embeddings",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("embedding failed after retries")


def main() -> None:
    docs = [json.loads(l) for l in KNOWLEDGE_PATH.open(encoding="utf-8")]
    print(f"Embedding {len(docs)} docs with {EMBED_MODEL}...")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i: i + BATCH_SIZE]
            texts = [d["content"] for d in batch]
            vecs = embed_batch(texts)
            for doc, vec in zip(batch, vecs):
                out.write(json.dumps({"doc_id": doc["doc_id"], "embedding": vec}, ensure_ascii=False) + "\n")
            written += len(batch)
            print(f"\r[{written}/{len(docs)}]", end="", flush=True)
            time.sleep(0.1)

    print(f"\nDone. {OUT_PATH}")


if __name__ == "__main__":
    main()
