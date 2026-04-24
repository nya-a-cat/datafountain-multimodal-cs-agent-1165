#!/usr/bin/env python3
"""Build knowledge chunks and embeddings directly from manual txt files."""

from __future__ import annotations
import json, os, re, time
import urllib.request
from pathlib import Path

API_BASE = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("SILICONFLOW_API_KEY", ""))
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
BATCH_SIZE = 32
MANUALS_DIR = Path("data/KownledgeBase/手册")
OUT_EMBED = Path("data/embeddings.jsonl")
OUT_KNOWLEDGE = Path("data/knowledge_v2.jsonl")


def _parse_manual(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()

    # extract text: everything between first `"` and last `"` before the image list
    # image list is a JSON array at the end: ["id1", "id2", ...]
    img_match = re.search(r'\[("[^"]*"(?:,\s*"[^"]*")*)\]\s*\]?\s*$', raw)
    image_ids: list[str] = []
    if img_match:
        image_ids = re.findall(r'"([^"]+)"', img_match.group(1))

    # extract the text portion: strip outer JSON string quotes
    text_match = re.match(r'^\[?"(.*)"(?:,|\s*\[)', raw, re.DOTALL)
    if not text_match:
        # fallback: just use raw stripped of brackets
        full_text = raw.strip('[]"')
    else:
        full_text = text_match.group(1)
        # unescape JSON string escapes
        full_text = full_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')

    manual_name = path.stem
    sections = re.split(r'(?=# )', full_text)
    pic_counter = 0
    chunks = []
    for idx, sec in enumerate(sections):
        sec = sec.strip()
        if not sec:
            continue
        pics_in_sec = sec.count("<PIC>")
        refs = [image_ids[pic_counter + i] for i in range(pics_in_sec)
                if pic_counter + i < len(image_ids)]
        pic_counter += pics_in_sec
        title = sec.split("\n")[0].lstrip("# ").strip()
        chunks.append({
            "doc_id": f"{manual_name}::s{idx:04d}",
            "title": f"{manual_name} / {title}",
            "content": sec,
            "embed_text": f"{manual_name} {title}\n{sec}",
            "image_refs": refs,
        })
    return chunks


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
    raise RuntimeError("embedding failed")


def main() -> None:
    all_chunks: list[dict] = []
    for txt in sorted(MANUALS_DIR.glob("*.txt")):
        chunks = _parse_manual(txt)
        all_chunks.extend(chunks)
        print(f"  {txt.name}: {len(chunks)} chunks")

    print(f"Total: {len(all_chunks)} chunks")

    with OUT_KNOWLEDGE.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Embedding with {EMBED_MODEL}...")
    # resume: load already-done doc_ids
    done: set[str] = set()
    if OUT_EMBED.exists():
        with OUT_EMBED.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["doc_id"])
                except Exception:
                    pass
    print(f"  resuming: {len(done)} already done")

    written = len(done)
    with OUT_EMBED.open("a", encoding="utf-8") as out:
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = [c for c in all_chunks[i: i + BATCH_SIZE] if c["doc_id"] not in done]
            if not batch:
                continue
            vecs = embed_batch([c.get("embed_text", c["content"]) for c in batch])
            for chunk, vec in zip(batch, vecs):
                out.write(json.dumps({"doc_id": chunk["doc_id"], "embedding": vec}, ensure_ascii=False) + "\n")
            written += len(batch)
            print(f"\r[{written}/{len(all_chunks)}]", end="", flush=True)
            time.sleep(0.1)

    print(f"\nDone. {written} embeddings")


if __name__ == "__main__":
    main()
