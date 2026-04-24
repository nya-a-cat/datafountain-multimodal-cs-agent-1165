#!/usr/bin/env python3
"""Build large-chunk embeddings: one chunk per manual (fits in 32K context)."""

from __future__ import annotations
import json, os, re, time
import urllib.request
from pathlib import Path

API_BASE = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("SILICONFLOW_API_KEY", ""))
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
MANUALS_DIR = Path("data/KownledgeBase/手册")
OUT_EMBED = Path("data/embeddings_large.jsonl")
OUT_KNOWLEDGE = Path("data/knowledge_large.jsonl")


def _parse_manual(path: Path) -> list[dict]:
    """One chunk per manual — full text with all image_refs."""
    raw = path.read_text(encoding="utf-8").strip()
    valid = set('"\\' + '/bfnrtu\n')
    buf, i = [], 0
    while i < len(raw):
        if raw[i] == '\\' and i + 1 < len(raw):
            if raw[i + 1] == '\\':
                buf.append('\\\\'); i += 2
            elif raw[i + 1] in valid:
                buf.append(raw[i]); i += 1
            else:
                buf.append('\\\\'); i += 1
        else:
            buf.append(raw[i]); i += 1
    fixed = ''.join(buf)
    decoder = json.JSONDecoder()
    pos = 0
    segments: list[tuple[str, list[str]]] = []
    while pos < len(fixed):
        chunk = fixed[pos:].lstrip()
        if not chunk:
            break
        try:
            data, end = decoder.raw_decode(chunk)
            segments.append((data[0], data[1]))
            pos += len(fixed[pos:]) - len(chunk) + end
        except (json.JSONDecodeError, IndexError):
            break
    if not segments:
        print(f"  SKIP: {path.name}")
        return []
    full_text = "\n".join(s[0] for s in segments)
    image_ids = []
    for s in segments:
        image_ids.extend(s[1])
    manual_name = path.stem
    return [{
        "doc_id": manual_name,
        "title": manual_name,
        "content": full_text,
        "image_refs": image_ids,
    }]


def embed_one(text: str) -> list[float]:
    payload = {"model": EMBED_MODEL, "input": [text[:20000]], "encoding_format": "float"}
    req = urllib.request.Request(
        f"{API_BASE.rstrip('/')}/embeddings",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["data"][0]["embedding"]
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
        if chunks:
            print(f"  {txt.name}: {len(chunks[0]['content'])} chars, {len(chunks[0]['image_refs'])} images")

    print(f"Total: {len(all_chunks)} chunks (one per manual)")

    with OUT_KNOWLEDGE.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    done = set()
    if OUT_EMBED.exists():
        with OUT_EMBED.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["doc_id"])
                except Exception:
                    pass
    print(f"Embedding {len(all_chunks) - len(done)} remaining...")

    with OUT_EMBED.open("a", encoding="utf-8") as out:
        for chunk in all_chunks:
            if chunk["doc_id"] in done:
                continue
            vec = embed_one(chunk["content"])
            out.write(json.dumps({"doc_id": chunk["doc_id"], "embedding": vec}, ensure_ascii=False) + "\n")
            print(f"  embedded: {chunk['doc_id']}")
            time.sleep(0.2)

    print("Done.")


if __name__ == "__main__":
    main()
