#!/usr/bin/env python3
"""Build retrievable knowledge chunks directly from manual txt files."""

from __future__ import annotations

import json
import re
from pathlib import Path

MANUALS_DIR = Path("data/KownledgeBase/手册")
OUT_KNOWLEDGE = Path("data/knowledge_v2.jsonl")
PROCEDURE_WINDOW = 5


def _is_heading_only(text: str) -> bool:
    body = text.strip().lstrip("# ").strip()
    return len(body) < 45 and "\n" not in body and "<PIC>" not in body


def _looks_like_procedure_heading(text: str) -> bool:
    title = text.strip().lstrip("# ").lower()
    keywords = (
        "install", "mount", "remove", "replace", "connect", "use", "set",
        "turn", "open", "close", "charging", "recharging", "operation",
        "安装", "拆卸", "连接", "使用", "设置", "打开", "关闭", "更换", "充电",
    )
    return _is_heading_only(text) and any(keyword in title for keyword in keywords)


def _fix_loose_json_escapes(raw: str) -> str:
    valid = set('"\\' + "/bfnrtu\n")
    buf: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "\\" and i + 1 < len(raw):
            if raw[i + 1] == "\\":
                buf.append("\\\\")
                i += 2
            elif raw[i + 1] in valid:
                buf.append(raw[i])
                i += 1
            else:
                buf.append("\\\\")
                i += 1
        else:
            buf.append(raw[i])
            i += 1
    return "".join(buf)


def _parse_manual_segments(raw: str) -> list[tuple[str, list[str]]]:
    fixed = _fix_loose_json_escapes(raw.strip())
    decoder = json.JSONDecoder()
    pos = 0
    segments: list[tuple[str, list[str]]] = []
    while pos < len(fixed):
        chunk = fixed[pos:].lstrip()
        if not chunk:
            break
        try:
            data, end = decoder.raw_decode(chunk)
        except json.JSONDecodeError:
            break
        if (
            isinstance(data, list)
            and len(data) >= 2
            and isinstance(data[0], str)
            and isinstance(data[1], list)
        ):
            segments.append((data[0], [str(item) for item in data[1]]))
        pos += len(fixed[pos:]) - len(chunk) + end
    if segments:
        return segments

    img_match = re.search(r'\[("[^"]*"(?:,\s*"[^"]*")*)\]\s*\]?\s*$', raw)
    image_ids = re.findall(r'"([^"]+)"', img_match.group(1)) if img_match else []
    text_match = re.match(r'^\[?"(.*)"(?:,|\s*\[)', raw, re.DOTALL)
    if text_match:
        full_text = text_match.group(1).replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
    else:
        full_text = raw.strip('[]"')
    return [(full_text, image_ids)]


def _parse_manual(path: Path) -> list[dict]:
    segments = _parse_manual_segments(path.read_text(encoding="utf-8"))
    manual_name = path.stem
    chunks: list[dict] = []
    atomic_chunks: list[dict] = []
    section_idx = 0
    for full_text, image_ids in segments:
        pic_counter = 0
        for sec in re.split(r"(?=# )", full_text):
            sec = sec.strip()
            if not sec:
                continue
            pics_in_sec = sec.count("<PIC>")
            refs = [
                image_ids[pic_counter + i]
                for i in range(pics_in_sec)
                if pic_counter + i < len(image_ids)
            ]
            pic_counter += pics_in_sec
            title = sec.split("\n")[0].lstrip("# ").strip()
            chunk = {
                "doc_id": f"{manual_name}::s{section_idx:04d}",
                "chunk_type": "atomic",
                "title": f"{manual_name} / {title}",
                "content": sec,
                "image_refs": refs,
            }
            chunks.append(chunk)
            atomic_chunks.append(chunk)
            section_idx += 1
    chunks.extend(_build_procedure_chunks(manual_name, atomic_chunks))
    return chunks


def _build_procedure_chunks(manual_name: str, atomic_chunks: list[dict]) -> list[dict]:
    procedure_chunks: list[dict] = []
    for idx, chunk in enumerate(atomic_chunks):
        if not _looks_like_procedure_heading(str(chunk["content"])):
            continue
        selected = [chunk]
        for nxt in atomic_chunks[idx + 1 : idx + PROCEDURE_WINDOW]:
            if _is_heading_only(str(nxt["content"])) and len(selected) > 1:
                break
            selected.append(nxt)
        if len(selected) < 2:
            continue
        image_refs: list[str] = []
        for item in selected:
            for ref in item.get("image_refs", []):
                if ref not in image_refs:
                    image_refs.append(ref)
        title = str(chunk["title"])
        content = "\n".join(str(item["content"]) for item in selected)
        procedure_chunks.append({
            "doc_id": f"{manual_name}::p{idx:04d}",
            "chunk_type": "procedure",
            "source_sections": [str(item["doc_id"]) for item in selected],
            "title": title,
            "content": content,
            "image_refs": image_refs,
        })
    return procedure_chunks


def main() -> None:
    all_chunks: list[dict] = []
    for txt in sorted(MANUALS_DIR.glob("*.txt")):
        chunks = _parse_manual(txt)
        all_chunks.extend(chunks)
        print(f"  {txt.name}: {len(chunks)} chunks")

    print(f"Total: {len(all_chunks)} chunks")
    with OUT_KNOWLEDGE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Wrote knowledge to {OUT_KNOWLEDGE}")


if __name__ == "__main__":
    main()
