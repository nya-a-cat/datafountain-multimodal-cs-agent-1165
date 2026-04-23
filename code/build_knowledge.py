#!/usr/bin/env python3
"""Build knowledge.jsonl from official KownledgeBase.zip."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import zipfile
from pathlib import Path


def _sanitize_invalid_escapes(raw: str) -> str:
    # Replace any backslash not followed by a valid JSON escape character
    return re.sub(r"\\([^\\\"/bfnrtu0-9])", r"\\\\\1", raw)


def _decode_json_values(raw: str) -> list[object]:
    decoder = json.JSONDecoder()
    values: list[object] = []
    idx = 0
    n = len(raw)
    while idx < n:
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break
        value, end = decoder.raw_decode(raw, idx)
        values.append(value)
        idx = end
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-file", type=Path, default=Path("data/KownledgeBase.zip"))
    parser.add_argument("--kb-dir", type=Path, default=None, help="Extracted KownledgeBase directory")
    parser.add_argument("--out-file", type=Path, default=Path("data/knowledge.jsonl"))
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=400)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(text: str) -> list[tuple[str, str]]:
    # Normalize inline "# Title content # Title2 content2" to newline-separated
    text = re.sub(r"\s+#\s+", "\n# ", text)
    sections: list[tuple[str, str]] = []
    current_title = "概述"
    buf: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line.startswith("#"):
            if buf:
                sections.append((current_title, "\n".join(buf).strip()))
                buf = []
            # Split "# Title remaining content" — title is first word group, rest is content
            rest = line.lstrip("#").strip()
            # Find where title ends: up to first sentence-ending punctuation or 20 chars
            m = re.match(r"^([^。！？!?\n]{1,30}?)\s+(.+)$", rest)
            if m:
                current_title = m.group(1).strip()
                buf.append(m.group(2).strip())
            else:
                current_title = rest or "未命名小节"
        else:
            buf.append(line)
    if buf:
        sections.append((current_title, "\n".join(buf).strip()))
    return sections


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？!?；;\n])", text)
    return [p.strip() for p in pieces if p.strip()]


def chunk_with_overlap(sentences: list[str], max_chars: int, overlap: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    i = 0
    while i < len(sentences):
        j, length = i, 0
        while j < len(sentences) and length + len(sentences[j]) <= max_chars:
            length += len(sentences[j])
            j += 1
        if j == i:
            j = i + 1
        chunks.append(sentences[i:j])
        if j >= len(sentences):
            break
        # step back by overlap chars to create overlap
        back, k = 0, j - 1
        while k > i and back < overlap:
            back += len(sentences[k])
            k -= 1
        i = k + 1 if back >= overlap else j
    return chunks


def build_records(name: str, text: str, image_ids: list[str], max_chars: int, overlap: int) -> list[dict]:
    manual_name = Path(name).stem.replace("手册", "").strip()
    slug = re.sub(r"[^\w一-鿿]+", "-", manual_name).strip("-") or "manual"

    sections = split_sections(normalize_text(text))
    records: list[dict] = []
    image_ptr = 0

    for sec_idx, (title, sec_text) in enumerate(sections, start=1):
        pic_count = sec_text.count("<PIC>")
        sec_image_ids = image_ids[image_ptr: image_ptr + pic_count]
        image_ptr += pic_count

        content = re.sub(r"\s*<PIC>\s*", " ", sec_text).strip()
        if not content:
            continue

        sentences = split_sentences(content)
        if not sentences:
            continue

        for chunk_idx, sents in enumerate(chunk_with_overlap(sentences, max_chars, overlap), start=1):
            chunk_text = " ".join(sents)
            records.append({
                "doc_id": f"{slug}-s{sec_idx:03d}-c{chunk_idx:02d}",
                "title": f"{manual_name} / {title}",
                "content": f"【{manual_name}】{title}：{chunk_text}",
                "image_refs": list(dict.fromkeys(sec_image_ids)),
            })

    return records


def load_manuals(zip_file: Path, kb_dir: Path | None = None) -> list[tuple[str, str, list[str]]]:
    out: list[tuple[str, str, list[str]]] = []

    def _parse_file(name: str, raw: str) -> None:
        try:
            values = _decode_json_values(raw)
        except json.JSONDecodeError:
            values = _decode_json_values(_sanitize_invalid_escapes(raw))
        for obj in values:
            if isinstance(obj, list) and len(obj) == 2:
                text, ids = obj
                if isinstance(text, str) and isinstance(ids, list):
                    out.append((name, text, [str(x) for x in ids]))

    if kb_dir is not None and kb_dir.exists():
        for p in sorted((kb_dir / "手册").glob("*手册.txt")):
            _parse_file(p.name, p.read_text(encoding="utf-8"))
    else:
        with zipfile.ZipFile(zip_file) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith("手册.txt"):
                    continue
                raw = zf.read(name).decode("utf-8")
                _parse_file(name, raw)
    return out


def main() -> None:
    args = parse_args()
    kb_dir = args.kb_dir
    if kb_dir is None and not args.zip_file.exists():
        # auto-detect extracted directory
        auto = args.zip_file.parent / "KownledgeBase"
        if auto.exists():
            kb_dir = auto

    manuals = load_manuals(args.zip_file, kb_dir)
    records: list[dict] = []
    for name, text, image_ids in manuals:
        records.extend(build_records(name, text, image_ids, args.max_chars, args.overlap))

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"manual_files={len(manuals)}")
    print(f"knowledge_docs={len(records)}")
    print(f"out_file={args.out_file}")


if __name__ == "__main__":
    main()
