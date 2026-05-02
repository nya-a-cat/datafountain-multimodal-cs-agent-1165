#!/usr/bin/env python3
"""Offline retrieval sanity checks.

This script does not call chat, embedding, rerank, or vision APIs. It only imports
the local knowledge base and validates that lexical/BM25 retrieval brings the
expected evidence into the compressed context.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("USE_QUERY_REWRITE", "0")

import baseline_api as api  # noqa: E402


CASES = [
    {
        "id": "lens_mount",
        "question": "How do you mount the lens of a camera when preparing for photography?",
        "must": ("Mounting a Lens", "Attach the lens", "EF-S"),
        "doc_prefix": "汇总英文手册::p",
    },
    {
        "id": "cf_card",
        "question": "How do you install the card into a camera before photography?",
        "must": ("Installing the Card", "Insert the CF card", "Close the cover"),
        "doc_prefix": "汇总英文手册::p",
    },
    {
        "id": "tpm",
        "question": "How do you know about TPM connector (14-1 pin TPM) for a motherboard?",
        "must": ("TPM connector", "Trusted Platform Module", "platform integrity"),
        "doc_prefix": "汇总英文手册::s",
    },
    {
        "id": "thermal_sensor",
        "question": "How do you know about Thermal sensor connector (2-pin T_SENSOR) of a motherboard?",
        "must": ("Thermal sensor connector", "T_SENSOR", "thermistor cable"),
        "doc_prefix": "汇总英文手册::s",
    },
    {
        "id": "t_rail",
        "question": "What are the detailed instructions for T-rail mounting of a camera or equipment?",
        "must": ("T-rail Mounting Instructions", "clips", "mount plate"),
        "doc_prefix": "汇总英文手册::s",
    },
    {
        "id": "dcb_lights",
        "question": "DCB107/DCB112指示灯含义？",
        "must": ("DCB107", "电池组充电中", "过热/过冷延迟"),
        "doc_prefix": "电钻手册::s",
    },
    {
        "id": "return_policy",
        "question": "请问你们家的商品支持7天无理由退换货吗？需要自己承担运费吗？",
        "must": ("7天无理由退换货", "非质量问题通常由买家承担", "质量问题由我们承担"),
        "doc_prefix": "",
    },
]


def main() -> None:
    failures: list[str] = []
    for case in CASES:
        question = case["question"]
        plan = api._classify_query(question, [])
        bundle = api._build_evidence_bundle(plan, [])
        text = bundle.compressed
        top_doc = bundle.docs[0].doc_id if bundle.docs else ""
        missing = [item for item in case["must"] if item not in text]
        wrong_doc = bool(case["doc_prefix"]) and not top_doc.startswith(case["doc_prefix"])
        status = "PASS" if not missing and not wrong_doc else "FAIL"
        print(f"{status} {case['id']} kind={plan.kind} top={top_doc}")
        if missing:
            print(f"  missing: {missing}")
        if wrong_doc:
            print(f"  expected top prefix: {case['doc_prefix']}")
        if status == "FAIL":
            failures.append(case["id"])
    if failures:
        raise SystemExit(f"offline retrieval sanity failed: {', '.join(failures)}")
    print(f"offline retrieval sanity passed: {len(CASES)} cases")


if __name__ == "__main__":
    main()
