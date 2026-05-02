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
    {
        "id": "missing_item_policy",
        "question": "我收到的商品少了一件，联系客服说会补发，但是过了一周还没补发！",
        "must": ("补发", "换货", "维修或退款"),
        "doc_prefix": "",
    },
    {
        "id": "blower_carburetor",
        "question": "使用吹风机时，如何调节化油器？",
        "must": ("化油器", "低速油针", "高速油针"),
        "doc_prefix": "吹风机手册::s",
    },
    {
        "id": "blower_cold_start",
        "question": "吹风机冷机时，该如何启动？",
        "must": ("冷机启动", "阻风门", "泵油"),
        "doc_prefix": "吹风机手册::s0034",
    },
    {
        "id": "blower_hot_start",
        "question": "吹风机热机时，该如何启动？",
        "must": ("热机启动", "阻风门", "启动油门"),
        "doc_prefix": "吹风机手册::s0034",
    },
    {
        "id": "blower_stop",
        "question": "该如何关闭吹风机？",
        "must": ("停机", "停机开关", "关闭发动机"),
        "doc_prefix": "吹风机手册::s0035",
    },
    {
        "id": "dishwasher_cutlery_basket",
        "question": "如何根据洗碗机型号安装餐具篮？",
        "must": ("餐具篮", "叉、勺", "视型号而定"),
        "doc_prefix": "洗碗机手册::s0038",
    },
    {
        "id": "watercraft_steering",
        "question": "How the ship steers?",
        "must": ("转向", "车把", "喷射推力喷嘴"),
        "doc_prefix": "摩托艇手册::s0038",
    },
    {
        "id": "jetski_start",
        "question": "How to start my jetski in different situations?",
        "must": ("启动发动机", "启动开关", "熄火绳"),
        "doc_prefix": "摩托艇手册::s0017",
    },
    {
        "id": "grill_safety_tips",
        "question": "Can you share me some safety tips using the grill?",
        "must": ("Safety Tips", "LP tank", "grease tray"),
        "doc_prefix": "汇总英文手册::s1236",
    },
    {
        "id": "function_keyboard_disclaimer",
        "question": "功能键盘的保修政策中，损害赔偿的除外责任或免责声明通常是什么？",
        "must": ("损害免责条款", "间接或继发性损害", "数据丢失"),
        "doc_prefix": "功能键盘手册::s0028",
    },
    {
        "id": "snowmobile_spark_plug",
        "question": "What are the steps to inspect the spark plug on a snowmobile?",
        "must": ("SPARK PLUG INSPECTION", "white porcelain insulator", "light tan color"),
        "doc_prefix": "汇总英文手册::s1402",
    },
    {
        "id": "lawn_mower_unload",
        "question": "How can you unload a lawn mower?",
        "must": ("Unloading the Machine", "ramp", "15 degrees"),
        "doc_prefix": "汇总英文手册::s2587",
    },
    {
        "id": "lawn_mower_engine_oil",
        "question": "How can you change the engine oil of a lawn mower?",
        "must": ("Changing the Engine Oil", "drains better", "Full mark"),
        "doc_prefix": "汇总英文手册::s2608",
    },
    {
        "id": "grill_leak_testing",
        "question": "To engure my safety when using the grill, how to do leak testing on valves, hose and regulatol?",
        "must": ("Leak Testing Valves", "Brush soapy solution", "growing"),
        "doc_prefix": "汇总英文手册::s1234",
    },
    {
        "id": "boat_factory_reset",
        "question": "In the boat's steering position, what does the factory reset screen show?",
        "must": ("Factory reset screen", "YES", "NO"),
        "doc_prefix": "汇总英文手册::s0757",
    },
    {
        "id": "boat_steering_check",
        "question": "How do I check the boat's steering system when I'm driving it?",
        "must": ("Steering system checks", "jet thrust nozzles", "starboard"),
        "doc_prefix": "汇总英文手册::s0865",
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
