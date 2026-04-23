#!/usr/bin/env python3
"""Validate submission CSV against public question file.

Checks:
1) Header must be: id,ret
2) Row count must match question file
3) Every id in question file appears exactly once
4) ret must be non-empty
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate submission csv format")
    parser.add_argument(
        "--question-file",
        type=Path,
        default=Path("data/question_public.csv"),
        help="Path to question_public.csv",
    )
    parser.add_argument(
        "--submission-file",
        type=Path,
        required=True,
        help="Path to generated submission csv",
    )
    return parser.parse_args()


def read_question_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("id", "")).strip()
            if qid:
                ids.append(qid)
    return ids


def validate_submission(question_ids: list[str], sub_file: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    seen: dict[str, int] = {}

    with sub_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["id", "ret"]:
            errors.append(f"invalid header: {header}, expected ['id', 'ret']")
            return False, errors

        rows = list(reader)

    if len(rows) != len(question_ids):
        errors.append(f"row count mismatch: submission={len(rows)} question={len(question_ids)}")

    for idx, row in enumerate(rows, start=2):
        if len(row) < 2:
            errors.append(f"line {idx}: column count < 2")
            continue
        qid = row[0].strip()
        ret = row[1].strip()
        if not qid:
            errors.append(f"line {idx}: empty id")
            continue
        if not ret:
            errors.append(f"line {idx}: empty ret")
        seen[qid] = seen.get(qid, 0) + 1

    for qid in question_ids:
        cnt = seen.get(qid, 0)
        if cnt == 0:
            errors.append(f"missing id: {qid}")
        elif cnt > 1:
            errors.append(f"duplicate id: {qid}, count={cnt}")

    return len(errors) == 0, errors


def main() -> None:
    args = parse_args()
    qids = read_question_ids(args.question_file)
    ok, errors = validate_submission(qids, args.submission_file)
    if ok:
        print("VALID: submission format passed")
        print(f"rows={len(qids)} file={args.submission_file}")
        return

    print("INVALID: submission format failed")
    for e in errors[:30]:
        print(f"- {e}")
    if len(errors) > 30:
        print(f"- ... and {len(errors) - 30} more")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
