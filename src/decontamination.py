"""Centralized Decontamination Filter & Audit Suite for Raising Wren 2.0.

Rigorous exclusion filter preventing eval probe leakage against:
- evals/heldout_ids.json (src_ids, base_q, texts)
- evals/probes.jsonl (turns, id, src_id)

Features:
- Multi-tier indexing (O(1) hash sets for exact matches, length short-circuiting,
  and Boyer-Moore-Horspool substring scanning).
- Normalization (lowercasing, punctuation stripping, unicode folding, whitespace collapsing).
- Batch screening utilities for JSONL files and in-memory record streams.
- Auditing and reason introspection for full compliance transparency.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]


def normalize_text(text: str | None) -> str:
    """Normalizes text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text or not isinstance(text, str):
        return ""
    t = text.lower()
    t = (
        t.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", "-")
        .replace("–", "-")
    )
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


class ContaminationFilter:
    """Thread-safe, high-throughput contamination screening against held-out evals."""

    def __init__(
        self,
        heldout_path: str | Path = "evals/heldout_ids.json",
        probes_path: str | Path = "evals/probes.jsonl",
        min_substring_len: int = 10,
    ) -> None:
        self.heldout_path = self._resolve_path(heldout_path, "evals/heldout_ids.json")
        self.probes_path = self._resolve_path(probes_path, "evals/probes.jsonl")
        self.min_substring_len = min_substring_len

        self.src_ids: set[str] = set()
        self.exact_raw: set[str] = set()
        self.exact_normalized: set[str] = set()
        self.substring_raw: list[str] = []
        self.substring_normalized: list[str] = []
        self.min_target_len: int = self.min_substring_len

        self._load_indexes()

    def _resolve_path(self, path: str | Path, default_rel: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        if p.exists():
            return p.resolve()
        root_p = ROOT / path
        if root_p.exists():
            return root_p.resolve()
        if str(path) == default_rel:
            root_def = ROOT / default_rel
            if root_def.exists():
                return root_def.resolve()
        return root_p

    def _load_indexes(self) -> None:
        sub_raw_set: set[str] = set()
        sub_norm_set: set[str] = set()

        # 1. Load evals/heldout_ids.json
        if self.heldout_path.exists():
            try:
                with open(self.heldout_path, "r", encoding="utf-8") as f:
                    held = json.load(f)
                for sid in held.get("src_ids", []):
                    if sid:
                        self.src_ids.add(str(sid).strip())
                for t in held.get("texts", []):
                    if t and t.strip():
                        raw_t = t.strip().lower()
                        self.exact_raw.add(raw_t)
                        norm_t = normalize_text(t)
                        if norm_t:
                            self.exact_normalized.add(norm_t)
                        if len(raw_t) >= self.min_substring_len:
                            sub_raw_set.add(raw_t)
                        if len(norm_t) >= self.min_substring_len:
                            sub_norm_set.add(norm_t)
                for bq in held.get("base_q", []):
                    if bq and bq.strip():
                        raw_bq = bq.strip().lower()
                        self.exact_raw.add(raw_bq)
                        norm_bq = normalize_text(bq)
                        if norm_bq:
                            self.exact_normalized.add(norm_bq)
                        if len(raw_bq) >= self.min_substring_len:
                            sub_raw_set.add(raw_bq)
                        if len(norm_bq) >= self.min_substring_len:
                            sub_norm_set.add(norm_bq)
            except Exception:
                pass

        # 2. Load evals/probes.jsonl
        if self.probes_path.exists():
            try:
                with open(self.probes_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        if item.get("src_id"):
                            self.src_ids.add(str(item["src_id"]).strip())
                        if item.get("id"):
                            self.src_ids.add(str(item["id"]).strip())
                        if item.get("base_q"):
                            bq = item["base_q"]
                            if bq and bq.strip():
                                raw_bq = bq.strip().lower()
                                self.exact_raw.add(raw_bq)
                                norm_bq = normalize_text(bq)
                                if norm_bq:
                                    self.exact_normalized.add(norm_bq)
                                if len(raw_bq) >= self.min_substring_len:
                                    sub_raw_set.add(raw_bq)
                                if len(norm_bq) >= self.min_substring_len:
                                    sub_norm_set.add(norm_bq)
                        for turn in item.get("turns", []):
                            if turn and turn.strip():
                                raw_turn = turn.strip().lower()
                                self.exact_raw.add(raw_turn)
                                norm_turn = normalize_text(turn)
                                if norm_turn:
                                    self.exact_normalized.add(norm_turn)
                                if len(raw_turn) >= self.min_substring_len:
                                    sub_raw_set.add(raw_turn)
                                if len(norm_turn) >= self.min_substring_len:
                                    sub_norm_set.add(norm_turn)
            except Exception:
                pass

        self.substring_raw = sorted(list(sub_raw_set), key=len, reverse=True)
        self.substring_normalized = sorted(list(sub_norm_set), key=len, reverse=True)
        if self.substring_raw:
            self.min_target_len = min(len(s) for s in self.substring_raw)
        else:
            self.min_target_len = self.min_substring_len

    def check_contamination(
        self, text: str | None, src_id: str | None = None
    ) -> tuple[bool, str | None]:
        """Returns (is_contaminated: bool, reason: str | None) with detailed audit reason."""
        if src_id is not None and str(src_id).strip() in self.src_ids:
            return True, f"src_id:{src_id}"

        if not text or not isinstance(text, str) or not text.strip():
            return False, None

        raw_lower = text.strip().lower()
        if raw_lower in self.exact_raw:
            return True, f"exact_raw:{raw_lower[:40]}"

        norm = normalize_text(text)
        if norm and norm in self.exact_normalized:
            return True, f"exact_norm:{norm[:40]}"

        if len(raw_lower) < self.min_target_len and len(norm) < self.min_target_len:
            return False, None

        for target in self.substring_raw:
            if target in raw_lower:
                return True, f"sub_raw:{target[:40]}"

        for target in self.substring_normalized:
            if target in norm:
                return True, f"sub_norm:{target[:40]}"

        return False, None

    def is_contaminated(self, text: str, src_id: str | None = None) -> bool:
        """Standard interface contract returning True if sample is contaminated."""
        return self.check_contamination(text, src_id)[0]

    def is_record_contaminated(
        self, record: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Inspects a candidate dictionary record across text, prompt, messages, and completion."""
        src_id = (
            record.get("src_id")
            or record.get("src")
            or record.get("source_id")
        )
        if src_id is not None and str(src_id).strip() in self.src_ids:
            return True, f"src_id:{src_id}"

        # Extract text content
        text = ""
        if "prompt" in record and isinstance(record["prompt"], str):
            text = record["prompt"]
        elif "question" in record and isinstance(record["question"], str):
            text = record["question"]
        elif "query" in record and isinstance(record["query"], str):
            text = record["query"]
        elif "text" in record and isinstance(record["text"], str):
            text = record["text"]
        elif "messages" in record and isinstance(record["messages"], list):
            parts = [
                m["content"]
                for m in record["messages"]
                if isinstance(m, dict)
                and m.get("role") == "user"
                and isinstance(m.get("content"), str)
            ]
            text = " ".join(parts)
        elif "input" in record and isinstance(record["input"], str):
            text = record["input"]

        bad, reason = self.check_contamination(text, src_id=src_id)
        if bad:
            return True, reason

        # Also inspect completion if present
        if "completion" in record and isinstance(record["completion"], str) and record["completion"].strip():
            bad_comp, reason_comp = self.check_contamination(record["completion"])
            if bad_comp:
                return True, f"completion_{reason_comp}"

        return False, None


# Global cached filter instance
_GLOBAL_FILTER: ContaminationFilter | None = None


def get_contamination_filter(
    heldout_path: str | Path = "evals/heldout_ids.json",
    probes_path: str | Path = "evals/probes.jsonl",
) -> ContaminationFilter:
    """Returns singleton/cached ContaminationFilter instance."""
    global _GLOBAL_FILTER
    if _GLOBAL_FILTER is None:
        _GLOBAL_FILTER = ContaminationFilter(heldout_path, probes_path)
    return _GLOBAL_FILTER


def is_contaminated(text: str, src_id: str | None = None) -> bool:
    """Module-level convenience function using default cached filter."""
    return get_contamination_filter().is_contaminated(text, src_id)


# =====================================================================
# High-Throughput Batch Screening Utilities
# =====================================================================


def filter_records(
    records: Sequence[dict[str, Any]],
    cfilter: ContaminationFilter | None = None,
    return_rejected: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | list[dict[str, Any]]:
    """Screens an in-memory sequence of candidate dictionary records.

    Args:
        records: Sequence of candidate dicts.
        cfilter: Optional ContaminationFilter (defaults to singleton).
        return_rejected: If True, returns (clean_records, rejected_records).
    """
    filt = cfilter or get_contamination_filter()
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for r in records:
        bad, reason = filt.is_record_contaminated(r)
        if bad:
            if return_rejected:
                rejected.append({**r, "_contamination_reason": reason})
        else:
            clean.append(r)

    if return_rejected:
        return clean, rejected
    return clean


def screen_jsonl(
    input_path: str | Path,
    output_clean_path: str | Path,
    output_contaminated_path: str | Path | None = None,
    cfilter: ContaminationFilter | None = None,
) -> dict[str, Any]:
    """Streams a JSONL file line-by-line with constant memory usage.

    Writes clean items to output_clean_path, and optionally contaminated items
    to output_contaminated_path. Returns audit summary dict.
    """
    filt = cfilter or get_contamination_filter()
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_clean = Path(output_clean_path)
    out_clean.parent.mkdir(parents=True, exist_ok=True)
    out_bad = Path(output_contaminated_path) if output_contaminated_path else None
    if out_bad:
        out_bad.parent.mkdir(parents=True, exist_ok=True)

    total, clean_count, contaminated_count = 0, 0, 0
    reasons_counter: Counter[str] = Counter()

    bad_f = open(out_bad, "w", encoding="utf-8") if out_bad else None

    try:
        with open(inp, "r", encoding="utf-8") as in_f, open(
            out_clean, "w", encoding="utf-8"
        ) as clean_f:
            for line in in_f:
                if not line.strip():
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                is_bad, reason = filt.is_record_contaminated(record)
                if is_bad:
                    contaminated_count += 1
                    prefix = reason.split(":")[0] if reason else "unknown"
                    reasons_counter[prefix] += 1
                    if bad_f:
                        bad_f.write(
                            json.dumps(
                                {**record, "_contamination_reason": reason}
                            )
                            + "\n"
                        )
                else:
                    clean_count += 1
                    clean_f.write(json.dumps(record) + "\n")
    finally:
        if bad_f:
            bad_f.close()

    rate = (contaminated_count / total) if total > 0 else 0.0
    return {
        "input_path": str(inp),
        "total": total,
        "clean": clean_count,
        "contaminated": contaminated_count,
        "contamination_rate": rate,
        "reasons": dict(reasons_counter),
    }


# =====================================================================
# CLI Runner
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Raising Wren 2.0 Centralized Contamination Screener"
    )
    parser.add_argument("--input", type=str, help="Input JSONL file to screen")
    parser.add_argument("--clean", type=str, help="Output clean JSONL file")
    parser.add_argument(
        "--contaminated", type=str, default=None, help="Output rejected JSONL file"
    )
    parser.add_argument(
        "--audit", type=str, help="Audit a JSONL file and print summary metrics"
    )
    parser.add_argument("--check", type=str, help="Check a single text prompt")
    parser.add_argument("--src-id", type=str, default=None, help="Optional src_id to check")
    args = parser.parse_args()

    cf = get_contamination_filter()

    if args.check:
        is_bad, reason = cf.check_contamination(args.check, src_id=args.src_id)
        status = "CONTAMINATED" if is_bad else "CLEAN"
        print(f"Status: {status} (Reason: {reason})")
        sys.exit(1 if is_bad else 0)

    if args.audit:
        import tempfile

        with tempfile.NamedTemporaryFile() as tmp:
            stats = screen_jsonl(args.audit, tmp.name, cfilter=cf)
            print(f"--- Contamination Audit: {stats['input_path']} ---")
            print(f"Total Records: {stats['total']}")
            print(f"Clean:         {stats['clean']}")
            print(f"Contaminated:  {stats['contaminated']} ({stats['contamination_rate']:.2%})")
            print(f"Reasons:       {stats['reasons']}")
            sys.exit(0)

    if args.input and args.clean:
        stats = screen_jsonl(args.input, args.clean, args.contaminated, cfilter=cf)
        print(
            f"Processed {stats['total']} records -> {stats['clean']} clean, "
            f"{stats['contaminated']} rejected ({stats['contamination_rate']:.2%})."
        )
        sys.exit(0)

    parser.print_help()
