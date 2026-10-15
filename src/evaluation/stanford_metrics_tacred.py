"""Stanford TACRED evaluation protocol implementation.

This module provides a `score` function that computes the TACRED-style
micro-averaged precision, recall and F1 (excluding the `no_relation` class),
plus per-relation precision/recall/F1 and overall accuracy.

The API is intentionally small:
- `score(golds, preds, no_relation='no_relation') -> dict`
- `classification_report(scores)` prints a readable per-relation report

The implementation follows the original TACRED evaluation script semantics:
- Counts gold, predicted and correct examples per relation
- Computes micro P/R/F1 ignoring the `no_relation` relation
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple
import argparse
import json
import sys


def score(golds: Iterable[str], preds: Iterable[str], no_relation: str = "no_relation") -> Dict:
	"""Compute TACRED-style scores.

	Args:
		golds: iterable of gold relation labels (strings)
		preds: iterable of predicted relation labels (strings)
		no_relation: label name used for the negative class (default: 'no_relation')

	Returns:
		A dictionary with keys:
		- 'precision', 'recall', 'f1' : micro-averaged scores excluding `no_relation`
		- 'accuracy' : overall accuracy across all labels
		- 'total' : total number of examples
		- 'per_relation' : mapping relation -> {'precision','recall','f1','pred_count','gold_count','correct'}
	"""
	golds = list(map(str, list(golds)))
	preds = list(map(str, list(preds)))
	if len(golds) != len(preds):
		raise ValueError("Length mismatch: golds (%d) vs preds (%d)" % (len(golds), len(preds)))

	gold_count: Dict[str, int] = defaultdict(int)
	pred_count: Dict[str, int] = defaultdict(int)
	correct_count: Dict[str, int] = defaultdict(int)

	total = len(golds)
	total_correct = 0

	for g, p in zip(golds, preds):
		gold_count[g] += 1
		pred_count[p] += 1
		if g == p:
			correct_count[g] += 1
			total_correct += 1

	# Micro-averaged excluding the no_relation class
	gold_excluding = sum(v for k, v in gold_count.items() if k != no_relation)
	pred_excluding = sum(v for k, v in pred_count.items() if k != no_relation)
	correct_excluding = sum(v for k, v in correct_count.items() if k != no_relation)

	precision = float(correct_excluding) / pred_excluding if pred_excluding > 0 else 0.0
	recall = float(correct_excluding) / gold_excluding if gold_excluding > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

	per_relation: Dict[str, Dict] = {}
	relations = sorted(set(list(gold_count.keys()) + list(pred_count.keys())))
	for r in relations:
		c = correct_count.get(r, 0)
		p_cnt = pred_count.get(r, 0)
		g_cnt = gold_count.get(r, 0)
		p = float(c) / p_cnt if p_cnt > 0 else 0.0
		rcall = float(c) / g_cnt if g_cnt > 0 else 0.0
		if (p + rcall) > 0:
			f = 2 * p * rcall / (p + rcall)
		else:
			f = 0.0
		per_relation[r] = {
			"precision": p,
			"recall": rcall,
			"f1": f,
			"pred_count": p_cnt,
			"gold_count": g_cnt,
			"correct": c,
		}

	return {
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"accuracy": float(total_correct) / total if total > 0 else 0.0,
		"total": total,
		"per_relation": per_relation,
	}


def classification_report(scores: Dict, sort: bool = True, show_support: bool = True) -> str:
	"""Create a simple text classification report from `score` output.

	Returns a multi-line string suitable for printing.
	"""
	per = scores.get("per_relation", {})
	lines = []
	header = f"{'relation':30s} {'prec':>7s} {'rec':>7s} {'f1':>7s} {'pred':>7s} {'gold':>7s}"
	lines.append(header)
	lines.append("-" * len(header))
	items = sorted(per.items(), key=lambda x: x[0])
	if sort:
		items = sorted(per.items(), key=lambda kv: kv[1]["f1"], reverse=True)
	for r, m in items:
		line = f"{r:30s} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} {m['pred_count']:7d} {m['gold_count']:7d}"
		lines.append(line)

	lines.append("")
	lines.append(f"micro p/r/f1 (excluding no_relation): {scores['precision']:.4f} / {scores['recall']:.4f} / {scores['f1']:.4f}")
	lines.append(f"accuracy: {scores['accuracy']:.4f}  total: {scores['total']}" )
	return "\n".join(lines)


def _read_label_file(path: str) -> List[str]:
	"""Load labels from a file. Supports:
	- plain text: one label per line
	- JSON list: entire file is a JSON array of labels
	- JSONL: one JSON value (string) per line
	"""
	with open(path, "r", encoding="utf-8") as fh:
		text = fh.read().strip()
		if not text:
			return []
		# Try JSON array first
		try:
			obj = json.loads(text)
			if isinstance(obj, list):
				return [str(x) for x in obj]
		except Exception:
			pass
		# Try JSONL or plain lines
		labels = []
		for line in text.splitlines():
			line = line.strip()
			if not line:
				continue
			try:
				val = json.loads(line)
				# If the JSON value is an object with 'predicted_relation' or similar,
				# users should pre-process; we only accept direct string values here.
				if isinstance(val, str):
					labels.append(val)
				else:
					labels.append(str(val))
			except Exception:
				labels.append(line)
		return labels


def main(argv: List[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="TACRED-style evaluation for predictions vs gold labels")
	parser.add_argument("gold_file", help="Path to gold labels (json list, jsonl, or one label per line)")
	parser.add_argument("pred_file", help="Path to predictions (same formats supported)")
	parser.add_argument("--no-relation", default="no_relation", help="Label used for no_relation (default: no_relation)")
	parser.add_argument("--print-report", action="store_true", help="Print per-relation report")
	args = parser.parse_args(argv)

	golds = _read_label_file(args.gold_file)
	preds = _read_label_file(args.pred_file)
	if len(golds) != len(preds):
		print(f"ERROR: golds ({len(golds)}) and preds ({len(preds)}) have different lengths", file=sys.stderr)
		return 2

	scores = score(golds, preds, no_relation=args.no_relation)
	print(f"micro p/r/f1 (excluding {args.no_relation}): {scores['precision']:.4f} / {scores['recall']:.4f} / {scores['f1']:.4f}")
	print(f"accuracy: {scores['accuracy']:.4f}  total: {scores['total']}")
	if args.print_report:
		print("")
		print(classification_report(scores))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

