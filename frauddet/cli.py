"""CLI: ``python -m frauddet.cli train --out model_dir`` and ``... score --model model_dir --input txns.csv``."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .data import generate_transactions
from .model import load, save, score, train


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="frauddet")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="generate synthetic data, train, evaluate, save")
    t.add_argument("--out", default="artifacts")
    t.add_argument("--n", type=int, default=20_000)
    t.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score", help="score a CSV of transactions")
    s.add_argument("--model", default="artifacts")
    s.add_argument("--input", required=True)
    args = p.parse_args(argv)
    try:
        if args.cmd == "train":
            df = generate_transactions(n=args.n, seed=args.seed)
            model, result = train(df)
            save(model, result, args.out)
            print(json.dumps(result.__dict__, indent=2, default=list))
        else:
            df = pd.read_csv(args.input).sort_values("ts", kind="mergesort").reset_index(drop=True)
            print(score(df, load(args.model)).to_csv(index=False), end="")
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
