"""
Stream metadata from a HuggingFace dataset, build balanced train/val/test splits (equal bonafide/spoof counts, spoof diversity-aware sampling), and write one CSV per split

Usage: python scripts/prepare_splits.py
Custom paths / ratios
python scripts/prepare_splits.py \
    --dataset-id AKCIT-Deepfake/BRSpeech-DF \
    --output-dir artifacts \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --seed 42
"""

import argparse
import collections
import csv
import math
import os
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

FIELDNAMES = ["split", "hf_index", "hf_split", "hf_config", "label", "model"]


def collect_bonafide(dataset_id: str) -> list[dict]:
    records = []
    for hf_split in ("train", "validation", "test"):
        ds = load_dataset(dataset_id, name="bonafide", split=hf_split, streaming=True)
        n = ds.info.splits[hf_split].num_examples if ds.info and ds.info.splits else None
        for idx, _ in tqdm(enumerate(ds), total=n, desc=f"bonafide/{hf_split}"):
            records.append({
                "hf_index":  idx,
                "hf_split":  hf_split,
                "hf_config": "bonafide",
                "label":     0,
                "model":     "bonafide",
            })
    return records


def collect_spoof(dataset_id: str) -> list[dict]:
    records = []
    for hf_split in ("train", "validation", "test"):
        ds = load_dataset(dataset_id, name="spoof", split=hf_split, streaming=True)
        n = ds.info.splits[hf_split].num_examples if ds.info and ds.info.splits else None
        for idx, example in tqdm(enumerate(ds), total=n, desc=f"spoof/{hf_split}"):
            records.append({
                "hf_index":  idx,
                "hf_split":  hf_split,
                "hf_config": "spoof",
                "label":     1,
                "model":     example.get("model", "unknown"),
            })
    return records

def proportional_sample(records: list[dict], budget: int, rng: random.Random) -> list[dict]:
    by_model: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)

    total = sum(len(v) for v in by_model.values())
    budget = min(budget, total)

    quotas = {m: round(len(v) / total * budget) for m, v in by_model.items()}
    diff = budget - sum(quotas.values())
    for m in list(quotas)[: abs(diff)]:
        quotas[m] += int(math.copysign(1, diff))

    sampled: list[dict] = []
    overflow = 0
    sampled_ids: set[int] = set()

    for model, recs in by_model.items():
        quota = quotas[model]
        if len(recs) <= quota:
            sampled.extend(recs)
            sampled_ids.update(id(r) for r in recs)
            overflow += quota - len(recs)
        else:
            chosen = rng.sample(recs, quota)
            sampled.extend(chosen)
            sampled_ids.update(id(r) for r in chosen)

    if overflow:
        surplus = [r for r in records if id(r) not in sampled_ids]
        sampled.extend(rng.sample(surplus, min(overflow, len(surplus))))

    return sampled

def make_balanced_splits(
    bonafide: list[dict],
    spoof: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:

    rng = random.Random(seed)
    rng.shuffle(bonafide)
    n = len(bonafide)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    bf_train = bonafide[:n_train]
    bf_val   = bonafide[n_train: n_train + n_val]
    bf_test  = bonafide[n_train + n_val:]

    print(f"Bonafide train: {len(bf_train):,}  val: {len(bf_val):,}  test: {len(bf_test):,}")

    rng.shuffle(spoof)
    sp_train = proportional_sample(spoof, len(bf_train), rng)

    sp_train_ids = {id(r) for r in sp_train}
    remaining = [r for r in spoof if id(r) not in sp_train_ids]
    sp_val = proportional_sample(remaining, len(bf_val), rng)

    sp_val_ids = {id(r) for r in sp_val}
    remaining2 = [r for r in remaining if id(r) not in sp_val_ids]
    sp_test = proportional_sample(remaining2, len(bf_test), rng)

    print(f"Spoof train: {len(sp_train):,}  val: {len(sp_val):,}  test: {len(sp_test):,}")

    def combine(bf, sp, tag):
        combined = bf + sp
        for r in combined:
            r["split"] = tag
        rng.shuffle(combined)
        return combined

    return combine(bf_train, sp_train, "train"), combine(bf_val, sp_val, "val"), combine(bf_test, sp_test, "test")

def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    print(f"  saved {path}  ({len(records):,} rows)")


def print_model_report(train: list[dict], val: list[dict], test: list[dict]) -> None:
    all_models = sorted(
        {r["model"] for split in (train, val, test) for r in split if r["label"] == 1}
    )
    header = f"{'model':<25} {'train':>8} {'val':>8} {'test':>8}"
    print("TTS model distribution in spoof examples")
    print(header)
    print("-" * len(header))
    for model in all_models:
        t = sum(1 for r in train if r["model"] == model)
        v = sum(1 for r in val   if r["model"] == model)
        s = sum(1 for r in test  if r["model"] == model)
        print(f"  {model:<23} {t:>8,} {v:>8,} {s:>8,}")

def main():
    parser = argparse.ArgumentParser(
        description="Build balanced train/val/test split CSVs from a HuggingFace dataset."
    )
    parser.add_argument(
        "--dataset-id",
        default="AKCIT-Deepfake/BRSpeech-DF",
        help="HuggingFace dataset repo ID (default: AKCIT-Deepfake/BRSpeech-DF)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write train.csv / val.csv / test.csv (default: artifacts/)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio",   type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    print("Collecting bonafide metadata...")
    bonafide = collect_bonafide(args.dataset_id)
    print(f"      {len(bonafide):,} bonafide examples")

    print("Collecting spoof metadata...")
    spoof = collect_spoof(args.dataset_id)
    print(f"      {len(spoof):,} spoof examples")

    print("Building balanced splits...")
    train_set, val_set, test_set = make_balanced_splits(
        bonafide, spoof,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    write_csv(train_set, args.output_dir / "train.csv")
    write_csv(val_set,   args.output_dir / "val.csv")
    write_csv(test_set,  args.output_dir / "test.csv")
    print_model_report(train_set, val_set, test_set)
    print("Done.")


if __name__ == "__main__":
    main()