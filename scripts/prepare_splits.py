"""
Stream metadata from the BRSpeech-DF HuggingFace dataset and build balanced,
speaker-disjoint train/val/test split CSVs.

For each original HuggingFace split (train / validation / test) we keep every
bonafide example (the scarce class) and sample an equal number of spoof examples,
distributed across TTS models proportionally so no single model dominates. Because
balancing happens *within* each original split, the resulting train/val/test sets
inherit the dataset's original speaker-disjoint partition: no speaker appears in
more than one split.

Usage:
    python scripts/prepare_splits.py
    python scripts/prepare_splits.py --output-dir artifacts
"""

import argparse
import collections
import csv
import math
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# --- fixed configuration (never changes for this project) ------------------
DATASET_ID = "AKCIT-Deepfake/BRSpeech-DF"
SEED = 42

FIELDNAMES = ["split", "hf_index", "hf_split", "hf_config", "label", "model"]
HF_TO_NEW = {"train": "train", "validation": "val", "test": "test"}


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
    """
    Sample `budget` records, distributing slots across TTS models proportionally
    to each model's count. If a model has fewer examples than its quota, all are
    taken and leftover slots are redistributed to the remaining pool.
    """
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


def build_balanced_splits(
    bonafide: list[dict],
    spoof: list[dict],
    seed: int,
) -> dict[str, list[dict]]:
    """
    Within each original HuggingFace split, keep all bonafide and sample an equal,
    model-diverse set of spoof. Returns {"train": [...], "val": [...], "test": [...]}.
    """
    rng = random.Random(seed)

    bona_by_split: dict[str, list[dict]] = collections.defaultdict(list)
    spoof_by_split: dict[str, list[dict]] = collections.defaultdict(list)
    for r in bonafide:
        bona_by_split[r["hf_split"]].append(r)
    for r in spoof:
        spoof_by_split[r["hf_split"]].append(r)

    out: dict[str, list[dict]] = {}
    for hf_split in ("train", "validation", "test"):
        bf = bona_by_split[hf_split]
        sp = proportional_sample(spoof_by_split[hf_split], len(bf), rng)
        tag = HF_TO_NEW[hf_split]

        combined = bf + sp
        for r in combined:
            r["split"] = tag
        rng.shuffle(combined)
        out[tag] = combined

        print(f"  {tag:5s}: bonafide {len(bf):,} + spoof {len(sp):,} = {len(combined):,}")
    return out


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
        description="Build balanced, speaker-disjoint train/val/test split CSVs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write train.csv / val.csv / test.csv (default: artifacts/)",
    )
    args = parser.parse_args()

    print("Collecting bonafide metadata...")
    bonafide = collect_bonafide(DATASET_ID)
    print(f"      {len(bonafide):,} bonafide examples")

    print("Collecting spoof metadata...")
    spoof = collect_spoof(DATASET_ID)
    print(f"      {len(spoof):,} spoof examples")

    print("Building balanced, speaker-disjoint splits...")
    splits = build_balanced_splits(bonafide, spoof, seed=SEED)

    write_csv(splits["train"], args.output_dir / "train.csv")
    write_csv(splits["val"],   args.output_dir / "val.csv")
    write_csv(splits["test"],  args.output_dir / "test.csv")
    print_model_report(splits["train"], splits["val"], splits["test"])
    print("Done.")


if __name__ == "__main__":
    main()