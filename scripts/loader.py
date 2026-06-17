from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from datasets import load_dataset, Audio


class SplitLoader:
    """
    Parameters
    dataset_id : str
        HuggingFace dataset repo, e.g. "AKCIT-Deepfake/BRSpeech-DF".
    splits_dir : str | Path
        Directory containing train.csv, val.csv, test.csv.
    features_dir : str | Path | None
        Root directory where <split>/y.npy will be written.
        Defaults to <splits_dir>/features.
    label_col : str
        CSV column for the ground-truth label. Defaults to "label".
    """

    _SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}

    def __init__(
        self,
        dataset_id: str,
        splits_dir: str | Path,
        features_dir: str | Path | None = None,
        label_col: str = "label",
    ):
        self.dataset_id = dataset_id
        self.splits_dir = Path(splits_dir)
        self.features_dir = (
            Path(features_dir) if features_dir is not None
            else self.splits_dir / "features"
        )
        self.label_col = label_col

        self._meta: dict[str, pd.DataFrame] = {}
        for split, fname in self._SPLIT_FILES.items():
            csv_path = self.splits_dir / fname
            if csv_path.exists():
                self._meta[split] = pd.read_csv(csv_path).reset_index(drop=True)

        if not self._meta:
            raise FileNotFoundError(
                f"No split CSVs found in {self.splits_dir}. "
                "Expected train.csv, val.csv, test.csv."
            )
          
    # Public API

    def stats(self, split: str) -> dict:
        df = self._get_meta(split)
        counts: dict = {"total": len(df)}
        if self.label_col in df.columns:
            vc = df[self.label_col].value_counts()
            counts["bonafide"] = int(vc.get(0, 0))
            counts["spoof"] = int(vc.get(1, 0))
        return counts

    def print_stats(self) -> None:
        print(f"{'split':<8} {'total':>8} {'bonafide':>10} {'spoof':>8}")
        print("-" * 38)
        for split in ("train", "val", "test"):
            if split in self._meta:
                s = self.stats(split)
                print(
                    f"{split:<8} {s['total']:>8} "
                    f"{s.get('bonafide', '?'):>10} "
                    f"{s.get('spoof', '?'):>8}"
                )

    def stream(self, split: str) -> Iterator[tuple[np.ndarray, int, int, dict]]:
        """
        Yield (audio, sample_rate, label, meta) for every row in the split
        CSV, in CSV row order.

        audio       : np.ndarray, float32, shape (n_samples,)
        sample_rate : int
        label       : int  (0 = bonafide, 1 = spoof)
        meta        : dict with all CSV columns for this row
        """
        df = self._get_meta(split)

        # audio_store[csv_row_position] = (array, sr)
        audio_store: dict[int, tuple[np.ndarray, int]] = {}

        # Group CSV rows by (hf_config, hf_split) so we only stream each
        # HF split once, sequentially
        groups = defaultdict(list)
        for csv_pos, row in df.iterrows():
            key = (str(row["hf_config"]), str(row["hf_split"]))
            groups[key].append((csv_pos, int(row["hf_index"])))

        for (hf_config, hf_split), members in groups.items():
            # Sort by hf_index so we stream sequentially — no skipping forward
            members_sorted = sorted(members, key=lambda x: x[1])
            needed = {hf_index: csv_pos for csv_pos, hf_index in members_sorted}

            print(f"Streaming HuggingFace (config={hf_config}, split={hf_split}), "
                  f"need {len(needed)} examples ...")

            ds = load_dataset(
                self.dataset_id,
                name=hf_config,
                split=hf_split,
                streaming=True,
            )
            ds = ds.cast_column("audio", Audio(sampling_rate=None))

            max_index = max(needed.keys())
            for current_index, example in enumerate(ds):
                if current_index in needed:
                    csv_pos = needed[current_index]
                    array = np.array(example["audio"]["array"], dtype=np.float32)
                    sr = int(example["audio"]["sampling_rate"])
                    audio_store[csv_pos] = (array, sr)
                if current_index >= max_index:
                    break  # no need to stream further

        # Yield in original CSV row order
        labels = []
        for csv_pos, row in df.iterrows():
            array, sr = audio_store[csv_pos]
            label = int(row[self.label_col])
            labels.append(label)
            yield array, sr, label, row.to_dict()

        self._save_y(split, np.array(labels, dtype=np.int64))

    # Internal helpers

    def _get_meta(self, split: str) -> pd.DataFrame:
        if split not in self._meta:
            raise KeyError(
                f"Split '{split}' not found. Available: {list(self._meta.keys())}"
            )
        return self._meta[split]

    def _save_y(self, split: str, labels: np.ndarray) -> None:
        out_path = self.features_dir / split / "y.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, labels)
        print(f"[{split}] saved y.npy  shape={labels.shape}  path={out_path}")
