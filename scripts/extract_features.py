import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from loader import SplitLoader
from extractors import EXTRACTORS

ARTIFACTS_DIR = Path("artifacts/features")
DATASET_ID = "AKCIT-Deepfake/BRSpeech-DF"


def extract_split(loader, extractor, split_name: str, with_features: bool = True):
    """Stream one split in CSV order and build its feature array.

    Labels are written by SplitLoader itself (it saves <split>/y.npy once the
    stream is fully consumed), so this function only builds X. When
    `with_features` is False the stream is still consumed -- so the loader can
    (re)write y.npy -- but the extractor is not run.
    """
    X = []
    n_examples = loader.stats(split_name)["total"]
    for audio, sr, label, meta in tqdm(
        loader.stream(split_name),
        total=n_examples,
        desc=f"{extractor.name} {split_name}",
    ):
        if with_features:
            X.append(extractor(audio, sr))
    return np.array(X, dtype=np.float32) if with_features else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        choices=EXTRACTORS.keys(),
        required=True,
        help="Feature representation to extract",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing the split CSVs (default: artifacts/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing feature files",
    )
    args = parser.parse_args()

    extractor = EXTRACTORS[args.representation]()
    print(f"Using extractor: {extractor.name}")

    loader = SplitLoader(
        dataset_id=DATASET_ID,
        splits_dir=str(args.meta_dir),
        features_dir=ARTIFACTS_DIR,
    )
    loader.print_stats()

    for split in ("train", "test"):
        split_dir = ARTIFACTS_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        x_file = split_dir / f"X_{extractor.name}.npy"
        y_file = split_dir / "y.npy"

        need_x = args.force or not x_file.exists()
        need_y = not y_file.exists()

        if not need_x and not need_y:
            print(f"[{split}] X and y exist -> skipping")
            continue

        # Consuming the stream makes SplitLoader (re)write y.npy in CSV order.
        X = extract_split(loader, extractor, split, with_features=need_x)

        if need_x:
            np.save(x_file, X)
            print(f"[{split}] saved {x_file.name} shape={X.shape}")
        else:
            print(f"[{split}] {x_file.name} exists -> kept (refreshed y.npy)")


if __name__ == "__main__":
    main()