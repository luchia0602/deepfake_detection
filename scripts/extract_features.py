import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from loader import SplitLoader
from extractors import EXTRACTORS


ARTIFACTS_DIR = Path("/content/drive/MyDrive/llm-project/artifacts/features")
DATASET_ID = "AKCIT-Deepfake/BRSpeech-DF"


def extract_split(loader, extractor, split_name: str):

    X = []
    n_examples = loader.stats(split_name)["total"]

    for audio, sr, label, meta in tqdm(
        loader.stream(split_name),
        total=n_examples,
        desc=f"{extractor.name} {split_name}",
    ):
        X.append(extractor(audio, sr))

    return np.array(X, dtype=np.float32)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--representation",
        choices=EXTRACTORS.keys(),
        required=True,
        help="Feature representation to extract",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )

    args = parser.parse_args()

    extractor = EXTRACTORS[
        args.representation
    ]()

    print(f"Using extractor: {extractor.name}")

    loader = SplitLoader(
        dataset_id=DATASET_ID,
        splits_dir="/content/drive/MyDrive/llm-project/artifacts",
    )

    loader.print_stats()

    for split in ("train", "val", "test"):

        out_file = (
            ARTIFACTS_DIR
            / split
            / f"X_{extractor.name}.npy"
        )

        out_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if out_file.exists() and not args.force:
            print(f"[{split}] exists -> skipping")
            continue

        X = extract_split(
            loader,
            extractor,
            split,
        )

        np.save(out_file, X)

        print(
            f"[{split}] saved "
            f"{out_file.name} "
            f"shape={X.shape}"
        )

if __name__ == "__main__":
    main()