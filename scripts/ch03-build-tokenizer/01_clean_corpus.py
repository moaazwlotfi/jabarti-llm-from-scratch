"""
01 — Download, Filter, and Clean the Corpus
============================================

Step 1 — download every pretrain shard from HuggingFace (~5 GB, cached).
Step 2 — measure whether Type-A cleaning still earns its place.
Step 3 — demo the Type-B normalizer (unchanged, still the golden rule).
Step 4 — write a SAMPLED corpus for the tokenizer to train on.

"""

import argparse
from pathlib import Path

import pandas as pd
from cleaning import clean_dataframe, normalize_text
from huggingface_hub import snapshot_download

REPO = "bakrianoo/jabarti-llm-dataset"

TRAIN_GLOB = "train-*.parquet"
EVAL_GLOB = "eval-*.parquet"

# KEEP_COLUMNS = ["text", "language", "article_id"]
KEEP_COLUMNS = ["text", "language"]   # language kept only to check the AR/EN mix

SAMPLE_SEED = 1234

DEFAULT_TOKENIZER_DOCS = 400_000

FINETUNE_FILES = {
    "ft_train": "finetune/train-00000-of-00001.parquet",
    "ft_eval":  "finetune/eval-00000-of-00001.parquet",
}

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_CORPUS = OUT_DIR / "tokenizer_corpus.parquet"

def shard_patterns(max_shards: int | None = None) -> list[str]:
    if max_shards is None:
        return ["pretrain/train-*.parquet"]
    return [f"pretrain/train-{i:05d}-of-*.parquet" for i in range(max_shards)]

def download_pretrain(max_shards: int | None = None) -> list[Path]:
    """Download the train shards and return their local paths."""
    root = Path(snapshot_download(
        repo_id=REPO, repo_type="dataset",
        allow_patterns=shard_patterns(max_shards),
    ))
    shards = sorted((root / "pretrain").glob(TRAIN_GLOB))
    return shards[:max_shards]  # snapshot dir may hold shards from earlier runs

def build_tokenizer_corpus(train_shards: list[Path], target_docs: int) -> None:
    """Sample evenly across shards, clean, and write the tokenizer's training text."""
    per_shard = -(-target_docs // len(train_shards))  # ceil division
    parts = []
    for i, shard in enumerate(train_shards):
        df = pd.read_parquet(shard, columns=KEEP_COLUMNS)
        df = df.sample(n=min(per_shard, len(df)), random_state=SAMPLE_SEED + i)
        parts.append(clean_dataframe(df).drop(columns="orig_len"))

    corpus = pd.concat(parts, ignore_index=True)
    corpus.to_parquet(TOKENIZER_CORPUS, index=False)
    print(f"tokenizer corpus: {len(corpus):,} docs, "
          f"{corpus['language'].value_counts().to_dict()}")

def demo_normalizer():
    print("TYPE B NORMALIZER (embeddable, runs at inference too)")

    samples = [
        "جُمْهُورِيَّةُ مِصْرَ الْعَرَبِيَّة",      # heavy diacritics
        "أحمد إبراهيم آدم علىّ",                   # alef + ya variants
        "The   United    States",                 # extra spaces
    ]

    for s in samples:
        print(f"  raw       : {s}")
        print(f"  normalized: {normalize_text(s)}")
        print("="*30)

def clean_phase(phase, input_path, output_path):

    print(f"==== clean_phase: {phase} ====")

    df = pd.read_parquet(input_path)

    print(f"  Source        : {input_path}")
    print(f"  Rows          : {len(df):,}")

    cleaned_df = clean_dataframe(df)

    removed_rows = len(df) - len(cleaned_df)
    print(f"  Rows kept        : {len(cleaned_df):,}  (dropped {removed_rows:,} empty rows)")

    cleaned_df.to_parquet(output_path, index=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-shards", type=int, default=None,
                        help="use only the first N train shards (classroom runs)")
    parser.add_argument("--tokenizer-docs", type=int, default=DEFAULT_TOKENIZER_DOCS,
                        help="documents to sample for tokenizer training")
    args = parser.parse_args()

    train_shards = download_pretrain(args.max_shards)
    build_tokenizer_corpus(train_shards, target_docs=args.tokenizer_docs)

if __name__ == "__main__":
    main()
    
