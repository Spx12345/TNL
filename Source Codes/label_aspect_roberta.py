"""
Aspect labeling via zero-shot classification (BART-large-MNLI).
TNL6323 — Malaysian eWallet Sentiment Analysis project.
"""

import warnings
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "Data" / "sentiment_data_cleaned.csv"
PROGRESS_PATH = PROJECT_ROOT / "Data" / "absa_progress.csv"
OUTPUT_PATH = PROJECT_ROOT / "Data" / "sentiment_data_absa_labeled.csv"

MODEL_NAME = "facebook/bart-large-mnli"
BATCH_SIZE = 16
SAVE_EVERY = 50

CANDIDATE_LABELS = ["payment", "ui", "service", "rewards", "general"]
HYPOTHESIS_TEMPLATE = "This review is about {}"


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def save_progress(df: pd.DataFrame) -> None:
    df.to_csv(PROGRESS_PATH, index=False, encoding="utf-8-sig")
    safe_print(f"  Checkpoint saved to {PROGRESS_PATH}")


def classify_batch(classifier, texts: list[str]) -> list[tuple[str, float]]:
    """Run zero-shot classification on a batch of reviews."""
    results = classifier(
        texts,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    if isinstance(results, dict):
        results = [results]

    output = []
    for result in results:
        output.append((result["labels"][0], float(result["scores"][0])))
    return output


def is_row_labeled(row: pd.Series) -> bool:
    return (
        pd.notna(row.get("aspect_label"))
        and str(row.get("aspect_label", "")).strip() != ""
        and pd.notna(row.get("aspect_confidence"))
    )


def main() -> None:
    print_section("Aspect Labeling — BART Zero-Shot (facebook/bart-large-mnli)")

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    safe_print(f"Device: {device_name}")

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    safe_print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    if "aspect_label" in df.columns:
        df["aspect_label_keyword"] = df["aspect_label"].copy()
    else:
        df["aspect_label_keyword"] = "general"

    df["aspect_confidence"] = pd.NA
    df["absa_sentiment"] = pd.NA
    df["absa_label"] = pd.NA

    if PROGRESS_PATH.exists():
        progress = pd.read_csv(PROGRESS_PATH, encoding="utf-8-sig")
        safe_print(f"Resuming from checkpoint: {PROGRESS_PATH}")
        for col in ["aspect_label", "aspect_confidence", "absa_sentiment", "absa_label"]:
            if col in progress.columns:
                merge_map = progress.set_index("id")[col].to_dict()
                df[col] = df["id"].map(merge_map).fillna(df[col])

    safe_print(f"Loading zero-shot classifier: {MODEL_NAME}")
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device,
    )

    total = len(df)
    pending_indices = [idx for idx in df.index if not is_row_labeled(df.loc[idx])]
    safe_print(f"Rows to process: {len(pending_indices)} / {total}")

    processed_this_run = 0

    for start in range(0, len(pending_indices), BATCH_SIZE):
        batch_indices = pending_indices[start : start + BATCH_SIZE]
        texts = []
        valid_indices = []

        for idx in batch_indices:
            content = df.at[idx, "content"]
            if pd.isna(content) or not str(content).strip():
                sentiment = str(df.at[idx, "sentiment"]) if pd.notna(df.at[idx, "sentiment"]) else "neutral"
                df.at[idx, "aspect_label"] = "general"
                df.at[idx, "aspect_confidence"] = 0.0
                df.at[idx, "absa_sentiment"] = sentiment
                df.at[idx, "absa_label"] = f"general_{sentiment}"
                processed_this_run += 1
                continue
            texts.append(str(content))
            valid_indices.append(idx)

        if texts:
            predictions = classify_batch(classifier, texts)
            for idx, (aspect, confidence) in zip(valid_indices, predictions):
                sentiment = str(df.at[idx, "sentiment"]) if pd.notna(df.at[idx, "sentiment"]) else "neutral"
                df.at[idx, "aspect_label"] = aspect
                df.at[idx, "aspect_confidence"] = confidence
                df.at[idx, "absa_sentiment"] = sentiment
                df.at[idx, "absa_label"] = f"{aspect}_{sentiment}"
                processed_this_run += 1

        labeled_count = df["aspect_confidence"].notna().sum()
        if labeled_count % SAVE_EVERY == 0 or labeled_count == total:
            safe_print(f"Progress: {labeled_count}/{total} rows done")
            save_progress(df)

    save_progress(df)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    safe_print(f"\nFinal dataset saved to {OUTPUT_PATH}")

    print_section("LABELING SUMMARY")
    safe_print("\nAspect distribution:")
    safe_print(df["aspect_label"].value_counts().to_string())

    safe_print("\nABSA label distribution:")
    safe_print(df["absa_label"].value_counts().head(15).to_string())

    safe_print("\nAverage confidence score per aspect:")
    avg_conf = df.groupby("aspect_label")["aspect_confidence"].mean().sort_values(ascending=False)
    for aspect, score in avg_conf.items():
        safe_print(f"  {aspect}: {score:.4f}")

    safe_print(f"\nOverall average confidence: {df['aspect_confidence'].mean():.4f}")


if __name__ == "__main__":
    main()
