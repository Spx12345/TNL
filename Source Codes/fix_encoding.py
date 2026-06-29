"""
Fix broken emoji and text encoding in project CSV files.
TNL6323 — Natural Language Processing
"""

import re
from pathlib import Path

import emoji
import ftfy
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"

FILES = [
    DATA_DIR / "raw_reviews.csv",
    DATA_DIR / "sentiment_data_cleaned.csv",
    DATA_DIR / "sentiment_data_absa_labeled.csv",
]

POSITIVE_EMOJIS = {
    "😊", "👍", "❤️", "😍", "✅", "👏", "🙏", "😁", "💪", "🥰",
    "😄", "🎉", "💯", "⭐", "🌟",
    "👍🏻", "👍🏼", "👍🏽", "👍🏾", "👍🏿",
}
NEGATIVE_EMOJIS = {
    "😡", "👎", "💔", "😤", "😭", "❌", "🤬", "😠", "🙄", "😒",
    "😞", "💩", "🤮", "😱", "😰",
    "👎🏻", "👎🏼", "👎🏽", "👎🏾", "👎🏿",
}

MOJIBAKE_PATTERN = re.compile(r"ð|â|ï|Ÿ|Ã|Â")


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def is_text_column(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or series.dtype == object


def fix_text(value):
    """Apply ftfy to a single cell; preserve NaN/empty values."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    if not text:
        return ""
    return ftfy.fix_text(text)


def extract_emojis_from_text(text: str) -> str:
    """Extract emojis from text and return comma-separated string."""
    if pd.isna(text) or not str(text).strip():
        return ""
    found = [item["emoji"] for item in emoji.emoji_list(str(text))]
    return ",".join(found)


def classify_emoji_sentiment(emojis_str: str) -> str:
    """Classify emoji sentiment from comma-separated emoji string."""
    if not emojis_str or not str(emojis_str).strip():
        return "neutral"

    emojis = [e.strip() for e in str(emojis_str).split(",") if e.strip()]
    pos = sum(1 for e in emojis if e in POSITIVE_EMOJIS)
    neg = sum(1 for e in emojis if e in NEGATIVE_EMOJIS)

    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def find_emoji_sample(df: pd.DataFrame, column: str) -> str:
    """Return a sample emoji string from a column, or empty if none."""
    if column not in df.columns:
        return ""
    series = df[column].fillna("").astype(str).str.strip()
    nonempty = series[series.ne("")]
    return nonempty.iloc[0] if len(nonempty) else ""


def count_changed_rows(before: pd.DataFrame, after: pd.DataFrame, columns: list[str]) -> int:
    changed = 0
    for col in columns:
        if col not in before.columns:
            continue
        before_vals = before[col].fillna("").astype(str)
        after_vals = after[col].fillna("").astype(str)
        changed += (before_vals != after_vals).sum()
    return int(changed)


def fix_csv_file(path: Path) -> None:
    safe_print(f"\nFixing: {path}")

    if not path.exists():
        safe_print(f"  Skipped — file not found: {path}")
        return

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        safe_print(f"  Error loading file: {exc}")
        return

    df_before = df.copy()
    sample_before = ""
    if "emojis_found" in df_before.columns:
        emoji_series = df_before["emojis_found"].fillna("").astype(str).str.strip()
        broken_emoji_rows = emoji_series[emoji_series.str.contains(MOJIBAKE_PATTERN, na=False)]
        if len(broken_emoji_rows):
            sample_before = broken_emoji_rows.iloc[0]
        elif emoji_series[emoji_series.ne("")].any():
            sample_before = emoji_series[emoji_series.ne("")].iloc[0]

    if not sample_before and "content" in df_before.columns:
        broken = df_before["content"].fillna("").astype(str)
        broken_rows = broken[broken.str.contains(MOJIBAKE_PATTERN, na=False)]
        if len(broken_rows):
            sample_before = broken_rows.iloc[0][:80]

    text_columns = [col for col in df.columns if is_text_column(df[col])]

    for col in text_columns:
        df[col] = df[col].apply(fix_text)

    if "content" in df.columns and "emojis_found" in df.columns:
        df["emojis_found"] = df["content"].apply(extract_emojis_from_text)

    if "emoji_sentiment" in df.columns and "emojis_found" in df.columns:
        df["emoji_sentiment"] = df["emojis_found"].apply(classify_emoji_sentiment)

    sample_after = find_emoji_sample(df, "emojis_found")
    rows_changed = count_changed_rows(df_before, df, text_columns)
    if "emojis_found" in df.columns:
        emoji_changed = (df_before["emojis_found"].fillna("").astype(str) != df["emojis_found"].fillna("").astype(str)).sum()
        rows_changed = max(rows_changed, int(emoji_changed))

    df.to_csv(path, index=False, encoding="utf-8-sig")

    safe_print(f"Fixed: {path}")
    safe_print(f"Sample emojis before: {sample_before or '(none)'}")
    safe_print(f"Sample emojis after: {sample_after or '(none)'}")
    safe_print(f"Total rows fixed: {len(df)} ({rows_changed} cells updated)")


def main() -> None:
    safe_print("=" * 60)
    safe_print("CSV Encoding Fix — Malaysian eWallet ABSA Project")
    safe_print("=" * 60)

    for csv_path in FILES:
        fix_csv_file(csv_path)

    safe_print("\n" + "=" * 60)
    safe_print("All files processed.")
    safe_print("=" * 60)


if __name__ == "__main__":
    main()
