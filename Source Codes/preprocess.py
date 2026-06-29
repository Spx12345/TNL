"""
Full preprocessing pipeline for Malaysian eWallet ABSA project.
TNL6323 - Natural Language Processing
"""

import html
import re
import time
import uuid
import warnings
from collections import Counter
from pathlib import Path

import emoji
import ftfy
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH     = PROJECT_ROOT / "Data" / "raw_reviews.csv"
OUTPUT_PATH  = PROJECT_ROOT / "Data" / "sentiment_data_cleaned.csv"
CHARTS_DIR   = PROJECT_ROOT / "Data" / "charts"

SENTIMENT_MODEL  = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE       = 32
TARGET_PER_CLASS = 220
RANDOM_STATE     = 42

APPS       = ["touchngo", "grabpay", "shopeepay"]
SENTIMENTS = ["positive", "neutral", "negative"]
ASPECT_LABELS = ["payment", "ui", "service", "rewards", "general"]

POSITIVE_EMOJIS = {
    "\U0001f60a", "\U0001f44d", "\u2764\ufe0f", "\U0001f60d", "\u2705",
    "\U0001f44f", "\U0001f64f", "\U0001f601", "\U0001f4aa", "\U0001f970",
    "\U0001f604", "\U0001f389", "\U0001f4af", "\u2b50", "\U0001f31f",
}
NEGATIVE_EMOJIS = {
    "\U0001f621", "\U0001f44e", "\U0001f494", "\U0001f624", "\U0001f62d",
    "\u274c", "\U0001f92c", "\U0001f620", "\U0001f644", "\U0001f612",
    "\U0001f61e", "\U0001f4a9", "\U0001f92e", "\U0001f631", "\U0001f630",
}

KEEP_STOPWORDS = {
    "not", "no", "never", "very", "too", "but", "however",
    "although", "despite", "without", "cannot", "cant",
    "wont", "dont", "didnt", "couldnt", "wouldnt",
}

MALAYSIAN_STOPWORDS = {
    "la", "lah", "je", "jer", "kan", "pun", "tau", "nak",
    "ada", "tapi", "ni", "tu", "yg", "dah", "boleh",
    "saya", "nya", "aje", "guna", "pakai", "letak",
    "ambik", "ambil", "pergi", "balik", "masuk", "keluar",
}

ASPECT_KEYWORDS = {
    "aspect_payment": [
        "crash", "error", "fail", "transaction", "reload",
        "payment", "pay", "money", "deduct", "balance",
        "topup", "top up", "refund", "stuck", "pending",
        "failed", "freeze", "hack", "scam", "stolen",
        "unauthorized", "security", "OTP", "transfer", "bug", "loading",
    ],
    "aspect_ui": [
        "design", "interface", "UI", "layout", "easy",
        "confusing", "navigate", "update", "button", "screen",
        "display", "simple", "complicated", "friendly",
        "smooth", "laggy", "lag", "fast", "slow", "load",
        "feature", "version", "icon", "menu", "page", "tab",
    ],
    "aspect_service": [
        "support", "service", "response", "staff", "help",
        "contact", "complaint", "customer", "agent", "reply",
        "resolve", "issue", "problem", "useless", "helpful",
        "rude", "professional", "chat", "hotline", "email",
        "feedback", "report",
    ],
    "aspect_rewards": [
        "cashback", "voucher", "promo", "promotion", "reward",
        "points", "discount", "offer", "deal", "free", "bonus",
        "redeem", "campaign", "rebate", "coin", "lucky draw",
        "prize", "earn", "collect",
    ],
}

ASPECT_LABEL_MAP = {
    "aspect_payment": "payment",
    "aspect_ui":      "ui",
    "aspect_service": "service",
    "aspect_rewards": "rewards",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def download_nltk_resources() -> None:
    for resource, kind in [
        ("punkt",                     "tokenizers"),
        ("punkt_tab",                 "tokenizers"),
        ("stopwords",                 "corpora"),
        ("averaged_perceptron_tagger","taggers"),
    ]:
        try:
            nltk.data.find(f"{kind}/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# STEP 1 — Encoding fix and emoji analysis
# ---------------------------------------------------------------------------
def fix_encoding(text: str) -> str:
    return ftfy.fix_text(str(text)) if pd.notna(text) else ""


def extract_emojis(text: str) -> list[str]:
    return [item["emoji"] for item in emoji.emoji_list(text)]


def classify_emoji_sentiment(emojis: list[str]) -> str:
    pos = sum(1 for e in emojis if e in POSITIVE_EMOJIS)
    neg = sum(1 for e in emojis if e in NEGATIVE_EMOJIS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# STEP 2 — Text cleaning (preserve casing and punctuation)
# ---------------------------------------------------------------------------
def reduce_repeated_chars(text: str) -> str:
    return re.sub(r"(.)\1{3,}", r"\1\1\1", text)


def clean_text(text: str) -> str:
    if pd.isna(text) or not str(text).strip():
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)     # URLs
    text = re.sub(r"<[^>]+>", " ", text)                    # HTML tags
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)  # control chars
    text = reduce_repeated_chars(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# STEP 4 — Tokenization and stopword removal
# ---------------------------------------------------------------------------
def build_stopword_set() -> set[str]:
    eng_sw = set(stopwords.words("english"))
    all_sw = (eng_sw | MALAYSIAN_STOPWORDS) - KEEP_STOPWORDS
    return all_sw


def tokenize_and_filter(text: str, stop_words: set[str]) -> list[str]:
    tokens = word_tokenize(text)
    return [
        t for t in tokens
        if (t.lower() not in stop_words or t.lower() in KEEP_STOPWORDS)
    ]


# ---------------------------------------------------------------------------
# STEP 5 — XLM-RoBERTa sentiment labeling
# ---------------------------------------------------------------------------
def score_to_sentiment_stars(score) -> str:
    try:
        s = int(score)
    except (ValueError, TypeError):
        return "neutral"
    return "positive" if s >= 4 else ("neutral" if s == 3 else "negative")


def label_with_xlm_roberta(texts: list[str]) -> list[str]:
    device = 0 if torch.cuda.is_available() else -1
    print(f"  Loading {SENTIMENT_MODEL} on {'cuda' if device == 0 else 'cpu'}...")

    classifier = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=device,
        truncation=True,
        max_length=512,
    )

    labels = []
    total  = len(texts)

    for start in range(0, total, BATCH_SIZE):
        end   = min(start + BATCH_SIZE, total)
        batch = [t if str(t).strip() else " " for t in texts[start:end]]
        results = classifier(batch)

        for r in results:
            label = r["label"].lower()
            if label not in ("positive", "neutral", "negative"):
                label = {"label_0": "negative",
                         "label_1": "neutral",
                         "label_2": "positive"}.get(label, "neutral")
            labels.append(label)

        if end % 200 == 0 or end == total:
            print(f"  Labeling row {end}/{total}...")

        time.sleep(0.1)

    return labels


# ---------------------------------------------------------------------------
# STEP 6 — Aspect detection
# ---------------------------------------------------------------------------
def count_keyword_matches(text: str, keywords: list[str]) -> int:
    tl = text.lower()
    return sum(tl.count(kw.lower()) for kw in keywords)


def detect_aspects(text: str) -> dict:
    match_counts = {
        aspect: count_keyword_matches(text, kws)
        for aspect, kws in ASPECT_KEYWORDS.items()
    }
    binary_flags = {asp: int(cnt > 0) for asp, cnt in match_counts.items()}
    detected = [ASPECT_LABEL_MAP[a] for a, c in match_counts.items() if c > 0]

    if not detected:
        label = "general"
    elif len(detected) == 1:
        label = detected[0]
    else:
        max_cnt = max(match_counts.values())
        label = "general"
        for asp, cnt in match_counts.items():
            if cnt == max_cnt:
                label = ASPECT_LABEL_MAP[asp]
                break

    return {**binary_flags, "aspect_label": label}


# ---------------------------------------------------------------------------
# STEP 7 — Balance dataset
# ---------------------------------------------------------------------------
def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for source in APPS:
        for sentiment in SENTIMENTS:
            subset    = df[(df["source"] == source) & (df["sentiment"] == sentiment)]
            available = len(subset)
            if available < TARGET_PER_CLASS:
                print(f"  WARNING: {source}/{sentiment} has only {available} "
                      f"samples (target: {TARGET_PER_CLASS}) — using all available")
            n = min(TARGET_PER_CLASS, available)
            if n > 0:
                parts.append(subset.sample(n=n, random_state=RANDOM_STATE))
    balanced = pd.concat(parts, ignore_index=True)
    return balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 8 — Visualizations
# ---------------------------------------------------------------------------
def plot_sentiment_per_app(df: pd.DataFrame) -> None:
    counts = (df.groupby(["source", "sentiment"]).size()
                .unstack(fill_value=0)
                .reindex(index=APPS, columns=SENTIMENTS, fill_value=0))
    counts.plot(kind="bar", figsize=(10, 6),
                color=["#2ecc71", "#f1c40f", "#e74c3c"], edgecolor="black", linewidth=0.5)
    plt.title("Sentiment Distribution per App")
    plt.xlabel("App"); plt.ylabel("Count")
    plt.xticks(rotation=0); plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "sentiment_per_app.png", dpi=150)
    plt.close()


def plot_aspect_per_app(df: pd.DataFrame) -> None:
    counts = (df.groupby(["source", "aspect_label"]).size()
                .unstack(fill_value=0)
                .reindex(index=APPS, columns=ASPECT_LABELS, fill_value=0))
    counts.plot(kind="bar", figsize=(12, 6), colormap="Set2",
                edgecolor="black", linewidth=0.5)
    plt.title("Aspect Distribution per App")
    plt.xlabel("App"); plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.legend(title="Aspect", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "aspect_per_app.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_aspect_sentiment_heatmap(df: pd.DataFrame) -> None:
    hmap = pd.crosstab(df["aspect_label"], df["sentiment"])
    hmap = hmap.reindex(
        index=[a for a in ASPECT_LABELS if a in hmap.index],
        columns=SENTIMENTS, fill_value=0,
    )
    plt.figure(figsize=(8, 5))
    sns.heatmap(hmap, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Review Count"})
    plt.title("Aspect vs Sentiment Heatmap")
    plt.xlabel("Sentiment"); plt.ylabel("Aspect")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "aspect_sentiment_heatmap.png", dpi=150)
    plt.close()


def plot_wordclouds(df: pd.DataFrame) -> None:
    for sentiment in SENTIMENTS:
        text = " ".join(df.loc[df["sentiment"] == sentiment, "filtered_tokens"])
        if not text.strip():
            print(f"  WARNING: no tokens for {sentiment} word cloud")
            continue
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap="viridis", max_words=100).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud — {sentiment.capitalize()} Reviews")
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"wordcloud_{sentiment}.png", dpi=150)
        plt.close()


def plot_emoji_distribution(df: pd.DataFrame) -> None:
    all_emojis = []
    for val in df["emojis_found"].fillna(""):
        if val:
            all_emojis.extend(val.split("|"))

    if not all_emojis:
        print("  No emojis found — skipping emoji distribution chart.")
        return

    top = Counter(all_emojis).most_common(15)
    labels, counts = zip(*top)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(labels)), counts, color="#3498db")
    plt.xticks(range(len(labels)), labels, fontsize=14)
    plt.title("Most Common Emojis in Reviews")
    plt.xlabel("Emoji"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "emoji_distribution.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print_section("eWallet ABSA - Full Preprocessing Pipeline")
    download_nltk_resources()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows from {RAW_PATH}")

    # -------------------------------------------------------------------
    # STEP 1 — Encoding fix and emoji analysis
    # -------------------------------------------------------------------
    print_section("STEP 1 - Fix Encoding and Emoji Analysis")
    original_content = df["content"].copy()
    df["content"] = df["content"].apply(fix_encoding)

    df["emojis_found"]    = df["content"].apply(extract_emojis)
    df["emoji_sentiment"] = df["emojis_found"].apply(classify_emoji_sentiment)
    df["emojis_found"]    = df["emojis_found"].apply(lambda x: "|".join(x) if x else "")

    print("Sample of 5 fixed rows (before -> after):")
    for idx in df.index[:5]:
        before = str(original_content.loc[idx])[:80]
        after  = str(df.loc[idx, "content"])[:80]
        emojis = df.loc[idx, "emojis_found"]
        safe_print(f"  [{idx}] BEFORE: {before}")
        safe_print(f"        AFTER:  {after}")
        safe_print(f"        EMOJIS: {emojis or '(none)'} | "
                   f"emoji_sentiment: {df.loc[idx, 'emoji_sentiment']}")
        safe_print("")

    # -------------------------------------------------------------------
    # STEP 2 — Text cleaning (no lowercasing, keep punctuation)
    # -------------------------------------------------------------------
    print_section("STEP 2 - Text Cleaning")
    df["cleaned_text"] = df["content"].apply(clean_text)
    before_len = len(df)
    df = df[df["cleaned_text"].str.split().str.len() >= 5].copy()
    print(f"Removed {before_len - len(df)} rows with fewer than 5 words")
    print(f"Remaining rows: {len(df)}")

    # -------------------------------------------------------------------
    # STEP 3 — Tokenization (keep punctuation)
    # -------------------------------------------------------------------
    print_section("STEP 3 - Tokenization")
    df["tokens"] = df["cleaned_text"].apply(word_tokenize)

    # -------------------------------------------------------------------
    # STEP 4 — Stopword removal (keep sentiment words)
    # -------------------------------------------------------------------
    print_section("STEP 4 - Stopword Removal")
    stop_words = build_stopword_set()
    df["filtered_tokens"] = df["cleaned_text"].apply(
        lambda t: tokenize_and_filter(t, stop_words)
    )

    # -------------------------------------------------------------------
    # STEP 5 — XLM-RoBERTa sentiment labeling
    # -------------------------------------------------------------------
    print_section("STEP 5 - XLM-RoBERTa Sentiment Labeling")
    df["sentiment_stars"] = df["score"].apply(score_to_sentiment_stars)
    start = time.time()
    df["sentiment"] = label_with_xlm_roberta(df["cleaned_text"].tolist())
    elapsed = time.time() - start
    print(f"Labeling completed in {elapsed:.1f} seconds")

    agreement = (df["sentiment"] == df["sentiment_stars"]).mean()
    print(f"\nAgreement rate (XLM-RoBERTa vs star ratings): {agreement * 100:.2f}%")
    print("\nXLM-RoBERTa sentiment distribution:")
    print(df["sentiment"].value_counts().to_string())
    print("\nStar-based sentiment distribution:")
    print(df["sentiment_stars"].value_counts().to_string())

    # -------------------------------------------------------------------
    # STEP 6 — Aspect detection
    # -------------------------------------------------------------------
    print_section("STEP 6 - Aspect Detection")
    aspect_df = df["cleaned_text"].apply(detect_aspects).apply(pd.Series)
    df = pd.concat([df, aspect_df], axis=1)
    print(df["aspect_label"].value_counts().to_string())

    # -------------------------------------------------------------------
    # STEP 7 — Balance and save
    # -------------------------------------------------------------------
    print_section("STEP 7 - Balance Dataset")
    df_bal = balance_dataset(df)
    print(f"Balanced dataset size: {len(df_bal)}")
    print("\nFinal balanced counts per app:")
    print(df_bal.groupby(["source", "sentiment"]).size().unstack(fill_value=0).to_string())

    df_bal["id"] = [str(uuid.uuid4()) for _ in range(len(df_bal))]
    df_bal["filtered_tokens"] = df_bal["filtered_tokens"].apply(
        lambda t: " ".join(t) if isinstance(t, list) else t
    )
    df_bal["tokens"] = df_bal["tokens"].apply(
        lambda t: " ".join(t) if isinstance(t, list) else t
    )

    final_cols = [
        "id", "source", "content", "cleaned_text", "tokens",
        "filtered_tokens", "emojis_found", "emoji_sentiment",
        "sentiment", "sentiment_stars",
        "aspect_payment", "aspect_ui", "aspect_service", "aspect_rewards",
        "aspect_label",
    ]
    df_final = df_bal[final_cols]
    df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved to {OUTPUT_PATH}")

    # -------------------------------------------------------------------
    # STEP 8 — Visualizations
    # -------------------------------------------------------------------
    print_section("STEP 8 - Visualizations")
    plot_sentiment_per_app(df_final)
    plot_aspect_per_app(df_final)
    plot_aspect_sentiment_heatmap(df_final)
    plot_wordclouds(df_final)
    plot_emoji_distribution(df_final)
    print(f"All charts saved to {CHARTS_DIR}")

    # Summary
    print_section("PREPROCESSING SUMMARY")
    print(f"Total rows: {len(df_final)}")
    print("\nRows per app:")
    print(df_final["source"].value_counts().reindex(APPS).to_string())
    print("\nSentiment per app:")
    print(df_final.groupby(["source", "sentiment"]).size()
                  .unstack(fill_value=0).to_string())
    print("\nAspect distribution:")
    print(df_final["aspect_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
