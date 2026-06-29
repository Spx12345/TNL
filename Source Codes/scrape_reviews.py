"""
Google Play Store review scraper for Malaysian eWallet apps.
TNL6323 — Aspect-Based Sentiment Analysis project.

All scraping strategies are consolidated in this single file.
Use --strategy to choose which approach to run:

  initial       Original scrape (300 en + 100 ms per app, newest sort)
  neutral       Boost 3-star neutral reviews (200 en + 200 ms per app)
  extra         Boost 1-2 star negative + 4-5 star positive (150 per score)
  targeted      Small targeted top-ups for specific app/sentiment gaps
  big           Large-scale scrape (~600+ per sentiment per app)
  more_positive Extra 4-5 star positive reviews (500 each, English only)

Examples:
  python "Source Codes/scrape_reviews.py" --strategy initial
  python "Source Codes/scrape_reviews.py" --strategy neutral
  python "Source Codes/scrape_reviews.py" --strategy big
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews

# ---------------------------------------------------------------------------
# Paths and shared constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "Data" / "raw_reviews.csv"

APPS = [
    {
        "app_id": "my.com.tngdigital.ewallet",
        "appName": "Touch 'n Go eWallet",
        "source": "touchngo",
    },
    {
        "app_id": "com.grabtaxi.passenger",
        "appName": "GrabPay",
        "source": "grabpay",
    },
    {
        "app_id": "com.shopee.my",
        "appName": "ShopeePay",
        "source": "shopeepay",
    },
]

REQUEST_DELAY = 2
BATCH_SIZE = 200
FIELDS = ["reviewId", "userName", "content", "score", "thumbsUpCount", "at", "appName"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def score_to_sentiment(score: int) -> str:
    if score >= 4:
        return "positive"
    if score == 3:
        return "neutral"
    return "negative"


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def load_existing() -> pd.DataFrame:
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
        safe_print(f"Loaded {len(df)} existing reviews from {RAW_PATH}")
        return df
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe_print("No existing raw_reviews.csv — starting fresh.")
    return pd.DataFrame(columns=FIELDS + ["source", "sentiment"])


def normalize_reviews(raw_reviews: list[dict]) -> pd.DataFrame:
    if not raw_reviews:
        return pd.DataFrame(columns=FIELDS + ["source", "sentiment"])

    df = pd.DataFrame(raw_reviews)
    for field in FIELDS:
        if field not in df.columns:
            df[field] = None
    df = df[FIELDS + ["source"]]
    df["sentiment"] = df["score"].apply(score_to_sentiment)
    return df


def merge_and_save(
    existing_df: pd.DataFrame,
    new_reviews: list[dict],
    min_words: int = 0,
) -> pd.DataFrame:
    """Append new reviews, deduplicate, clean, and save to raw_reviews.csv."""
    new_df = normalize_reviews(new_reviews)
    combined = pd.concat([existing_df, new_df], ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["reviewId"], keep="first")
    combined = combined.dropna(subset=["content"])
    combined = combined[combined["content"].astype(str).str.strip() != ""]
    if min_words > 0:
        combined = combined[combined["content"].astype(str).str.split().str.len() >= min_words]
    combined["sentiment"] = combined["score"].apply(score_to_sentiment)
    combined = combined.reset_index(drop=True)

    safe_print(f"Combined before dedup/clean: {before}")
    safe_print(f"After dedup & cleaning: {len(combined)}")
    safe_print(f"Net new rows added: {len(combined) - len(existing_df)}")

    combined.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
    safe_print(f"Saved to {RAW_PATH}")
    return combined


def scrape_paginated(
    app_id: str,
    lang: str,
    country: str,
    target_count: int,
    score_filter: int | None = None,
) -> list[dict]:
    """Generic paginated scraper with optional star-rating filter."""
    collected: list[dict] = []
    continuation_token = None

    while len(collected) < target_count:
        remaining = target_count - len(collected)
        batch_size = min(BATCH_SIZE, remaining)
        kwargs = {
            "lang": lang,
            "country": country,
            "sort": Sort.NEWEST,
            "count": batch_size,
            "continuation_token": continuation_token,
        }
        if score_filter is not None:
            kwargs["filter_score_with"] = score_filter

        try:
            batch, continuation_token = reviews(app_id, **kwargs)
        except Exception as exc:
            safe_print(f"    ERROR (score={score_filter}, lang={lang}): {exc}")
            break

        if not batch:
            break

        collected.extend(batch)
        if len(collected) >= target_count or continuation_token is None:
            break

        time.sleep(REQUEST_DELAY)

    result = collected[:target_count]
    if len(result) < target_count:
        safe_print(
            f"    WARNING: score={score_filter} lang={lang} "
            f"returned {len(result)}/{target_count} reviews"
        )
    return result


def tag_reviews(batch: list[dict], app_config: dict) -> None:
    for review in batch:
        review["appName"] = app_config["appName"]
        review["source"] = app_config["source"]


def print_summary(df: pd.DataFrame, title: str = "SCRAPING SUMMARY") -> None:
    safe_print("\n" + "=" * 60)
    safe_print(title)
    safe_print("=" * 60)
    safe_print(f"\nTotal rows: {len(df)}")

    safe_print("\nRows per app:")
    for source in sorted(df["source"].unique()):
        safe_print(f"  {source}: {len(df[df['source'] == source])}")

    safe_print("\nSentiment distribution per app:")
    for source in sorted(df["source"].unique()):
        subset = df[df["source"] == source]
        safe_print(f"\n  {source} (n={len(subset)}):")
        for sentiment in ["positive", "neutral", "negative"]:
            count = len(subset[subset["sentiment"] == sentiment])
            pct = count / len(subset) * 100 if len(subset) else 0
            safe_print(f"    {sentiment}: {count} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# STRATEGY 1 — Initial scrape (formerly scrape_reviews.py)
# First data collection: 300 English + 100 Malay reviews per app.
# No star filter — newest reviews regardless of rating.
# Used to build the foundation raw_reviews.csv dataset.
# ---------------------------------------------------------------------------
INITIAL_CONFIG = [
    {"lang": "en", "country": "my", "target_count": 300},
    {"lang": "ms", "country": "my", "target_count": 100},
]


def run_initial_scrape() -> pd.DataFrame:
    safe_print("\n" + "=" * 60)
    safe_print("STRATEGY 1 — Initial Scrape")
    safe_print("300 en + 100 ms per app | newest sort | no star filter")
    safe_print("=" * 60)

    all_reviews: list[dict] = []

    for app_config in APPS:
        safe_print(f"\nScraping {app_config['appName']} ({app_config['source']})...")
        for cfg in INITIAL_CONFIG:
            safe_print(
                f"  lang={cfg['lang']}, target={cfg['target_count']}..."
            )
            batch = scrape_paginated(
                app_config["app_id"],
                lang=cfg["lang"],
                country=cfg["country"],
                target_count=cfg["target_count"],
            )
            tag_reviews(batch, app_config)
            all_reviews.extend(batch)
            safe_print(f"    Retrieved {len(batch)} reviews")
            time.sleep(REQUEST_DELAY)

    df = normalize_reviews(all_reviews)
    df = df.drop_duplicates(subset=["reviewId"], keep="first")
    df = df.dropna(subset=["content"])
    df = df[df["content"].astype(str).str.strip() != ""]
    df = df[df["content"].astype(str).str.split().str.len() >= 5]
    df = df.reset_index(drop=True)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
    safe_print(f"\nSaved to {RAW_PATH}")
    print_summary(df, "INITIAL SCRAPE SUMMARY")
    return df


# ---------------------------------------------------------------------------
# STRATEGY 2 — Neutral boost (formerly scrape_neutral.py)
# Target 3-star reviews only using filter_score_with=3.
# 200 English + 200 Malay per app to increase neutral class balance.
# Merges with existing raw_reviews.csv.
# ---------------------------------------------------------------------------
NEUTRAL_CONFIG = [
    {"lang": "en", "country": "my", "target_count": 200},
    {"lang": "ms", "country": "my", "target_count": 200},
]


def run_neutral_scrape() -> pd.DataFrame:
    safe_print("\n" + "=" * 60)
    safe_print("STRATEGY 2 — Neutral Boost (3-star reviews)")
    safe_print("200 en + 200 ms per app | filter_score_with=3")
    safe_print("=" * 60)

    existing_df = load_existing()
    new_reviews: list[dict] = []

    for app_config in APPS:
        safe_print(f"\nScraping 3-star {app_config['source']}...")
        for cfg in NEUTRAL_CONFIG:
            batch = scrape_paginated(
                app_config["app_id"],
                lang=cfg["lang"],
                country=cfg["country"],
                target_count=cfg["target_count"],
                score_filter=3,
            )
            tag_reviews(batch, app_config)
            new_reviews.extend(batch)
            safe_print(f"    Retrieved {len(batch)} three-star reviews")
            time.sleep(REQUEST_DELAY)

    combined = merge_and_save(existing_df, new_reviews)
    print_summary(combined, "NEUTRAL BOOST SUMMARY")
    return combined


# ---------------------------------------------------------------------------
# STRATEGY 3 — Extra positive/negative (formerly scrape_extra.py)
# Boost 1-2 star (negative) and 4-5 star (positive) reviews.
# 150 reviews per score rating, both English and Malay per app.
# Merges with existing raw_reviews.csv.
# ---------------------------------------------------------------------------
EXTRA_LANG_CONFIG = [{"lang": "en", "country": "my"}, {"lang": "ms", "country": "my"}]
EXTRA_NEGATIVE_SCORES = [1, 2]
EXTRA_POSITIVE_SCORES = [4, 5]
EXTRA_TARGET_PER_SCORE = 150


def run_extra_scrape() -> pd.DataFrame:
    safe_print("\n" + "=" * 60)
    safe_print("STRATEGY 3 — Extra Positive / Negative Boost")
    safe_print("150 per score | 1-2 star negative + 4-5 star positive")
    safe_print("=" * 60)

    existing_df = load_existing()
    new_reviews: list[dict] = []

    for label, scores in [("negative", EXTRA_NEGATIVE_SCORES), ("positive", EXTRA_POSITIVE_SCORES)]:
        safe_print(f"\nScraping extra {label} reviews...")
        for app_config in APPS:
            safe_print(f"  {app_config['source']}:")
            for score_filter in scores:
                for lang_cfg in EXTRA_LANG_CONFIG:
                    batch = scrape_paginated(
                        app_config["app_id"],
                        lang=lang_cfg["lang"],
                        country=lang_cfg["country"],
                        target_count=EXTRA_TARGET_PER_SCORE,
                        score_filter=score_filter,
                    )
                    tag_reviews(batch, app_config)
                    new_reviews.extend(batch)
                    safe_print(f"    score={score_filter} lang={lang_cfg['lang']}: {len(batch)}")
                    time.sleep(REQUEST_DELAY)

    combined = merge_and_save(existing_df, new_reviews)
    print_summary(combined, "EXTRA SCRAPE SUMMARY")
    return combined


# ---------------------------------------------------------------------------
# STRATEGY 4 — Targeted top-ups (formerly scrape_extra_reviews.py)
# Small targeted scrapes for specific app/sentiment gaps:
#   +20 touchngo positive (4-5 star)
#   +15 grabpay neutral (3 star)
# Skips reviews already in the dataset (dedup by reviewId during scrape).
# ---------------------------------------------------------------------------
TARGETED_SCRAPES = [
    {
        "app_id": "my.com.tngdigital.ewallet",
        "appName": "Touch 'n Go eWallet",
        "source": "touchngo",
        "score_filters": [4, 5],
        "target_new": 20,
        "label": "positive (4-5 star)",
    },
    {
        "app_id": "com.grabtaxi.passenger",
        "appName": "GrabPay",
        "source": "grabpay",
        "score_filters": [3],
        "target_new": 15,
        "label": "neutral (3 star)",
    },
]


def scrape_unique_reviews(
    app_id: str,
    app_name: str,
    source: str,
    score_filters: list[int],
    target_new: int,
    existing_ids: set[str],
    lang: str = "en",
    country: str = "my",
) -> list[dict]:
    """Scrape until target_new unique (unseen) reviews are collected."""
    new_reviews: list[dict] = []
    seen_ids = set(existing_ids)

    for score_filter in score_filters:
        if len(new_reviews) >= target_new:
            break

        continuation_token = None
        while len(new_reviews) < target_new:
            try:
                batch, continuation_token = reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.NEWEST,
                    count=BATCH_SIZE,
                    filter_score_with=score_filter,
                    continuation_token=continuation_token,
                )
            except Exception as exc:
                safe_print(f"    ERROR: {exc}")
                break

            if not batch:
                break

            for review in batch:
                rid = review["reviewId"]
                if rid in seen_ids:
                    continue
                review["appName"] = app_name
                review["source"] = source
                new_reviews.append(review)
                seen_ids.add(rid)
                if len(new_reviews) >= target_new:
                    break

            if len(new_reviews) >= target_new or continuation_token is None:
                break

            time.sleep(REQUEST_DELAY)

        time.sleep(REQUEST_DELAY)

    return new_reviews[:target_new]


def run_targeted_scrape() -> pd.DataFrame:
    safe_print("\n" + "=" * 60)
    safe_print("STRATEGY 4 — Targeted Top-ups")
    safe_print("+20 touchngo positive | +15 grabpay neutral")
    safe_print("=" * 60)

    existing_df = load_existing()
    existing_ids = set(existing_df["reviewId"].astype(str))
    all_new: list[dict] = []

    for target in TARGETED_SCRAPES:
        safe_print(
            f"\nScraping {target['label']} for {target['source']} "
            f"(target: {target['target_new']} new)..."
        )
        batch = scrape_unique_reviews(
            app_id=target["app_id"],
            app_name=target["appName"],
            source=target["source"],
            score_filters=target["score_filters"],
            target_new=target["target_new"],
            existing_ids=existing_ids,
        )
        for review in batch:
            existing_ids.add(review["reviewId"])
        all_new.extend(batch)
        safe_print(f"  Added {len(batch)} new reviews")

    combined = merge_and_save(existing_df, all_new)
    print_summary(combined, "TARGETED SCRAPE SUMMARY")
    return combined


# ---------------------------------------------------------------------------
# STRATEGY 5 — Large-scale scrape (formerly scrape_big.py)
# Comprehensive scrape targeting 600+ reviews per sentiment per app.
# Positive: 5-star (400 en + 200 ms) + 4-star (400 en + 200 ms)
# Neutral:  3-star (400 en + 400 ms)
# Negative: 1-star (400 en + 200 ms) + 2-star (400 en + 200 ms)
# Applies >=5 word content filter on merge.
# ---------------------------------------------------------------------------
BIG_POSITIVE_BATCHES = [(5, 400, "en", "my"), (5, 200, "ms", "my"), (4, 400, "en", "my"), (4, 200, "ms", "my")]
BIG_NEUTRAL_BATCHES = [(3, 400, "en", "my"), (3, 400, "ms", "my")]
BIG_NEGATIVE_BATCHES = [(1, 400, "en", "my"), (1, 200, "ms", "my"), (2, 400, "en", "my"), (2, 200, "ms", "my")]


def run_big_scrape() -> pd.DataFrame:
    safe_print("\n" + "=" * 60)
    safe_print("STRATEGY 5 — Large-Scale Scrape")
    safe_print("Target: 600+ per sentiment per app")
    safe_print("=" * 60)

    existing_df = load_existing()
    all_new: list[dict] = []

    for app_config in APPS:
        safe_print(f"\nScraping {app_config['appName']} ({app_config['source']})...")
        for label, batches in [
            ("POSITIVE", BIG_POSITIVE_BATCHES),
            ("NEUTRAL", BIG_NEUTRAL_BATCHES),
            ("NEGATIVE", BIG_NEGATIVE_BATCHES),
        ]:
            safe_print(f"  {label}:")
            for score_filter, target, lang, country in batches:
                safe_print(f"    score={score_filter}, lang={lang}, target={target}...")
                batch = scrape_paginated(
                    app_config["app_id"],
                    lang=lang,
                    country=country,
                    target_count=target,
                    score_filter=score_filter,
                )
                tag_reviews(batch, app_config)
                all_new.extend(batch)
                safe_print(f"      Retrieved {len(batch)} reviews")
                time.sleep(REQUEST_DELAY)

    combined = merge_and_save(existing_df, all_new, min_words=5)
    print_summary(combined, "LARGE-SCALE SCRAPE SUMMARY")
    return combined


# ---------------------------------------------------------------------------
# STRATEGY 6 — More positive reviews (formerly scrape_more_positive.py)
# Extra 4-5 star positive boost: 500 five-star + 500 four-star per app.
# English only (lang=en, country=my). Merges with existing dataset.
# ---------------------------------------------------------------------------
MORE_POSITIVE_TARGETS = [(5, 500), (4, 500)]
MORE_POSITIVE_LANG = "en"
MORE_POSITIVE_COUNTRY = "my"


def run_more_positive_scrape() -> pd.DataFrame:
    safe_print("\n" + "=" * 60)
    safe_print("STRATEGY 6 — More Positive Reviews")
    safe_print("500 x 5-star + 500 x 4-star per app | English only")
    safe_print("=" * 60)

    existing_df = load_existing()
    all_new: list[dict] = []

    for app_config in APPS:
        safe_print(f"\nScraping {app_config['appName']} ({app_config['source']})...")
        for score_filter, target in MORE_POSITIVE_TARGETS:
            safe_print(f"  score={score_filter}, target={target}...")
            batch = scrape_paginated(
                app_config["app_id"],
                lang=MORE_POSITIVE_LANG,
                country=MORE_POSITIVE_COUNTRY,
                target_count=target,
                score_filter=score_filter,
            )
            tag_reviews(batch, app_config)
            all_new.extend(batch)
            safe_print(f"    Retrieved {len(batch)} reviews")
            time.sleep(REQUEST_DELAY)

    combined = merge_and_save(existing_df, all_new)
    print_summary(combined, "MORE POSITIVE SCRAPE SUMMARY")
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
STRATEGIES = {
    "initial": run_initial_scrape,
    "neutral": run_neutral_scrape,
    "extra": run_extra_scrape,
    "targeted": run_targeted_scrape,
    "big": run_big_scrape,
    "more_positive": run_more_positive_scrape,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Google Play Store reviews for Malaysian eWallet apps."
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="initial",
        help=(
            "Scraping strategy to run: "
            "initial | neutral | extra | targeted | big | more_positive"
        ),
    )
    args = parser.parse_args()

    safe_print(f"Starting scrape — strategy: {args.strategy}")
    STRATEGIES[args.strategy]()
    safe_print("\nDone.")


if __name__ == "__main__":
    main()
