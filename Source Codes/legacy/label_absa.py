"""
Label aspects and aspect-level sentiment using Groq API for ABSA.
TNL6323 — Malaysian eWallet Sentiment Analysis project.
"""

import json
import logging
import os
import re
import sys
import threading
import time
import warnings
from pathlib import Path

import pandas as pd
from groq import Groq

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "Data" / "sentiment_data_cleaned.csv"
PROGRESS_PATH = PROJECT_ROOT / "Data" / "absa_progress.csv"
OUTPUT_PATH = PROJECT_ROOT / "Data" / "sentiment_data_absa_labeled.csv"

MODEL_NAME = "llama-3.1-8b-instant"
API_DELAY = 0.5
SAVE_EVERY = 25
API_TIMEOUT = 10

VALID_ASPECTS = {"payment", "ui", "service", "rewards", "general"}
VALID_SENTIMENTS = {"positive", "neutral", "negative"}

PROMPT_TEMPLATE = """You are a sentiment analysis expert for Malaysian eWallet apps (Touch n Go, GrabPay, ShopeePay).

Read this review and respond with ONLY a valid JSON object, no extra text, no markdown:
{{"aspect": "payment|ui|service|rewards|general", "aspect_sentiment": "positive|neutral|negative"}}

Aspect definitions:
- payment: transactions, money, reload, deduction, refund, crash, error, failed payment, security, OTP
- ui: app design, layout, navigation, ease of use, update, interface, button, screen, laggy, smooth
- service: customer support, complaint, agent, response, help, contact, resolve, rude, helpful
- rewards: cashback, voucher, promo, points, discount, offer, bonus, redeem, campaign, rebate, coin
- general: overall opinion not fitting above categories

Review: {review}"""


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def call_groq(client: Groq, prompt: str, timeout: int = API_TIMEOUT) -> str | None:
    result = [None]
    error = [None]

    def target():
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a sentiment analysis expert. "
                            "Always respond with valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=100,
            )
            result[0] = response.choices[0].message.content
        except Exception as exc:
            error[0] = str(exc)

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None
    if error[0]:
        raise Exception(error[0])
    return result[0]


def parse_json_response(text: str) -> dict | None:
    """Extract and validate JSON from model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]+\}", text)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    aspect = str(data.get("aspect", "")).lower().strip()
    sentiment = str(data.get("aspect_sentiment", "")).lower().strip()

    if aspect not in VALID_ASPECTS or sentiment not in VALID_SENTIMENTS:
        return None

    return {"aspect": aspect, "aspect_sentiment": sentiment}


def label_review(
    client: Groq,
    review_text: str,
    fallback_aspect: str,
    fallback_sentiment: str,
) -> tuple[str, str, str, bool]:
    """Return (aspect_label, absa_sentiment, absa_label, used_groq)."""
    prompt = PROMPT_TEMPLATE.format(review=review_text)

    for attempt in range(2):
        try:
            response_text = call_groq(client, prompt)
            if response_text:
                result = parse_json_response(response_text)
                if result:
                    aspect = result["aspect"]
                    sentiment = result["aspect_sentiment"]
                    return aspect, sentiment, f"{aspect}_{sentiment}", True
        except Exception as exc:
            if attempt == 0:
                safe_print(f"  API error (retrying): {exc}")
            else:
                safe_print(f"  API error (using fallback): {exc}")

        if attempt == 0:
            time.sleep(1)

    aspect = fallback_aspect if fallback_aspect in VALID_ASPECTS else "general"
    sentiment = fallback_sentiment if fallback_sentiment in VALID_SENTIMENTS else "neutral"
    return aspect, sentiment, f"{aspect}_{sentiment}", False


def save_progress(df: pd.DataFrame) -> None:
    df.to_csv(PROGRESS_PATH, index=False, encoding="utf-8-sig")
    safe_print(f"  Progress saved to {PROGRESS_PATH} ({len(df)} rows)")


def main() -> None:
    print_section("ABSA Labeling with Groq API")

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set!")
        print('Run: $env:GROQ_API_KEY = "your-key-here"')
        sys.exit(1)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    safe_print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    df["aspect_label_keyword"] = df["aspect_label"].copy()

    if PROGRESS_PATH.exists():
        progress = pd.read_csv(PROGRESS_PATH, encoding="utf-8-sig")
        processed_ids = set(progress["id"].astype(str))
        safe_print(f"Resuming from progress file: {len(processed_ids)} rows already processed")

        for col in ["aspect_label", "absa_sentiment", "absa_label"]:
            if col in progress.columns:
                merge_col = progress.set_index("id")[col].to_dict()
                df[col] = df["id"].map(merge_col).fillna(df.get(col, pd.NA))
    else:
        processed_ids = set()
        df["absa_sentiment"] = None
        df["absa_label"] = None

    total = len(df)
    newly_processed = 0
    groq_labeled = 0
    keyword_fallback = 0

    for idx, row in df.iterrows():
        row_id = str(row["id"])
        if row_id in processed_ids and pd.notna(row.get("absa_label")):
            continue

        review = str(row["content"]) if pd.notna(row["content"]) else ""
        fallback_aspect = str(row.get("aspect_label_keyword", "general"))
        fallback_sentiment = str(row.get("sentiment", "neutral"))

        aspect, sentiment, absa_label, used_groq = label_review(
            client, review, fallback_aspect, fallback_sentiment
        )

        df.at[idx, "aspect_label"] = aspect
        df.at[idx, "absa_sentiment"] = sentiment
        df.at[idx, "absa_label"] = absa_label
        processed_ids.add(row_id)
        newly_processed += 1

        if used_groq:
            groq_labeled += 1
        else:
            keyword_fallback += 1

        if newly_processed % SAVE_EVERY == 0:
            pct = len(processed_ids) / total * 100
            safe_print(
                f"Progress: {len(processed_ids)}/{total} ({pct:.1f}%) | "
                f"Groq labels: {groq_labeled} | Keyword fallback: {keyword_fallback}"
            )
            save_progress(df)

        time.sleep(API_DELAY)

    save_progress(df)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    safe_print(f"\nFinal dataset saved to {OUTPUT_PATH}")

    print_section("LABELING SUMMARY")
    safe_print(f"Total rows: {len(df)}")
    safe_print(f"Newly processed this run: {newly_processed}")
    safe_print(f"Total Groq labeled: {groq_labeled}")
    safe_print(f"Total keyword fallback: {keyword_fallback}")
    if newly_processed > 0:
        success_rate = groq_labeled / newly_processed * 100
        safe_print(f"Groq success rate: {success_rate:.1f}%")
    else:
        safe_print("Groq success rate: N/A (no new rows processed)")

    safe_print("\nAspect distribution (Groq + fallback):")
    safe_print(df["aspect_label"].value_counts().to_string())

    safe_print("\nABSA label distribution:")
    safe_print(df["absa_label"].value_counts().head(15).to_string())

    agreement = (df["aspect_label"] == df["aspect_label_keyword"]).mean()
    safe_print(f"\nAgreement rate (keyword vs Groq aspect): {agreement * 100:.2f}%")

    safe_print("\nSample of 5 labeled rows:")
    sample = df.sample(n=min(5, len(df)), random_state=42)
    for _, r in sample.iterrows():
        preview = str(r["content"]).replace("\n", " ")[:100]
        safe_print(f"  Review: {preview}...")
        safe_print(f"  Aspect: {r['aspect_label']} | ABSA sentiment: {r['absa_sentiment']}")
        safe_print("")


if __name__ == "__main__":
    main()
