"""
Technology-Focused Malaysian eWallet ABSA — Streamlit Web Application
TNL6323 — Natural Language Processing
"""

import html
import re
from pathlib import Path

import emoji
import ftfy
import numpy as np
import onnxruntime as ort
import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import BertTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data" / "sentiment_data_absa_labeled.csv"
CHARTS_DIR = PROJECT_ROOT / "Data" / "charts"
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_PATH = MODELS_DIR / "model_results.txt"

ONNX_PATH = MODELS_DIR / "mbert_model.onnx"
TOKENIZER_DIR = MODELS_DIR / "mbert_tokenizer"

MAX_LENGTH = 128
SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
ASPECT_LABELS = {0: "payment", 1: "ui", 2: "service", 3: "rewards", 4: "general"}

APP_OPTIONS = {
    "Touch 'n Go eWallet": "touchngo",
    "GrabPay": "grabpay",
    "ShopeePay": "shopeepay",
}
APP_DISPLAY = {v: k for k, v in APP_OPTIONS.items()}

ASPECT_INFO = {
    "payment": ("💳", "Payment & Reliability"),
    "ui": ("📱", "User Interface"),
    "service": ("🎧", "Customer Service"),
    "rewards": ("🎁", "Promotions & Rewards"),
    "general": ("📝", "General Feedback"),
}

SENTIMENT_STYLE = {
    "positive": {"emoji": "😊", "color": "#d4edda", "border": "#28a745", "label": "Positive"},
    "neutral": {"emoji": "😐", "color": "#fff3cd", "border": "#ffc107", "label": "Neutral"},
    "negative": {"emoji": "😠", "color": "#f8d7da", "border": "#dc3545", "label": "Negative"},
}

POSITIVE_EMOJIS = {
    "😊", "👍", "❤️", "😍", "✅", "👏", "🙏", "😁", "💪", "🥰",
    "😄", "🎉", "💯", "⭐", "🌟",
}
NEGATIVE_EMOJIS = {
    "😡", "👎", "💔", "😤", "😭", "❌", "🤬", "😠", "🙄", "😒",
    "😞", "💩", "🤮", "😱", "😰",
}

st.set_page_config(
    page_title="Malaysian FinTech eWallet Sentiment Analyser",
    page_icon="🇲🇾",
    layout="wide",
)


def reduce_repeated_chars(text: str) -> str:
    return re.sub(r"(.)\1{3,}", r"\1\1\1", text)


def clean_input_text(text: str) -> str:
    if not text or not str(text).strip():
        return ""
    text = ftfy.fix_text(str(text))
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = reduce_repeated_chars(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_emojis(text: str) -> tuple[list[str], str]:
    found = [item["emoji"] for item in emoji.emoji_list(text)]
    pos = sum(1 for e in found if e in POSITIVE_EMOJIS)
    neg = sum(1 for e in found if e in NEGATIVE_EMOJIS)
    if pos > neg:
        signal = "positive"
    elif neg > pos:
        signal = "negative"
    elif found:
        signal = "neutral"
    else:
        signal = ""
    return found, signal


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def models_available() -> bool:
    return ONNX_PATH.exists() and TOKENIZER_DIR.exists()


def get_model_metrics() -> tuple[str, str]:
    """Read sentiment/aspect accuracy from model_results.txt if available."""
    sent_acc, asp_acc = "—", "—"
    if RESULTS_PATH.exists():
        text = RESULTS_PATH.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "mBERT (Sentiment)" in line and "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    sent_acc = parts[1]
            if "mBERT (Aspect)" in line and "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    asp_acc = parts[1]
    return sent_acc, asp_acc


def build_combined_insight(overall_sentiment: str, aspect: str, app_name: str) -> str:
    aspect_phrases = {
        "payment": {
            "positive": "particularly praising the payment and transaction reliability",
            "neutral": "with comments focused on payment and transaction features",
            "negative": "with specific complaints about payment and transaction reliability",
        },
        "ui": {
            "positive": "particularly praising the user interface and app design",
            "neutral": "with comments focused on the user interface and app experience",
            "negative": "with specific complaints about the user interface and app experience",
        },
        "service": {
            "positive": "particularly praising the customer service and support quality",
            "neutral": "with comments focused on customer service and support",
            "negative": "with specific complaints about customer service and support",
        },
        "rewards": {
            "positive": "particularly praising the promotions, cashback and reward features",
            "neutral": "with comments focused on promotions and reward features",
            "negative": "with specific complaints about promotions and reward offers",
        },
        "general": {
            "positive": "reflecting broadly positive feedback about the overall eWallet experience",
            "neutral": "with general, balanced feedback about the overall eWallet experience",
            "negative": "reflecting broadly negative feedback about the overall eWallet experience",
        },
    }
    sentiment_intros = {
        "positive": "This review expresses overall **POSITIVE** sentiment",
        "neutral": "This review has a **NEUTRAL** tone toward",
        "negative": "This review expresses overall **NEGATIVE** sentiment",
    }
    aspect_detail = aspect_phrases.get(aspect, aspect_phrases["general"])[overall_sentiment]
    intro = sentiment_intros[overall_sentiment]
    if overall_sentiment == "neutral":
        return f"{intro} **{app_name}**, {aspect_detail}."
    return f"{intro}, {aspect_detail} of **{app_name}**."


@st.cache_resource
def load_onnx_session():
    if not ONNX_PATH.exists():
        return None
    return ort.InferenceSession(str(ONNX_PATH))


@st.cache_resource
def load_tokenizer():
    if not TOKENIZER_DIR.exists():
        return None
    return BertTokenizer.from_pretrained(str(TOKENIZER_DIR))


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["app_display"] = df["source"].map(APP_DISPLAY)
    return df


def predict_dual_head(cleaned_text: str) -> dict:
    """Run dual-head mBERT ONNX — returns sentiment and aspect in one pass."""
    session = load_onnx_session()
    tokenizer = load_tokenizer()
    if session is None or tokenizer is None:
        raise FileNotFoundError("Dual-head mBERT ONNX model or tokenizer not found.")

    encoding = tokenizer(
        cleaned_text,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="np",
    )
    outputs = session.run(
        None,
        {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
        },
    )
    sentiment_logits = outputs[0][0]
    aspect_logits = outputs[1][0]

    sent_probs = softmax(sentiment_logits)
    asp_probs = softmax(aspect_logits)
    sent_id = int(np.argmax(sent_probs))
    asp_id = int(np.argmax(asp_probs))

    return {
        "sentiment": SENTIMENT_LABELS[sent_id],
        "sentiment_confidence": float(sent_probs[sent_id]),
        "aspect": ASPECT_LABELS[asp_id],
        "aspect_confidence": float(asp_probs[asp_id]),
    }


def render_overall_sentiment(sentiment: str, confidence: float) -> None:
    sent_acc, _ = get_model_metrics()
    style = SENTIMENT_STYLE[sentiment]
    st.markdown(f"#### Overall Sentiment (Dual-Head mBERT — {sent_acc} accuracy)")
    st.markdown(
        f"""
        <div style="
            background-color:{style['color']};
            border:2px solid {style['border']};
            border-radius:12px;
            padding:28px;
            text-align:center;
            margin-bottom:12px;">
            <h2 style="margin:0;color:#333;">{style['label']} {style['emoji']}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")


def render_aspect_analysis(aspect: str, sentiment: str, aspect_confidence: float) -> None:
    _, asp_acc = get_model_metrics()
    icon, aspect_name = ASPECT_INFO.get(aspect, ("📝", aspect.title()))
    asp_style = SENTIMENT_STYLE.get(sentiment, SENTIMENT_STYLE["neutral"])

    st.markdown(f"#### Aspect Analysis (Dual-Head mBERT — {asp_acc} accuracy)")
    st.markdown(
        f"""
        <div style="
            background-color:#f8f9fa;
            border:2px solid #dee2e6;
            border-radius:12px;
            padding:24px;
            margin-bottom:12px;">
            <h3 style="margin:0 0 10px 0;">{icon} {aspect_name}</h3>
            <p style="margin:0 0 8px 0;font-size:1.05em;">
                Detected aspect with
                <strong>{aspect_confidence * 100:.1f}%</strong> confidence
            </p>
            <p style="margin:0;font-size:1.05em;">
                Overall review sentiment:
                <strong style="color:{asp_style['border']};">
                    {asp_style['label']} {asp_style['emoji']}
                </strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if aspect_confidence < 0.50:
        st.warning(
            "Low-confidence aspect prediction. Treat this category as an exploratory "
            "suggestion rather than a definitive classification."
        )


def render_combined_insight(
    overall_sentiment: str,
    aspect: str,
    app_name: str,
    sentiment_confidence: float,
    aspect_confidence: float,
    emojis: list[str],
    emoji_signal: str,
) -> None:
    st.markdown("#### Combined Insight")
    st.markdown(build_combined_insight(overall_sentiment, aspect, app_name))
    st.caption(f"eWallet analysed: **{app_name}**")
    st.caption(
        f"Sentiment confidence: **{sentiment_confidence * 100:.1f}%** | "
        f"Aspect confidence: **{aspect_confidence * 100:.1f}%**"
    )
    if emojis:
        emoji_str = " ".join(emojis)
        tone_hints = {
            "positive": "suggesting positive tone",
            "negative": "suggesting negative tone",
            "neutral": "suggesting neutral tone",
        }
        hint = tone_hints.get(emoji_signal, "detected in review")
        st.info(f"Emojis found: {emoji_str} {hint}")


def page_sentiment_analyser() -> None:
    st.title("🇲🇾 Malaysian FinTech eWallet Sentiment Analyser")
    st.caption("Technology (Malaysia Tech Scene) · Touch 'n Go · GrabPay · ShopeePay")

    if not models_available():
        st.error("Model files not found. Please run train_models.py first.")
        return

    review_text = st.text_area(
        "Enter your eWallet review here",
        placeholder=(
            "e.g. Touch n Go app keeps crashing when I try to reload my balance..."
        ),
        height=150,
    )
    selected_app = st.selectbox("Select eWallet", list(APP_OPTIONS.keys()))

    if st.button("Analyse Sentiment", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Please enter a review before analysing.")
            return

        with st.spinner("Analysing your review..."):
            cleaned = clean_input_text(review_text)
            emojis_found, emoji_signal = detect_emojis(review_text)

            try:
                result = predict_dual_head(cleaned)
            except FileNotFoundError as exc:
                st.error(f"{exc} Please run train_models.py first.")
                return
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        st.divider()
        render_overall_sentiment(result["sentiment"], result["sentiment_confidence"])
        st.divider()
        render_aspect_analysis(
            result["aspect"],
            result["sentiment"],
            result["aspect_confidence"],
        )
        st.divider()
        render_combined_insight(
            result["sentiment"],
            result["aspect"],
            selected_app,
            result["sentiment_confidence"],
            result["aspect_confidence"],
            emojis_found,
            emoji_signal,
        )


def compute_conclusion(df: pd.DataFrame) -> str:
    reviews_per_app = df.groupby("source").size()
    balanced_per_app = int(reviews_per_app.min()) if len(reviews_per_app) else 0

    overall_aspects = df["aspect_label"].value_counts()
    top_aspect = overall_aspects.index[0] if len(overall_aspects) else "general"
    top_aspect_count = int(overall_aspects.iloc[0]) if len(overall_aspects) else 0

    negative_rows = df[df["absa_sentiment"] == "negative"]
    neg_aspects = negative_rows["aspect_label"].value_counts()
    top_complaint = neg_aspects.index[0] if len(neg_aspects) else "general"
    top_complaint_count = int(neg_aspects.iloc[0]) if len(neg_aspects) else 0

    ui_counts = df[df["aspect_label"] == "ui"]["sentiment"].value_counts(normalize=True)

    top_aspect_name = ASPECT_INFO.get(top_aspect, ("", top_aspect.title()))[1]
    complaint_name = ASPECT_INFO.get(top_complaint, ("", top_complaint.title()))[1]

    return f"""📊 **Key Findings from {len(df):,} Malaysian FinTech eWallet Reviews:**

⚖️ The dataset is intentionally balanced at **{balanced_per_app} reviews per application**, with equal positive, neutral and negative sentiment counts. The dashboard therefore does not rank one eWallet as the most positive.

🔎 **{top_aspect_name}** is the most frequently assigned dominant aspect with **{top_aspect_count}** reviews.

⚠️ **{complaint_name}** is the most frequent aspect among negative reviews with **{top_complaint_count}** mentions.

📱 Among UI-related reviews, the sentiment mix is **{ui_counts.get('positive', 0) * 100:.0f}% positive**, **{ui_counts.get('neutral', 0) * 100:.0f}% neutral**, and **{ui_counts.get('negative', 0) * 100:.0f}% negative**.

ℹ️ Aspect findings are exploratory because the aspect labels were produced through zero-shot pseudo-labelling and include low-confidence cases."""


def page_dashboard() -> None:
    st.title("📊 Malaysian FinTech eWallet Sentiment Dashboard")

    df = load_dataset()
    if df.empty:
        st.error(f"Dataset not found at {DATA_PATH}. Please run label_aspect_roberta.py first.")
        return

    reviews_per_app = df.groupby("source").size()
    balanced_reviews_per_app = int(reviews_per_app.min()) if len(reviews_per_app) else 0

    neg_aspects = df[df["absa_sentiment"] == "negative"]["aspect_label"].value_counts()
    most_complained = ASPECT_INFO.get(
        neg_aspects.index[0], ("", neg_aspects.index[0].title())
    )[1]

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Reviews Analysed", f"{len(df):,}")
    m2.metric("Balanced Reviews per App", f"{balanced_reviews_per_app:,}")
    m3.metric("Most Complained Aspect", most_complained)

    st.info(
        "The dataset contains equal positive, neutral and negative counts for each "
        "application. App-level positive-rate rankings are therefore intentionally "
        "not shown."
    )

    st.divider()

    st.subheader("Sentiment Comparison")
    sent_counts = df.groupby(["source", "sentiment"]).size().reset_index(name="count")
    sent_counts["app"] = sent_counts["source"].map(APP_DISPLAY)
    fig_sent = px.bar(
        sent_counts,
        x="app",
        y="count",
        color="sentiment",
        barmode="group",
        title="Sentiment Distribution by eWallet App",
        color_discrete_map={"positive": "#28a745", "neutral": "#ffc107", "negative": "#dc3545"},
        category_orders={"sentiment": ["positive", "neutral", "negative"], "app": list(APP_OPTIONS.keys())},
    )
    fig_sent.update_layout(xaxis_title="eWallet App", yaxis_title="Count", legend_title="Sentiment")
    st.plotly_chart(fig_sent, use_container_width=True)

    st.subheader("Aspect Breakdown")
    aspect_counts = df.groupby(["aspect_label", "source"]).size().reset_index(name="count")
    aspect_counts["app"] = aspect_counts["source"].map(APP_DISPLAY)
    fig_aspect = px.bar(
        aspect_counts,
        x="aspect_label",
        y="count",
        color="app",
        barmode="group",
        title="Aspect Distribution by eWallet App",
        labels={"aspect_label": "Aspect", "count": "Count"},
        category_orders={"app": list(APP_OPTIONS.keys())},
    )
    st.plotly_chart(fig_aspect, use_container_width=True)

    st.subheader("Aspect vs Sentiment Heatmap")
    heatmap_data = df.groupby(["aspect_label", "absa_sentiment"]).size().reset_index(name="count")
    heatmap_pivot = heatmap_data.pivot(index="aspect_label", columns="absa_sentiment", values="count").fillna(0)
    for col in ["positive", "neutral", "negative"]:
        if col not in heatmap_pivot.columns:
            heatmap_pivot[col] = 0
    heatmap_pivot = heatmap_pivot[["positive", "neutral", "negative"]]
    fig_heat = px.imshow(
        heatmap_pivot.values,
        x=["Positive", "Neutral", "Negative"],
        y=[ASPECT_INFO.get(a, ("", a))[1] for a in heatmap_pivot.index],
        text_auto=True,
        color_continuous_scale="YlOrRd",
        title="Aspect-Sentiment Heatmap",
        labels=dict(color="Count"),
    )
    fig_heat.update_layout(xaxis_title="Sentiment", yaxis_title="Aspect")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Word Clouds")
    tab_pos, tab_neu, tab_neg = st.tabs(["Positive Reviews", "Neutral Reviews", "Negative Reviews"])
    wordcloud_files = {
        "positive": CHARTS_DIR / "wordcloud_positive.png",
        "neutral": CHARTS_DIR / "wordcloud_neutral.png",
        "negative": CHARTS_DIR / "wordcloud_negative.png",
    }
    for tab, sentiment in zip([tab_pos, tab_neu, tab_neg], wordcloud_files):
        with tab:
            wc_path = wordcloud_files[sentiment]
            if wc_path.exists():
                st.image(str(wc_path), use_container_width=True)
            else:
                st.info(f"Word cloud not found at {wc_path.name}. Run preprocess.py to generate charts.")

    st.divider()
    st.subheader("Auto Conclusion")
    st.markdown(compute_conclusion(df))


def page_about() -> None:
    st.title("ℹ️ About This Project")
    sent_acc, asp_acc = get_model_metrics()

    with st.expander("Project Overview", expanded=True):
        st.markdown(
            """
            **Domain:** Technology (Malaysia Tech Scene) — Financial Technology

            **Title:** Aspect-Based Sentiment Analysis of Malaysian FinTech eWallet Applications

            **Objective:** To analyse how payment reliability, user interface, customer service
            and promotional rewards are associated with positive and negative sentiment among
            Malaysian eWallet users.
            """
        )

    with st.expander("Dataset Information"):
        st.markdown(
            """
            - **Technology focus:** Malaysian financial technology and mobile payment applications
            - **Source:** Google Play Store Malaysia
            - **Apps:** Touch 'n Go eWallet, GrabPay, ShopeePay
            - **Total reviews:** 1,980 (balanced)
            - **Per app:** 660 reviews (220 per sentiment class)
            - **Language:** English and Bahasa Malaysia
            - **Sentiment labeling:** XLM-RoBERTa teacher model
            - **Aspect labeling:** BART-large-MNLI zero-shot classification
            """
        )

    with st.expander("Methodology"):
        st.markdown(
            f"""
            - **Data Collection:** google-play-scraper (no API key needed)
            - **Preprocessing:** ftfy encoding fix, conservative cleaning,
              NLTK tokenization, Malaysian stopword removal
            - **Aspect Labeling:** `facebook/bart-large-mnli` zero-shot classifier
            - **Sentiment + Aspect Model:** Dual-head mBERT (`bert-base-multilingual-cased`)
              — Sentiment accuracy: {sent_acc}, Aspect accuracy: {asp_acc}
            - **Architecture:** Shared BERT encoder with two output heads
              (sentiment: 3 classes, aspect: 5 classes)
            - **Deployment:** Streamlit web application
            - **Model optimization:** ONNX export for faster inference
            """
        )

    with st.expander("Advanced Features"):
        st.markdown(
            """
            - ✅ Dual-head transformer model (mBERT)
            - ✅ Zero-shot aspect labeling (BART-MNLI)
            - ✅ Aspect-Based Sentiment Analysis (ABSA)
            - ✅ Emoji sentiment detection
            - ✅ ONNX model optimization
            - ✅ Multilingual support (English + Bahasa Malaysia)
            """
        )

    with st.expander("Group Members"):
        st.markdown(
            """
            - **Member 1:** Sia Poh Xiang
            - **Member 2:** Shi Hong Yi
            - **Member 3:** Lim Chin Chen
            - **Member 4:** Joshua Wong Yong
            """
        )


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Sentiment Analyser", "FinTech Comparison Dashboard", "About This Project"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption("TNL6323 NLP — Technology / Malaysian FinTech")
    st.sidebar.caption("Touch 'n Go · GrabPay · ShopeePay")

    if page == "Sentiment Analyser":
        page_sentiment_analyser()
    elif page == "FinTech Comparison Dashboard":
        page_dashboard()
    else:
        page_about()


if __name__ == "__main__":
    main()
