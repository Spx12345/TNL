"""
Retrain sentiment models with hyperparameter search and binary classification.
TNL6323 — eWallet Aspect-Based Sentiment Analysis
"""

import copy
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data" / "sentiment_data_cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_PATH = MODELS_DIR / "model_results.txt"

SENTIMENT_MAP_3 = {"negative": 0, "neutral": 1, "positive": 2}
SENTIMENT_MAP_2 = {"negative": 0, "positive": 1}
SENTIMENT_NAMES_3 = ["negative", "neutral", "positive"]
SENTIMENT_NAMES_2 = ["negative", "positive"]

MBERT_MODEL_NAME = "bert-base-multilingual-cased"
MBERT_EPOCHS = 5
MBERT_BATCH_SIZE = 16
MBERT_MAX_LENGTH = 128
MBERT_LR = 2e-5
MBERT_WARMUP = 100
EARLY_STOPPING_PATIENCE = 2
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def format_time(seconds: float) -> str:
    return f"{seconds / 60:.1f} mins" if seconds >= 60 else f"{seconds:.1f} secs"


def prepare_texts(df: pd.DataFrame, text_col: str, use_source: bool) -> pd.Series:
    texts = df[text_col].fillna("").astype(str)
    if use_source:
        return df["source"].astype(str) + " " + texts
    return texts


def filter_task_data(df: pd.DataFrame, task: str):
    if task == "binary":
        subset = df[df["sentiment"].isin(["positive", "negative"])].copy()
        labels = subset["sentiment"].map(SENTIMENT_MAP_2).values
        label_names = SENTIMENT_NAMES_2
    else:
        subset = df.copy()
        labels = subset["sentiment"].map(SENTIMENT_MAP_3).values
        label_names = SENTIMENT_NAMES_3
    return subset, labels, label_names


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_predictions(y_true, y_pred, label_names: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def config_label(task: str, text_col: str, use_source: bool) -> str:
    return f"{task} | text={text_col} | source={'yes' if use_source else 'no'}"


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------
def train_lr_variation(texts: pd.Series, labels: np.ndarray, label_names: list[str], config_name: str) -> dict:
    start = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    grid = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        param_grid={"C": [0.1, 1, 10]},
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    grid.fit(X_train_tfidf, y_train)

    y_pred = grid.predict(X_test_tfidf)
    metrics = evaluate_predictions(y_test, y_pred, label_names)

    return {
        "model_type": "Logistic Regression",
        "config": config_name,
        "metrics": metrics,
        "training_time": time.time() - start,
        "best_C": grid.best_params_["C"],
        "model": grid.best_estimator_,
        "vectorizer": vectorizer,
    }


def run_all_lr_experiments(df: pd.DataFrame) -> list[dict]:
    print_section("LOGISTIC REGRESSION — ALL VARIATIONS")
    results = []

    for text_col in ["cleaned_text", "filtered_tokens"]:
        for use_source in [False, True]:
            for task in ["3-class", "binary"]:
                subset, labels, label_names = filter_task_data(df, task)
                texts = prepare_texts(subset, text_col, use_source)
                name = config_label(task, text_col, use_source)
                print(f"\nTraining: {name}")
                result = train_lr_variation(texts, labels, label_names, name)
                result.update({"task": task, "text_col": text_col, "use_source": use_source})
                results.append(result)
                m = result["metrics"]
                print(
                    f"  Best C={result['best_C']} | Acc={m['accuracy']*100:.2f}% | F1={m['f1_weighted']*100:.2f}%"
                )

    print("\n--- LR Results Table ---")
    print(f"{'Configuration':<55} | {'Acc':>6} | {'F1':>6} | Time")
    print("-" * 85)
    for r in results:
        m = r["metrics"]
        print(
            f"{r['config']:<55} | {m['accuracy']*100:5.1f}% | "
            f"{m['f1_weighted']*100:5.1f}% | {format_time(r['training_time'])}"
        )
    return results


# ---------------------------------------------------------------------------
# mBERT
# ---------------------------------------------------------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def run_epoch(model, loader, optimizer, scheduler, class_weights, train: bool) -> float:
    model.train() if train else model.eval()
    total_loss, batches = 0.0, 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss(weight=class_weights)(outputs.logits, labels)
            if train:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / max(batches, 1)


def predict_mbert(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(batch["labels"].numpy())
    return labels, preds


def train_mbert_variation(
    texts: pd.Series,
    labels: np.ndarray,
    label_names: list[str],
    config_name: str,
    num_labels: int,
    verbose: bool = True,
) -> dict:
    start = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
    )

    tokenizer = BertTokenizer.from_pretrained(MBERT_MODEL_NAME)
    train_loader = DataLoader(
        SentimentDataset(X_tr.tolist(), y_tr.tolist(), tokenizer, MBERT_MAX_LENGTH),
        batch_size=MBERT_BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        SentimentDataset(X_val.tolist(), y_val.tolist(), tokenizer, MBERT_MAX_LENGTH),
        batch_size=MBERT_BATCH_SIZE, shuffle=False,
    )
    test_loader = DataLoader(
        SentimentDataset(X_test.tolist(), y_test.tolist(), tokenizer, MBERT_MAX_LENGTH),
        batch_size=MBERT_BATCH_SIZE, shuffle=False,
    )

    model = BertForSequenceClassification.from_pretrained(
        MBERT_MODEL_NAME, num_labels=num_labels
    ).to(device)

    class_weights = compute_class_weights(y_tr, num_labels)
    optimizer = AdamW(model.parameters(), lr=MBERT_LR)
    total_steps = len(train_loader) * MBERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, MBERT_WARMUP, total_steps)

    best_val_loss = float("inf")
    best_state = None
    patience = 0
    epochs_run = 0

    for epoch in range(MBERT_EPOCHS):
        epochs_run = epoch + 1
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, class_weights, True)
        val_loss = run_epoch(model, val_loader, None, None, class_weights, False)
        if verbose:
            print(f"    Epoch {epoch + 1} | train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                if verbose:
                    print(f"    Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    y_true, y_pred = predict_mbert(model, test_loader)
    metrics = evaluate_predictions(y_true, y_pred, label_names)

    return {
        "model_type": "mBERT",
        "config": config_name,
        "metrics": metrics,
        "training_time": time.time() - start,
        "epochs_run": epochs_run,
        "best_val_loss": best_val_loss,
        "model": model,
        "tokenizer": tokenizer,
        "num_labels": num_labels,
    }


def run_all_mbert_experiments(df: pd.DataFrame) -> list[dict]:
    print_section("mBERT — ALL VARIATIONS")
    print(f"Device: {device} | Max epochs: {MBERT_EPOCHS} | LR: {MBERT_LR} | Warmup: {MBERT_WARMUP}")
    results = []

    for text_col in ["cleaned_text", "filtered_tokens"]:
        for use_source in [False, True]:
            for task in ["3-class", "binary"]:
                subset, labels, label_names = filter_task_data(df, task)
                texts = prepare_texts(subset, text_col, use_source)
                name = config_label(task, text_col, use_source)
                print(f"\nTraining: {name}")
                try:
                    result = train_mbert_variation(
                        texts, labels, label_names, name, len(label_names)
                    )
                    result.update({"task": task, "text_col": text_col, "use_source": use_source})
                    # Keep only metrics in results list; model saved separately for best
                    model = result.pop("model")
                    tokenizer = result.pop("tokenizer")
                    result["_model"] = model
                    result["_tokenizer"] = tokenizer
                    results.append(result)
                    m = result["metrics"]
                    print(f"  Acc={m['accuracy']*100:.2f}% | F1={m['f1_weighted']*100:.2f}%")
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print("\n--- mBERT Results Table ---")
    print(f"{'Configuration':<55} | {'Acc':>6} | {'F1':>6} | Time")
    print("-" * 85)
    for r in results:
        m = r["metrics"]
        print(
            f"{r['config']:<55} | {m['accuracy']*100:5.1f}% | "
            f"{m['f1_weighted']*100:5.1f}% | {format_time(r['training_time'])}"
        )
    return results


def pick_best(results: list[dict]) -> dict:
    return max(results, key=lambda r: r["metrics"]["f1_weighted"])


def save_best_models(best_lr: dict, best_mbert: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_lr["model"], MODELS_DIR / "logistic_regression_model.pkl")
    joblib.dump(best_lr["vectorizer"], MODELS_DIR / "tfidf_vectorizer.pkl")
    best_mbert["_model"].save_pretrained(MODELS_DIR / "mbert_model")
    best_mbert["_tokenizer"].save_pretrained(MODELS_DIR / "mbert_tokenizer")
    print(f"\nSaved best LR model + vectorizer to {MODELS_DIR}")
    print(f"Saved best mBERT model + tokenizer to {MODELS_DIR}")


def save_results_file(
    lr_results: list[dict],
    mbert_results: list[dict],
    best_lr: dict,
    best_mbert: dict,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("eWallet ABSA — Retrained Model Results\n")
        f.write(f"Training Date: {timestamp}\n")
        f.write(f"Dataset: {DATA_PATH}\n")
        f.write(f"Device: {device}\n\n")
        f.write("NOTE: Aspect Classifier kept as-is from previous training (58.33% accuracy).\n\n")

        f.write("=" * 70 + "\n")
        f.write("LOGISTIC REGRESSION — ALL VARIATIONS\n")
        f.write("=" * 70 + "\n")
        for r in lr_results:
            m = r["metrics"]
            f.write(f"\n{r['config']}\n")
            f.write(f"  Best C: {r['best_C']} | Acc: {m['accuracy']:.4f} | F1: {m['f1_weighted']:.4f}\n")
            f.write(f"  Time: {format_time(r['training_time'])}\n")
            f.write(f"  Report:\n{r['metrics']['classification_report']}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("mBERT — ALL VARIATIONS\n")
        f.write("=" * 70 + "\n")
        for r in mbert_results:
            m = r["metrics"]
            f.write(f"\n{r['config']}\n")
            f.write(f"  Epochs run: {r['epochs_run']} | Val loss: {r['best_val_loss']:.4f}\n")
            f.write(f"  Acc: {m['accuracy']:.4f} | F1: {m['f1_weighted']:.4f}\n")
            f.write(f"  Time: {format_time(r['training_time'])}\n")
            f.write(f"  Report:\n{r['metrics']['classification_report']}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("BEST MODELS SELECTED (by weighted F1)\n")
        f.write("=" * 70 + "\n")
        f.write(f"\nBest Logistic Regression: {best_lr['config']}\n")
        f.write(f"  C={best_lr['best_C']} | Acc={best_lr['metrics']['accuracy']:.4f} | ")
        f.write(f"F1={best_lr['metrics']['f1_weighted']:.4f}\n")
        f.write(f"\nBest mBERT: {best_mbert['config']}\n")
        f.write(f"  Acc={best_mbert['metrics']['accuracy']:.4f} | ")
        f.write(f"F1={best_mbert['metrics']['f1_weighted']:.4f}\n")

        f.write("\n\nCOMPARISON TABLE (BEST VERSIONS)\n")
        f.write("+---------------------------+-----------+----------+---------------+\n")
        f.write("| Model                     | Accuracy  | F1 Score | Training Time |\n")
        f.write("+---------------------------+-----------+----------+---------------+\n")
        for r in [best_lr, best_mbert]:
            m = r["metrics"]
            f.write(
                f"| {r['model_type']:<25} | {m['accuracy']*100:7.2f}% | "
                f"{m['f1_weighted']*100:6.2f}% | {format_time(r['training_time']):>13} |\n"
            )
        f.write("| Aspect Classifier (prev)  |   58.33% |  52.39% |     (unchanged)|\n")
        f.write("+---------------------------+-----------+----------+---------------+\n")

    print(f"Results saved to {RESULTS_PATH}")


def main() -> None:
    print_section("SENTIMENT MODEL RETRAINING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Device: {device}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    lr_results = run_all_lr_experiments(df)
    mbert_results = run_all_mbert_experiments(df)

    best_lr = pick_best(lr_results)
    best_mbert = pick_best(mbert_results)

    print_section("BEST MODELS (selected by weighted F1)")
    print(f"Best LR:    {best_lr['config']}")
    print(f"  Acc={best_lr['metrics']['accuracy']*100:.2f}% F1={best_lr['metrics']['f1_weighted']*100:.2f}% C={best_lr['best_C']}")
    print(f"Best mBERT: {best_mbert['config']}")
    print(f"  Acc={best_mbert['metrics']['accuracy']*100:.2f}% F1={best_mbert['metrics']['f1_weighted']*100:.2f}%")

    # 3-class vs binary summary
    print_section("3-CLASS vs BINARY SUMMARY")
    for model_type, results in [("Logistic Regression", lr_results), ("mBERT", mbert_results)]:
        print(f"\n{model_type}:")
        for task in ["3-class", "binary"]:
            task_results = [r for r in results if r["task"] == task]
            if task_results:
                best = pick_best(task_results)
                m = best["metrics"]
                print(
                    f"  {task:<8} best F1={m['f1_weighted']*100:.1f}% Acc={m['accuracy']*100:.1f}% "
                    f"({best['config']})"
                )

    save_best_models(best_lr, best_mbert)
    save_results_file(lr_results, mbert_results, best_lr, best_mbert)

    print_section("RETRAINING COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
