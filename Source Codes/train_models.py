"""
Train dual-head mBERT for sentiment + aspect classification.
TNL6323 — Natural Language Processing

Single mBERT model with two output heads:
  - Sentiment: positive / neutral / negative
  - Aspect: payment / ui / service / rewards / general
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel, BertTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data" / "sentiment_data_absa_labeled.csv"
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_PATH = MODELS_DIR / "model_results.txt"

MBERT_MODEL_NAME = "bert-base-multilingual-cased"
MBERT_EPOCHS = 5
MBERT_BATCH_SIZE = 16
MBERT_MAX_LENGTH = 128
MBERT_LR = 2e-5
MBERT_WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
EARLY_STOPPING_PATIENCE = 2
FREEZE_LAYERS = 6
RANDOM_STATE = 42

SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}
SENTIMENT_NAMES = ["negative", "neutral", "positive"]
ASPECT_MAP = {"payment": 0, "ui": 1, "service": 2, "rewards": 3, "general": 4}
ASPECT_NAMES = ["payment", "ui", "service", "rewards", "general"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = "GPU" if torch.cuda.is_available() else "CPU"


class DualHeadBert(nn.Module):
    """mBERT encoder with sentiment and aspect classification heads."""

    def __init__(self, model_name: str, freeze_layers: int = FREEZE_LAYERS):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, 3)
        self.aspect_head = nn.Linear(self.bert.config.hidden_size, 5)
        self._freeze_lower_layers(freeze_layers)

    def _freeze_lower_layers(self, num_layers: int) -> None:
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentiment_labels: torch.Tensor | None = None,
        aspect_labels: torch.Tensor | None = None,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        sentiment_logits = self.sentiment_head(pooled)
        aspect_logits = self.aspect_head(pooled)

        loss = None
        if sentiment_labels is not None and aspect_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            sentiment_loss = loss_fn(sentiment_logits, sentiment_labels)
            aspect_loss = loss_fn(aspect_logits, aspect_labels)
            loss = sentiment_loss + aspect_loss

        return sentiment_logits, aspect_logits, loss

    def save_pretrained(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")
        self.bert.config.save_pretrained(save_dir)
        config = {
            "model_name": MBERT_MODEL_NAME,
            "architecture": "dual_head_bert",
            "sentiment_labels": SENTIMENT_NAMES,
            "aspect_labels": ASPECT_NAMES,
            "freeze_layers": FREEZE_LAYERS,
        }
        with open(save_dir / "dual_head_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_pretrained(cls, load_dir: Path) -> "DualHeadBert":
        """Load fine-tuned dual-head weights from saved checkpoint (not base BERT)."""
        config_path = load_dir / "dual_head_config.json"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        state_dict = torch.load(
            load_dir / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )

        if (load_dir / "config.json").exists():
            bert_config = BertConfig.from_pretrained(str(load_dir))
        else:
            bert_config = BertConfig.from_pretrained(config.get("model_name", MBERT_MODEL_NAME))

        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.bert = BertModel(bert_config)
        model.sentiment_head = nn.Linear(bert_config.hidden_size, 3)
        model.aspect_head = nn.Linear(bert_config.hidden_size, 5)
        model.load_state_dict(state_dict)
        model.eval()
        return model


class DualHeadOnnxWrapper(nn.Module):
    """ONNX export wrapper returning both logits tensors."""

    def __init__(self, model: DualHeadBert):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        sentiment_logits, aspect_logits, _ = self.model(input_ids, attention_mask)
        return sentiment_logits, aspect_logits


class DualHeadDataset(Dataset):
    def __init__(
        self,
        texts,
        sentiment_labels,
        aspect_labels,
        tokenizer,
        max_length,
    ):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.aspect_labels = aspect_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentiment_labels": torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
            "aspect_labels": torch.tensor(self.aspect_labels[idx], dtype=torch.long),
        }


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def format_time(seconds: float) -> str:
    if seconds >= 60:
        return f"{seconds / 60:.1f} mins"
    return f"{seconds:.1f} secs"


def encode_sentiment(series: pd.Series) -> np.ndarray:
    return series.map(SENTIMENT_MAP).values


def encode_aspect(series: pd.Series) -> np.ndarray:
    return series.map(ASPECT_MAP).values


def evaluate_task(y_true, y_pred, label_names, title: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{title}")
    print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Weighted F1: {f1:.4f} ({f1 * 100:.2f}%)")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def run_epoch(model, data_loader, optimizer=None, scheduler=None) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    batch_count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment_labels"].to(device)
            aspect_labels = batch["aspect_labels"].to(device)

            if is_train:
                optimizer.zero_grad()

            sentiment_logits, aspect_logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment_labels=sentiment_labels,
                aspect_labels=aspect_labels,
            )

            if is_train:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += loss.item()
            else:
                total_loss += loss.item()

            batch_count += 1

    return total_loss / max(batch_count, 1)


def predict_batch(model, data_loader) -> tuple[list[int], list[int], list[int], list[int]]:
    model.eval()
    sentiment_preds, sentiment_true = [], []
    aspect_preds, aspect_true = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment_labels"].to(device)
            aspect_labels = batch["aspect_labels"].to(device)

            sentiment_logits, aspect_logits, _ = model(input_ids, attention_mask)
            sentiment_preds.extend(torch.argmax(sentiment_logits, dim=1).cpu().numpy())
            aspect_preds.extend(torch.argmax(aspect_logits, dim=1).cpu().numpy())
            sentiment_true.extend(sentiment_labels.cpu().numpy())
            aspect_true.extend(aspect_labels.cpu().numpy())

    return sentiment_true, sentiment_preds, aspect_true, aspect_preds


def export_to_onnx(model_dir: Path, tokenizer, onnx_path: Path) -> None:
    print_section("ONNX Conversion — Dual-Head mBERT")

    model = DualHeadBert.load_pretrained(model_dir)
    model.cpu()
    wrapper = DualHeadOnnxWrapper(model)
    wrapper.cpu()
    wrapper.eval()

    sample_encoding = tokenizer(
        "Touch n Go payment is very fast and easy to use.",
        max_length=MBERT_MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = sample_encoding["input_ids"]
    attention_mask = sample_encoding["attention_mask"]

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(onnx_path),
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["sentiment_logits", "aspect_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "sentiment_logits": {0: "batch_size"},
            "aspect_logits": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"Saved ONNX model to {onnx_path}")

    session = ort.InferenceSession(str(onnx_path))
    sample_reviews = [
        "Touch n Go payment is very fast and easy to use.",
        "The app keeps crashing whenever I try to top up.",
        "It works fine, nothing special about the rewards.",
    ]

    print("\nPrediction comparison (PyTorch vs ONNX):")
    for review in sample_reviews:
        encoding = tokenizer(
            review,
            max_length=MBERT_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = encoding["input_ids"]
        mask = encoding["attention_mask"]

        with torch.no_grad():
            pt_sent, pt_asp = wrapper(ids, mask)
        pt_sent_label = SENTIMENT_NAMES[int(torch.argmax(pt_sent, dim=1))]
        pt_asp_label = ASPECT_NAMES[int(torch.argmax(pt_asp, dim=1))]

        onnx_out = session.run(
            None,
            {"input_ids": ids.numpy(), "attention_mask": mask.numpy()},
        )
        onnx_sent_label = SENTIMENT_NAMES[int(np.argmax(onnx_out[0], axis=1)[0])]
        onnx_asp_label = ASPECT_NAMES[int(np.argmax(onnx_out[1], axis=1)[0])]

        sent_ok = "OK" if pt_sent_label == onnx_sent_label else "MISMATCH"
        asp_ok = "OK" if pt_asp_label == onnx_asp_label else "MISMATCH"
        print(
            f"  [{sent_ok}/{asp_ok}] Sentiment: {pt_sent_label}/{onnx_sent_label} | "
            f"Aspect: {pt_asp_label}/{onnx_asp_label} | {review[:45]}..."
        )

    print("\nONNX conversion successful!")


def train_dual_head_mbert(
    X: pd.Series,
    y_sentiment: np.ndarray,
    y_aspect: np.ndarray,
) -> dict:
    print_section("Dual-Head mBERT Training")
    print(f"Base model: {MBERT_MODEL_NAME}")
    print(f"Device: {device} ({DEVICE_NAME})")

    start_time = time.time()

    indices = np.arange(len(X))
    idx_train_full, idx_test, ys_train_full, ys_test, ya_train_full, ya_test = train_test_split(
        indices,
        y_sentiment,
        y_aspect,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_sentiment,
    )
    idx_train, idx_val, ys_train, ys_val, ya_train, ya_val = train_test_split(
        idx_train_full,
        ys_train_full,
        ya_train_full,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=ys_train_full,
    )

    X_train = X.iloc[idx_train]
    X_val = X.iloc[idx_val]
    X_test = X.iloc[idx_test]

    print(
        f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )

    tokenizer = BertTokenizer.from_pretrained(MBERT_MODEL_NAME)
    train_loader = DataLoader(
        DualHeadDataset(
            X_train.tolist(), ys_train.tolist(), ya_train.tolist(), tokenizer, MBERT_MAX_LENGTH
        ),
        batch_size=MBERT_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        DualHeadDataset(
            X_val.tolist(), ys_val.tolist(), ya_val.tolist(), tokenizer, MBERT_MAX_LENGTH
        ),
        batch_size=MBERT_BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        DualHeadDataset(
            X_test.tolist(), ys_test.tolist(), ya_test.tolist(), tokenizer, MBERT_MAX_LENGTH
        ),
        batch_size=MBERT_BATCH_SIZE,
        shuffle=False,
    )

    model = DualHeadBert(MBERT_MODEL_NAME).to(device)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=MBERT_LR,
        weight_decay=MBERT_WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * MBERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(MBERT_EPOCHS):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler)
        val_loss = run_epoch(model, val_loader)

        s_true, s_pred, a_true, a_pred = predict_batch(model, val_loader)
        val_sent_acc = accuracy_score(s_true, s_pred)
        val_asp_acc = accuracy_score(a_true, a_pred)

        print(f"\nEpoch {epoch + 1}/{MBERT_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Sentiment Accuracy: {val_sent_acc * 100:.2f}%")
        print(f"Val Aspect Accuracy: {val_asp_acc * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best checkpoint from epoch {best_epoch}")

    print("\nEvaluating on test set...")
    s_true, s_pred, a_true, a_pred = predict_batch(model, test_loader)

    sentiment_metrics = evaluate_task(
        s_true, s_pred, SENTIMENT_NAMES, "mBERT — Sentiment Test Results"
    )
    aspect_metrics = evaluate_task(
        a_true, a_pred, ASPECT_NAMES, "mBERT — Aspect Test Results"
    )

    combined_acc = np.mean(
        [(s == sp) and (a == ap) for s, sp, a, ap in zip(s_true, s_pred, a_true, a_pred)]
    )
    print(f"\nCombined Accuracy (both correct): {combined_acc * 100:.2f}%")

    elapsed = time.time() - start_time

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mbert_model_dir = MODELS_DIR / "mbert_model"
    mbert_tokenizer_dir = MODELS_DIR / "mbert_tokenizer"
    model.save_pretrained(mbert_model_dir)
    tokenizer.save_pretrained(mbert_tokenizer_dir)
    with open(MODELS_DIR / "mbert_training_time.txt", "w", encoding="utf-8") as f:
        f.write(str(elapsed))

    print(f"\nSaved model to {mbert_model_dir}")
    print(f"Saved tokenizer to {mbert_tokenizer_dir}")
    print(f"Training time: {format_time(elapsed)}")

    export_to_onnx(mbert_model_dir, tokenizer, MODELS_DIR / "mbert_model.onnx")

    return {
        "sentiment_metrics": sentiment_metrics,
        "aspect_metrics": aspect_metrics,
        "combined_accuracy": combined_acc,
        "training_time": elapsed,
    }


def print_comparison_table(sentiment_metrics: dict, aspect_metrics: dict) -> str:
    print_section("FINAL MODEL COMPARISON")

    header = (
        "+----------------------+-----------+----------+\n"
        "| Model                | Accuracy  | F1 Score |\n"
        "+----------------------+-----------+----------+"
    )
    print(header)

    rows = [
        ("mBERT (Sentiment)", sentiment_metrics),
        ("mBERT (Aspect)", aspect_metrics),
    ]
    lines = [header]
    for name, metrics in rows:
        acc = metrics["accuracy"] * 100
        f1 = metrics["f1_weighted"] * 100
        row = f"| {name:<20} | {acc:>7.2f}% | {f1:>6.2f}% |"
        print(row)
        lines.append(row)

    footer = "+----------------------+-----------+----------+"
    print(footer)
    lines.append(footer)
    return "\n".join(lines)


def save_results(results: dict, comparison_table: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("eWallet ABSA — Dual-Head mBERT Training Results\n")
        f.write(f"Training Date: {timestamp}\n")
        f.write("Dataset: 1980 rows, 3 apps, 220 per sentiment per app\n")
        f.write(f"Device used: {DEVICE_NAME}\n")
        f.write(f"Training Time: {format_time(results['training_time'])}\n")
        f.write(f"Combined Accuracy (both correct): {results['combined_accuracy'] * 100:.2f}%\n\n")
        f.write(comparison_table + "\n\n")

        for name, key in [
            ("mBERT (Sentiment)", "sentiment_metrics"),
            ("mBERT (Aspect)", "aspect_metrics"),
        ]:
            metrics = results[key]
            f.write("=" * 60 + "\n")
            f.write(f"{name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Weighted F1: {metrics['f1_weighted']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(metrics["classification_report"] + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(metrics["confusion_matrix"]) + "\n\n")

    print(f"\nResults saved to {RESULTS_PATH}")


def main() -> None:
    print_section("eWallet ABSA — Dual-Head mBERT Training Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device} ({DEVICE_NAME})")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    X_text = df["cleaned_text"]
    y_sentiment = encode_sentiment(df["sentiment"])
    y_aspect = encode_aspect(df["aspect_label"])

    results = train_dual_head_mbert(X_text, y_sentiment, y_aspect)
    comparison_table = print_comparison_table(
        results["sentiment_metrics"],
        results["aspect_metrics"],
    )
    save_results(results, comparison_table)

    print_section("TRAINING COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
