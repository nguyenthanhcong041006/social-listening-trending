"""
XLM-RoBERTa Fine-Tuning CLI with Automatic Checkpoint Versioning (v1, v2, v3,...)
and Comprehensive Evaluation Metrics (Loss, Accuracy, Macro F1, Confusion Matrix).

Features:
- Train / Validation Split (85% / 15%) to prevent overfitting.
- Evaluates per epoch: Train Loss, Validation Loss, Validation Accuracy, Macro F1.
- Final evaluation: Full classification report & confusion matrix saved to version_info.json.
- Standalone evaluation mode: `python scripts/finetune_xlm_roberta.py --evaluate`
- Overview of all versions and their scores: `python scripts/finetune_xlm_roberta.py --list-versions`
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from loguru import logger

# Add root directory to python path and anchor working directory to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.basename(os.getcwd()) == "scripts":
    os.chdir(PROJECT_ROOT)

from src.stage_03_multilingual_sentiment.versioning import (
    save_fine_tuned_checkpoint,
    resolve_model_path,
    list_available_versions,
    DEFAULT_EMBEDDING_BASE_DIR,
    DEFAULT_SENTIMENT_BASE_DIR,
    DEFAULT_FALLBACK_EMBEDDING,
    DEFAULT_FALLBACK_SENTIMENT,
)
from src.utils.file_io import resolve_path, load_dataframe


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in item.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_training_data(data_path: str):
    """Loads dataframe from parquet or csv gracefully."""
    resolved = resolve_path(data_path)
    if not os.path.exists(resolved):
        candidates = [
            "data/02_preprocessed/preprocessed_posts.parquet",
            "data/03_embeddings_sentiment/posts_with_sentiment.parquet",
            "data/01_raw/social_listening_30days.csv",
            "data/01_raw/social_listening_dataset.csv"
        ]
        for c in candidates:
            rc = resolve_path(c)
            if os.path.exists(rc):
                resolved = rc
                break
    
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Cannot find training data file at {data_path} or fallback paths.")
        
    logger.info(f"Loading dataset from: {resolved}")
    df = load_dataframe(resolved)
    return df


def evaluate_sentiment_model(model, tokenizer, texts, labels, batch_size=32, max_length=128, device="cuda"):
    """Evaluates sentiment model on a given dataset and returns comprehensive metrics."""
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_targets = []
    total_val_loss = 0.0
    step_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            total_val_loss += outputs.loss.item()
            step_count += 1

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    avg_val_loss = float(total_val_loss / max(1, step_count))

    target_names = ["negative", "neutral", "positive"]
    report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return {
        "val_loss": round(avg_val_loss, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm
    }


def finetune_sentiment(args):
    """Fine-tunes XLM-RoBERTa for 3-class sentiment analysis with validation split and auto-versioning."""
    logger.info("=" * 70)
    logger.info("STARTING XLM-RoBERTa SENTIMENT FINE-TUNING")
    logger.info("=" * 70)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target compute device: {device}")

    # Resolve base model to train from
    base_model = args.base_model
    if not base_model:
        base_model = resolve_model_path(
            DEFAULT_SENTIMENT_BASE_DIR,
            default_fallback=DEFAULT_FALLBACK_SENTIMENT,
            task="Sentiment Classifier"
        )
    logger.info(f"Base model checkpoint: '{base_model}'")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3).to(device)

    # Load data
    df = load_training_data(args.data)
    text_col = "cleaned_text" if "cleaned_text" in df.columns else ("text" if "text" in df.columns else df.columns[0])
    logger.info(f"Using text column: '{text_col}' (Total samples: {len(df)})")

    # Map labels: 0: negative, 1: neutral, 2: positive
    if "sentiment_label" in df.columns:
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        labels = df["sentiment_label"].astype(str).str.lower().map(label_map).fillna(1).astype(int).tolist()
    elif "sentiment_score" in df.columns:
        scores = df["sentiment_score"].fillna(0.0).values
        labels = np.where(scores > 0.05, 2, np.where(scores < -0.05, 0, 1)).tolist()
    else:
        logger.warning("No explicit sentiment labels found. Using neutral (1) placeholder for domain adaptation.")
        labels = [1] * len(df)

    # Train / Validation Split (85% train, 15% validation)
    texts_all = df[text_col].tolist()
    if len(texts_all) > 20:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts_all, labels, test_size=0.15, random_state=42, stratify=labels if len(set(labels)) > 1 else None
        )
    else:
        train_texts, val_texts, train_labels, val_labels = texts_all, texts_all, labels, labels

    logger.info(f"Dataset split -> Train samples: {len(train_texts)} | Validation samples: {len(val_texts)}")

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        step_count = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step_count += 1

        avg_train_loss = total_loss / max(1, step_count)

        # Evaluate on validation set
        val_metrics = evaluate_sentiment_model(
            model, tokenizer, val_texts, val_labels, batch_size=args.batch_size, max_length=args.max_length, device=device
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_metrics["val_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"]
        })

        logger.info(
            f"Epoch [{epoch}/{args.epochs}] -> "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.1%} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.3f}"
        )

    # Final evaluation summary
    final_metrics = evaluate_sentiment_model(
        model, tokenizer, val_texts, val_labels, batch_size=args.batch_size, max_length=args.max_length, device=device
    )

    print("\n" + "=" * 70)
    print("FINAL VALIDATION EVALUATION RESULTS")
    print("=" * 70)
    print(f"- Accuracy:   {final_metrics['accuracy']:.1%}")
    print(f"- Macro F1:   {final_metrics['macro_f1']:.3f}")
    print(f"- Val Loss:   {final_metrics['val_loss']:.4f}")
    print("- Confusion Matrix:")
    print("      [Pred: Neg, Neu, Pos]")
    for idx, row in enumerate(final_metrics["confusion_matrix"]):
        class_name = ["Neg", "Neu", "Pos"][idx]
        print(f"  {class_name}: {row}")
    print("=" * 70 + "\n")

    metadata = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": final_metrics["val_loss"],
        "accuracy": final_metrics["accuracy"],
        "macro_f1": final_metrics["macro_f1"],
        "weighted_f1": final_metrics["weighted_f1"],
        "confusion_matrix": final_metrics["confusion_matrix"],
        "history": history,
        "dataset_samples": len(df),
        "source_data": args.data,
        "base_model": base_model
    }

    saved_path = save_fine_tuned_checkpoint(
        model=model,
        tokenizer=tokenizer,
        task="sentiment",
        base_dir=DEFAULT_SENTIMENT_BASE_DIR,
        metadata=metadata
    )
    logger.info(f"Successfully saved fine-tuned sentiment version to: '{saved_path}'")
    return saved_path


def finetune_embedding(args):
    """Fine-tunes XLM-RoBERTa for domain contextual embeddings and auto-versions the checkpoint."""
    logger.info("=" * 70)
    logger.info("STARTING XLM-RoBERTa EMBEDDING FINE-TUNING")
    logger.info("=" * 70)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target compute device: {device}")

    base_model = args.base_model
    if not base_model:
        base_model = resolve_model_path(
            DEFAULT_EMBEDDING_BASE_DIR,
            default_fallback=DEFAULT_FALLBACK_EMBEDDING,
            task="BERTopic Embeddings"
        )
    logger.info(f"Base model checkpoint: '{base_model}'")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModel.from_pretrained(base_model).to(device)

    df = load_training_data(args.data)
    text_col = "cleaned_text" if "cleaned_text" in df.columns else ("text" if "text" in df.columns else df.columns[0])
    logger.info(f"Using text column: '{text_col}' (Total samples: {len(df)})")

    dataset = TextClassificationDataset(df[text_col].tolist(), None, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps
    )

    model.train()
    history = []
    cos_sim = torch.nn.CosineSimilarity(dim=-1)
    cross_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        step_count = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out1 = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            out2 = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

            temp = 0.05
            sim_matrix = cos_sim(out1.unsqueeze(1), out2.unsqueeze(0)) / temp
            labels = torch.arange(len(out1), device=device)
            loss = cross_entropy(sim_matrix, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step_count += 1

        avg_loss = total_loss / max(1, step_count)
        history.append({"epoch": epoch, "loss": round(avg_loss, 4)})
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Contrastive Loss: {avg_loss:.4f}")

    metadata = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_loss": round(history[-1]["loss"], 4) if history else None,
        "history": history,
        "dataset_samples": len(df),
        "source_data": args.data,
        "base_model": base_model,
        "objective": "SimCSE contrastive domain adaptation"
    }

    saved_path = save_fine_tuned_checkpoint(
        model=model,
        tokenizer=tokenizer,
        task="embedding",
        base_dir=DEFAULT_EMBEDDING_BASE_DIR,
        metadata=metadata
    )
    logger.info(f"Successfully fine-tuned and saved embedding checkpoint to: '{saved_path}'")
    return saved_path


def run_standalone_evaluation(args):
    """Evaluates the latest version or a specific checkpoint on the dataset and prints all metrics."""
    logger.info("=" * 70)
    logger.info("EVALUATION MODE: Evaluating Sentiment Model")
    logger.info("=" * 70)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.base_model or resolve_model_path(
        DEFAULT_SENTIMENT_BASE_DIR,
        default_fallback=DEFAULT_FALLBACK_SENTIMENT,
        task="Sentiment Classifier"
    )
    logger.info(f"Evaluating Model: '{model_path}' on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    df = load_training_data(args.data)
    text_col = "cleaned_text" if "cleaned_text" in df.columns else ("text" if "text" in df.columns else df.columns[0])

    if "sentiment_label" in df.columns:
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        labels = df["sentiment_label"].astype(str).str.lower().map(label_map).fillna(1).astype(int).tolist()
    elif "sentiment_score" in df.columns:
        scores = df["sentiment_score"].fillna(0.0).values
        labels = np.where(scores > 0.05, 2, np.where(scores < -0.05, 0, 1)).tolist()
    else:
        labels = [1] * len(df)

    metrics = evaluate_sentiment_model(
        model, tokenizer, df[text_col].tolist(), labels, batch_size=args.batch_size, max_length=args.max_length, device=device
    )

    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT FOR: {model_path}")
    print("=" * 70)
    print(f"- Overall Accuracy: {metrics['accuracy']:.1%}")
    print(f"- Macro F1 Score:   {metrics['macro_f1']:.3f}")
    print(f"- Weighted F1:      {metrics['weighted_f1']:.3f}")
    print(f"- Evaluation Loss:  {metrics['val_loss']:.4f}")
    print("\n- Class-by-Class Breakdown:")
    for cls in ["negative", "neutral", "positive"]:
        cls_data = metrics["classification_report"].get(cls, {})
        print(f"  * {cls.capitalize():8s} -> Precision: {cls_data.get('precision', 0):.1%} | Recall: {cls_data.get('recall', 0):.1%} | F1: {cls_data.get('f1-score', 0):.3f}")

    print("\n- Confusion Matrix [Rows: Actual, Columns: Predicted]:")
    print("           Negative  Neutral  Positive")
    for idx, row in enumerate(metrics["confusion_matrix"]):
        class_name = ["Negative", "Neutral ", "Positive"][idx]
        print(f"  {class_name}    {row[0]:6d}   {row[1]:6d}   {row[2]:6d}")
    print("=" * 70 + "\n")


def show_versions():
    """Prints a clear tabular view of all existing versions and their performance scores."""
    print("=" * 70)
    print("SAVED XLM-RoBERTa MODEL VERSIONS OVERVIEW")
    print("=" * 70)

    for task_name, base_dir in [("Embedding", DEFAULT_EMBEDDING_BASE_DIR), ("Sentiment", DEFAULT_SENTIMENT_BASE_DIR)]:
        print(f"\n[Task: {task_name}] -> Directory: {base_dir}")
        versions = list_available_versions(base_dir)
        if not versions:
            print("  (No fine-tuned versions found yet. Currently using default Hugging Face base model.)")
        else:
            for v in versions:
                v_name = v.get("version", "unknown")
                meta = v.get("metadata", {})
                saved_at = v.get("saved_at", "N/A")[:16].replace("T", " ")
                loss = meta.get("final_val_loss") or meta.get("final_loss") or "N/A"
                acc = f"{meta.get('accuracy', 0):.1%}" if meta.get("accuracy") is not None else "N/A"
                f1 = f"{meta.get('macro_f1', 0):.3f}" if meta.get("macro_f1") is not None else "N/A"
                epochs = meta.get("epochs", "N/A")
                print(
                    f"  * {v_name:4s} | Path: {v['path']:36s} | Loss: {str(loss):6s} | Acc: {acc:6s} | F1: {f1:6s} | Epochs: {str(epochs):2s} | Date: {saved_at}"
                )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="XLM-RoBERTa Auto-Versioning Fine-Tuning CLI")
    parser.add_argument("--task", type=str, choices=["sentiment", "embedding", "both"], default="sentiment",
                        help="Fine-tuning target task: 'sentiment', 'embedding', or 'both'")
    parser.add_argument("--data", type=str, default="data/02_preprocessed/preprocessed_posts.parquet",
                        help="Path to training data (parquet or csv)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model checkpoint or Hugging Face ID (default: auto-resolved)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max token sequence length")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--list-versions", action="store_true", help="Display all saved versions and exit")
    parser.add_argument("--evaluate", action="store_true", help="Run standalone evaluation on dataset without training")
    args = parser.parse_args()

    if args.list_versions:
        show_versions()
        return

    if args.evaluate:
        run_standalone_evaluation(args)
        return

    if args.task in ("sentiment", "both"):
        finetune_sentiment(args)
    if args.task in ("embedding", "both"):
        finetune_embedding(args)


if __name__ == "__main__":
    main()
