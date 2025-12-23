#!/usr/bin/env python3
"""
train_tiny_sentiment_small.py - Train TinySentiment with Reduced Vocabulary

This script trains a TinySentiment model with a reduced vocabulary size (5000 tokens)
to reduce memory usage during FIONA Spike simulation.

Key differences from train_tiny_sentiment.py:
- vocab_size=5000 (instead of 30522)
- Maps out-of-vocabulary tokens to [UNK] (token id 100)
- Significantly reduces embedding memory: 30522*128*4 = 15.6MB -> 5000*128*4 = 2.5MB

Usage:
    python train_tiny_sentiment_small.py --output_dir ./tiny_sentiment_small

Author: FIONA Project
Date: 2025-12-23
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import json


class TinySentimentConfig:
    """Configuration for TinySentiment model with reduced vocabulary."""

    def __init__(
        self,
        vocab_size=5000,
        max_position=64,
        d_model=128,
        n_heads=2,
        d_k=64,
        d_ff=256,
        n_layers=2,
        num_labels=2,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.max_position = max_position
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.num_labels = num_labels
        self.dropout = dropout


class TinySentimentModel(nn.Module):
    """Minimal Transformer for sentiment classification with reduced vocabulary."""

    def __init__(self, config: TinySentimentConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_position, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Pooler and classifier
        self.pooler = nn.Linear(config.d_model, config.d_model)
        self.pooler_activation = nn.Tanh()
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        embeddings = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        embeddings = self.embedding_dropout(embeddings)

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        cls_hidden = hidden_states[:, 0, :]
        pooled = self.pooler_activation(self.pooler(cls_hidden))

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def build_reduced_vocabulary(tokenizer, dataset, vocab_size=5000):
    """
    Build a reduced vocabulary from the training data.

    Returns:
        old_to_new: mapping from original token id to new token id
        new_to_old: mapping from new token id to original token id
    """
    from collections import Counter

    print(f"Building reduced vocabulary (target size: {vocab_size})...")

    # Count token frequencies in training data
    token_counts = Counter()
    for example in dataset["train"]:
        tokens = tokenizer(example["sentence"], truncation=True, max_length=64)["input_ids"]
        token_counts.update(tokens)

    # Special tokens to always include
    special_tokens = {
        tokenizer.cls_token_id,    # [CLS] = 101
        tokenizer.sep_token_id,    # [SEP] = 102
        tokenizer.pad_token_id,    # [PAD] = 0
        tokenizer.unk_token_id,    # [UNK] = 100
        tokenizer.mask_token_id,   # [MASK] = 103
    }

    # Get most common tokens
    most_common = token_counts.most_common(vocab_size - len(special_tokens))

    # Build mapping
    old_to_new = {}
    new_to_old = {}

    # Reserve index 0 for [PAD]
    new_idx = 0
    for special_id in sorted(special_tokens):
        if special_id is not None:
            old_to_new[special_id] = new_idx
            new_to_old[new_idx] = special_id
            new_idx += 1

    # Add most common tokens
    for token_id, count in most_common:
        if token_id not in old_to_new:
            old_to_new[token_id] = new_idx
            new_to_old[new_idx] = token_id
            new_idx += 1
            if new_idx >= vocab_size:
                break

    # UNK token for mapping unknown tokens
    unk_new_id = old_to_new[tokenizer.unk_token_id]

    print(f"  Original vocab size: {tokenizer.vocab_size}")
    print(f"  Reduced vocab size: {len(old_to_new)}")
    print(f"  UNK token (new id): {unk_new_id}")

    return old_to_new, new_to_old, unk_new_id


def remap_tokens(input_ids, old_to_new, unk_new_id):
    """Remap token IDs from original vocabulary to reduced vocabulary."""
    return [old_to_new.get(tid, unk_new_id) for tid in input_ids]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    """Train TinySentiment with reduced vocabulary on SST-2."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from tqdm import tqdm

    print("=" * 60)
    print("TinySentiment Training (Reduced Vocabulary)")
    print("=" * 60)
    print(f"Target vocabulary size: {args.vocab_size}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print("\nLoading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Build reduced vocabulary
    old_to_new, new_to_old, unk_new_id = build_reduced_vocabulary(
        tokenizer, dataset, args.vocab_size
    )

    # Custom dataset class
    class SST2Dataset(Dataset):
        def __init__(self, hf_dataset, tokenizer, old_to_new, unk_new_id, max_length):
            self.data = hf_dataset
            self.tokenizer = tokenizer
            self.old_to_new = old_to_new
            self.unk_new_id = unk_new_id
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            example = self.data[idx]

            # Tokenize
            encoded = self.tokenizer(
                example["sentence"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

            # Remap to reduced vocabulary
            input_ids = remap_tokens(encoded["input_ids"], self.old_to_new, self.unk_new_id)

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                "label": torch.tensor(example["label"], dtype=torch.long),
            }

    # Create datasets
    train_dataset = SST2Dataset(dataset["train"], tokenizer, old_to_new, unk_new_id, args.max_length)
    val_dataset = SST2Dataset(dataset["validation"], tokenizer, old_to_new, unk_new_id, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2)

    # Create model with reduced vocabulary
    actual_vocab_size = len(old_to_new)
    config = TinySentimentConfig(
        vocab_size=actual_vocab_size,
        max_position=args.max_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        num_labels=2,
    )
    model = TinySentimentModel(config).to(device)

    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"  Token embeddings: {actual_vocab_size} x {config.d_model} = {actual_vocab_size * config.d_model:,}")
    print(f"  Memory savings: {(30522 - actual_vocab_size) * config.d_model * 4 / 1024 / 1024:.1f} MB")
    print(f"Config: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"d_ff={config.d_ff}, n_layers={config.n_layers}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask)
                predictions = outputs["logits"].argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f} ({correct}/{total})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save model
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "accuracy": accuracy,
                "old_to_new": old_to_new,
                "new_to_old": new_to_old,
                "unk_new_id": unk_new_id,
            }, output_path / "model.pt")

            # Save vocabulary mapping as JSON for C++ export
            with open(output_path / "vocab_mapping.json", "w") as f:
                json.dump({
                    "old_to_new": {str(k): v for k, v in old_to_new.items()},
                    "new_to_old": {str(k): v for k, v in new_to_old.items()},
                    "unk_new_id": unk_new_id,
                    "vocab_size": actual_vocab_size,
                }, f, indent=2)

            print(f"Saved best model (accuracy: {accuracy:.4f})")

    print("\n" + "=" * 60)
    print(f"Training complete! Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"Vocabulary size: {actual_vocab_size}")
    print("=" * 60)

    return model, config, old_to_new, new_to_old


def main():
    parser = argparse.ArgumentParser(description="Train TinySentiment with reduced vocabulary")

    parser.add_argument("--output_dir", type=str, default="./tiny_sentiment_small",
                        help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=5000,
                        help="Target vocabulary size")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
