#!/usr/bin/env python3
"""
train_tiny_sentiment.py - Train TinySentiment for SST-2 (Sentiment Classification)

This script trains a minimal Transformer model (~1.2M parameters) optimized for
FIONA-workload execution. The model is designed to fit within FIONA's constraints
while achieving reasonable accuracy on sentiment classification.

Usage:
    python train_tiny_sentiment.py --output_dir ./tiny_sentiment

Requirements:
    pip install torch transformers datasets

Author: FIONA Project
Date: 2025-12-23
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path


class TinySentimentConfig:
    """Configuration for TinySentiment model optimized for FIONA."""

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
    """
    Minimal Transformer for sentiment classification.

    Architecture:
    - Token + Position Embedding
    - N Transformer Encoder layers
    - [CLS] token pooling
    - Classification head

    Total parameters: ~1.2M
    """

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
            norm_first=True,  # Pre-LN (matches FIONA implementation)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Pooler and classifier
        self.pooler = nn.Linear(config.d_model, config.d_model)
        self.pooler_activation = nn.Tanh()
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values."""
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

        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        embeddings = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        embeddings = self.embedding_dropout(embeddings)

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert padding mask: 1 for padding positions
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Encode
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Pool [CLS] token (first position)
        cls_hidden = hidden_states[:, 0, :]
        pooled = self.pooler_activation(self.pooler(cls_hidden))

        # Classify
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    """Train TinySentiment on SST-2."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm

    print("=" * 60)
    print("TinySentiment Training")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print("\nLoading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")

    # Load tokenizer (use BERT's vocab, but we'll limit it)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize function
    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    # Tokenize dataset
    print("Tokenizing...")
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # DataLoaders
    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=args.batch_size * 2,
    )

    # Create model
    config = TinySentimentConfig(
        vocab_size=tokenizer.vocab_size,  # Use full vocab for training
        max_position=args.max_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        num_labels=2,
    )
    model = TinySentimentModel(config).to(device)

    print(f"\nModel parameters: {count_parameters(model):,}")
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

        # Train
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

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save model
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "accuracy": accuracy,
            }, output_path / "model.pt")

            print(f"Saved best model (accuracy: {accuracy:.4f})")

    print("\n" + "=" * 60)
    print(f"Training complete! Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)

    return model, config


def demo(args):
    """Demo mode: create a small model without training."""
    print("=" * 60)
    print("TinySentiment Demo Mode")
    print("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    config = TinySentimentConfig(
        vocab_size=tokenizer.vocab_size,
        max_position=args.max_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
    )
    model = TinySentimentModel(config)

    print(f"\nModel parameters: {count_parameters(model):,}")

    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "accuracy": 0.0,  # Random weights
    }, output_path / "model.pt")

    print(f"\nDemo model saved to: {args.output_dir}")
    print("Note: This model has random weights. Use --train for actual training.")


def main():
    parser = argparse.ArgumentParser(description="Train TinySentiment for SST-2")

    # Paths
    parser.add_argument("--output_dir", type=str, default="./tiny_sentiment",
                        help="Output directory for saved model")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=128,
                        help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=2,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=256,
                        help="FFN hidden dimension")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--max_length", type=int, default=64,
                        help="Maximum sequence length")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate")

    # Mode
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: create model without training")

    args = parser.parse_args()

    if args.demo:
        demo(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
