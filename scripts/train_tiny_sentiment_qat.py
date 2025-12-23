#!/usr/bin/env python3
"""
train_tiny_sentiment_qat.py - QAT Training for TinySentiment

Quantization-Aware Training (QAT) implementation that simulates FIONA's
DAC/ADC quantization during training. This allows the model to learn
representations that are robust to low-bit quantization.

Key Features:
- Fake quantization matching FIONA's photonic_models.py implementation
- Configurable bit-width (2-16 bits)
- Straight-Through Estimator (STE) for gradient flow
- Same model architecture as train_tiny_sentiment.py

Usage:
    # Train with 8-bit quantization (default)
    python train_tiny_sentiment_qat.py --output_dir ./tiny_sentiment_qat

    # Train with 4-bit quantization
    python train_tiny_sentiment_qat.py --quant_bits 4 --output_dir ./tiny_sentiment_qat_4bit

    # Compare with baseline (no quantization during training)
    python train_tiny_sentiment.py --output_dir ./tiny_sentiment_baseline

Requirements:
    pip install torch transformers datasets

Author: FIONA Project
Date: 2025-12-23
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ============================================================
# Fake Quantization (Matching FIONA's Implementation)
# ============================================================

class FakeQuantize(torch.autograd.Function):
    """
    Fake quantization with Straight-Through Estimator (STE).

    Forward: Apply quantization (non-differentiable)
    Backward: Pass gradients through unchanged (STE)

    This matches FIONA's apply_quantization() in photonic_models.py:
    - Normalize by max absolute value
    - Scale to 2^bits levels
    - Round to nearest integer
    - Scale back
    """

    @staticmethod
    def forward(ctx, x, bits):
        if bits >= 16:
            return x

        # Handle zero tensor
        max_val = torch.max(torch.abs(x))
        if max_val == 0:
            return x

        # Quantize (matching FIONA's implementation)
        levels = 2 ** bits
        normalized = x / max_val
        quantized = torch.round(normalized * (levels / 2)) / (levels / 2)

        return quantized * max_val

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass gradients unchanged
        return grad_output, None


def fake_quantize(x, bits):
    """Apply fake quantization with STE."""
    return FakeQuantize.apply(x, bits)


# ============================================================
# Quantization-Aware Linear Layer
# ============================================================

class QATLinear(nn.Module):
    """
    Linear layer with fake quantization for QAT.

    Simulates FIONA's photonic MVM pipeline:
    - DAC quantization on input
    - Matrix-vector multiplication
    - ADC quantization on output

    During training, fake quantization is applied to both input and output.
    During inference, the layer behaves like a normal linear layer
    (actual quantization is applied by FIONA hardware/simulator).
    """

    def __init__(self, in_features, out_features, bias=True, quant_bits=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.quant_bits = quant_bits
        self.training_with_quant = True

    def forward(self, x):
        if self.training and self.training_with_quant:
            # DAC quantization (input)
            x_quant = fake_quantize(x, self.quant_bits)

            # Linear transformation
            output = self.linear(x_quant)

            # ADC quantization (output)
            output_quant = fake_quantize(output, self.quant_bits)

            return output_quant
        else:
            # Normal linear (for inference or non-QAT mode)
            return self.linear(x)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


# ============================================================
# QAT Transformer Components
# ============================================================

class QATMultiHeadAttention(nn.Module):
    """Multi-head attention with QAT linear layers."""

    def __init__(self, d_model, n_heads, dropout=0.1, quant_bits=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.quant_bits = quant_bits

        # Q, K, V projections (quantized)
        self.W_q = QATLinear(d_model, d_model, quant_bits=quant_bits)
        self.W_k = QATLinear(d_model, d_model, quant_bits=quant_bits)
        self.W_v = QATLinear(d_model, d_model, quant_bits=quant_bits)

        # Output projection (quantized)
        self.W_o = QATLinear(d_model, d_model, quant_bits=quant_bits)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Q, K, V projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores (electronic - no quantization)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Softmax (electronic - no quantization)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output (electronic - no quantization)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection (quantized)
        output = self.W_o(attn_output)

        return output


class QATFeedForward(nn.Module):
    """Feed-forward network with QAT linear layers."""

    def __init__(self, d_model, d_ff, dropout=0.1, quant_bits=8):
        super().__init__()
        self.linear1 = QATLinear(d_model, d_ff, quant_bits=quant_bits)
        self.linear2 = QATLinear(d_ff, d_model, quant_bits=quant_bits)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # First linear (quantized)
        hidden = self.linear1(x)

        # GELU activation (electronic - no quantization)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        # Second linear (quantized)
        output = self.linear2(hidden)

        return output


class QATTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with QAT."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, quant_bits=8):
        super().__init__()
        self.self_attn = QATMultiHeadAttention(d_model, n_heads, dropout, quant_bits)
        self.ffn = QATFeedForward(d_model, d_ff, dropout, quant_bits)

        # Layer norms (electronic - no quantization)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN architecture (matches FIONA implementation)
        # Self-attention with residual
        normed = self.norm1(x)
        attn_output = self.self_attn(normed, mask)
        x = x + self.dropout1(attn_output)

        # FFN with residual
        normed = self.norm2(x)
        ffn_output = self.ffn(normed)
        x = x + self.dropout2(ffn_output)

        return x


# ============================================================
# QAT Model
# ============================================================

class TinySentimentConfig:
    """Configuration for TinySentiment model."""

    def __init__(
        self,
        vocab_size=30522,
        max_position=64,
        d_model=128,
        n_heads=2,
        d_k=64,
        d_ff=256,
        n_layers=2,
        num_labels=2,
        dropout=0.1,
        quant_bits=8,
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
        self.quant_bits = quant_bits


class TinySentimentQATModel(nn.Module):
    """
    TinySentiment with Quantization-Aware Training.

    Architecture matches train_tiny_sentiment.py but with:
    - QATLinear layers instead of nn.Linear for all MVM operations
    - Fake quantization simulating DAC/ADC during training

    Quantized operations (photonic):
    - Q, K, V projections
    - Attention output projection
    - FFN linear layers
    - Pooler
    - Classifier

    Non-quantized operations (electronic):
    - Embeddings
    - Layer normalization
    - Softmax
    - GELU activation
    - Residual additions
    """

    def __init__(self, config: TinySentimentConfig):
        super().__init__()
        self.config = config

        # Embeddings (electronic - no quantization)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_position, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Transformer encoder layers (with QAT)
        self.encoder_layers = nn.ModuleList([
            QATTransformerEncoderLayer(
                config.d_model, config.n_heads, config.d_ff,
                config.dropout, config.quant_bits
            )
            for _ in range(config.n_layers)
        ])

        # Pooler and classifier (quantized)
        self.pooler = QATLinear(config.d_model, config.d_model, quant_bits=config.quant_bits)
        self.pooler_activation = nn.Tanh()
        self.classifier = QATLinear(config.d_model, config.num_labels, quant_bits=config.quant_bits)

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
            elif isinstance(module, QATLinear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings (no quantization)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        embeddings = self.embedding_dropout(embeddings)

        # Create attention mask
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Encoder layers
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, key_padding_mask)

        # Pool [CLS] token
        cls_hidden = hidden_states[:, 0, :]
        pooled = self.pooler(cls_hidden)
        pooled = self.pooler_activation(pooled)

        # Classify
        logits = self.classifier(pooled)

        # Compute loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def set_quant_training(self, enabled=True):
        """Enable/disable quantization during training."""
        for module in self.modules():
            if isinstance(module, QATLinear):
                module.training_with_quant = enabled


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_qat_layers(model):
    """Count QAT linear layers."""
    return sum(1 for m in model.modules() if isinstance(m, QATLinear))


# ============================================================
# Training
# ============================================================

def train(args):
    """Train TinySentiment with QAT on SST-2."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm

    print("=" * 60)
    print("TinySentiment QAT Training")
    print("=" * 60)
    print(f"Quantization bits: {args.quant_bits}")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print("\nLoading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")

    # Load tokenizer
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
        vocab_size=tokenizer.vocab_size,
        max_position=args.max_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        num_labels=2,
        quant_bits=args.quant_bits,
    )
    model = TinySentimentQATModel(config).to(device)

    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"QAT linear layers: {count_qat_layers(model)}")
    print(f"Config: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"d_ff={config.d_ff}, n_layers={config.n_layers}, quant_bits={config.quant_bits}")

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
                "quant_bits": args.quant_bits,
                "qat_enabled": True,
            }, output_path / "model.pt")

            print(f"Saved best model (accuracy: {accuracy:.4f})")

    print("\n" + "=" * 60)
    print(f"QAT Training complete!")
    print(f"Quantization bits: {args.quant_bits}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)

    return model, config


def evaluate_quantization_robustness(args):
    """Evaluate model robustness to different quantization levels."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    print("=" * 60)
    print("Quantization Robustness Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = Path(args.output_dir) / "model.pt"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    config = TinySentimentConfig(**checkpoint["config"])
    model = TinySentimentQATModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded model trained with {checkpoint.get('quant_bits', 'unknown')} bits")

    # Load validation data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("glue", "sst2")

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=config.max_position,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=64)

    # Evaluate at different bit widths
    print("\nEvaluating at different quantization levels...")
    print("-" * 40)

    bit_widths = [2, 3, 4, 5, 6, 7, 8, 16]
    results = []

    for bits in bit_widths:
        # Update quantization bits
        for module in model.modules():
            if isinstance(module, QATLinear):
                module.quant_bits = bits

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask)
                predictions = outputs["logits"].argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        results.append((bits, accuracy))
        print(f"  {bits:2d} bits: {accuracy:.4f} ({correct}/{total})")

    print("-" * 40)
    print("\nSummary:")
    for bits, acc in results:
        bar = "*" * int(acc * 50)
        print(f"  {bits:2d} bits: {acc:.4f} |{bar}")

    return results


def demo(args):
    """Demo mode: create a QAT model without training."""
    print("=" * 60)
    print("TinySentiment QAT Demo Mode")
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
        quant_bits=args.quant_bits,
    )
    model = TinySentimentQATModel(config)

    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"QAT linear layers: {count_qat_layers(model)}")
    print(f"Quantization bits: {args.quant_bits}")

    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "accuracy": 0.0,
        "quant_bits": args.quant_bits,
        "qat_enabled": True,
    }, output_path / "model.pt")

    print(f"\nDemo model saved to: {args.output_dir}")
    print("Note: This model has random weights. Remove --demo for actual training.")


def main():
    parser = argparse.ArgumentParser(description="Train TinySentiment with QAT")

    # Paths
    parser.add_argument("--output_dir", type=str, default="./tiny_sentiment_qat",
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

    # Quantization
    parser.add_argument("--quant_bits", type=int, default=8,
                        help="Quantization bit width (2-16)")

    # Training
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (more for QAT)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate (lower for QAT)")

    # Mode
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: create model without training")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate quantization robustness of trained model")

    args = parser.parse_args()

    if args.demo:
        demo(args)
    elif args.evaluate:
        evaluate_quantization_robustness(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
