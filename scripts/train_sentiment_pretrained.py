#!/usr/bin/env python3
"""
train_sentiment_pretrained.py - Fine-tune pre-trained BERT-tiny on SST-2

This script loads pre-trained weights from Hugging Face's bert-tiny model
and fine-tunes it on SST-2 sentiment classification task.

Key difference from train_tiny_sentiment.py:
  - Loads pre-trained weights instead of random initialization
  - Expected accuracy: 80-85% (vs 65-75% from scratch)

Usage:
    python train_sentiment_pretrained.py --output_dir ./sentiment_pretrained

Requirements:
    pip install torch transformers datasets tqdm

Author: FIONA Project
Date: 2025-12-23
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class PretrainedSentimentConfig:
    """Configuration matching bert-tiny architecture."""

    def __init__(
        self,
        vocab_size=30522,        # BERT vocabulary size
        max_position=512,        # BERT default (we'll use 64 for inference)
        d_model=128,             # bert-tiny hidden_size
        n_heads=2,               # bert-tiny num_attention_heads
        d_k=64,                  # d_model / n_heads
        d_ff=512,                # bert-tiny intermediate_size
        n_layers=2,              # bert-tiny num_hidden_layers
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


class PretrainedSentimentModel(nn.Module):
    """
    Sentiment classifier using pre-trained BERT-tiny weights.

    Architecture matches bert-tiny for weight compatibility:
    - Token + Position Embedding
    - N Transformer Encoder layers (Pre-LN)
    - [CLS] token pooling
    - Classification head
    """

    def __init__(self, config: PretrainedSentimentConfig):
        super().__init__()
        self.config = config

        # Embeddings (BERT-style)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_position, config.d_model)
        self.token_type_embedding = nn.Embedding(2, config.d_model)  # BERT has token types
        self.embedding_layernorm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # BERT uses Post-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Pooler (BERT-style)
        self.pooler = nn.Linear(config.d_model, config.d_model)
        self.pooler_activation = nn.Tanh()

        # Classifier head
        self.classifier_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Token type IDs (default to 0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Embeddings
        embeddings = (
            self.token_embedding(input_ids) +
            self.position_embedding(position_ids) +
            self.token_type_embedding(token_type_ids)
        )
        embeddings = self.embedding_layernorm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        # Create attention mask for transformer
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Encode
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Pool [CLS] token
        cls_hidden = hidden_states[:, 0, :]
        pooled = self.pooler_activation(self.pooler(cls_hidden))

        # Classify
        pooled = self.classifier_dropout(pooled)
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def load_pretrained_bert(self, bert_model):
        """
        Load weights from a Hugging Face BERT model.

        Args:
            bert_model: A BertModel instance from transformers
        """
        print("Loading pre-trained BERT weights...")

        # Embeddings
        self.token_embedding.weight.data.copy_(
            bert_model.embeddings.word_embeddings.weight.data
        )
        self.position_embedding.weight.data.copy_(
            bert_model.embeddings.position_embeddings.weight.data
        )
        self.token_type_embedding.weight.data.copy_(
            bert_model.embeddings.token_type_embeddings.weight.data
        )
        self.embedding_layernorm.weight.data.copy_(
            bert_model.embeddings.LayerNorm.weight.data
        )
        self.embedding_layernorm.bias.data.copy_(
            bert_model.embeddings.LayerNorm.bias.data
        )

        # Transformer layers
        for layer_idx in range(self.config.n_layers):
            bert_layer = bert_model.encoder.layer[layer_idx]
            our_layer = self.encoder.layers[layer_idx]

            # Self-attention
            # BERT stores Q, K, V separately; PyTorch TransformerEncoderLayer uses in_proj
            q_weight = bert_layer.attention.self.query.weight.data
            k_weight = bert_layer.attention.self.key.weight.data
            v_weight = bert_layer.attention.self.value.weight.data
            q_bias = bert_layer.attention.self.query.bias.data
            k_bias = bert_layer.attention.self.key.bias.data
            v_bias = bert_layer.attention.self.value.bias.data

            # Concatenate Q, K, V into in_proj
            our_layer.self_attn.in_proj_weight.data.copy_(
                torch.cat([q_weight, k_weight, v_weight], dim=0)
            )
            our_layer.self_attn.in_proj_bias.data.copy_(
                torch.cat([q_bias, k_bias, v_bias], dim=0)
            )

            # Output projection
            our_layer.self_attn.out_proj.weight.data.copy_(
                bert_layer.attention.output.dense.weight.data
            )
            our_layer.self_attn.out_proj.bias.data.copy_(
                bert_layer.attention.output.dense.bias.data
            )

            # FFN
            our_layer.linear1.weight.data.copy_(
                bert_layer.intermediate.dense.weight.data
            )
            our_layer.linear1.bias.data.copy_(
                bert_layer.intermediate.dense.bias.data
            )
            our_layer.linear2.weight.data.copy_(
                bert_layer.output.dense.weight.data
            )
            our_layer.linear2.bias.data.copy_(
                bert_layer.output.dense.bias.data
            )

            # LayerNorms (BERT Post-LN: after attention and after FFN)
            our_layer.norm1.weight.data.copy_(
                bert_layer.attention.output.LayerNorm.weight.data
            )
            our_layer.norm1.bias.data.copy_(
                bert_layer.attention.output.LayerNorm.bias.data
            )
            our_layer.norm2.weight.data.copy_(
                bert_layer.output.LayerNorm.weight.data
            )
            our_layer.norm2.bias.data.copy_(
                bert_layer.output.LayerNorm.bias.data
            )

        # Pooler
        self.pooler.weight.data.copy_(bert_model.pooler.dense.weight.data)
        self.pooler.bias.data.copy_(bert_model.pooler.dense.bias.data)

        print("Pre-trained weights loaded successfully!")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    """Fine-tune pre-trained BERT-tiny on SST-2."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm

    print("=" * 60)
    print("Pre-trained BERT-tiny Fine-tuning for SST-2")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pre-trained model and tokenizer
    print(f"\nLoading pre-trained model: {args.pretrained_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    bert_model = AutoModel.from_pretrained(args.pretrained_model)

    # Print BERT config
    print(f"BERT config: hidden_size={bert_model.config.hidden_size}, "
          f"num_layers={bert_model.config.num_hidden_layers}, "
          f"num_heads={bert_model.config.num_attention_heads}, "
          f"intermediate_size={bert_model.config.intermediate_size}")

    # Create our model with matching config
    config = PretrainedSentimentConfig(
        vocab_size=bert_model.config.vocab_size,
        max_position=bert_model.config.max_position_embeddings,
        d_model=bert_model.config.hidden_size,
        n_heads=bert_model.config.num_attention_heads,
        d_k=bert_model.config.hidden_size // bert_model.config.num_attention_heads,
        d_ff=bert_model.config.intermediate_size,
        n_layers=bert_model.config.num_hidden_layers,
        num_labels=2,
    )
    model = PretrainedSentimentModel(config)

    # Load pre-trained weights
    model.load_pretrained_bert(bert_model)
    model = model.to(device)

    print(f"\nModel parameters: {count_parameters(model):,}")

    # Load dataset
    print("\nLoading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")

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
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

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
            outputs = model(input_ids, attention_mask, labels=labels)
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
                "pretrained_model": args.pretrained_model,
            }, output_path / "model.pt")

            print(f"Saved best model (accuracy: {accuracy:.4f})")

    print("\n" + "=" * 60)
    print(f"Training complete! Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)

    # Export info for FIONA
    print("\n" + "=" * 60)
    print("Next steps for FIONA integration:")
    print("=" * 60)
    print(f"""
1. Export weights to C headers:
   python export_sentiment_pretrained.py \\
       --model_path {args.output_dir}/model.pt \\
       --output_dir ../app/sentiment_classifier/weights

2. Rebuild sentiment_classifier:
   cd ../fiona-workload && make clean && make

3. Run on Spike:
   spike --extension=fiona pk build/sentiment_classifier.elf
""")

    return model, config


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune pre-trained BERT-tiny on SST-2"
    )

    # Paths
    parser.add_argument("--output_dir", type=str, default="./sentiment_pretrained",
                        help="Output directory for saved model")
    parser.add_argument("--pretrained_model", type=str, default="prajjwal1/bert-tiny",
                        help="Pre-trained model name on Hugging Face")

    # Training
    parser.add_argument("--max_length", type=int, default=64,
                        help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (lower for fine-tuning)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
