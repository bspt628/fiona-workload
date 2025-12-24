#!/usr/bin/env python3
"""
verify_pretrained_accuracy.py - Verify pretrained model accuracy on SST-2

This script loads the saved model and runs inference on SST-2 validation set
to confirm the expected ~80% accuracy.
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizer
from tqdm import tqdm


class PretrainedSentimentConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 30522)
        self.max_position = kwargs.get('max_position', 512)
        self.d_model = kwargs.get('d_model', 128)
        self.n_heads = kwargs.get('n_heads', 2)
        self.d_k = kwargs.get('d_k', 64)
        self.d_ff = kwargs.get('d_ff', 512)
        self.n_layers = kwargs.get('n_layers', 2)
        self.num_labels = kwargs.get('num_labels', 2)
        self.dropout = kwargs.get('dropout', 0.1)


class PretrainedSentimentModel(nn.Module):
    def __init__(self, config: PretrainedSentimentConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_position, config.d_model)
        self.token_type_embedding = nn.Embedding(2, config.d_model)
        self.embedding_layernorm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Transformer encoder (Post-LN, norm_first=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # Post-LN (BERT style)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Pooler
        self.pooler = nn.Linear(config.d_model, config.d_model)
        self.pooler_activation = nn.Tanh()

        # Classifier
        self.classifier_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
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

        # Attention mask
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Encode
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Pool [CLS]
        cls_hidden = hidden_states[:, 0, :]
        pooled = self.pooler_activation(self.pooler(cls_hidden))

        # Classify
        pooled = self.classifier_dropout(pooled)
        logits = self.classifier(pooled)

        return logits


def main():
    print("=" * 60)
    print("Verifying Pretrained Model Accuracy")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load('sentiment_pretrained/model.pt', map_location='cpu', weights_only=False)

    config_dict = checkpoint['config']
    config = PretrainedSentimentConfig(**config_dict)

    model = PretrainedSentimentModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Saved accuracy: {checkpoint['accuracy']:.4f}")
    print(f"  Config: d_model={config.d_model}, d_ff={config.d_ff}, vocab={config.vocab_size}")

    # Load tokenizer and dataset
    print("\nLoading tokenizer and dataset...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("glue", "sst2", split="validation")
    print(f"  Samples: {len(dataset)}")

    # Run inference
    print("\nRunning inference...")
    correct = 0
    total = 0

    with torch.no_grad():
        for example in tqdm(dataset):
            text = example["sentence"]
            label = example["label"]

            # Tokenize
            encoding = tokenizer(
                text,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # Forward
            logits = model(input_ids, attention_mask)
            pred = logits.argmax(dim=-1).item()

            if pred == label:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"\n{'=' * 60}")
    print(f"RESULT: {accuracy:.4f} ({correct}/{total})")
    print(f"{'=' * 60}")

    if accuracy >= 0.79:
        print("\nPython inference confirms ~80% accuracy.")
        print("If C implementation shows lower accuracy, there's still a bug in C code.")
    else:
        print("\nUnexpected: Python accuracy is also low.")
        print("The model may not have been trained correctly.")


if __name__ == "__main__":
    main()
