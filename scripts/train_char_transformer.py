#!/usr/bin/env python3
"""
Character-level Transformer Training Script

Trains a small Transformer model for character-level text generation,
then exports weights to C header files for use in fiona-workload.

Usage:
    python train_char_transformer.py --data <text_file> --epochs 100

Architecture must match text_gen_transformer/main.cc:
    VOCAB_SIZE = 128 (ASCII)
    SEQ_LEN = 32
    D_MODEL = 64
    D_K = 32
    D_FF = 128
    N_LAYERS = 2
"""

import argparse
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Model Configuration (must match C++ code)
# ============================================================

VOCAB_SIZE = 128   # ASCII characters
SEQ_LEN = 32       # Context window
D_MODEL = 64       # Embedding dimension
D_K = 32           # Attention key/query dimension
D_FF = 128         # FFN hidden dimension
N_LAYERS = 2       # Number of Transformer blocks


# ============================================================
# Dataset
# ============================================================

class CharDataset(Dataset):
    """Character-level text dataset."""

    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len
        # Filter to ASCII only
        self.chars = [c for c in text if ord(c) < VOCAB_SIZE]
        self.data = [ord(c) for c in self.chars]

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ============================================================
# Model Definition
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class SimpleAttention(nn.Module):
    """Single-head attention matching C++ implementation."""

    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_k, bias=False)
        self.Wo = nn.Linear(d_k, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        Q = self.Wq(x)  # (batch, seq_len, d_k)
        K = self.Wk(x)
        V = self.Wv(x)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return self.Wo(out)


class TransformerBlock(nn.Module):
    """Transformer block matching C++ implementation."""

    def __init__(self, d_model, d_k, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SimpleAttention(d_model, d_k)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    """Character-level Transformer matching C++ implementation."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_encoding = PositionalEncoding(D_MODEL, SEQ_LEN)
        self.layers = nn.ModuleList([
            TransformerBlock(D_MODEL, D_K, D_FF) for _ in range(N_LAYERS)
        ])
        self.output = nn.Linear(D_MODEL, VOCAB_SIZE)

        # Create causal mask
        mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN))
        self.register_buffer('mask', mask)

    def forward(self, x):
        # x: (batch, seq_len) of token indices
        seq_len = x.size(1)

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Causal mask for current sequence length
        mask = self.mask[:seq_len, :seq_len]

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Project to vocabulary
        logits = self.output(x)
        return logits


# ============================================================
# Training
# ============================================================

def train(model, dataloader, epochs, lr=0.001, device='cpu'):
    """Train the model."""
    import sys
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)

            # Reshape for loss: (batch * seq_len, vocab_size)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            sys.stdout.flush()

    return model


def generate(model, prompt, max_len=50, temperature=1.0, device='cpu'):
    """Generate text from prompt."""
    model.eval()

    # Encode prompt
    indices = [ord(c) for c in prompt if ord(c) < VOCAB_SIZE]

    with torch.no_grad():
        for _ in range(max_len):
            # Prepare input (last SEQ_LEN tokens)
            if len(indices) > SEQ_LEN:
                input_indices = indices[-SEQ_LEN:]
            else:
                input_indices = [0] * (SEQ_LEN - len(indices)) + indices

            x = torch.tensor([input_indices], dtype=torch.long, device=device)
            logits = model(x)

            # Get last token prediction
            logits = logits[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()

            indices.append(next_token)

    return ''.join(chr(i) if 32 <= i < 127 else '?' for i in indices)


# ============================================================
# Weight Export
# ============================================================

def export_weights_to_header(model, output_dir):
    """Export model weights to C header files."""
    os.makedirs(output_dir, exist_ok=True)

    def array_to_c(name, arr, file):
        """Write numpy array as C static array."""
        arr = arr.flatten().astype(np.float32)
        file.write(f"static const float {name}[{len(arr)}] = {{\n")
        for i, val in enumerate(arr):
            if i % 8 == 0:
                file.write("    ")
            file.write(f"{val:.8f}f")
            if i < len(arr) - 1:
                file.write(", ")
            if (i + 1) % 8 == 0:
                file.write("\n")
        if len(arr) % 8 != 0:
            file.write("\n")
        file.write("};\n\n")

    # Export embedding weights
    with open(os.path.join(output_dir, "weights_embedding.h"), 'w') as f:
        f.write("// Auto-generated embedding weights\n")
        f.write(f"// Shape: [{VOCAB_SIZE} x {D_MODEL}]\n\n")
        f.write("#pragma once\n\n")
        arr = model.embedding.weight.detach().cpu().numpy()
        array_to_c("PRETRAINED_EMBEDDING", arr, f)

    # Export output layer weights
    with open(os.path.join(output_dir, "weights_output.h"), 'w') as f:
        f.write("// Auto-generated output layer weights\n")
        f.write(f"// Weight shape: [{VOCAB_SIZE} x {D_MODEL}]\n")
        f.write(f"// Bias shape: [{VOCAB_SIZE}]\n\n")
        f.write("#pragma once\n\n")
        arr = model.output.weight.detach().cpu().numpy()
        array_to_c("PRETRAINED_OUTPUT_WEIGHT", arr, f)
        arr = model.output.bias.detach().cpu().numpy()
        array_to_c("PRETRAINED_OUTPUT_BIAS", arr, f)

    # Export transformer layer weights
    for l, layer in enumerate(model.layers):
        with open(os.path.join(output_dir, f"weights_layer{l}.h"), 'w') as f:
            f.write(f"// Auto-generated transformer layer {l} weights\n\n")
            f.write("#pragma once\n\n")

            # LayerNorm 1
            arr = layer.ln1.weight.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_LN1_GAMMA", arr, f)
            arr = layer.ln1.bias.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_LN1_BETA", arr, f)

            # Attention weights
            arr = layer.attn.Wq.weight.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_WQ", arr, f)
            arr = layer.attn.Wk.weight.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_WK", arr, f)
            arr = layer.attn.Wv.weight.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_WV", arr, f)
            arr = layer.attn.Wo.weight.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_WO", arr, f)

            # LayerNorm 2
            arr = layer.ln2.weight.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_LN2_GAMMA", arr, f)
            arr = layer.ln2.bias.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_LN2_BETA", arr, f)

            # FFN weights
            arr = layer.ffn[0].weight.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_FFN_W1", arr, f)
            arr = layer.ffn[0].bias.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_FFN_B1", arr, f)
            arr = layer.ffn[2].weight.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_FFN_W2", arr, f)
            arr = layer.ffn[2].bias.detach().cpu().numpy()
            array_to_c(f"PRETRAINED_L{l}_FFN_B2", arr, f)

    # Create main include header
    with open(os.path.join(output_dir, "pretrained_weights.h"), 'w') as f:
        f.write("// Auto-generated pretrained weights\n")
        f.write("// Include this file to use pretrained weights\n\n")
        f.write("#pragma once\n\n")
        f.write('#include "weights_embedding.h"\n')
        f.write('#include "weights_output.h"\n')
        for l in range(N_LAYERS):
            f.write(f'#include "weights_layer{l}.h"\n')

    print(f"Weights exported to {output_dir}/")
    print("Files created:")
    print("  - pretrained_weights.h (main include)")
    print("  - weights_embedding.h")
    print("  - weights_output.h")
    for l in range(N_LAYERS):
        print(f"  - weights_layer{l}.h")


def save_pytorch_model(model, path):
    """Save PyTorch model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': VOCAB_SIZE,
            'seq_len': SEQ_LEN,
            'd_model': D_MODEL,
            'd_k': D_K,
            'd_ff': D_FF,
            'n_layers': N_LAYERS,
        }
    }, path)
    print(f"PyTorch model saved to {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train character-level Transformer')
    parser.add_argument('--data', type=str, default=None, help='Path to training text file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='../app/text_gen_transformer/weights',
                        help='Output directory for C headers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample text')
    parser.add_argument('--max-chars', type=int, default=0,
                        help='Maximum characters to use (0 = all)')
    args = parser.parse_args()

    print(f"Model Configuration:")
    print(f"  VOCAB_SIZE: {VOCAB_SIZE}")
    print(f"  SEQ_LEN: {SEQ_LEN}")
    print(f"  D_MODEL: {D_MODEL}")
    print(f"  D_K: {D_K}")
    print(f"  D_FF: {D_FF}")
    print(f"  N_LAYERS: {N_LAYERS}")
    print(f"  Device: {args.device}")
    print()

    # Load or create sample data
    if args.data:
        with open(args.data, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        if args.max_chars > 0 and len(text) > args.max_chars:
            text = text[:args.max_chars]
            print(f"Loaded {len(text)} characters from {args.data} (truncated to {args.max_chars})")
        else:
            print(f"Loaded {len(text)} characters from {args.data}")
    elif args.demo:
        # Sample text for demo
        text = """
        The quick brown fox jumps over the lazy dog.
        Hello world! This is a test of the character-level transformer.
        Machine learning is a subset of artificial intelligence.
        Deep learning uses neural networks with multiple layers.
        Natural language processing helps computers understand text.
        """ * 100  # Repeat for more training data
        print(f"Using demo text ({len(text)} characters)")
    else:
        print("Error: Please provide --data <file> or use --demo")
        return

    # Create dataset and dataloader
    dataset = CharDataset(text, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset: {len(dataset)} samples")
    print()

    # Create and train model
    model = CharTransformer()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()

    print("Training...")
    model = train(model, dataloader, args.epochs, args.lr, args.device)
    print()

    # Generate sample text
    print("Sample generation:")
    for prompt in ["The ", "Hello ", "Machine "]:
        generated = generate(model, prompt, max_len=50, temperature=0.8, device=args.device)
        print(f'  "{prompt}" -> "{generated}"')
    print()

    # Export weights
    export_weights_to_header(model, args.output_dir)

    # Save PyTorch model
    save_pytorch_model(model, os.path.join(args.output_dir, "model.pt"))


if __name__ == "__main__":
    main()
