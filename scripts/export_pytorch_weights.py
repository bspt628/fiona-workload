#!/usr/bin/env python3
"""
PyTorch MLP weights to FIONA C header converter.

Usage:
    python export_pytorch_weights.py --model checkpoint.pth --output weights.h
    python export_pytorch_weights.py --benchmark mnist --output mnist_weights.h
    python export_pytorch_weights.py --benchmark cifar10 --hidden 256,128,64 --output cifar_weights.h

Supports:
    - Custom PyTorch models
    - Pre-trained benchmarks (MNIST, CIFAR-10, Fashion-MNIST)
    - Automatic quantization to int16
"""

import argparse
import numpy as np
import struct
import os

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Only header generation from numpy will work.")


class SimpleMLP(nn.Module):
    """Simple MLP for benchmarks."""
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def train_benchmark_model(benchmark, hidden_sizes, epochs=10, device='cpu'):
    """Train a simple MLP on a benchmark dataset."""

    print(f"Training MLP on {benchmark}...")

    # Dataset configuration
    if benchmark == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        input_size = 28 * 28
        output_size = 10

    elif benchmark == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform)
        input_size = 28 * 28
        output_size = 10

    elif benchmark == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        input_size = 32 * 32 * 3
        output_size = 10
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Create model
    model = SimpleMLP(input_size, hidden_sizes, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Extract sample test data
    test_samples = []
    test_labels = []
    for i, (data, label) in enumerate(test_dataset):
        if i >= 100:  # Get 100 test samples
            break
        test_samples.append(data.numpy().flatten())
        test_labels.append(label)

    return model, np.array(test_samples), np.array(test_labels)


def quantize_weights(weights, bits=8):
    """Quantize float weights to int16 with scaling."""
    max_val = np.max(np.abs(weights))
    if max_val == 0:
        return np.zeros_like(weights, dtype=np.int16), 1.0

    # Scale to fit in int range with some margin
    max_int = (1 << (bits - 1)) - 1
    scale = max_val / max_int
    quantized = np.clip(np.round(weights / scale), -max_int, max_int).astype(np.int16)

    return quantized, scale


def extract_weights_from_pytorch(model):
    """Extract weights and biases from PyTorch model."""
    weights = []
    biases = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.detach().cpu().numpy()  # (out, in)
            b = module.bias.detach().cpu().numpy() if module.bias is not None else None
            weights.append(w)
            biases.append(b)

    return weights, biases


def generate_c_header(weights, biases, test_x, test_y, output_file,
                      quant_bits=8, model_name="mlp_bench"):
    """Generate C header file with quantized weights."""

    # Quantize all weights
    q_weights = []
    q_biases = []
    w_scales = []
    b_scales = []

    for i, (w, b) in enumerate(zip(weights, biases)):
        qw, sw = quantize_weights(w, quant_bits)
        q_weights.append(qw)
        w_scales.append(sw)

        if b is not None:
            # Bias needs different scaling (accumulator scale)
            qb, sb = quantize_weights(b, quant_bits + 4)  # More bits for bias
            q_biases.append(qb)
            b_scales.append(sb)
        else:
            q_biases.append(None)
            b_scales.append(None)

    # Quantize test data
    q_test_x, test_scale = quantize_weights(test_x, quant_bits)

    # Generate header
    with open(output_file, 'w') as f:
        f.write(f"""/**
 * @file {os.path.basename(output_file)}
 * @brief Auto-generated MLP weights for benchmark
 *
 * Architecture: {' -> '.join([str(w.shape[1]) for w in weights] + [str(weights[-1].shape[0])])}
 * Quantization: {quant_bits}-bit
 *
 * Generated by export_pytorch_weights.py
 */

#ifndef {model_name.upper()}_WEIGHTS_H
#define {model_name.upper()}_WEIGHTS_H

#include "base/config.h"

// Quantization info
#define {model_name.upper()}_QUANT_BITS {quant_bits}
""")

        # Write weight scales
        for i, s in enumerate(w_scales):
            f.write(f"#define {model_name.upper()}_W{i+1}_SCALE {s:.6f}f\n")
        f.write(f"#define {model_name.upper()}_INPUT_SCALE {test_scale:.6f}f\n")
        f.write("\n")

        # Write bias scales
        for i, s in enumerate(b_scales):
            if s is not None:
                f.write(f"#define {model_name.upper()}_B{i+1}_SCALE {s:.6f}f\n")
        f.write("\n")

        # Write weights
        for i, qw in enumerate(q_weights):
            out_size, in_size = qw.shape
            f.write(f"// FC{i+1} weights: {out_size} x {in_size}\n")
            f.write(f"static const elem_t {model_name}_w{i+1}[{out_size}][{in_size}] = {{\n")
            for row in qw:
                f.write("    {" + ", ".join(map(str, row)) + "},\n")
            f.write("};\n\n")

        # Write biases
        for i, qb in enumerate(q_biases):
            if qb is not None:
                f.write(f"// FC{i+1} biases: {len(qb)}\n")
                f.write(f"static const elem_t {model_name}_b{i+1}[{len(qb)}] = {{")
                f.write(", ".join(map(str, qb)))
                f.write("};\n\n")

        # Write test data
        num_samples = min(64, len(test_y))  # Limit samples
        input_size = q_test_x.shape[1]

        f.write(f"// Test data: {num_samples} samples x {input_size} features\n")
        f.write(f"#define {model_name.upper()}_NUM_TEST {num_samples}\n")
        f.write(f"#define {model_name.upper()}_INPUT_SIZE {input_size}\n")
        f.write(f"static const elem_t {model_name}_test_X[{num_samples}][{input_size}] = {{\n")
        for j in range(num_samples):
            row = q_test_x[j]
            # Output all elements (no truncation for valid C code)
            f.write("    {" + ", ".join(map(str, row)) + "},\n")
        f.write("};\n\n")

        # Test labels
        f.write(f"// Test labels\n")
        f.write(f"static const elem_t {model_name}_test_Y[{num_samples}] = {{")
        f.write(", ".join(map(str, test_y[:num_samples])))
        f.write("};\n\n")

        f.write(f"#endif /* {model_name.upper()}_WEIGHTS_H */\n")

    print(f"Generated: {output_file}")

    # Print summary
    total_params = sum(w.size for w in q_weights) + sum(b.size for b in q_biases if b is not None)
    print(f"Total parameters: {total_params}")
    print(f"Architecture: {' -> '.join([str(w.shape[1]) for w in weights] + [str(weights[-1].shape[0])])}")


def generate_binary_file(weights, biases, output_file):
    """Generate FIONA binary weight file."""

    MAGIC = 0x464E4E57  # "FNNW"
    VERSION = 1

    with open(output_file, 'wb') as f:
        # Header
        num_layers = len(weights) + 1  # Including input layer
        f.write(struct.pack('IIII', MAGIC, VERSION, len(weights), 0))

        # Architecture
        layer_sizes = [weights[0].shape[1]]  # Input size
        for w in weights:
            layer_sizes.append(w.shape[0])

        f.write(struct.pack('I', num_layers))
        for size in layer_sizes:
            f.write(struct.pack('I', size))

        # Weights and biases
        for i, (w, b) in enumerate(zip(weights, biases)):
            meta = struct.pack('IIII', w.shape[1], w.shape[0], 1 if b is not None else 0, 0)
            f.write(meta)
            f.write(w.astype(np.float32).tobytes())
            if b is not None:
                f.write(b.astype(np.float32).tobytes())

    print(f"Generated binary: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch weights to FIONA format')
    parser.add_argument('--model', type=str, help='Path to PyTorch model checkpoint')
    parser.add_argument('--benchmark', type=str, choices=['mnist', 'fashion_mnist', 'cifar10'],
                        help='Train on benchmark dataset')
    parser.add_argument('--hidden', type=str, default='256,128',
                        help='Hidden layer sizes (comma-separated)')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--format', type=str, choices=['header', 'binary'], default='header',
                        help='Output format')
    parser.add_argument('--quant-bits', type=int, default=8, help='Quantization bits')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs for benchmark')
    parser.add_argument('--name', type=str, default='mlp_bench', help='Model name prefix')

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for this script")
        return 1

    hidden_sizes = [int(x) for x in args.hidden.split(',')]

    if args.benchmark:
        # Train on benchmark
        model, test_x, test_y = train_benchmark_model(
            args.benchmark, hidden_sizes, epochs=args.epochs)
        weights, biases = extract_weights_from_pytorch(model)

    elif args.model:
        # Load existing model
        checkpoint = torch.load(args.model, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Try to infer architecture from state dict
        # This is a simplified approach - might need customization
        print("Loading from checkpoint...")
        model = SimpleMLP(784, hidden_sizes, 10)  # Default architecture
        model.load_state_dict(state_dict)
        weights, biases = extract_weights_from_pytorch(model)
        test_x = np.random.randn(100, 784) * 0.3  # Placeholder
        test_y = np.zeros(100, dtype=np.int32)
    else:
        print("Error: Either --model or --benchmark is required")
        return 1

    # Generate output
    if args.format == 'header':
        generate_c_header(weights, biases, test_x, test_y,
                         args.output, args.quant_bits, args.name)
    else:
        generate_binary_file(weights, biases, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
