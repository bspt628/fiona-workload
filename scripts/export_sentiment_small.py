#!/usr/bin/env python3
"""
export_sentiment_small.py - Export reduced vocabulary model to FIONA C headers

This script exports both weights and test data with the reduced vocabulary mapping.

Usage:
    python export_sentiment_small.py \
        --model_path ./tiny_sentiment_small/model.pt \
        --output_dir ../app/sentiment_photonic/

Author: FIONA Project
Date: 2025-12-23
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer


def write_weight_header(filepath: Path, name: str, arr: np.ndarray, description: str = ""):
    """Write a weight array as a C header file."""
    with open(filepath, "w") as f:
        guard = f"{name.upper()}_H"
        f.write(f"/**\n")
        f.write(f" * @file {filepath.name}\n")
        f.write(f" * @brief {description}\n")
        f.write(f" * @note Auto-generated\n")
        f.write(f" *\n")
        f.write(f" * Shape: {list(arr.shape)}\n")
        f.write(f" * Elements: {arr.size:,}\n")
        f.write(f" * Memory: {arr.nbytes:,} bytes\n")
        f.write(f" */\n\n")

        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")

        flat = arr.flatten()
        f.write(f"// Shape: {list(arr.shape)}\n")
        f.write(f"static const float {name}[{len(flat)}] = {{\n")

        for i in range(0, len(flat), 8):
            chunk = flat[i:i + 8]
            values = ", ".join(f"{v:.8f}f" for v in chunk)
            f.write(f"    {values},\n")

        f.write("};\n\n")
        f.write(f"#endif // {guard}\n")


def export_weights(state_dict: dict, output_path: Path, config: dict):
    """Export all model weights."""
    weights_path = output_path / "weights"
    weights_path.mkdir(parents=True, exist_ok=True)

    # Token embedding
    token_emb = state_dict["token_embedding.weight"].numpy()
    write_weight_header(
        weights_path / "embedding_token.h",
        "token_embedding",
        token_emb,
        f"Token embedding [{token_emb.shape[0]} x {token_emb.shape[1]}]"
    )
    print(f"  Token embedding: {token_emb.shape} ({token_emb.nbytes / 1024:.1f} KB)")

    # Position embedding
    pos_emb = state_dict["position_embedding.weight"].numpy()
    write_weight_header(
        weights_path / "embedding_position.h",
        "position_embedding",
        pos_emb,
        f"Position embedding [{pos_emb.shape[0]} x {pos_emb.shape[1]}]"
    )

    # Transformer layers
    n_layers = config.get("n_layers", 2)
    for layer_idx in range(n_layers):
        export_layer_weights(state_dict, weights_path, layer_idx, config)

    # Pooler and classifier
    export_head_weights(state_dict, weights_path, config)

    # Master header
    generate_master_header(weights_path, config)

    print(f"  Weights exported to: {weights_path}")


def export_layer_weights(state_dict: dict, output_path: Path, layer_idx: int, config: dict):
    """Export weights for a single transformer layer."""
    prefix = f"encoder.layers.{layer_idx}"

    in_proj = state_dict[f"{prefix}.self_attn.in_proj_weight"].numpy()
    in_proj_bias = state_dict[f"{prefix}.self_attn.in_proj_bias"].numpy()

    Wq, Wk, Wv = np.split(in_proj, 3, axis=0)
    bq, bk, bv = np.split(in_proj_bias, 3, axis=0)

    Wo = state_dict[f"{prefix}.self_attn.out_proj.weight"].numpy()
    bo = state_dict[f"{prefix}.self_attn.out_proj.bias"].numpy()

    W1 = state_dict[f"{prefix}.linear1.weight"].numpy()
    b1 = state_dict[f"{prefix}.linear1.bias"].numpy()
    W2 = state_dict[f"{prefix}.linear2.weight"].numpy()
    b2 = state_dict[f"{prefix}.linear2.bias"].numpy()

    ln1_gamma = state_dict[f"{prefix}.norm1.weight"].numpy()
    ln1_beta = state_dict[f"{prefix}.norm1.bias"].numpy()
    ln2_gamma = state_dict[f"{prefix}.norm2.weight"].numpy()
    ln2_beta = state_dict[f"{prefix}.norm2.bias"].numpy()

    layer_name = f"layer{layer_idx}"

    weights = [
        (f"{layer_name}_Wq", Wq, "Query projection weight"),
        (f"{layer_name}_bq", bq, "Query projection bias"),
        (f"{layer_name}_Wk", Wk, "Key projection weight"),
        (f"{layer_name}_bk", bk, "Key projection bias"),
        (f"{layer_name}_Wv", Wv, "Value projection weight"),
        (f"{layer_name}_bv", bv, "Value projection bias"),
        (f"{layer_name}_Wo", Wo, "Output projection weight"),
        (f"{layer_name}_bo", bo, "Output projection bias"),
        (f"{layer_name}_W1", W1, "FFN layer 1 weight"),
        (f"{layer_name}_b1", b1, "FFN layer 1 bias"),
        (f"{layer_name}_W2", W2, "FFN layer 2 weight"),
        (f"{layer_name}_b2", b2, "FFN layer 2 bias"),
        (f"{layer_name}_ln1_gamma", ln1_gamma, "LayerNorm 1 gamma"),
        (f"{layer_name}_ln1_beta", ln1_beta, "LayerNorm 1 beta"),
        (f"{layer_name}_ln2_gamma", ln2_gamma, "LayerNorm 2 gamma"),
        (f"{layer_name}_ln2_beta", ln2_beta, "LayerNorm 2 beta"),
    ]

    with open(output_path / f"{layer_name}.h", "w") as f:
        guard = f"{layer_name.upper()}_H"
        f.write(f"/**\n")
        f.write(f" * @file {layer_name}.h\n")
        f.write(f" * @brief Transformer layer {layer_idx} weights\n")
        f.write(f" * @note Auto-generated\n")
        f.write(f" */\n\n")
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")

        for name, arr, desc in weights:
            flat = arr.flatten()
            f.write(f"// {desc}, Shape: {list(arr.shape)}\n")
            f.write(f"static const float {name}[{len(flat)}] = {{\n")
            for i in range(0, len(flat), 8):
                chunk = flat[i:i + 8]
                values = ", ".join(f"{v:.8f}f" for v in chunk)
                f.write(f"    {values},\n")
            f.write("};\n\n")

        f.write(f"#endif // {guard}\n")


def export_head_weights(state_dict: dict, output_path: Path, config: dict):
    """Export pooler and classifier weights."""
    pooler_weight = state_dict["pooler.weight"].numpy()
    pooler_bias = state_dict["pooler.bias"].numpy()
    classifier_weight = state_dict["classifier.weight"].numpy()
    classifier_bias = state_dict["classifier.bias"].numpy()

    with open(output_path / "classifier.h", "w") as f:
        f.write(f"/**\n")
        f.write(f" * @file classifier.h\n")
        f.write(f" * @brief Pooler and classifier head weights\n")
        f.write(f" * @note Auto-generated\n")
        f.write(f" */\n\n")
        f.write(f"#ifndef CLASSIFIER_H\n")
        f.write(f"#define CLASSIFIER_H\n\n")

        weights = [
            ("pooler_weight", pooler_weight, "Pooler weight"),
            ("pooler_bias", pooler_bias, "Pooler bias"),
            ("classifier_weight", classifier_weight, "Classifier weight"),
            ("classifier_bias", classifier_bias, "Classifier bias"),
        ]

        for name, arr, desc in weights:
            flat = arr.flatten()
            f.write(f"// {desc}, Shape: {list(arr.shape)}\n")
            f.write(f"static const float {name}[{len(flat)}] = {{\n")
            for i in range(0, len(flat), 8):
                chunk = flat[i:i + 8]
                values = ", ".join(f"{v:.8f}f" for v in chunk)
                f.write(f"    {values},\n")
            f.write("};\n\n")

        f.write(f"#endif // CLASSIFIER_H\n")


def generate_master_header(output_path: Path, config: dict):
    """Generate master header."""
    n_layers = config.get("n_layers", 2)

    with open(output_path / "weights.h", "w") as f:
        f.write(f"/**\n")
        f.write(f" * @file weights.h\n")
        f.write(f" * @brief Master header for TinySentiment weights (reduced vocabulary)\n")
        f.write(f" * @note Auto-generated\n")
        f.write(f" */\n\n")
        f.write(f"#ifndef WEIGHTS_H\n")
        f.write(f"#define WEIGHTS_H\n\n")

        f.write(f"// Model configuration (reduced vocabulary)\n")
        f.write(f"#define MODEL_VOCAB_SIZE   {config.get('vocab_size', 5000)}\n")
        f.write(f"#define MODEL_MAX_POSITION {config.get('max_position', 64)}\n")
        f.write(f"#define MODEL_D_MODEL      {config.get('d_model', 128)}\n")
        f.write(f"#define MODEL_N_HEADS      {config.get('n_heads', 2)}\n")
        f.write(f"#define MODEL_D_K          {config.get('d_k', 64)}\n")
        f.write(f"#define MODEL_D_FF         {config.get('d_ff', 256)}\n")
        f.write(f"#define MODEL_N_LAYERS     {config.get('n_layers', 2)}\n")
        f.write(f"#define MODEL_NUM_LABELS   {config.get('num_labels', 2)}\n\n")

        f.write(f"// Include all weight files\n")
        f.write(f"#include \"embedding_token.h\"\n")
        f.write(f"#include \"embedding_position.h\"\n")
        for i in range(n_layers):
            f.write(f"#include \"layer{i}.h\"\n")
        f.write(f"#include \"classifier.h\"\n\n")

        f.write(f"#endif // WEIGHTS_H\n")


def export_testdata(output_path: Path, old_to_new: dict, unk_new_id: int,
                    max_length: int = 64, num_samples: int = None):
    """Export SST-2 test data with remapped token IDs."""
    testdata_path = output_path / "testdata"
    testdata_path.mkdir(parents=True, exist_ok=True)

    print("\nExporting test data with remapped vocabulary...")

    # Load dataset and tokenizer
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    validation = dataset["validation"]
    if num_samples is not None:
        validation = validation.select(range(min(num_samples, len(validation))))

    n_samples = len(validation)
    print(f"  Samples: {n_samples}")

    # Remap function
    def remap_tokens(input_ids):
        return [old_to_new.get(tid, unk_new_id) for tid in input_ids]

    # Collect data
    all_token_ids = []
    all_labels = []

    for example in validation:
        encoded = tokenizer(
            example["sentence"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        remapped = remap_tokens(encoded["input_ids"])
        all_token_ids.append(remapped)
        all_labels.append(example["label"])

    # Write header file
    with open(testdata_path / "sst2_testdata.h", "w") as f:
        f.write(f"/**\n")
        f.write(f" * @file sst2_testdata.h\n")
        f.write(f" * @brief SST-2 validation data with remapped vocabulary\n")
        f.write(f" * @note Auto-generated\n")
        f.write(f" *\n")
        f.write(f" * Samples: {n_samples}\n")
        f.write(f" * Sequence length: {max_length}\n")
        f.write(f" * Vocabulary: reduced (remapped to new IDs)\n")
        f.write(f" */\n\n")

        f.write(f"#ifndef SST2_TESTDATA_H\n")
        f.write(f"#define SST2_TESTDATA_H\n\n")

        f.write(f"#define SST2_NUM_SAMPLES {n_samples}\n")
        f.write(f"#define SST2_SEQ_LEN {max_length}\n\n")

        # Token IDs
        f.write(f"static const int sst2_token_ids[SST2_NUM_SAMPLES][SST2_SEQ_LEN] = {{\n")
        for i, tokens in enumerate(all_token_ids):
            tokens_str = ", ".join(str(t) for t in tokens)
            f.write(f"    {{ {tokens_str} }},\n")
        f.write(f"}};\n\n")

        # Labels
        f.write(f"static const int sst2_labels[SST2_NUM_SAMPLES] = {{\n")
        for i in range(0, n_samples, 20):
            chunk = all_labels[i:i + 20]
            labels_str = ", ".join(str(l) for l in chunk)
            f.write(f"    {labels_str},\n")
        f.write(f"}};\n\n")

        f.write(f"#endif // SST2_TESTDATA_H\n")

    print(f"  Test data exported to: {testdata_path}")


def main():
    parser = argparse.ArgumentParser(description="Export reduced vocabulary model")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model.pt checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (e.g., ../app/sentiment_photonic/)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of test samples (default: all 872)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: export only 4 test samples")

    args = parser.parse_args()

    if args.quick:
        args.num_samples = 4

    print("=" * 60)
    print("TinySentiment Exporter (Reduced Vocabulary)")
    print("=" * 60)

    # Load checkpoint
    print(f"\nLoading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})
    old_to_new = checkpoint.get("old_to_new", {})
    unk_new_id = checkpoint.get("unk_new_id", 0)

    # Convert keys to int if they're strings
    old_to_new = {int(k): v for k, v in old_to_new.items()}

    print(f"Config: vocab_size={config.get('vocab_size')}, d_model={config.get('d_model')}")
    print(f"Vocabulary mapping: {len(old_to_new)} tokens")

    output_path = Path(args.output_dir)

    # Export weights
    print("\nExporting weights...")
    export_weights(state_dict, output_path, config)

    # Export test data
    export_testdata(
        output_path,
        old_to_new,
        unk_new_id,
        max_length=config.get("max_position", 64),
        num_samples=args.num_samples
    )

    print("\n" + "=" * 60)
    print(f"Export complete!")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
