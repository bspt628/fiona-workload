#!/usr/bin/env python3
"""
Download standard text datasets for character-level language modeling.

Available datasets:
- tiny_shakespeare: 1.1MB, Shakespeare plays (good for small models)
- wikitext2: 10MB, Wikipedia articles (standard benchmark)
- text8: 100MB, Wikipedia text (larger benchmark)

Usage:
    python download_dataset.py --dataset tiny_shakespeare
    python download_dataset.py --dataset wikitext2
"""

import argparse
import os
import urllib.request
import zipfile

DATASETS = {
    'tiny_shakespeare': {
        'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'filename': 'tiny_shakespeare.txt',
        'description': 'Shakespeare plays (~1.1MB, 1.1M chars)',
        'is_zip': False,
    },
    'wikitext2_raw': {
        'url': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip',
        'filename': 'wikitext-2-raw-v1.zip',
        'description': 'WikiText-2 raw (~12MB unzipped)',
        'is_zip': True,
        'extract_file': 'wikitext-2-raw/wiki.train.raw',
    },
    'text8': {
        'url': 'http://mattmahoney.net/dc/text8.zip',
        'filename': 'text8.zip',
        'description': 'Text8 Wikipedia (~100MB unzipped)',
        'is_zip': True,
        'extract_file': 'text8',
    },
}

def download_file(url, filepath):
    """Download a file with progress indicator."""
    print(f"Downloading {url}...")

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r  Progress: {percent:.1f}% ({downloaded // 1024} KB)", end='')

    urllib.request.urlretrieve(url, filepath, show_progress)
    print()

def main():
    parser = argparse.ArgumentParser(description='Download text datasets')
    parser.add_argument('--dataset', type=str, default='tiny_shakespeare',
                        choices=list(DATASETS.keys()),
                        help='Dataset to download')
    parser.add_argument('--output-dir', type=str, default='../data',
                        help='Output directory')
    parser.add_argument('--list', action='store_true',
                        help='List available datasets')
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name}: {info['description']}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = DATASETS[args.dataset]
    filepath = os.path.join(args.output_dir, dataset['filename'])

    print(f"Dataset: {args.dataset}")
    print(f"Description: {dataset['description']}")
    print()

    # Download
    if not os.path.exists(filepath):
        download_file(dataset['url'], filepath)
    else:
        print(f"File already exists: {filepath}")

    # Extract if zip
    if dataset['is_zip']:
        print(f"Extracting {filepath}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(args.output_dir)

        # Find the extracted file
        extract_path = os.path.join(args.output_dir, dataset['extract_file'])
        if os.path.exists(extract_path):
            # Read and save as single text file
            with open(extract_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            output_file = os.path.join(args.output_dir, f"{args.dataset}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"Saved to: {output_file}")
            print(f"Size: {len(text):,} characters ({len(text) / 1024 / 1024:.2f} MB)")
    else:
        # Just report size
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        print(f"Size: {len(text):,} characters ({len(text) / 1024 / 1024:.2f} MB)")

    print()
    print("Done! You can now train with:")
    print(f"  python train_char_transformer.py --data {args.output_dir}/{args.dataset}.txt --epochs 100")

if __name__ == "__main__":
    main()
