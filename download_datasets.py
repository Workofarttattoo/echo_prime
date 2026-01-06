#!/usr/bin/env python3
"""
ECH0-PRIME Dataset Downloader
Downloads full benchmark datasets for comprehensive evaluation
"""

import os
import json
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import argparse

class DatasetDownloader:
    """Download and prepare benchmark datasets"""

    def __init__(self):
        self.datasets_dir = Path("datasets")
        self.datasets_dir.mkdir(exist_ok=True)

        # Dataset configurations
        self.datasets = {
            'gsm8k': {
                'type': 'jsonl',
                'urls': {
                    'train': 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl',
                    'test': 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'
                }
            },
            'arc_easy': {
                'type': 'huggingface',
                'dataset': 'ai2_arc',
                'config': 'ARC-Easy',
                'split': 'test'
            },
            'arc_challenge': {
                'type': 'huggingface',
                'dataset': 'ai2_arc',
                'config': 'ARC-Challenge',
                'split': 'test'
            },
            'math': {
                'type': 'tar.gz',
                'url': 'https://people.eecs.berkeley.edu/~hendrycks/MATH.tar.gz',
                'extract_to': 'MATH'
            },
            'mmlu': {
                'type': 'huggingface',
                'dataset': 'cais/mmlu',
                'config': 'all',
                'split': 'test'
            },
            'gooaq': {
                'type': 'huggingface',
                'dataset': 'sentence-transformers/gooaq',
                'split': 'train'
            },
            'hellaswag': {
                'type': 'huggingface',
                'dataset': 'Rowan/hellaswag',
                'config': 'default',
                'split': 'validation'
            },
            'truthful_qa': {
                'type': 'huggingface',
                'dataset': 'truthful_qa',
                'config': 'multiple_choice',
                'split': 'validation'
            },
            'winogrande': {
                'type': 'huggingface',
                'dataset': 'winogrande',
                'config': 'winogrande_xl',
                'split': 'validation'
            }
        }

    def download_all(self, datasets: list = None) -> None:
        """Download all specified datasets"""
        if datasets is None:
            datasets = list(self.datasets.keys())

        print("📥 Downloading benchmark datasets...")
        print(f"Datasets to download: {', '.join(datasets)}")
        print("=" * 60)

        for dataset in datasets:
            if dataset not in self.datasets:
                print(f"⚠️ Unknown dataset: {dataset}")
                continue

            try:
                print(f"\n📦 Downloading {dataset}...")
                self._download_dataset(dataset)
                print(f"✅ {dataset} downloaded successfully")
            except Exception as e:
                print(f"❌ Failed to download {dataset}: {e}")

        print("\n" + "=" * 60)
        print("🎉 Dataset download complete!")

    def _download_dataset(self, dataset: str) -> None:
        """Download a specific dataset"""
        config = self.datasets[dataset]

        if config['type'] == 'jsonl':
            self._download_jsonl_files(dataset, config)
        elif config['type'] == 'zip':
            self._download_zip_file(dataset, config)
        elif config['type'] == 'tar.gz':
            self._download_tar_file(dataset, config)
        elif config['type'] == 'huggingface':
            self._download_huggingface_dataset(dataset, config)

    def _download_huggingface_dataset(self, dataset: str, config: dict) -> None:
        """Download dataset from HuggingFace and save to JSON"""
        try:
            from datasets import load_dataset
            print(f"  Downloading from HuggingFace: {config['dataset']}...")
            
            if 'config' in config:
                print(f"  Using config: {config['config']}")
                ds = load_dataset(config['dataset'], config['config'], split=config['split'], trust_remote_code=True)
            else:
                ds = load_dataset(config['dataset'], split=config['split'], trust_remote_code=True)
            
            output_file = self.datasets_dir / f"{dataset}_{config['split']}.json"
            print(f"  Saving to {output_file}...")
            
            # Save first 10k samples as a representative sample
            samples = []
            for i, example in enumerate(ds):
                if i >= 10000: break
                samples.append(example)
                
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=2)
                
            print(f"  Successfully saved {len(samples)} samples")
            
        except ImportError:
            print("  ❌ Error: 'datasets' library not installed. Run 'pip install datasets'")
        except Exception as e:
            print(f"  ❌ Error downloading HuggingFace dataset: {e}")

    def _download_jsonl_files(self, dataset: str, config: dict) -> None:
        """Download JSONL files and convert to JSON"""
        for split, url in config['urls'].items():
            filename = f"{dataset}_{split}.jsonl"
            filepath = self.datasets_dir / filename

            print(f"  Downloading {split} split...")
            urllib.request.urlretrieve(url, filepath)

            # Convert to JSON array
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))

            json_filepath = filepath.with_suffix('.json')
            with open(json_filepath, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"  Converted to JSON: {json_filepath}")

    def _download_zip_file(self, dataset: str, config: dict) -> None:
        """Download and extract ZIP file"""
        zip_filename = f"{dataset}.zip"
        zip_filepath = self.datasets_dir / zip_filename

        print(f"  Downloading {dataset}.zip...")
        urllib.request.urlretrieve(config['url'], zip_filepath)

        extract_to = self.datasets_dir / config['extract_to']
        print(f"  Extracting to {extract_to}...")

        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(self.datasets_dir)

        print(f"  Extracted {len(zip_ref.namelist())} files")

    def _download_tar_file(self, dataset: str, config: dict) -> None:
        """Download and extract TAR.GZ file"""
        tar_filename = f"{dataset}.tar.gz"
        tar_filepath = self.datasets_dir / tar_filename

        print(f"  Downloading {dataset}.tar.gz...")
        urllib.request.urlretrieve(config['url'], tar_filepath)

        extract_to = self.datasets_dir / config['extract_to']
        print(f"  Extracting to {extract_to}...")

        with tarfile.open(tar_filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(self.datasets_dir)

        print(f"  Extracted {len(tar_ref.getmembers())} files")

    def list_available_datasets(self) -> None:
        """List all available datasets"""
        print("📚 Available Datasets:")
        print("=" * 40)

        for name, config in self.datasets.items():
            dataset_type = config['type']
            if 'urls' in config:
                files = list(config['urls'].keys())
            elif 'extract_to' in config:
                files = [config['extract_to']]
            else:
                files = ['direct file']

            print(f"  {name}: {dataset_type} ({', '.join(files)})")

    def check_downloaded_datasets(self) -> dict:
        """Check which datasets have been downloaded"""
        status = {}

        for dataset in self.datasets.keys():
            config = self.datasets[dataset]

            if config['type'] == 'jsonl':
                # Check for JSON files
                has_train = (self.datasets_dir / f"{dataset}_train.json").exists()
                has_test = (self.datasets_dir / f"{dataset}_test.json").exists()
                status[dataset] = {'train': has_train, 'test': has_test, 'ready': has_train and has_test}
            elif config['type'] in ['zip', 'tar.gz']:
                extract_path = self.datasets_dir / config.get('extract_to', dataset)
                status[dataset] = {'extracted': extract_path.exists(), 'ready': extract_path.exists()}
            else:
                file_path = self.datasets_dir / config.get('filename', f"{dataset}.json")
                status[dataset] = {'downloaded': file_path.exists(), 'ready': file_path.exists()}

        return status

def main():
    """Download benchmark datasets"""
    parser = argparse.ArgumentParser(description="ECH0-PRIME Dataset Downloader")
    parser.add_argument("--datasets", nargs="+",
                       help="Specific datasets to download")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    parser.add_argument("--check", action="store_true",
                       help="Check download status")

    args = parser.parse_args()

    downloader = DatasetDownloader()

    if args.list:
        downloader.list_available_datasets()
    elif args.check:
        status = downloader.check_downloaded_datasets()
        print("📊 Dataset Download Status:")
        print("=" * 40)
        for dataset, info in status.items():
            ready_status = "✅ Ready" if info.get('ready', False) else "❌ Not Ready"
            print(f"  {dataset}: {ready_status}")
            if 'train' in info:
                print(f"    Train: {'✅' if info['train'] else '❌'}")
            if 'test' in info:
                print(f"    Test: {'✅' if info['test'] else '❌'}")
            if 'extracted' in info:
                print(f"    Extracted: {'✅' if info['extracted'] else '❌'}")
            if 'downloaded' in info:
                print(f"    Downloaded: {'✅' if info['downloaded'] else '❌'}")
    else:
        datasets_to_download = args.datasets if args.datasets else None
        downloader.download_all(datasets_to_download)

if __name__ == "__main__":
    main()


