"""
MIMII Dataset Downloader and Extractor Script for AcousticIQ.
Official Zenodo Record: https://zenodo.org/records/3384388

Supports:
- Inspecting available MIMII archives and file sizes.
- Targeted download of specific machine categories and SNR levels.
- Resumable chunked streaming download with progress tracking.
- ZIP extraction into configured local dataset directory.
- Validation of audio files.
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional, Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from backend.app.core.config import settings
    DEFAULT_RAW_DIR = settings.DATASET_RAW_DIR
except Exception:
    DEFAULT_RAW_DIR = BASE_DIR / "ml" / "data" / "raw" / "MIMII"

# Official Zenodo Record 3384388 File Catalog & Approximate Sizes
MIMII_ARCHIVES: Dict[str, Dict[str, Any]] = {
    "valve_6dB": {
        "file_name": "6_dB_valve.zip",
        "url": "https://zenodo.org/records/3384388/files/6_dB_valve.zip?download=1",
        "size_gb": 6.9,
        "machine": "valve",
        "snr": "6_dB",
    },
    "slider_6dB": {
        "file_name": "6_dB_slider.zip",
        "url": "https://zenodo.org/records/3384388/files/6_dB_slider.zip?download=1",
        "size_gb": 7.1,
        "machine": "slider",
        "snr": "6_dB",
    },
    "pump_6dB": {
        "file_name": "6_dB_pump.zip",
        "url": "https://zenodo.org/records/3384388/files/6_dB_pump.zip?download=1",
        "size_gb": 7.7,
        "machine": "pump",
        "snr": "6_dB",
    },
    "fan_6dB": {
        "file_name": "6_dB_fan.zip",
        "url": "https://zenodo.org/records/3384388/files/6_dB_fan.zip?download=1",
        "size_gb": 10.2,
        "machine": "fan",
        "snr": "6_dB",
    },
    "valve_0dB": {
        "file_name": "0_dB_valve.zip",
        "url": "https://zenodo.org/records/3384388/files/0_dB_valve.zip?download=1",
        "size_gb": 7.5,
        "machine": "valve",
        "snr": "0_dB",
    },
    "pump_0dB": {
        "file_name": "0_dB_pump.zip",
        "url": "https://zenodo.org/records/3384388/files/0_dB_pump.zip?download=1",
        "size_gb": 7.9,
        "machine": "pump",
        "snr": "0_dB",
    },
    "slider_0dB": {
        "file_name": "0_dB_slider.zip",
        "url": "https://zenodo.org/records/3384388/files/0_dB_slider.zip?download=1",
        "size_gb": 7.5,
        "machine": "slider",
        "snr": "0_dB",
    },
    "fan_0dB": {
        "file_name": "0_dB_fan.zip",
        "url": "https://zenodo.org/records/3384388/files/0_dB_fan.zip?download=1",
        "size_gb": 10.4,
        "machine": "fan",
        "snr": "0_dB",
    },
}


def print_catalog():
    print("\n=== MIMII Dataset (Zenodo 3384388) Archive Catalog ===")
    print(f"{'Archive Key':<15} | {'Machine':<10} | {'SNR':<8} | {'Est. Size':<10} | {'Filename'}")
    print("-" * 65)
    for key, item in MIMII_ARCHIVES.items():
        print(f"{key:<15} | {item['machine']:<10} | {item['snr']:<8} | {item['size_gb']:>5.1f} GB   | {item['file_name']}")
    print("-" * 65)
    print("License: Creative Commons Attribution 4.0 International (CC BY 4.0)")
    print("Authors: Harsh Purohit et al. (Hitachi, Ltd.)\n")


def download_archive(archive_key: str, raw_dir: Path, extract: bool = True):
    if archive_key not in MIMII_ARCHIVES:
        print(f"Error: Unknown archive key '{archive_key}'. Available: {list(MIMII_ARCHIVES.keys())}")
        return False

    item = MIMII_ARCHIVES[archive_key]
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / item["file_name"]

    print(f"\n[MIMII Downloader] Preparing to download {item['file_name']} (~{item['size_gb']} GB)...")
    print(f"Destination: {zip_path}")
    print(f"Target raw directory: {raw_dir}")
    print(f"URL: {item['url']}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AcousticIQ/1.0"
    }

    # Resumable download check
    initial_bytes = 0
    if zip_path.exists():
        initial_bytes = zip_path.stat().st_size
        print(f"Existing file found: {initial_bytes / (1024*1024):.2f} MB. Attempting to resume...")
        headers["Range"] = f"bytes={initial_bytes}-"

    req = urllib.request.Request(item["url"], headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            mode = "ab" if initial_bytes > 0 else "wb"
            total_size = resp.headers.get("Content-Length")
            total_size = int(total_size) + initial_bytes if total_size else None

            downloaded = initial_bytes
            chunk_size = 1024 * 1024  # 1 MB chunks

            with open(zip_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\rDownloading: {mb_done:.1f} / {mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
                    else:
                        print(f"\rDownloaded: {downloaded / (1024 * 1024):.1f} MB", end="", flush=True)

        print("\nDownload finished successfully.")

        if extract:
            print(f"Extracting {zip_path} into {raw_dir} (preserving category structure)...")
            from scripts.extract_dataset import extract_archive
            extract_archive(zip_path, raw_dir, machine_category=item["machine"])
            print("Extraction and read-only preservation complete.")

        return True
    except Exception as e:
        print(f"\nDownload interrupted or failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="MIMII Dataset Downloader for AcousticIQ")
    parser.add_argument("--info", action="store_true", help="Print available MIMII archives and sizes")
    parser.add_argument("--archive", type=str, default="valve_6dB", help="Archive key to download (e.g. valve_6dB)")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR), help="Destination raw dataset directory (default: ml/data/raw/MIMII)")
    parser.add_argument("--target-dir", type=str, default=None, help="Alias for --raw-dir")
    parser.add_argument("--no-extract", action="store_true", help="Do not extract ZIP archive after download")

    args = parser.parse_args()

    if args.info:
        print_catalog()
    else:
        dest_dir = Path(args.target_dir or args.raw_dir)
        download_archive(args.archive, dest_dir, extract=not args.no_extract)


if __name__ == "__main__":
    main()
