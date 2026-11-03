"""
Dataset Extraction Utility for AcousticIQ.
Extracts downloaded MIMII archives into ml/data/raw/MIMII/ organized by machine category.

IMPORTANT: Original raw recordings are preserved in their exact uncompressed state
and marked as read-only where permitted to prevent accidental alteration.
"""

import os
import sys
import stat
import argparse
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.core.config import settings


def make_readonly(path: Path):
    """Sets file permission to read-only to safeguard raw recordings."""
    try:
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except Exception:
        pass


def extract_archive(
    archive_path: Path,
    target_raw_dir: Path,
    machine_category: Optional[str] = None,
    enforce_readonly: bool = True,
) -> Dict[str, Any]:
    """
    Extracts a MIMII ZIP archive into the raw dataset directory.
    Target directory structure:
        raw/MIMII/<machine_category>/<model_id>/<normal|anomaly>/*.wav
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive file not found: {archive_path}")

    target_raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Dataset Extractor] Inspecting archive: {archive_path.name}")
    print(f"Destination root: {target_raw_dir}")

    extracted_files: List[Path] = []
    category_counts: Dict[str, int] = {}

    with zipfile.ZipFile(archive_path, "r") as z:
        file_list = [f for f in z.infolist() if not f.is_dir()]
        total_files = len(file_list)
        print(f"Archive contains {total_files} files. Extracting...")

        for idx, member in enumerate(file_list, 1):
            # Member filename is typically:
            # - "6_dB_valve/valve/id_00/normal/00000000.wav"
            # - or "valve/id_00/normal/00000000.wav"
            parts = Path(member.filename).parts
            
            # Identify machine category from path or fallback argument
            cat = machine_category
            for part in parts:
                p_lower = part.lower()
                for known in ["valve", "pump", "fan", "slider"]:
                    if known in p_lower:
                        cat = known
                        break
                if cat:
                    break
            cat = cat or "valve"

            # Reconstruct clean relative path: <machine>/<id>/<condition>/<name>
            # Locate condition ('normal' or 'anomaly')
            cond = "normal"
            model_id = "id_00"
            for p in parts:
                if p.lower() in ["normal", "anomaly"]:
                    cond = p.lower()
                elif "id_" in p.lower():
                    model_id = p.lower()

            rel_dest = Path(cat) / model_id / cond / Path(member.filename).name
            dest_file = target_raw_dir / rel_dest
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            with z.open(member) as src, open(dest_file, "wb") as dst:
                dst.write(src.read())

            if enforce_readonly:
                make_readonly(dest_file)

            extracted_files.append(dest_file)
            category_counts[cat] = category_counts.get(cat, 0) + 1

            if idx % 100 == 0 or idx == total_files:
                print(f"\rExtracted: {idx} / {total_files} files ({(idx / total_files) * 100:.1f}%)", end="", flush=True)

    print("\nExtraction complete.")
    for cat, count in category_counts.items():
        print(f"  - Category '{cat}': {count} recordings written to {target_raw_dir / cat}")

    return {
        "archive": str(archive_path),
        "target_directory": str(target_raw_dir),
        "total_extracted": len(extracted_files),
        "categories": category_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract MIMII archives into ml/data/raw/MIMII/")
    parser.add_argument("--archive", type=str, required=True, help="Path to the downloaded ZIP archive")
    parser.add_argument("--raw-dir", type=str, default=str(settings.DATASET_RAW_DIR), help="Destination raw directory")
    parser.add_argument("--machine", type=str, default=None, help="Explicit machine category override (valve, pump, fan, slider)")
    parser.add_argument("--no-readonly", action="store_true", help="Do not mark extracted files as read-only")

    args = parser.parse_args()

    extract_archive(
        archive_path=Path(args.archive),
        target_raw_dir=Path(args.raw_dir),
        machine_category=args.machine,
        enforce_readonly=not args.no_readonly,
    )


if __name__ == "__main__":
    main()
