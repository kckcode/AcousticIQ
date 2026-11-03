"""
Secure file storage and upload handling service for AcousticIQ.
"""

import os
import re
import uuid
import shutil
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile, HTTPException
from backend.app.core.config import settings


class StorageService:
    ALLOWED_EXTENSIONS = {".wav"}
    MAX_FILE_SIZE = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024  # bytes

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitizes incoming file names to prevent path traversal or special character attacks."""
        name = Path(filename).name
        # Remove non-alphanumeric except dots, dashes, underscores
        clean_name = re.sub(r"[^\w\.-]", "_", name)
        return clean_name or "recording.wav"

    @classmethod
    async def save_upload_file(cls, upload_file: UploadFile) -> Tuple[Path, str]:
        """
        Validates and safely saves an uploaded audio file into temporary storage.
        Returns:
            temp_path (Path): Path to the saved temporary file.
            sanitized_name (str): Safe filename.
        """
        filename = upload_file.filename or "unknown.wav"
        sanitized_name = cls.sanitize_filename(filename)
        ext = Path(sanitized_name).suffix.lower()

        if ext not in cls.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Only standard uncompressed WAV files (.wav) are accepted."
            )

        unique_id = uuid.uuid4().hex
        dest_filename = f"{unique_id}_{sanitized_name}"
        dest_path = settings.UPLOAD_DIR / dest_filename

        # Write in chunks while enforcing size bounds
        size_read = 0
        try:
            with open(dest_path, "wb") as out_file:
                while True:
                    chunk = await upload_file.read(1024 * 64) # 64KB
                    if not chunk:
                        break
                    size_read += len(chunk)
                    if size_read > cls.MAX_FILE_SIZE:
                        out_file.close()
                        if dest_path.exists():
                            dest_path.unlink()
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds maximum allowed upload size of {settings.MAX_UPLOAD_SIZE_MB} MB."
                        )
                    out_file.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            if dest_path.exists():
                dest_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

        return dest_path, sanitized_name

    @staticmethod
    def remove_file(file_path: Path):
        """Safely removes a temporary file."""
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass
