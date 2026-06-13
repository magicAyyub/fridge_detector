#!/usr/bin/env python3
"""Download model checkpoints from Amazon S3 on container startup.

Reads environment variables:
  - BUCKET_NAME: Name of the Amazon S3 bucket. If empty, downloads are skipped.
  - S3_PREFIX: Prefix/folder in the S3 bucket (default: "checkpoints").
"""

import os
import sys
from pathlib import Path

# Files to download
CHECKPOINTS = [
    "best.pt",
    "sam2.1_hiera_tiny.pt"
]

def download_checkpoints() -> None:
    bucket = os.environ.get("BUCKET_NAME")
    if not bucket:
        print("[INFO] BUCKET_NAME env var is not set. Skipping S3 download.")
        return

    prefix = os.environ.get("S3_PREFIX", "checkpoints").strip("/")
    
    # Resolve checkpoints directory relative to root
    root_dir = Path(__file__).resolve().parent.parent
    checkpoints_dir = root_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
    except ImportError:
        print("[ERROR] boto3 is not installed. Cannot download from S3.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Initializing S3 client to download from bucket: {bucket}")
    s3 = boto3.client("s3")

    for filename in CHECKPOINTS:
        target_path = checkpoints_dir / filename
        s3_key = f"{prefix}/{filename}" if prefix else filename

        if target_path.exists():
            print(f"[INFO] File already exists, skipping S3 download: {target_path}")
            continue

        print(f"[INFO] Downloading s3://{bucket}/{s3_key} -> {target_path} ...")
        try:
            s3.download_file(bucket, s3_key, str(target_path))
            print(f"[INFO] Successfully downloaded: {filename}")
        except NoCredentialsError:
            print("[ERROR] AWS credentials not found. Make sure your App Runner IAM Instance Role has S3 Read access.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    print("[INFO] All checkpoints checked and ready.")

if __name__ == "__main__":
    download_checkpoints()
