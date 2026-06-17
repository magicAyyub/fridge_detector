#!/usr/bin/env python3
"""Upload model checkpoints and data files to Amazon S3.

Usage:
  python scripts/upload_assets.py --bucket my-s3-bucket-name
"""

import os
import sys
import argparse
from pathlib import Path

# Files to upload: (local_path, s3_filename)
FILES_TO_UPLOAD = [
    ("checkpoints/best.pt", "best.pt"),
    ("checkpoints/sam2.1_hiera_tiny.pt", "sam2.1_hiera_tiny.pt"),
    ("data/recipes.json", "recipes.json")
]

def upload_assets() -> None:
    parser = argparse.ArgumentParser(description="Upload checkpoints and recipes dataset to S3.")
    parser.add_argument("--bucket", type=str, default=os.environ.get("BUCKET_NAME"),
                        help="S3 bucket name. Can also be set via BUCKET_NAME env var.")
    parser.add_argument("--prefix", type=str, default=os.environ.get("S3_PREFIX", "checkpoints"),
                        help="S3 folder prefix (default: checkpoints).")
    args = parser.parse_args()

    bucket = args.bucket
    if not bucket:
        print("[ERROR] S3 bucket name is missing. Please provide it via --bucket or BUCKET_NAME env var.", file=sys.stderr)
        print("Example: python scripts/upload_assets.py --bucket my-bucket-name", file=sys.stderr)
        sys.exit(1)

    prefix = args.prefix.strip("/")

    # Resolve project root directory
    root_dir = Path(__file__).resolve().parent.parent

    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
    except ImportError:
        print("[ERROR] boto3 is not installed. Run: pip install boto3", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Initializing S3 client for bucket: {bucket}")
    s3 = boto3.client("s3")

    # Verify bucket accessibility/credentials
    try:
        s3.head_bucket(Bucket=bucket)
    except NoCredentialsError:
        print("[ERROR] AWS credentials not found. Run 'aws configure' first or set AWS env vars.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Cannot access bucket '{bucket}': {e}", file=sys.stderr)
        print("[ERROR] Please verify the bucket exists and that your credentials have access.", file=sys.stderr)
        sys.exit(1)

    for local_rel_path, s3_filename in FILES_TO_UPLOAD:
        source_path = root_dir / local_rel_path
        s3_key = f"{prefix}/{s3_filename}" if prefix else s3_filename

        if not source_path.exists():
            print(f"[WARNING] Local file does not exist, skipping upload: {source_path}")
            continue

        print(f"[INFO] Uploading {source_path} -> s3://{bucket}/{s3_key} ...")
        
        # Define progress callback
        file_size = source_path.stat().st_size
        uploaded_bytes = 0

        def progress_callback(bytes_amount):
            nonlocal uploaded_bytes
            uploaded_bytes += bytes_amount
            percent = (uploaded_bytes / file_size) * 100
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({uploaded_bytes}/{file_size} bytes)")
            sys.stdout.flush()

        try:
            s3.upload_file(
                Filename=str(source_path),
                Bucket=bucket,
                Key=s3_key,
                Callback=progress_callback
            )
            print(f"\n[INFO] Successfully uploaded: {s3_filename}\n")
        except Exception as e:
            print(f"\n[ERROR] Failed to upload {s3_filename}: {e}", file=sys.stderr)
            sys.exit(1)

    print("[INFO] All assets successfully uploaded to S3.")

if __name__ == "__main__":
    upload_assets()
