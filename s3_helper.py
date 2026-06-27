"""
S3 Helper — Handles static asset storage for deployed environment.
Falls back to local filesystem when S3_BUCKET is not set (local dev).
"""

import os
import io
import boto3
from botocore.exceptions import ClientError

S3_BUCKET = os.environ.get("S3_BUCKET", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")

_s3_client = None

def _get_s3():
    global _s3_client
    if _s3_client is None and S3_BUCKET:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def is_s3_enabled():
    """Check if S3 storage is configured."""
    return bool(S3_BUCKET)


def get_file_bytes(path):
    """
    Read a file from S3 (if configured) or local filesystem.
    path: relative path like 'cases/case-lumbar-approve_bundle.pdf'
    Returns: bytes
    """
    if S3_BUCKET:
        try:
            s3 = _get_s3()
            response = s3.get_object(Bucket=S3_BUCKET, Key=path)
            return response["Body"].read()
        except ClientError as e:
            print(f"S3 read error for {path}: {e}")
            # Fall back to local
            pass

    # Local filesystem fallback
    local_path = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return f.read()
    return None


def get_file_text(path, encoding="utf-8"):
    """
    Read a text file from S3 or local filesystem.
    Returns: string or None
    """
    data = get_file_bytes(path)
    if data:
        return data.decode(encoding)
    return None


def file_exists(path):
    """Check if a file exists in S3 or locally."""
    if S3_BUCKET:
        try:
            s3 = _get_s3()
            s3.head_object(Bucket=S3_BUCKET, Key=path)
            return True
        except ClientError:
            pass

    local_path = os.path.join(os.path.dirname(__file__), path)
    return os.path.exists(local_path)


def get_presigned_url(path, expiration=3600):
    """
    Generate a presigned URL for S3 assets (PDFs, videos, images).
    For local dev, returns the relative path as-is.
    """
    if S3_BUCKET:
        try:
            s3 = _get_s3()
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": path},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            print(f"Presigned URL error for {path}: {e}")

    # Local: return relative path for direct serving
    return f"./{path}"


def list_files(prefix=""):
    """List files in S3 bucket under a prefix, or local directory."""
    if S3_BUCKET:
        try:
            s3 = _get_s3()
            response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except ClientError as e:
            print(f"S3 list error: {e}")

    # Local fallback
    local_dir = os.path.join(os.path.dirname(__file__), prefix)
    if os.path.isdir(local_dir):
        return [os.path.join(prefix, f) for f in os.listdir(local_dir)]
    return []
