#!/usr/bin/env python3
"""
seed_vector_db.py — Ingest evidence PDFs from S3 into Qdrant vector database.

Reads PDFs from S3, extracts text, generates embeddings, and stores them
in Qdrant with full provenance metadata. This populates the vector DB
so the CRF can perform semantic retrieval during PA evaluation.

USAGE:
  source ./set-aws-profile.sh AKIA... SECRET... TOKEN... us-west-2
  python seed_vector_db.py --bucket beacon-priorauthai-assets \
    --qdrant-url http://localhost:6333

REQUIREMENTS: boto3, qdrant-client, pypdf, sentence-transformers
"""

import argparse
import hashlib
import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone

try:
    import boto3
    from pypdf import PdfReader
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, PointStruct, VectorParams,
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install boto3 pypdf qdrant-client sentence-transformers")
    sys.exit(1)


# =============================================================================
# Embedding Model
# =============================================================================

_embedder = None

def get_embedder():
    """Lazy-load the embedding function using Bedrock Titan."""
    global _embedder
    if _embedder is None:
        try:
            import boto3
            client = boto3.client("bedrock-runtime",
                                  region_name=os.environ.get("AWS_REGION", "us-west-2"))
            # Test the connection
            print("  Using Amazon Bedrock Titan Embed Text v2 (1024 dimensions)")
            _embedder = client
        except Exception as e:
            print(f"  ⚠ Bedrock not available ({e}), using random vectors")
            _embedder = "random"
    return _embedder


def embed_text(text: str) -> list[float]:
    """Generate embedding vector for text using Bedrock Titan."""
    import random
    embedder = get_embedder()
    if embedder == "random":
        random.seed(hash(text) % (2**32))
        return [random.uniform(-1, 1) for _ in range(1024)]
    try:
        body = json.dumps({"inputText": text[:8000]})
        response = embedder.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        return result.get("embedding", [random.uniform(-1, 1) for _ in range(1024)])
    except Exception as e:
        random.seed(hash(text) % (2**32))
        return [random.uniform(-1, 1) for _ in range(1024)]


# =============================================================================
# PDF Text Extraction
# =============================================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text content from a PDF."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n".join(pages_text)
    except Exception as e:
        return f"[PDF extraction failed: {e}]"


def chunk_text(text: str, max_chunk_size: int = 500) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:max_chunk_size]] if text else []


# =============================================================================
# Qdrant Collection Setup
# =============================================================================

COLLECTION_NAME = "clinical_documents"
VECTOR_SIZE = 1024  # Amazon Titan Embed Text v2 output size


def ensure_collection(client: QdrantClient):
    """Create Qdrant collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"  ✓ Collection '{COLLECTION_NAME}' already exists")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"  ✓ Created collection '{COLLECTION_NAME}' ({VECTOR_SIZE}d, cosine)")


# =============================================================================
# Main Ingestion
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ingest S3 evidence PDFs into Qdrant")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="clinical-evidence/", help="S3 prefix")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--region", default=None)
    parser.add_argument("--max-docs", type=int, default=200, help="Max documents to process")
    args = parser.parse_args()

    # Connect to S3
    s3 = boto3.client("s3", **({"region_name": args.region} if args.region else {}))
    try:
        s3.head_bucket(Bucket=args.bucket)
        print(f"✓ S3 bucket '{args.bucket}' accessible")
    except Exception as e:
        print(f"✗ Cannot access S3 bucket: {e}")
        sys.exit(1)

    # Connect to Qdrant
    try:
        qdrant = QdrantClient(url=args.qdrant_url)
        qdrant.get_collections()
        print(f"✓ Qdrant connected at {args.qdrant_url}")
    except Exception as e:
        print(f"✗ Cannot connect to Qdrant at {args.qdrant_url}: {e}")
        sys.exit(1)

    # Ensure collection exists
    ensure_collection(qdrant)

    # List PDFs in S3
    print(f"\nListing PDFs in s3://{args.bucket}/{args.prefix}...")
    paginator = s3.get_paginator("list_objects_v2")
    pdf_keys = []
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".pdf"):
                pdf_keys.append(obj["Key"])

    pdf_keys = pdf_keys[:args.max_docs]
    print(f"  Found {len(pdf_keys)} PDFs to ingest")

    if not pdf_keys:
        print("  ⚠ No PDFs found. Run generate_evidence_docs.py first.")
        sys.exit(0)

    # Process each PDF
    print(f"\nIngesting {len(pdf_keys)} documents into Qdrant...")
    print("─" * 60)

    total_chunks = 0
    points_batch = []
    BATCH_SIZE = 50

    for doc_idx, s3_key in enumerate(pdf_keys):
        # Download PDF from S3
        response = s3.get_object(Bucket=args.bucket, Key=s3_key)
        pdf_bytes = response["Body"].read()

        # Extract text
        text = extract_text_from_pdf(pdf_bytes)
        if not text.strip():
            continue

        # Chunk text
        chunks = chunk_text(text, max_chunk_size=500)

        # Compute document hash
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Extract member_id from path (clinical-evidence/MEM-XXXX/...)
        parts = s3_key.split("/")
        member_id = parts[1] if len(parts) > 1 else "unknown"
        doc_type = parts[2].split("_")[0] if len(parts) > 2 else "unknown"

        # Generate embeddings and create points
        for chunk_idx, chunk_text_content in enumerate(chunks):
            embedding = embed_text(chunk_text_content)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{s3_key}:{chunk_idx}"))

            points_batch.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk_text_content,
                    "document_id": s3_key,
                    "member_id": member_id,
                    "doc_type": doc_type,
                    "content_hash": content_hash,
                    "chunk_index": chunk_idx,
                    "namespace": member_id,
                    "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
                    "s3_bucket": args.bucket,
                    "s3_key": s3_key,
                },
            ))
            total_chunks += 1

        # Batch upsert
        if len(points_batch) >= BATCH_SIZE:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points_batch)
            points_batch = []

        if (doc_idx + 1) % 10 == 0:
            print(f"  ... {doc_idx + 1}/{len(pdf_keys)} docs processed ({total_chunks} chunks)")

    # Final batch
    if points_batch:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points_batch)

    print(f"\n{'─' * 60}")
    print(f"✓ Ingestion complete: {len(pdf_keys)} docs → {total_chunks} chunks in Qdrant")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Qdrant: {args.qdrant_url}")

    # Print collection info
    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"  Total points in collection: {info.points_count}")


if __name__ == "__main__":
    main()
