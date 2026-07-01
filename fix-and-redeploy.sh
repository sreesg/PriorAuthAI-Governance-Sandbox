#!/bin/zsh
# Fix all issues and redeploy: Neo4j, Qdrant persistence, re-seed, app image
set -e

REGION="us-west-2"
ACCOUNT_ID="747675253087"
ECR_REPO="beacon-priorauth"
NAMESPACE="beacon"
S3_BUCKET="beacon-priorauthai-assets"

echo "═══════════════════════════════════════════"
echo "  Fix & Redeploy — All Services"
echo "═══════════════════════════════════════════"

# ─── Step 1: Fix Neo4j (delete crashing pod, apply fixed manifest) ───
echo ""
echo "▶ Step 1: Fixing Neo4j..."
kubectl delete deployment neo4j -n $NAMESPACE 2>/dev/null || true
kubectl apply -f k8s/neo4j.yaml
echo "  ✓ Neo4j manifest applied (waiting for startup...)"

# ─── Step 2: Fix Qdrant (add persistent storage) ───
echo ""
echo "▶ Step 2: Applying Qdrant PVC + updated manifest..."
kubectl apply -f k8s/qdrant-pvc.yaml
kubectl delete deployment qdrant -n $NAMESPACE 2>/dev/null || true
kubectl apply -f k8s/qdrant.yaml
echo "  ✓ Qdrant with persistent storage applied"

# ─── Step 3: Build and push new app image ───
echo ""
echo "▶ Step 3: Building Docker image..."
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker build --platform linux/amd64 -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${ECR_REPO}:latest .
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${ECR_REPO}:latest
echo "  ✓ Image pushed"

# ─── Step 4: Deploy app ───
echo ""
echo "▶ Step 4: Deploying app..."
kubectl apply -f k8s/deployment.yaml
kubectl rollout restart deployment/beacon-priorauth -n $NAMESPACE
echo "  Waiting for all pods..."
kubectl rollout status deployment/qdrant -n $NAMESPACE --timeout=120s || true
kubectl rollout status deployment/beacon-priorauth -n $NAMESPACE --timeout=180s

# ─── Step 5: Wait for Neo4j to be ready ───
echo ""
echo "▶ Step 5: Waiting for Neo4j (90s startup)..."
sleep 30
kubectl rollout status deployment/neo4j -n $NAMESPACE --timeout=120s || echo "  ⚠ Neo4j still starting"

# ─── Step 6: Seed Qdrant (with Bedrock Titan embeddings) ───
echo ""
echo "▶ Step 6: Seeding Qdrant with Bedrock Titan embeddings..."
kubectl port-forward -n $NAMESPACE svc/qdrant 6333:6333 &
PF1=$!
sleep 5

python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='http://localhost:6333')
try:
    c.delete_collection('clinical_documents')
    print('  Old collection deleted')
except: pass
" 2>/dev/null || true

python seed_vector_db.py \
  --bucket $S3_BUCKET \
  --prefix clinical-evidence/ \
  --qdrant-url http://localhost:6333 \
  --region $REGION || echo "  ⚠ Qdrant seeding failed"

kill $PF1 2>/dev/null || true

# ─── Step 7: Seed Neo4j ───
echo ""
echo "▶ Step 7: Seeding Neo4j graph..."
kubectl port-forward -n $NAMESPACE svc/neo4j 7687:7687 &
PF2=$!
sleep 10

python seed_graph_db.py \
  --neo4j-uri bolt://localhost:7687 \
  --clear || echo "  ⚠ Neo4j seeding failed (may still be starting)"

kill $PF2 2>/dev/null || true

# ─── Done ───
echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ All services fixed and redeployed!"
echo ""
echo "  kubectl get pods -n $NAMESPACE"
echo "  URL: https://beacon.sbxaws.medeanalytics.zone"
echo "═══════════════════════════════════════════"
