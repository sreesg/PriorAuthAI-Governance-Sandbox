#!/bin/zsh
# Quick redeploy — rebuild app image and restart pod (skips infra/seeding)
set -e

REGION="us-west-2"
ACCOUNT_ID="747675253087"
ECR_REPO="beacon-priorauth"
NAMESPACE="beacon"
IMAGE_TAG="latest"
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"

echo "═══════════════════════════════════════════"
echo "  Quick Redeploy — App Image Only"
echo "═══════════════════════════════════════════"

echo ""
echo "▶ Step 1: ECR Login..."
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

echo ""
echo "▶ Step 2: Building Docker image..."
docker build --platform linux/amd64 -t $ECR_REPO:$IMAGE_TAG .

echo ""
echo "▶ Step 3: Pushing to ECR..."
docker tag $ECR_REPO:$IMAGE_TAG $ECR_URI
docker push $ECR_URI

echo ""
echo "▶ Step 4: Restarting pod..."
kubectl rollout restart deployment/beacon-priorauth -n $NAMESPACE
kubectl rollout status deployment/beacon-priorauth -n $NAMESPACE --timeout=120s

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Redeploy complete!"
echo "  URL: https://beacon.sbxaws.medeanalytics.zone"
echo "═══════════════════════════════════════════"
