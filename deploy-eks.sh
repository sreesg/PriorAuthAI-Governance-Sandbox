#!/bin/bash
# Deploy Beacon PriorAuthAI to EKS
# Prerequisites: aws cli, kubectl, eksctl, docker configured
set -e

REGION="us-west-2"
ACCOUNT_ID="747675253087"
CLUSTER="sandbox-mededatalake-cluster"
ECR_REPO="beacon-priorauth"
S3_BUCKET="beacon-priorauthai-assets"
NAMESPACE="beacon"
IMAGE_TAG="latest"

echo "═══════════════════════════════════════════"
echo "  Beacon PriorAuthAI — EKS Deployment"
echo "═══════════════════════════════════════════"

# ─── Step 1: Create S3 bucket for static assets ───
echo ""
echo "▶ Step 1: Creating S3 bucket for assets..."
aws s3api create-bucket \
  --bucket $S3_BUCKET \
  --region $REGION \
  --create-bucket-configuration LocationConstraint=$REGION \
  2>/dev/null || echo "  Bucket already exists"

# Upload static assets to S3
echo "  Uploading PDFs, video, and images to S3..."
aws s3 sync ./cases/ s3://$S3_BUCKET/cases/ --region $REGION
aws s3 sync ./policies/ s3://$S3_BUCKET/policies/ --include "*.pdf" --region $REGION
aws s3 cp "./PA agent.mp4" "s3://$S3_BUCKET/PA agent.mp4" --region $REGION
aws s3 cp "./PA Agentic architecture.png" "s3://$S3_BUCKET/PA Agentic architecture.png" --region $REGION
aws s3 cp ./real_payer_policy_uhc.pdf s3://$S3_BUCKET/real_payer_policy_uhc.pdf --region $REGION
aws s3 cp ./medical_necessity_rules.pdf s3://$S3_BUCKET/medical_necessity_rules.pdf --region $REGION
echo "  ✓ Assets uploaded"

# ─── Step 2: Create ECR repository ───
echo ""
echo "▶ Step 2: Creating ECR repository..."
aws ecr create-repository \
  --repository-name $ECR_REPO \
  --region $REGION \
  2>/dev/null || echo "  Repository already exists"

# ─── Step 3: Build and push Docker image ───
echo ""
echo "▶ Step 3: Building and pushing Docker image..."
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

docker build --platform linux/amd64 -t $ECR_REPO:$IMAGE_TAG .
docker tag $ECR_REPO:$IMAGE_TAG $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
echo "  ✓ Image pushed to ECR"

# ─── Step 4: Update kubeconfig ───
echo ""
echo "▶ Step 4: Configuring kubectl..."
aws eks update-kubeconfig --name $CLUSTER --region $REGION
echo "  ✓ kubectl configured for $CLUSTER"

# ─── Step 5: Create node group (if not exists) ───
echo ""
echo "▶ Step 5: Creating beacon-llm node group (m5.2xlarge, 32GB)..."
eksctl create nodegroup \
  --cluster $CLUSTER \
  --region $REGION \
  --name beacon-llm \
  --node-type m5.2xlarge \
  --nodes 1 \
  --nodes-min 1 \
  --nodes-max 2 \
  --node-volume-size 50 \
  --node-labels "node-group=beacon-llm" \
  2>/dev/null || echo "  Node group already exists"

echo "  Waiting for node to be ready..."
kubectl wait --for=condition=Ready nodes -l node-group=beacon-llm --timeout=300s 2>/dev/null || true

# ─── Step 6: Deploy to Kubernetes ───
echo ""
echo "▶ Step 6: Deploying to Kubernetes..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

echo "  Waiting for pods to be ready..."
kubectl rollout status deployment/beacon-priorauth -n $NAMESPACE --timeout=300s

# ─── Step 7: Create Route 53 record ───
echo ""
echo "▶ Step 7: Creating Route 53 DNS record..."
HOSTED_ZONE_ID="Z02419242UVTP3JVANNPO"
ALB_DNS=$(kubectl get ingress beacon-priorauth-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)

if [ -n "$ALB_DNS" ]; then
  aws route53 change-resource-record-sets \
    --hosted-zone-id $HOSTED_ZONE_ID \
    --change-batch '{
      "Changes": [{
        "Action": "UPSERT",
        "ResourceRecordSet": {
          "Name": "beacon.sbxaws.medeanalytics.zone",
          "Type": "CNAME",
          "TTL": 300,
          "ResourceRecords": [{"Value": "'"$ALB_DNS"'"}]
        }
      }]
    }' --region $REGION
  echo "  ✓ DNS record created: beacon.sbxaws.medeanalytics.zone → $ALB_DNS"
else
  echo "  ⚠ ALB not ready yet. Run this after ingress is provisioned:"
  echo "    kubectl get ingress -n $NAMESPACE"
fi

# ─── Done ───
echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Deployment complete!"
echo ""
echo "  URL: https://beacon.sbxaws.medeanalytics.zone"
echo "  (accessible via VPN / private network)"
echo ""
echo "  Useful commands:"
echo "    kubectl get pods -n $NAMESPACE"
echo "    kubectl logs -f deployment/beacon-priorauth -n $NAMESPACE -c priorauth-app"
echo "    kubectl logs -f deployment/beacon-priorauth -n $NAMESPACE -c ollama"
echo "═══════════════════════════════════════════"
