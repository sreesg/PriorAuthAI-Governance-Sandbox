#!/bin/bash
# Run this from inside the VPC (on VPN) once the EKS endpoint update completes
# This applies the service account and restarts the pod with S3 access

set -e

echo "Applying IRSA service account..."
kubectl apply -f k8s/serviceaccount.yaml

echo "Applying updated deployment with serviceAccountName..."
kubectl apply -f k8s/deployment.yaml

echo "Restarting pods to pick up new service account..."
kubectl rollout restart deployment/beacon-priorauth -n beacon

echo "Waiting for rollout..."
kubectl rollout status deployment/beacon-priorauth -n beacon --timeout=300s

echo "Verifying S3 access..."
sleep 30
kubectl exec -n beacon deployment/beacon-priorauth -c priorauth-app -- python3 -c "
from s3_helper import get_file_bytes
data = get_file_bytes('PA Agentic architecture.png')
print(f'S3 access OK: {len(data)} bytes')
"

echo "✅ Done! S3-served assets should now work at https://beacon.sbxaws.medeanalytics.zone"
