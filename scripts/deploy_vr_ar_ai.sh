#!/bin/bash
# deploy_vr_ar_ai.sh (Deployment Script for VR/AR AI)

set -euo pipefail
IFS=$'
	'

echo "Deploying VR/AR AI..."

kubectl apply -f deployment/k8s/vr_ar_ai_deployment.yaml

echo "VR/AR AI Deployment Complete."

