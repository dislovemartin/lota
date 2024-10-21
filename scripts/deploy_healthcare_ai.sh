#!/bin/bash
# deploy_healthcare_ai.sh (Deployment Script for Healthcare AI)

set -euo pipefail
IFS=$'
	'

echo "Deploying Healthcare AI..."

kubectl apply -f deployment/k8s/healthcare_ai_deployment.yaml

echo "Healthcare AI Deployment Complete."

