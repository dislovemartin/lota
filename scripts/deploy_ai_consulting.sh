#!/bin/bash
# deploy_ai_consulting.sh (Deployment Script for AI Consulting)

set -euo pipefail
IFS=$'
	'

echo "Deploying AI Consulting..."

kubectl apply -f deployment/k8s/ai_consulting_deployment.yaml

echo "AI Consulting Deployment Complete."

