#!/bin/bash
# deploy_content_creation_ai.sh (Deployment Script for Content Creation AI)

set -euo pipefail
IFS=$'
	'

echo "Deploying Content Creation AI..."

kubectl apply -f deployment/k8s/content_creation_ai_deployment.yaml

echo "Content Creation AI Deployment Complete."

