#!/bin/bash
# deploy_personalization.sh (Deployment Script for Personalization Engine)

set -euo pipefail
IFS=$'
	'

echo "Deploying Personalization Engine..."

kubectl apply -f deployment/k8s/personalization_engine_deployment.yaml

echo "Personalization Engine Deployment Complete."

