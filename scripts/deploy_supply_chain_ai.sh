#!/bin/bash
# deploy_supply_chain_ai.sh (Deployment Script for Supply Chain AI)

set -euo pipefail
IFS=$'
	'

echo "Deploying Supply Chain AI..."

kubectl apply -f deployment/k8s/supply_chain_ai_deployment.yaml

echo "Supply Chain AI Deployment Complete."

