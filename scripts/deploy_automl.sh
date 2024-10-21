#!/bin/bash
# deploy_automl.sh (Deployment Script for AutoML)

set -euo pipefail
IFS=$'
	'

echo "Deploying AutoML..."

kubectl apply -f deployment/k8s/automl_deployment.yaml

echo "AutoML Deployment Complete."

