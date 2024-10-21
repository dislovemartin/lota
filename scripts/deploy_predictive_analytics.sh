#!/bin/bash
# deploy_predictive_analytics.sh (Deployment Script for Predictive Analytics)

set -euo pipefail
IFS=$'
	'

echo "Deploying Predictive Analytics..."

kubectl apply -f deployment/k8s/predictive_analytics_deployment.yaml

echo "Predictive Analytics Deployment Complete."

