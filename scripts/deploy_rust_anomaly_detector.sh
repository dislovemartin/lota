#!/bin/bash
# deploy_rust_anomaly_detector.sh (Deployment Script for Rust Anomaly Detector)

set -euo pipefail
IFS=$'
	'

echo "Deploying Rust Anomaly Detector..."

kubectl apply -f deployment/k8s/rust_anomaly_detector_deployment.yaml

echo "Rust Anomaly Detector Deployment Complete."

