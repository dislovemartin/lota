#!/bin/bash
# deploy_cpp_anomaly_detector.sh (Deployment Script for C++ Anomaly Detector)

set -euo pipefail
IFS=$'
	'

echo "Deploying C++ Anomaly Detector..."

kubectl apply -f deployment/k8s/cpp_module_deployment.yaml

echo "C++ Anomaly Detector Deployment Complete."

