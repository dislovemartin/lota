#!/bin/bash
# deploy.sh (General Deployment Script)

set -euo pipefail
IFS=$'
	'

echo "Starting Deployment of AI Platform..."

bash scripts/deploy_personalization.sh
bash scripts/deploy_rust_anomaly_detector.sh
bash scripts/deploy_predictive_analytics.sh
bash scripts/deploy_cpp_anomaly_detector.sh
bash scripts/deploy_content_creation_ai.sh
bash scripts/deploy_healthcare_ai.sh
bash scripts/deploy_ai_consulting.sh
bash scripts/deploy_vr_ar_ai.sh
bash scripts/deploy_supply_chain_ai.sh
bash scripts/deploy_automl.sh

echo "All Deployments Completed Successfully."

