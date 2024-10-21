#!/bin/bash
# setup_env.sh (Environment Setup Script)

set -euo pipefail
IFS=$'
	'

echo "Setting up environment variables..."

# Load production environment variables from prod.yaml
export DATABASE_URL=
export OPENAI_API_KEY=
export KAFKA_BROKER=

echo "Environment variables set successfully."

