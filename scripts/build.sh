#!/bin/bash
# build.sh (Build Orchestration Script)

set -euo pipefail
IFS=$'
	'

echo "Starting the build process..."

# Source environment variables
bash scripts/setup_env.sh

# Install dependencies and build projects
bash scripts/install_dependencies.sh

echo "Build process completed successfully."

