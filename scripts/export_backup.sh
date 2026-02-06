#!/bin/bash
# Export MLflow data to a backup zip file using Docker

set -e

# Default values
OUTPUT_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        *)
            OUTPUT_NAME="$1"
            shift
            ;;
    esac
done

echo "🔄 Exporting MLflow data via Docker..."
echo ""

# Run export
if [ -n "$OUTPUT_NAME" ]; then
    DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose run --rm \
        -e BACKUP_NAME="$OUTPUT_NAME" \
        export python scripts/export_mlflow.py --output-dir /app/backups --name "$OUTPUT_NAME"
else
    DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose run --rm export
fi

echo ""
echo "✅ Export complete! Backup saved in ./backups/"
ls -lh backups/*.zip | tail -1
