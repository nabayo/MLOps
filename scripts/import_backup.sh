#!/bin/bash
# Import MLflow data from a backup zip file using Docker

set -e

if [ -z "$1" ]; then
    echo "❌ Error: Backup file not specified"
    echo ""
    echo "Usage:"
    echo "  ./scripts/docker_import.sh <backup_file.zip> [--dry-run] [--overwrite]"
    echo ""
    echo "Examples:"
    echo "  ./scripts/docker_import.sh my_backup.zip"
    echo "  ./scripts/docker_import.sh my_backup.zip --dry-run"
    echo "  ./scripts/docker_import.sh my_backup.zip --overwrite"
    echo ""
    exit 1
fi

BACKUP_FILE="$1"
shift  # Remove first argument

# Handle potential path input (e.g. "backups/file.zip" vs "file.zip")
BASENAME=$(basename "$BACKUP_FILE")

# Check if file exists in backups/ directory
if [ -f "backups/$BASENAME" ]; then
    # File exists in correct location
    TARGET_FILE="$BASENAME"
elif [ -f "$BACKUP_FILE" ]; then
    # File exists but might be elsewhere or user gave full path
    # Check if it's actually inside backups/
    REALPATH=$(realpath "$BACKUP_FILE")
    BACKUPS_DIR=$(realpath "backups")
    
    if [[ "$REALPATH" == "$BACKUPS_DIR"* ]]; then
        TARGET_FILE="$BASENAME"
    else
        echo "❌ Error: Backup file must be inside the 'backups/' directory."
        echo "   Docker container cannot verify files outside this directory."
        echo "   Please move '$BACKUP_FILE' to 'backups/' and try again."
        exit 1
    fi
else
    echo "❌ Error: Backup file not found: backups/$BASENAME"
    echo ""
    echo "Available backups:"
    ls -1 backups/*.zip 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "🔄 Importing MLflow data from: $TARGET_FILE"
echo ""

# Run import with remaining arguments
export BACKUP_FILE="$TARGET_FILE"

# Use docker compose (modern) with BuildKit enabled
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose run --rm \
    -e BACKUP_FILE="$BACKUP_FILE" \
    import python scripts/import_mlflow.py /app/backups/"$BACKUP_FILE" "$@"

echo ""
echo "✅ Import complete!"
