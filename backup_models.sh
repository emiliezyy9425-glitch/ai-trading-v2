#!/bin/bash

shopt -s nullglob

# Ensure the backup directory exists
mkdir -p /app/backups

# Generate timestamp and archive name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="model_backup_$TIMESTAMP.tar.gz"

MODEL_FILES=(models/*.pkl models/*.keras models/*.h5 models/*.zip)

if [ ${#MODEL_FILES[@]} -eq 0 ]; then
  echo "[WARN] No model artifacts found to back up. Skipping archive creation."
else
  # Archive and save model files (including .pkl, .keras, .h5, and .zip)
  tar -czf /app/backups/$ARCHIVE_NAME -C /app "${MODEL_FILES[@]}"
  echo "[INFO] Backup created: $ARCHIVE_NAME"
fi

# Clean up backups older than 30 days
find /app/backups -name "model_backup_*.tar.gz" -mtime +30 -delete