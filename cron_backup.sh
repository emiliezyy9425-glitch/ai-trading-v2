#!/bin/bash

# List of config files for each model
CONFIG_FILES=(
    "rf_config.json"
    "xgb_config.json"
    "lgb_config.json"
    "lstm_config.json"
    "ppo_config.json"
)

# Check if individual config files exist, otherwise use config.json
for CONFIG in "${CONFIG_FILES[@]}"; do
    if [ -f "/app/$CONFIG" ]; then
        echo "[INFO] Training with config $CONFIG..."
        python /app/retrain_models.py --config "/app/$CONFIG" >> /app/logs/cron.log 2>&1
    else
        echo "[WARNING] Config file $CONFIG not found, skipping..." >> /app/logs/cron.log
    fi
done

# Fallback to config.json if it exists and no individual configs are found
if [ ! -f "/app/rf_config.json" ] && [ -f "/app/config.json" ]; then
    echo "[INFO] No individual config files found, using config.json as fallback" >> /app/logs/cron.log
    python /app/retrain_models.py --config /app/config.json >> /app/logs/cron.log 2>&1
fi

# Run backup script
echo "[INFO] Running backup_models.sh..."
/app/backup_models.sh >> /app/logs/cron.log 2>&1