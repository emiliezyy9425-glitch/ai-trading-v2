# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel

ENV TZ=Asia/Singapore \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash cron tzdata sudo curl ca-certificates git python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user + directories
RUN useradd -ms /bin/bash appuser \
    && mkdir -p /app/logs /app/models /app/data /app/backups \
    && chown -R appuser:appuser /app

WORKDIR /app

# Switch to appuser EARLY and install everything as the final user
USER appuser

# Upgrade pip and install requirements as appuser
COPY --chown=appuser:appuser requirements.txt .
RUN python3 -m pip install --user --upgrade pip setuptools wheel \
    && python3 -m pip install --user --no-cache-dir -r requirements.txt

# Make sure .local/bin is in PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Verify pandas works
RUN python3 -c "import pandas as pd; print('pandas OK →', pd.__version__)"

# Copy source code
COPY --chown=appuser:appuser . /app

# Make scripts executable
RUN chmod +x /app/launch_all.sh /app/self_learn_cron.sh /app/ai_agent_cron.sh

EXPOSE 8050
CMD ["/bin/bash", "/app/launch_all.sh"]
