# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel

# === 环境配置 ===
ENV TZ=Asia/Singapore \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH="/home/appuser/.local/bin:${PATH}"

# === 系统依赖 ===
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash cron tzdata sudo curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && echo "appuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# === 创建用户 + 目录 ===
RUN useradd -ms /bin/bash appuser \
    && mkdir -p /app/logs /app/models /app/data /app/backups \
    && chown -R appuser:appuser /app

WORKDIR /app
USER appuser

# === 第1步：先复制依赖（最大化缓存）===
COPY --chown=appuser:appuser requirements.txt .

# 升级 pip + 安装所有 Python 包（包括你原来的所有依赖）
RUN python3 -m pip install --user --upgrade pip setuptools wheel && \
    python3 -m pip install --user --no-cache-dir -r requirements.txt && \
    python3 -c "import pandas, torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# === 第2步：复制完整源码（包括 train_tcn.py）===
COPY --chown=appuser:appuser . /app

# === 权限设置 ===
RUN chmod +x /app/launch_all.sh \
             /app/self_learn_cron.sh \
             /app/ai_agent_cron.sh \
             /app/train_tcn.py 2>/dev/null || true

# === 验证 TCN 脚本存在 ===
RUN ls -la /app/train_tcn.py && echo "train_tcn.py 成功复制！"

EXPOSE 8050
CMD ["/bin/bash", "/app/launch_all.sh"]
