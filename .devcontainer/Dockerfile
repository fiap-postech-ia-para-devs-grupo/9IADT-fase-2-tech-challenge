FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV NVM_DIR=/home/vscode/.nvm

USER vscode
RUN curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash \
    && . "$NVM_DIR/nvm.sh" \
    && nvm install --lts \
    && nvm use --lts \
    && nvm alias default lts/*

USER root

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -eu' \
    'prompt=${1:-}' \
    'case "$prompt" in' \
    '  *Username*) printf "%s\n" "x-access-token" ;;' \
    '  *Password*) printf "%s\n" "${GITHUB_TOKEN:-}" ;;' \
    '  *) printf "\n" ;;' \
    'esac' \
    > /usr/local/bin/github-askpass \
    && chmod +x /usr/local/bin/github-askpass


WORKDIR /workspaces/9IADT-fase-2-tech-challenge
