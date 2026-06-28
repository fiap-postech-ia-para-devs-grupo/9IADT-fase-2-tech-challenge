#!/usr/bin/env bash

set -e

bash .devcontainer/scripts/configure-git.sh

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m ipykernel install --user --name fiap-tech-challenge --display-name "Python (fiap-tech-challenge)"

# Create .claude/skills symlink pointing to .agents/skills
mkdir -p /workspaces/9IADT-tech-challenge/.agents/skills
mkdir -p /workspaces/9IADT-tech-challenge/.claude
ln -sf /workspaces/9IADT-tech-challenge/.agents/skills /workspaces/9IADT-tech-challenge/.claude/skills