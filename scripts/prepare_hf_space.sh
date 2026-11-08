#!/usr/bin/env bash
# Copy exactly what the API container needs into a local clone of your HF Space
# and push it. Usage:
#   scripts/prepare_hf_space.sh <hf-username>/<space-name>
set -euo pipefail

SPACE="${1:?usage: scripts/prepare_hf_space.sh <hf-username>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/../hf-space-$(basename "$SPACE")"

if [ ! -d "$DEST/.git" ]; then
  git clone "https://huggingface.co/spaces/$SPACE" "$DEST"
fi

rsync -a --delete --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' \
  "$ROOT/src/" "$DEST/src/"
mkdir -p "$DEST/config"
cp "$ROOT/config/retrieval_config.json" "$DEST/config/"
cp "$ROOT/Dockerfile" "$ROOT/requirements-api.txt" "$DEST/"
cp "$ROOT/deploy/hf_space/README.md" "$DEST/README.md"

cd "$DEST"
git add -A
git commit -m "Deploy FinanceRAG API from $(git -C "$ROOT" rev-parse --short HEAD)" || echo "nothing to commit"
git push
echo "Pushed to https://huggingface.co/spaces/$SPACE — watch the build under the Space's 'Logs' tab."
