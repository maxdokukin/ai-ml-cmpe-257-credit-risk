#!/usr/bin/env bash
# keep_empty_dirs.sh — run from repo root

set -euo pipefail

# Create .gitkeep in every empty dir except .git and .venv
find . -type d -empty \
  -not -path './.git*' \
  -not -path './.venv*' \
  -print0 | while IFS= read -r -d '' d; do
    : > "$d/.gitkeep"
    echo "keep $d"
  done

# Stage only the .gitkeep files we just created; commit only if there are changes
find . -type f -name '.gitkeep' \
  -not -path './.git/*' \
  -not -path './.idea/*' \
  -not -path './.venv/*' -print0 |
  while IFS= read -r -d '' f; do git add "$f"; done

if ! git diff --cached --quiet; then
  git commit -m "chore: keep empty dirs (exclude .venv)"
else
  echo "nothing to commit"
fi
