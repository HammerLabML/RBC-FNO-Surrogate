#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_dir>"
    exit 1
fi

DATASET_INPUT="$1"

# Directory where the script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve dataset path (portable: macOS + Linux)
DATASET="$(cd "$DATASET_INPUT" 2>/dev/null && pwd)"

if [ -z "$DATASET" ] || [ ! -d "$DATASET" ]; then
    echo "Error: Dataset directory does not exist: $DATASET_INPUT"
    exit 1
fi

# Use the dataset folder name as link name
LINK_NAME="$(basename "$DATASET")"
LINK_PATH="$SCRIPT_DIR/$LINK_NAME"

if [ -e "$LINK_PATH" ] || [ -L "$LINK_PATH" ]; then
    echo "Error: Link already exists: $LINK_PATH"
    exit 1
fi

ln -s "$DATASET" "$LINK_PATH"

echo "Created symlink:"
echo "  $LINK_PATH -> $DATASET"