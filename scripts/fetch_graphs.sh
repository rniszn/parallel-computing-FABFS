#!/usr/bin/env bash
set -euo pipefail

# Create data directory if missing
mkdir -p data

FILES=(
  "roadNet-CA.txt.gz"
  "web-Google.txt.gz"
  "com-lj.ungraph.txt.gz"
  "com-orkut.ungraph.txt.gz"
  "roadNet-PA.txt.gz"
  "com-amazon.ungraph.txt.gz"
  "com-dblp.ungraph.txt.gz"
)

BASE_URL="https://snap.stanford.edu/data"

for gz_file in "${FILES[@]}"; do
  txt_file="${gz_file%.gz}"
  
  if [[ -f "data/$txt_file" ]]; then
    echo "[fetch] data/$txt_file already present, skipping."
    continue
  fi

  echo "[fetch] Downloading $BASE_URL/$gz_file ..."
  if ! curl --fail --retry 3 --retry-delay 2 -L -o "data/$gz_file" "$BASE_URL/$gz_file"; then
    echo "ERROR: Failed to download $gz_file from $BASE_URL" >&2
    exit 1
  fi

  echo "[fetch] Uncompressing data/$gz_file ..."
  gzip -d -f "data/$gz_file"
  echo "[fetch] Ready: data/$txt_file"
done

echo "[fetch] All SNAP graph files are present in data/"
