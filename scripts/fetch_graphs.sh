#!/usr/bin/env bash
set -euo pipefail

# Create data directory if missing
mkdir -p data

# Pair: URL and local gz filename
URLS=(
  "https://snap.stanford.edu/data/roadNet-CA.txt.gz"
  "https://snap.stanford.edu/data/web-Google.txt.gz"
  "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz"
  "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz"
  "https://snap.stanford.edu/data/roadNet-PA.txt.gz"
  "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz"
  "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz"
)

for url in "${URLS[@]}"; do
  gz_file=$(basename "$url")
  txt_file="${gz_file%.gz}"
  
  if [[ -f "data/$txt_file" ]]; then
    echo "[fetch] data/$txt_file already present, skipping."
    continue
  fi

  echo "[fetch] Downloading $url ..."
  if ! curl --fail --retry 3 --retry-delay 2 -L -o "data/$gz_file" "$url"; then
    echo "ERROR: Failed to download $gz_file from $url" >&2
    exit 1
  fi

  echo "[fetch] Uncompressing data/$gz_file ..."
  gzip -d -f "data/$gz_file"
  echo "[fetch] Ready: data/$txt_file"
done

echo "[fetch] All SNAP graph files are present in data/"
