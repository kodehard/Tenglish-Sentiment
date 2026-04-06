#!/bin/bash
# Download CMTET dataset from GitHub

set -e

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

URL="https://raw.githubusercontent.com/ksubbu199/cmtet-sentiment/master/codemix_sentiment_data.txt"
DEST="$RAW_DIR/codemix_sentiment_data.txt"

echo "Downloading CMTET dataset from GitHub..."
curl -L "$URL" -o "$DEST"

# Verify line count (expect ~19,857 sentences)
LINE_COUNT=$(wc -l < "$DEST")
echo "Downloaded: $DEST ($LINE_COUNT lines)"

if [ "$LINE_COUNT" -lt 10000 ]; then
    echo "ERROR: Line count ($LINE_COUNT) is lower than expected (~19,857). Download may have failed."
    exit 1
fi

echo "CMTET dataset downloaded successfully."
