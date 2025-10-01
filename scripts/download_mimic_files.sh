#!/bin/bash

# Config
BASE_URL="https://physionet.org/files/mimiciii/1.4"
FILE_LIST="scripts/mimic_files.txt"
TARGET_DIR="data/mimic-iii-dataset"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Loop through each file and download it
while IFS= read -r FILE; do
  # Username snd password must be in .netrc file
  # Check if .netrc file exists
  if [[ ! -f ~/.netrc ]]; then
    echo "Error: .netrc file not found. Please create it with your PhysioNet credentials."
    exit 1
  fi

    # Determine full and uncompressed file paths
  COMPRESSED_PATH="$TARGET_DIR/$FILE"
  UNCOMPRESSED_PATH="${COMPRESSED_PATH%.gz}"

  # Check if either version exists
  if [[ -f "$COMPRESSED_PATH" || -f "$UNCOMPRESSED_PATH" ]]; then
    echo "$FILE or its uncompressed version already exists. Skipping download."
    continue
  fi

  echo "Downloading $FILE..."
  # Download file to the target directory
  wget --no-verbose -P "$TARGET_DIR" "$BASE_URL/$FILE"

  # If it's a .gz file, unzip it in place
  if [[ "$FILE" == *.gz ]]; then
    echo "Unzipping $FILE..."
    gunzip -f "$TARGET_DIR/$FILE"
  fi

done < "$FILE_LIST"
