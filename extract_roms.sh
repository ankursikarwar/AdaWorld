#!/bin/bash
# Extracts No-Intro ROM archives (which are zip-of-zips) into flat ROM files
# Usage: bash extract_roms.sh

ROM_DIR=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/roms/No-Intro-Collection_2016-01-03_Fixed
EXTRACT_DIR=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/roms/extracted

mkdir -p $EXTRACT_DIR

ARCHIVES=(
    "Nintendo - Game Boy.zip"
    "Nintendo - Nintendo Entertainment System.zip"
    "Nintendo - Super Nintendo Entertainment System.zip"
    "Sega - Master System - Mark III.zip"
    "Sega - Mega Drive - Genesis.zip"
)

for archive in "${ARCHIVES[@]}"; do
    echo "=== Extracting outer archive: $archive ==="
    SYSTEM_DIR="$EXTRACT_DIR/$(basename "$archive" .zip)"
    mkdir -p "$SYSTEM_DIR"

    # Extract outer zip (contains one .zip per ROM)
    unzip -o -q "$ROM_DIR/$archive" -d "$SYSTEM_DIR"

    echo "=== Extracting inner ROM zips for: $archive ==="
    # Each inner zip contains the actual ROM file — extract in place
    find "$SYSTEM_DIR" -name "*.zip" | while read inner_zip; do
        unzip -o -q "$inner_zip" -d "$SYSTEM_DIR" && rm "$inner_zip"
    done

    echo "Done: $archive"
done

echo "=== All ROMs extracted to $EXTRACT_DIR ==="
find $EXTRACT_DIR -type f ! -name "*.zip" | wc -l
echo "ROM files total"
