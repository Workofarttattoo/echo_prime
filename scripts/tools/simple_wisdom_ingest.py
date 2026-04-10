#!/usr/bin/env python3
"""
Simple Wisdom Ingestion Script

Directly copies wisdom files from external drives to the research_drop directory
without requiring full AGI initialization. This allows basic ingestion while
the AGI system is being debugged.
"""

import os
import sys
import shutil
import time
from pathlib import Path

def ensure_directories():
    """Ensure necessary directories exist"""
    dirs = [
        'research_drop',
        'research_drop/pdfs',
        'research_drop/json',
        'research_drop/logs'
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Ensured directory: {dir_path}")

def scan_for_drives():
    """Scan for mounted external drives"""
    volumes_path = "/Volumes"
    base_volumes = {'Macintosh HD', '.timemachine', 'com.apple.TimeMachine.localsnapshots'}

    try:
        all_volumes = set(os.listdir(volumes_path))
        new_volumes = all_volumes - base_volumes

        return [os.path.join(volumes_path, vol) for vol in new_volumes if os.path.exists(os.path.join(volumes_path, vol))]
    except Exception as e:
        print(f"❌ Error scanning volumes: {e}")
        return []

def count_wisdom_files(drive_path):
    """Count wisdom files on a drive"""
    pdf_count = 0
    json_count = 0

    for root, dirs, files in os.walk(drive_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_count += 1
            elif file.lower().endswith('.json'):
                json_count += 1

    return pdf_count, json_count

def ingest_wisdom_files(drive_path, max_files=100):
    """Ingest wisdom files from drive to research_drop"""
    print(f"🧠 Starting wisdom ingestion from: {drive_path}")

    ingested_count = 0
    pdf_count = 0
    json_count = 0

    start_time = time.time()

    for root, dirs, files in os.walk(drive_path):
        for file in files:
            if ingested_count >= max_files:
                print(f"⚠️ Reached max files limit: {max_files}")
                break

            file_path = os.path.join(root, file)
            file_name = os.path.basename(file_path)

            # Determine destination based on file type
            if file.lower().endswith('.pdf'):
                dest_dir = 'research_drop/pdfs'
                pdf_count += 1
            elif file.lower().endswith('.json'):
                dest_dir = 'research_drop/json'
                json_count += 1
            else:
                continue  # Skip non-wisdom files

            # Copy file
            dest_path = os.path.join(dest_dir, file_name)

            try:
                shutil.copy2(file_path, dest_path)
                ingested_count += 1

                if ingested_count <= 5:  # Show first 5 files
                    print(f"📄 Ingested: {file_name}")

            except Exception as e:
                print(f"❌ Failed to copy {file_name}: {e}")

        if ingested_count >= max_files:
            break

    elapsed = time.time() - start_time

    print(f"✅ Ingestion complete in {elapsed:.2f}s")
    print(f"📊 Files ingested: {ingested_count}")
    print(f"📚 PDFs: {pdf_count}")
    print(f"📋 JSONs: {json_count}")

    return ingested_count, pdf_count, json_count

def log_ingestion(drive_name, stats):
    """Log the ingestion results"""
    log_file = 'research_drop/logs/ingestion_log.txt'

    with open(log_file, 'a') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n[{timestamp}] Drive: {drive_name}\n")
        f.write(f"  Files: {stats[0]}, PDFs: {stats[1]}, JSONs: {stats[2]}\n")

    print(f"📝 Ingestion logged to: {log_file}")

def main():
    print("🧠 ECH0-PRIME Simple Wisdom Ingestion")
    print("=" * 40)

    # Ensure directories exist
    ensure_directories()

    # Scan for drives
    drives = scan_for_drives()

    if not drives:
        print("❌ No external drives detected")
        print("💡 Please ensure your external drive is mounted")
        return

    print(f"🎯 Found {len(drives)} external drive(s)")

    total_ingested = 0

    for drive_path in drives:
        drive_name = os.path.basename(drive_path)
        print(f"\n🔍 Processing drive: {drive_name}")
        print(f"📂 Path: {drive_path}")

        # Count wisdom files
        pdf_count, json_count = count_wisdom_files(drive_path)
        print(f"📊 Wisdom files found: {pdf_count} PDFs, {json_count} JSONs")

        if pdf_count + json_count == 0:
            print("⚠️ No wisdom files found on this drive")
            continue

        # Ingest files
        stats = ingest_wisdom_files(drive_path, max_files=1000)  # Allow up to 1000 files
        total_ingested += stats[0]

        # Log results
        log_ingestion(drive_name, stats)

    if total_ingested > 0:
        print(f"\n🎉 SUCCESS! Ingested {total_ingested} wisdom files")
        print("📁 Files stored in: research_drop/")
        print("🧠 Ready for AGI processing when system is online")
    else:
        print("\n❌ No wisdom files were ingested")

if __name__ == "__main__":
    main()
