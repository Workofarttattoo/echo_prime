#!/usr/bin/env python3
"""
Stream the entire Materials Project summary dataset to a local JSONL (optionally gzipped) file.

Example:
  MP_API_KEY=... poetry run python scripts/download-mp-dataset.py \
    --output data/materials/mp-summary.jsonl.gz --chunk-size 1000
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Dict, Iterable, Optional

import requests

API_URL = "https://api.materialsproject.org/materials/summary"


def log_info(message: str) -> None:
  print(f"[info] {message}")


def log_warn(message: str) -> None:
  print(f"[warn] {message}")


def log_error(message: str) -> None:
  print(f"[error] {message}")


def repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def open_sink(path: Path) -> IO[str]:
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.suffix == ".gz":
    return gzip.open(path, "at", encoding="utf-8")
  return path.open("a", encoding="utf-8")


def send_chunk_request(
  api_key: str,
  chunk_index: int,
  chunk_size: int,
  extra_properties: Optional[Iterable[str]] = None,
  rate_limit_hz: Optional[float] = None,
) -> Dict[str, Any]:
  if rate_limit_hz and rate_limit_hz > 0:
    time.sleep(1.0 / rate_limit_hz)

  properties = [
    "material_id",
    "formula_pretty",
    "formula_anonymous",
    "density",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "volume",
    "nsites",
    "last_updated",
  ]
  if extra_properties:
    properties.extend(extra_properties)

  params = {
    "_fields": ",".join(properties),
    "_all_fields": "false",
    "_per_page": chunk_size,
    "_page": chunk_index,
  }
  response = requests.get(
    API_URL,
    headers={"X-API-KEY": api_key},
    params=params,
    timeout=120,
  )
  if response.status_code != 200:
    raise RuntimeError(
      f"Materials Project API error ({response.status_code}) on chunk {chunk_index}: {response.text}"
    )
  return response.json()


def download_dataset(
  api_key: str,
  output_path: Path,
  *,
  chunk_size: int,
  start_chunk: int,
  max_chunks: Optional[int],
  rate_limit_hz: Optional[float],
  extra_properties: Optional[Iterable[str]],
) -> Dict[str, Any]:
  stats = {
    "generatedAt": datetime.now(tz=timezone.utc).isoformat(),
    "output": str(output_path),
    "chunkSize": chunk_size,
    "startChunk": start_chunk,
    "chunksFetched": 0,
    "materialsWritten": 0,
  }

  with open_sink(output_path) as sink:
    chunk = start_chunk
    while True:
      if max_chunks is not None and stats["chunksFetched"] >= max_chunks:
        log_warn("Reached max-chunks limit; stopping download.")
        break
      log_info(f"Requesting chunk {chunk} (size={chunk_size})")
      data = send_chunk_request(
        api_key,
        chunk,
        chunk_size,
        extra_properties=extra_properties,
        rate_limit_hz=rate_limit_hz,
      )
      materials = data.get("data") or []
      if not materials:
        log_info("No more materials returned; download complete.")
        break

      for entry in materials:
        sink.write(json.dumps(entry))
        sink.write("\n")
      stats["materialsWritten"] += len(materials)
      stats["chunksFetched"] += 1
      log_info(
        f"Wrote {len(materials)} records from chunk {chunk} "
        f"(total={stats['materialsWritten']})"
      )
      chunk += 1

  return stats


def parse_args() -> argparse.Namespace:
  default_output = repo_root() / "qulab-infinite" / "data" / "materials" / "mp-summary.jsonl.gz"
  parser = argparse.ArgumentParser(
    description="Download the Materials Project summary dataset into a local JSONL file."
  )
  parser.add_argument(
    "output",
    nargs="?",
    default=str(default_output),
    help=f"Output file path (default: {default_output})",
  )
  parser.add_argument(
    "--chunk-size",
    dest="chunk_size",
    type=int,
    default=1000,
    help="Number of materials to request per chunk (default: %(default)s).",
  )
  parser.add_argument(
    "--start-chunk",
    dest="start_chunk",
    type=int,
    default=1,
    help="Chunk index to start from (default: 1).",
  )
  parser.add_argument(
    "--max-chunks",
    dest="max_chunks",
    type=int,
    default=None,
    help="Optional limit on the number of chunks to download.",
  )
  parser.add_argument(
    "--rate-limit-hz",
    dest="rate_limit_hz",
    type=float,
    default=5.0,
    help="Requests per second throttle (default: %(default)s).",
  )
  parser.add_argument(
    "--extra-property",
    dest="extra_properties",
    action="append",
    help="Additional Materials Project fields to include (can be repeated).",
  )
  parser.add_argument(
    "--mp-api-key",
    dest="mp_api_key",
    default=os.environ.get("MP_API_KEY"),
    help="Materials Project API key (falls back to MP_API_KEY env var).",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  if not args.mp_api_key:
    log_error("MP_API_KEY is required (pass --mp-api-key or export the env var).")
    return 1

  output_path = Path(args.output).expanduser().resolve()
  log_info(f"Streaming Materials Project data to {output_path}")
  stats = download_dataset(
    args.mp_api_key,
    output_path,
    chunk_size=args.chunk_size,
    start_chunk=args.start_chunk,
    max_chunks=args.max_chunks,
    rate_limit_hz=args.rate_limit_hz,
    extra_properties=args.extra_properties,
  )
  manifest_path = output_path.with_suffix(output_path.suffix + ".meta.json")
  manifest_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
  log_info(
    f"Completed download: {stats['materialsWritten']} materials across "
    f"{stats['chunksFetched']} chunk(s). Manifest written to {manifest_path}"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
