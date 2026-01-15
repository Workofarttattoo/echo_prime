#!/usr/bin/env python3
"""
Download and validate extended material databases against the Materials Project API.

The script scans for `*.db.json` files, extracts Materials Project IDs, fetches
reference data, and compares overlapping properties (density, band gap, etc.)
within configurable tolerances. Results are written to
`logs/materials/mp-validation-report.json` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
  import requests
except ImportError as exc:  # pragma: no cover - handled at runtime
  raise SystemExit(
    "[error] The validate-extended-materials script requires the `requests` package. "
    "Install project dependencies with Poetry (`poetry install`) or `pip install requests`."
  ) from exc

try:
  import ijson
except ImportError:
  ijson = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATTERN = "*db.json"
MP_ID_PATTERN = re.compile(r"mp-\d+", re.IGNORECASE)
EV_TO_J = 1.602176634e-19
ANGSTROM3_TO_M3 = 1e-30
NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def log_info(message: str) -> None:
  print(f"[info] {message}")


def log_warn(message: str) -> None:
  print(f"[warn] {message}")


def log_error(message: str) -> None:
  print(f"[error] {message}")


def coerce_float(value: Any) -> Optional[float]:
  if value is None:
    return None
  if isinstance(value, (int, float)):
    if isinstance(value, bool):
      return float(value)
    return float(value)
  if isinstance(value, str):
    stripped = value.strip()
    if not stripped or stripped.lower() in {"na", "n/a", "null", "none"}:
      return None
    match = NUMERIC_PATTERN.search(stripped.replace(",", ""))
    if match:
      try:
        return float(match.group())
      except ValueError:
        return None
  return None


def coerce_bool(value: Any) -> Optional[bool]:
  if isinstance(value, bool):
    return value
  if isinstance(value, (int, float)):
    return bool(value)
  if isinstance(value, str):
    normalized = value.strip().lower()
    if not normalized:
      return None
    if normalized in {"true", "yes", "y", "stable", "pass", "1"}:
      return True
    if normalized in {"false", "no", "n", "unstable", "fail", "0"}:
      return False
  return None


def repo_relative(path: Path) -> str:
  try:
    return str(path.relative_to(REPO_ROOT))
  except ValueError:
    return str(path)


def extract_mp_id(entry: Dict[str, Any]) -> Optional[str]:
  candidate_keys = (
    "mp_id",
    "mpId",
    "material_id",
    "materialsProjectId",
    "materials_project_id",
    "id",
  )
  for key in candidate_keys:
    value = entry.get(key)
    if isinstance(value, str):
      match = MP_ID_PATTERN.search(value)
      if match:
        return match.group(0).lower()
  provenance = entry.get("provenance", {})
  if isinstance(provenance, dict):
    for prov_key in ("url", "link"):
      prov_value = provenance.get(prov_key)
      if isinstance(prov_value, str):
        match = MP_ID_PATTERN.search(prov_value)
        if match:
          return match.group(0).lower()
  references = entry.get("references")
  if isinstance(references, list):
    for reference in references:
      if isinstance(reference, dict):
        for ref_key in ("url", "link", "source", "notes"):
          ref_val = reference.get(ref_key)
          if isinstance(ref_val, str):
            match = MP_ID_PATTERN.search(ref_val)
            if match:
              return match.group(0).lower()
      elif isinstance(reference, str):
        match = MP_ID_PATTERN.search(reference)
        if match:
          return match.group(0).lower()
  description = entry.get("description")
  if isinstance(description, str):
    match = MP_ID_PATTERN.search(description)
    if match:
      return match.group(0).lower()
  return None


STREAMING_UNAVAILABLE_WARNING_EMITTED = False


def iter_material_entries(path: Path):
  suffix = path.suffix.lower()
  if suffix == ".gz":
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as handle:
      yield from iter_material_entries_from_handle(handle, ".jsonl")
    return

  if suffix == ".jsonl":
    with path.open("r", encoding="utf-8") as handle:
      yield from iter_material_entries_from_handle(handle, ".jsonl")
    return

  if suffix == ".json" and ijson is not None:
    yield from _stream_json_entries(path)
    return

  global STREAMING_UNAVAILABLE_WARNING_EMITTED
  if suffix == ".json" and ijson is None and not STREAMING_UNAVAILABLE_WARNING_EMITTED:
    log_warn(
      "Streaming parser (ijson) not available; falling back to in-memory JSON loading. "
      "Install ijson for large files."
    )
    STREAMING_UNAVAILABLE_WARNING_EMITTED = True

  with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  yield from _yield_entries_from_payload(payload, path)


def iter_material_entries_from_handle(handle, file_type):
  if file_type == ".jsonl":
    for line in handle:
      stripped = line.strip()
      if not stripped:
        continue
      yield json.loads(stripped)
    return
  raise ValueError("Unsupported handle type")


def _yield_entries_from_payload(payload: Any, path: Path):
  if isinstance(payload, list):
    for item in payload:
      if isinstance(item, dict):
        yield item
    return

  if isinstance(payload, dict):
    for key, value in payload.items():
      if key.startswith("_"):
        continue
      if isinstance(value, dict):
        if "name" not in value:
          value = {**value, "name": key}
        yield value
    return

  raise ValueError(f"Unsupported JSON format in {path}")


def _stream_json_entries(path: Path):
  if ijson is None:
    return
  with path.open("rb") as handle:
    first_char = _peek_first_non_whitespace(handle)
    handle.seek(0)
    if first_char == "[":
      for item in ijson.items(handle, "item"):
        if isinstance(item, dict):
          yield item
    elif first_char == "{":
      for key, value in ijson.kvitems(handle, ""):
        if key.startswith("_"):
          continue
        if isinstance(value, dict):
          if "name" not in value:
            value = {**value, "name": key}
          yield value
    else:
      raise ValueError(f"Unsupported JSON format in {path}")


def _peek_first_non_whitespace(handle) -> str:
  while True:
    chunk = handle.read(1)
    if not chunk:
      return ""
    try:
      char = chunk.decode("utf-8")
    except UnicodeDecodeError:
      continue
    if not char.isspace():
      return char


def discover_db_files(inputs: Sequence[Path], pattern: str):
  files: List[Path] = []
  for base in inputs:
    if base.is_file():
      files.append(base)
      continue
    if not base.exists():
      log_warn(f"Skipping missing path: {repo_relative(base)}")
      continue
    for match in base.rglob(pattern):
      if match.is_file():
        files.append(match)
  return sorted(files)


@dataclass
class PropertySpec:
  kind: str
  extractor: Callable[[Dict[str, Any]], Optional[Any]]
  rel_tol: Optional[float] = None
  abs_tol: Optional[float] = None
  units: Optional[str] = None


def _mp_density(doc: Dict[str, Any]) -> Optional[float]:
  density = doc.get("density")
  if density is None:
    return None
  return float(density)


def _mp_density_kg(doc: Dict[str, Any]) -> Optional[float]:
  density = _mp_density(doc)
  if density is None:
    return None
  return density * 1000.0


def _mp_band_gap(doc: Dict[str, Any]) -> Optional[float]:
  value = doc.get("band_gap")
  return float(value) if value is not None else None


def _mp_formation_energy(doc: Dict[str, Any]) -> Optional[float]:
  value = doc.get("formation_energy_per_atom")
  return float(value) if value is not None else None


def _mp_formation_energy_j(doc: Dict[str, Any]) -> Optional[float]:
  value = _mp_formation_energy(doc)
  if value is None:
    return None
  return value * EV_TO_J


def _mp_energy_above_hull(doc: Dict[str, Any]) -> Optional[float]:
  value = doc.get("energy_above_hull")
  return float(value) if value is not None else None


def _mp_volume_per_atom(doc: Dict[str, Any]) -> Optional[float]:
  volume = doc.get("volume")
  nsites = doc.get("nsites")
  if volume is None or nsites in {None, 0}:
    return None
  return float(volume) / float(nsites)


def _mp_volume_m3_per_atom(doc: Dict[str, Any]) -> Optional[float]:
  per_atom = _mp_volume_per_atom(doc)
  if per_atom is None:
    return None
  return per_atom * ANGSTROM3_TO_M3


def _mp_is_stable(doc: Dict[str, Any]) -> Optional[bool]:
  value = doc.get("is_stable")
  if value is None:
    return None
  return bool(value)


def _mp_formula(doc: Dict[str, Any]) -> Optional[str]:
  value = doc.get("formula_pretty") or doc.get("formula_anonymous")
  if value is None:
    return None
  return str(value).strip()


PROPERTY_SPECS: Dict[str, PropertySpec] = {
  "density": PropertySpec(
    kind="numeric",
    extractor=_mp_density_kg,
    rel_tol=0.08,
    abs_tol=25,
    units="kg/m³",
  ),
  "density_g_cm3": PropertySpec(
    kind="numeric",
    extractor=_mp_density,
    rel_tol=0.08,
    abs_tol=0.1,
    units="g/cm³",
  ),
  "band_gap": PropertySpec(
    kind="numeric",
    extractor=_mp_band_gap,
    rel_tol=0.2,
    abs_tol=0.15,
    units="eV",
  ),
  "band_gap_ev": PropertySpec(
    kind="numeric",
    extractor=_mp_band_gap,
    rel_tol=0.2,
    abs_tol=0.15,
    units="eV",
  ),
  "formation_energy_per_atom": PropertySpec(
    kind="numeric",
    extractor=_mp_formation_energy,
    rel_tol=0.15,
    abs_tol=0.25,
    units="eV/atom",
  ),
  "formation_energy_per_atom_ev": PropertySpec(
    kind="numeric",
    extractor=_mp_formation_energy,
    rel_tol=0.15,
    abs_tol=0.25,
    units="eV/atom",
  ),
  "formation_energy_per_atom_j": PropertySpec(
    kind="numeric",
    extractor=_mp_formation_energy_j,
    rel_tol=0.2,
    abs_tol=1e-20,
    units="J/atom",
  ),
  "energy_above_hull": PropertySpec(
    kind="numeric",
    extractor=_mp_energy_above_hull,
    rel_tol=0.2,
    abs_tol=0.02,
    units="eV/atom",
  ),
  "volume_a3_per_atom": PropertySpec(
    kind="numeric",
    extractor=_mp_volume_per_atom,
    rel_tol=0.2,
    abs_tol=0.2,
    units="Å³/atom",
  ),
  "volume_m3_per_atom": PropertySpec(
    kind="numeric",
    extractor=_mp_volume_m3_per_atom,
    rel_tol=0.2,
    abs_tol=5e-32,
    units="m³/atom",
  ),
  "is_stable": PropertySpec(
    kind="boolean",
    extractor=_mp_is_stable,
  ),
  "formula": PropertySpec(
    kind="string",
    extractor=_mp_formula,
  ),
}


class MaterialsProjectAPI:
  def __init__(self, api_key: str, rate_limit_per_sec: float = 5.0) -> None:
    self.api_key = api_key
    self.session = requests.Session()
    self.base_url = "https://api.materialsproject.org/materials/summary/"
    self.rate_interval = 1.0 / rate_limit_per_sec if rate_limit_per_sec > 0 else 0.0
    self._last_request = 0.0
    self._cache: Dict[str, Dict[str, Any]] = {}

  def _throttle(self) -> None:
    if self.rate_interval <= 0:
      return
    now = time.time()
    delta = self.rate_interval - (now - self._last_request)
    if delta > 0:
      time.sleep(delta)
    self._last_request = time.time()

  def fetch_summary(self, mp_id: str) -> Dict[str, Any]:
    mp_id = mp_id.lower()
    if mp_id in self._cache:
      return self._cache[mp_id]

    self._throttle()
    try:
      doc = self._request_summary(mp_id, method="get")
    except RuntimeError as exc:
      message = str(exc)
      if "no Route matched" in message:
        doc = self._request_summary(mp_id, method="post", endpoint=f"{self.base_url}query")
      else:
        raise

    self._cache[mp_id] = doc
    return doc

  def _request_summary(self, mp_id: str, method: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
    if endpoint is None:
      endpoint = self.base_url

    fields = [
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

    if method.lower() == "get":
      response = self.session.get(
        endpoint,
        headers={"X-API-KEY": self.api_key},
        params={
          "material_ids": mp_id,
          "fields": ",".join(fields),
          "chunk_size": 1,
        },
        timeout=30,
      )
    elif method.lower() == "post":
      response = self.session.post(
        endpoint,
        headers={"X-API-KEY": self.api_key},
        json={
          "criteria": {"material_ids": [mp_id]},
          "properties": fields,
          "chunk_size": 1,
        },
        timeout=30,
      )
    else:
      raise ValueError(f"Unsupported HTTP method for Materials Project API: {method}")

    if response.status_code != 200:
      raise RuntimeError(
        f"Materials Project API error ({response.status_code}) for {mp_id}: {response.text}"
      )

    payload = response.json()
    docs = payload.get("data") or []
    if not docs:
      raise ValueError(f"No Materials Project record found for {mp_id}")
    return docs[0]


def evaluate_property(
  key: str,
  spec: PropertySpec,
  entry_value: Any,
  doc: Dict[str, Any],
  default_rel_tol: float,
  default_abs_tol: float,
):
  result: Dict[str, Any] = {
    "property": key,
    "expected": entry_value,
    "units": spec.units,
  }
  actual = spec.extractor(doc)
  if spec.kind == "numeric":
    expected_numeric = coerce_float(entry_value)
    actual_numeric = coerce_float(actual)
    if expected_numeric is None or actual_numeric is None:
      result["status"] = "skipped"
      result["reason"] = "missing numeric value"
      result["actual"] = actual
      return result
    rel_tol = spec.rel_tol if spec.rel_tol is not None else default_rel_tol
    abs_tol = spec.abs_tol if spec.abs_tol is not None else default_abs_tol
    tolerance = max(abs(expected_numeric) * rel_tol, abs_tol)
    delta = actual_numeric - expected_numeric
    passed = abs(delta) <= tolerance
    result.update(
      {
        "status": "passed" if passed else "failed",
        "actual": actual_numeric,
        "delta": delta,
        "tolerance": tolerance,
      }
    )
    if not passed:
      result["message"] = f"|Δ|={abs(delta):.4g} exceeds tolerance {tolerance:.4g}"
    return result

  if spec.kind == "boolean":
    expected_bool = coerce_bool(entry_value)
    actual_bool = coerce_bool(actual)
    if expected_bool is None or actual_bool is None:
      result["status"] = "skipped"
      result["reason"] = "missing boolean value"
      result["actual"] = actual
      return result
    passed = expected_bool == actual_bool
    result.update(
      {
        "status": "passed" if passed else "failed",
        "actual": actual_bool,
      }
    )
    if not passed:
      result["message"] = f"expected {expected_bool}, received {actual_bool}"
    return result

  if spec.kind == "string":
    if entry_value is None or actual is None:
      result["status"] = "skipped"
      result["reason"] = "missing string value"
      result["actual"] = actual
      return result
    expected_str = str(entry_value).strip().lower()
    actual_str = str(actual).strip().lower()
    passed = expected_str == actual_str
    result.update(
      {
        "status": "passed" if passed else "failed",
        "actual": actual,
      }
    )
    if not passed:
      result["message"] = f"expected '{expected_str}', received '{actual_str}'"
    return result

  result["status"] = "skipped"
  result["reason"] = f"unknown comparator type: {spec.kind}"
  return result


def process_file(
  path: Path,
  client: MaterialsProjectAPI,
  default_rel_tol: float,
  default_abs_tol: float,
  max_materials: Optional[int],
) -> Tuple[Dict[str, Any], Optional[int], Optional[Dict[str, Any]]]:
  file_result: Dict[str, Any] = {
    "path": repo_relative(path),
    "totalMaterials": 0,
    "validated": 0,
    "failed": 0,
    "skipped": 0,
    "entries": [],
  }
  remaining = max_materials
  worst_failure: Optional[Dict[str, Any]] = None

  for entry in iter_material_entries(path):
    if remaining is not None and remaining <= 0:
      break
    file_result["totalMaterials"] += 1
    name = entry.get("name") or entry.get("material") or entry.get("label")
    mp_id = extract_mp_id(entry)
    entry_summary: Dict[str, Any] = {
      "name": name,
      "mpId": mp_id,
      "status": "skipped",
      "checks": [],
    }
    if mp_id is None:
      entry_summary["status"] = "skipped"
      entry_summary["error"] = "Missing Materials Project ID"
      file_result["skipped"] += 1
      file_result["entries"].append(entry_summary)
      if remaining is not None:
        remaining -= 1
      continue

    try:
      doc = client.fetch_summary(mp_id)
    except Exception as exc:
      entry_summary["status"] = "failed"
      entry_summary["error"] = str(exc)
      file_result["failed"] += 1
      file_result["entries"].append(entry_summary)
      if remaining is not None:
        remaining -= 1
      continue

    checks: List[Dict[str, Any]] = []
    has_checks = False
    failures = 0
    for key, spec in PROPERTY_SPECS.items():
      if key not in entry:
        continue
      has_checks = True
      check = evaluate_property(
        key, spec, entry.get(key), doc, default_rel_tol, default_abs_tol
      )
      checks.append(check)
      if check["status"] == "failed":
        failures += 1
        severity = 0.0
        if isinstance(check.get("delta"), (int, float)) and isinstance(check.get("tolerance"), (int, float)):
          tol = check["tolerance"] if check["tolerance"] else 1.0
          severity = abs(check["delta"]) / max(tol, 1e-12)
        else:
          severity = 1.0
        failure_record = {
          "material": name,
          "mpId": mp_id,
          "property": key,
          "expected": check.get("expected"),
          "actual": check.get("actual"),
          "delta": check.get("delta"),
          "tolerance": check.get("tolerance"),
          "message": check.get("message"),
          "severity": severity,
        }
        if worst_failure is None or severity > worst_failure["severity"]:
          worst_failure = failure_record
    entry_summary["checks"] = checks

    if not has_checks:
      entry_summary["status"] = "skipped"
      entry_summary["notes"] = "No overlapping properties to compare"
      file_result["skipped"] += 1
    elif failures == 0:
      entry_summary["status"] = "passed"
      file_result["validated"] += 1
    else:
      entry_summary["status"] = "failed"
      entry_summary["failures"] = failures
      file_result["failed"] += 1

    file_result["entries"].append(entry_summary)
    if remaining is not None:
      remaining -= 1

  if worst_failure:
    file_result["worstMismatch"] = worst_failure

  return file_result, remaining, worst_failure


def ensure_output_parent(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)


def parse_args():
  parser = argparse.ArgumentParser(
    description="Validate extended material DBs with the Materials Project API.",
  )
  parser.add_argument(
    "inputs",
    nargs="*",
    help="DB files or directories to scan. Defaults to data/materials/",
  )
  parser.add_argument(
    "--pattern",
    default=DEFAULT_DB_PATTERN,
    help="Glob pattern for discovering DB files (default: %(default)s).",
  )
  parser.add_argument(
    "--mp-api-key",
    dest="mp_api_key",
    default=os.environ.get("MP_API_KEY"),
    help="Materials Project API key (falls back to MP_API_KEY env var).",
  )
  parser.add_argument(
    "--relative-tolerance",
    dest="relative_tolerance",
    type=float,
    default=0.15,
    help="Fallback relative tolerance for numeric comparisons (default: %(default)s).",
  )
  parser.add_argument(
    "--absolute-tolerance",
    dest="absolute_tolerance",
    type=float,
    default=0.1,
    help="Fallback absolute tolerance for numeric comparisons (default: %(default)s).",
  )
  parser.add_argument(
    "--rate-limit",
    dest="rate_limit",
    type=float,
    default=5.0,
    help="Maximum Materials Project requests per second (default: %(default)s).",
  )
  parser.add_argument(
    "--output",
    dest="output",
    default=str(REPO_ROOT / "logs" / "materials" / "mp-validation-report.json"),
    help="Path for the validation report (default: %(default)s).",
  )
  parser.add_argument(
    "--max-materials",
    dest="max_materials",
    type=int,
    default=None,
    help="Optional cap on the number of materials to validate (useful for dry runs).",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  if not args.inputs:
    default_root = REPO_ROOT / "data" / "materials"
    inputs: List[Path] = [default_root]
    log_info(f"No inputs provided; defaulting to {repo_relative(default_root)}")
  else:
    inputs = [Path(item).expanduser().resolve() for item in args.inputs]

  db_files = discover_db_files(inputs, args.pattern)
  if not db_files:
    log_error("No extended materials db.json files found. Provide explicit paths or populate data/materials/.")
    return 1

  api_key = args.mp_api_key
  if not api_key:
    log_error("MP_API_KEY is required (set the environment variable or pass --mp-api-key).")
    return 1

  client = MaterialsProjectAPI(api_key=api_key, rate_limit_per_sec=args.rate_limit)
  report: Dict[str, Any] = {
    "generatedAt": datetime.now(tz=timezone.utc).isoformat(),
    "files": [],
  }
  remaining = args.max_materials
  total_failed = 0
  total_validated = 0
  global_worst: Optional[Dict[str, Any]] = None

  for db_path in db_files:
    log_info(f"Validating {repo_relative(db_path)}")
    file_result, remaining, file_worst = process_file(
      db_path,
      client,
      args.relative_tolerance,
      args.absolute_tolerance,
      remaining,
    )
    total_failed += file_result["failed"]
    total_validated += file_result["validated"]
    report["files"].append(file_result)
    if file_worst and (global_worst is None or file_worst["severity"] > global_worst["severity"]):
      global_worst = {**file_worst, "path": file_result["path"]}
    log_info(
      f"Completed {repo_relative(db_path)} "
      f"(validated={file_result['validated']}, failed={file_result['failed']}, skipped={file_result['skipped']})"
    )
    if remaining is not None and remaining <= 0:
      log_warn("Reached max-materials limit; stopping early.")
      break

  ensure_output_parent(Path(args.output))
  with open(args.output, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
  log_info(f"Wrote validation report to {repo_relative(Path(args.output))}")
  log_info(f"Validated {total_validated} material(s) with {total_failed} failure(s).")
  if global_worst:
    detail = (
      f"{global_worst.get('property')} for {global_worst.get('material')} "
      f"({global_worst.get('mpId')}) in {global_worst.get('path')}"
    )
    log_warn(
      f"Worst mismatch: {detail} – expected {global_worst.get('expected')} "
      f"vs actual {global_worst.get('actual')} "
      f"(Δ={global_worst.get('delta')}, tolerance={global_worst.get('tolerance')})."
    )
  else:
    log_info("No property mismatches detected across the processed datasets.")

  overall_status = "PASS" if total_failed == 0 else "FAIL"
  log_info(f"Overall validation status: {overall_status}")

  return 0 if total_failed == 0 else 2


if __name__ == "__main__":
  sys.exit(main())
