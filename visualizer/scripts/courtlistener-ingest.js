import { promises as fs } from "node:fs";
import path from "node:path";
import process from "node:process";

import {
  CourtListenerClient,
  COURT_LISTENER_ENDPOINTS,
} from "./courtlistener-client.js";
import {
  createSupabase,
  hasSupabaseConfig,
} from "./supabase-client.js";

const RECORDS_TABLE = "courtlistener_records";
const MISSING_PDFS_TABLE = "recap_missing_documents";
const SUPPORTED_ENDPOINTS = Array.from(COURT_LISTENER_ENDPOINTS).join(", ");

function parseArgs(argv) {
  const result = {};

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (!token.startsWith("--")) {
      continue;
    }

    const name = token.slice(2);
    const next = argv[index + 1];

    if (!next || next.startsWith("--")) {
      result[name] = true;
      continue;
    }

    result[name] = next;
    index += 1;
  }

  return result;
}

function parseSupabaseOverrides(flags) {
  const overrides = {};

  if (flags["supabase-url"]) {
    overrides.url = flags["supabase-url"];
  }

  if (flags["supabase-key"]) {
    overrides.key = flags["supabase-key"];
  }

  return overrides;
}

function parseQueryParams(rawParams) {
  if (!rawParams) {
    return {};
  }

  const params = new URLSearchParams(rawParams);
  const output = {};

  params.forEach((value, key) => {
    output[key] = value;
  });

  return output;
}

function chunkRecords(records, size) {
  if (!size || size <= 0) {
    return [records];
  }

  const chunks = [];

  for (let index = 0; index < records.length; index += size) {
    chunks.push(records.slice(index, index + size));
  }

  return chunks;
}

function resolveApiKey(flags) {
  return (
    flags["api-key"] ||
    process.env.COURT_LISTENER_API_KEY ||
    process.env.COURTLISTENER_API_KEY ||
    ""
  ).trim();
}

function toNumber(value, label) {
  if (value === undefined || value === true) {
    return undefined;
  }

  const parsed = Number(value);

  if (Number.isNaN(parsed)) {
    throw new Error(`Expected ${label} to be numeric. Received "${value}".`);
  }

  return parsed;
}

function buildOutputPath(endpoint, explicitPath) {
  if (explicitPath) {
    return path.resolve(process.cwd(), explicitPath);
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const logsDir = path.resolve(process.cwd(), "logs", "courtlistener");

  return path.join(logsDir, `${endpoint}-${timestamp}.json`);
}

function buildRecordPointers(record) {
  return {
    id: record.id || null,
    absoluteUrl: record.absolute_url || null,
    docket: record.docket || null,
    recapApiUrl: record.url || record.resource_uri || null,
    downloadUrl: record.download_url || null,
    pdfUrl: record.pdf_url || null,
    pacerCaseId: record.pacer_case_id || null,
    pacerDocId: record.pacer_doc_id || null,
  };
}

function normalizeRecordId(record) {
  if (record.id !== undefined && record.id !== null) {
    return String(record.id);
  }

  if (record.uuid) {
    return String(record.uuid);
  }

  if (record.absolute_url) {
    return record.absolute_url;
  }

  if (record.url) {
    return record.url;
  }

  return null;
}

function mapRecordForSupabase(record, endpoint, fetchedAt, params) {
  const recordId = normalizeRecordId(record);
  return {
    endpoint,
    record_id: recordId,
    docket: record.docket || null,
    absolute_url: record.absolute_url || null,
    pacer_case_id: record.pacer_case_id || null,
    pacer_doc_id: record.pacer_doc_id || null,
    fetched_at: fetchedAt,
    params,
    payload: record,
  };
}

function mapMissingPdfForSupabase(pointer, fetchedAt, endpoint) {
  return {
    endpoint,
    record_id: pointer.id,
    docket: pointer.docket,
    absolute_url: pointer.absoluteUrl,
    recap_api_url: pointer.recapApiUrl,
    download_url: pointer.downloadUrl,
    pacer_case_id: pointer.pacerCaseId,
    pacer_doc_id: pointer.pacerDocId,
    flagged_at: fetchedAt,
  };
}

async function writeOutput(outputPath, payload) {
  const directory = path.dirname(outputPath);
  await fs.mkdir(directory, { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(payload, null, 2), "utf8");
}

function printUsage() {
  console.log(
    [
      "[info] CourtListener ingestion helper",
      "",
      "Usage:",
      "  node scripts/courtlistener-ingest.js --endpoint dockets --params 'court=nvb' [--max-pages 2]",
      "",
      "Flags:",
      "  --endpoint       One of: dockets, docket-entries, opinions, recap-documents",
      "  --params         URL query string appended to the request (e.g., 'court=nvb&order_by=-date_filed')",
      "  --page-size      Number of records per page (CourtListener defaults to 20)",
      "  --max-pages      Maximum number of pages to request",
      "  --api-key        Override COURT_LISTENER_API_KEY environment variable",
      "  --output         Optional custom output path (defaults to logs/courtlistener/)",
      "  --supabase-url   Optional Supabase URL override (uses SUPABASE_URL env by default)",
      "  --supabase-key   Optional Supabase key override (uses SUPABASE_SERVICE_ROLE_KEY env)",
      "  --skip-supabase  Skip Supabase sync even if credentials are present",
      "  --require-supabase  Fail if Supabase credentials are missing",
      "  --help           Show this message",
    ].join("\n")
  );
}

function isPdfAvailable(record) {
  return Boolean(
    record.pdf_url ||
      record.download_url ||
      (record.absolute_url && record.absolute_url.endsWith(".pdf"))
  );
}

async function syncToSupabase({
  endpoint,
  fetchedAt,
  params,
  records,
  missingPdfs,
  overrides,
  chunkSize = 500,
}) {
  if (!hasSupabaseConfig(overrides)) {
    return { attempted: false };
  }

  const supabase = createSupabase(overrides);

  const recordRows = records
    .map((record) => mapRecordForSupabase(record, endpoint, fetchedAt, params))
    .filter((row) => row.record_id);

  const missingRows = missingPdfs.map((pointer) =>
    mapMissingPdfForSupabase(pointer, fetchedAt, endpoint)
  );

  if (recordRows.length === 0) {
    console.log("[info] No records with identifiers available for Supabase sync.");
  }

  for (const chunk of chunkRecords(recordRows, chunkSize)) {
    if (chunk.length === 0) {
      continue;
    }

    const { error } = await supabase
      .from(RECORDS_TABLE)
      .upsert(chunk, { onConflict: "endpoint,record_id" });

    if (error) {
      throw new Error(`Supabase record upsert failed: ${error.message}`);
    }
  }

  for (const chunk of chunkRecords(missingRows, chunkSize)) {
    if (chunk.length === 0) {
      continue;
    }

    const { error } = await supabase
      .from(MISSING_PDFS_TABLE)
      .upsert(chunk, { onConflict: "record_id" });

    if (error) {
      throw new Error(`Supabase missing PDF upsert failed: ${error.message}`);
    }
  }

  console.log(
    `[info] Synced ${recordRows.length} records and ${missingRows.length} missing PDFs to Supabase.`
  );

  return { attempted: true };
}

async function ingest() {
  const flags = parseArgs(process.argv.slice(2));

  if (flags.help) {
    printUsage();
    return;
  }

  const endpoint = flags.endpoint;

  if (!endpoint) {
    console.error("[error] --endpoint is required.");
    printUsage();
    process.exitCode = 1;
    return;
  }

  if (!COURT_LISTENER_ENDPOINTS.has(endpoint)) {
    console.error(
      `[error] Unsupported endpoint "${endpoint}". Choose one of: ${SUPPORTED_ENDPOINTS}.`
    );
    process.exitCode = 1;
    return;
  }

  const apiKey = resolveApiKey(flags);

  if (!apiKey) {
    console.error(
      "[error] Provide a CourtListener API key via --api-key or COURT_LISTENER_API_KEY."
    );
    process.exitCode = 1;
    return;
  }

  const supabaseOverrides = parseSupabaseOverrides(flags);
  const supabaseConfigured = hasSupabaseConfig(supabaseOverrides);
  const skipSupabase = Boolean(flags["skip-supabase"]);
  const requireSupabase = Boolean(flags["require-supabase"]);

  if (requireSupabase && !supabaseConfigured) {
    console.error(
      "[error] Supabase credentials are required (set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY or use --supabase-url/--supabase-key)."
    );
    process.exitCode = 1;
    return;
  }

  const params = parseQueryParams(flags.params);
  const pageSize = toNumber(flags["page-size"], "--page-size");
  const maxPages = toNumber(flags["max-pages"], "--max-pages");

  if (pageSize) {
    params.page_size = pageSize;
  }

  const client = new CourtListenerClient({
    apiKey,
  });

  const collected = [];
  const missingPdfs = [];

  let pageCounter = 0;
  let recordCounter = 0;

  try {
    for await (const page of client.paginate(endpoint, {
      params,
      maxPages,
    })) {
      pageCounter += 1;
      const records = Array.isArray(page.results) ? page.results : [];
      recordCounter += records.length;

      console.log(
        `[info] Received ${records.length} records from page ${pageCounter}.`
      );

      records.forEach((record) => {
        collected.push(record);

        if (endpoint === "recap-documents" && !isPdfAvailable(record)) {
          missingPdfs.push(buildRecordPointers(record));
        }
      });
    }
  } catch (error) {
    console.error(`[error] CourtListener ingestion failed: ${error.message}`);
    process.exitCode = 1;
    return;
  }

  console.log(`[info] Fetched ${recordCounter} records across ${pageCounter} pages.`);

  if (missingPdfs.length > 0) {
    console.warn(
      `[warn] Identified ${missingPdfs.length} RECAP documents without PDFs.`
    );
  }

  const fetchedAt = new Date().toISOString();
  const outputPath = buildOutputPath(endpoint, flags.output);
  const payload = {
    endpoint,
    fetchedAt,
    params,
    totalRecords: collected.length,
    missingPdfCount: missingPdfs.length,
    missingPdfPointers: missingPdfs,
    records: collected,
  };

  try {
    await writeOutput(outputPath, payload);
  } catch (error) {
    console.error(`[error] Unable to write output file: ${error.message}`);
    process.exitCode = 1;
    return;
  }

  console.log(`[info] Wrote CourtListener payload to ${outputPath}`);

  if (skipSupabase) {
    console.log("[info] Supabase sync skipped via --skip-supabase flag.");
    return;
  }

  if (!supabaseConfigured) {
    console.log("[info] Supabase credentials not provided. Skipping sync.");
    return;
  }

  try {
    await syncToSupabase({
      endpoint,
      fetchedAt,
      params,
      records: collected,
      missingPdfs,
      overrides: supabaseOverrides,
    });
  } catch (error) {
    console.error(`[error] Supabase sync failed: ${error.message}`);
    process.exitCode = 1;
  }
}

ingest().catch((error) => {
  console.error("[error] Unexpected failure", error);
  process.exitCode = 1;
});
