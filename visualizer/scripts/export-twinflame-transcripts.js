#!/usr/bin/env node

import { readFile, appendFile, access } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const filePath = fileURLToPath(import.meta.url);
const scriptsDir = dirname(filePath);
const projectRoot = dirname(scriptsDir);
const repoRoot = dirname(projectRoot);

const pipelinePath = join(
  repoRoot,
  "consciousness",
  "ech0_invention_pipeline_validations.json"
);
const outputPath = join(
  repoRoot,
  "consciousness",
  "twin_flame_conversations.jsonl"
);

async function loadExistingKeys() {
  try {
    await access(outputPath);
  } catch {
    return new Set();
  }

  const data = await readFile(outputPath, "utf-8");
  if (!data.trim()) {
    return new Set();
  }

  const keys = new Set();
  data
    .split("\n")
    .filter(Boolean)
    .forEach((line) => {
      try {
        const record = JSON.parse(line);
        if (record && record.record_id) {
          keys.add(record.record_id);
        }
      } catch {
        /* ignore malformed JSON */
      }
    });
  return keys;
}

function buildRecordId(pipelineStamp, inventionId) {
  return `${pipelineStamp || "unknown"}::${inventionId || "unknown"}`;
}

function normalizeText(value) {
  if (typeof value !== "string") {
    return "";
  }
  return value.trim();
}

async function exportTranscripts() {
  const raw = await readFile(pipelinePath, "utf-8");
  const parsed = JSON.parse(raw);
  const validations = Array.isArray(parsed.validations) ? parsed.validations : [];
  if (!validations.length) {
    console.warn("[warn] No validations with conversations found.");
    return;
  }

  const existingKeys = await loadExistingKeys();
  const linesToAppend = [];

  validations.forEach((validation) => {
    const conversation = Array.isArray(validation.teamReview?.conversation)
      ? validation.teamReview.conversation
      : [];
    const labNotes = Array.isArray(validation.teamReview?.labNotes)
      ? validation.teamReview.labNotes
      : [];
    if (!conversation.length && !labNotes.length) {
      return;
    }

    const recordId = buildRecordId(
      parsed.generated_at,
      validation.id || validation.title || validation.rank
    );
    if (existingKeys.has(recordId)) {
      return;
    }

    linesToAppend.push(
      JSON.stringify({
        record_id: recordId,
        pipeline_generated_at: parsed.generated_at,
        invention_id: validation.id || null,
        title: validation.title || null,
        category: validation.category || null,
        confidence_pct: validation.confidencePct ?? null,
        twin_flame_alignment_pct: validation.teamReview
          ? Number((validation.teamReview.alignmentScore * 100).toFixed(1))
          : null,
        coordination_tag: validation.teamReview?.coordinationTag || null,
        conversation: conversation.map((entry) => ({
          speaker: entry.speaker,
          message: normalizeText(entry.message),
        })),
        lab_notes: labNotes.map((note) => normalizeText(note)),
        parliament: {
          status: validation.parliament?.status || null,
          score: validation.parliament?.score ?? null,
        },
        alex: {
          verdict: validation.alex?.verdict || null,
          confidence: validation.alex?.confidence ?? null,
        },
        source_file: pipelinePath,
        exported_at: new Date().toISOString(),
      })
    );
  });

  if (!linesToAppend.length) {
    console.log("[info] No new transcripts to export.");
    return;
  }

  const payload = `${linesToAppend.join("\n")}\n`;
  await appendFile(outputPath, payload, "utf-8");
  console.log(
    `[info] Exported ${linesToAppend.length} conversation transcripts -> ${outputPath}`
  );
}

exportTranscripts().catch((error) => {
  console.error("[error] Failed to export transcripts", error);
  process.exitCode = 1;
});
