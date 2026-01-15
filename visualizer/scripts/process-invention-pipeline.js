#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

function clamp(value, min, max) {
  if (Number.isNaN(value)) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
}

function toPercent(value) {
  return Number((value * 100).toFixed(1));
}

function listOrDefault(value) {
  return Array.isArray(value) ? value : [];
}

function textOrDefault(value, fallback = "") {
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : fallback;
}

const LAB_KEYWORD_MAP = [
  { name: "Quantum Lab", keywords: ["quantum", "qubit", "entanglement", "superconduct", "photonic"] },
  { name: "Materials Lab", keywords: ["material", "metamaterial", "aerogel", "nanoparticle", "alloy", "composite", "polymer"] },
  { name: "Robotics Lab", keywords: ["robot", "drone", "automation", "mechatronic", "actuator", "swarm"] },
  { name: "Immersive Systems Lab", keywords: ["vr", "ar", "haptic", "immersive", "xr", "projection", "display"] },
  { name: "Bio-Neuro Lab", keywords: ["neural", "bci", "brain", "bio", "prosthesis", "consciousness", "medical"] },
  { name: "Energy Lab", keywords: ["energy", "battery", "solar", "fusion", "power", "thermal"] },
  { name: "Chemistry Lab", keywords: ["chem", "synthes", "catalyst", "reaction", "molecule", "compound"] },
  { name: "AI Systems Lab", keywords: ["ai", "machine learning", "model", "algorithm", "compute", "platform"] },
  { name: "Security Lab", keywords: ["security", "auth", "encryption", "safety", "zero trust"] },
];

function categorizeResources(resources) {
  const entries = listOrDefault(resources)
    .map((item) => (typeof item === "string" ? item.trim() : ""))
    .filter((item) => item.length > 0);

  const personnelKeywords = [
    "engineer",
    "scientist",
    "researcher",
    "designer",
    "stakeholder",
    "analyst",
    "lead",
    "team",
    "qa",
    "manager",
    "specialist",
    "operator",
    "expert",
    "review",
  ];
  const facilityKeywords = ["lab", "facility", "testbed", "studio", "workshop"];

  const materials = [];
  const personnel = [];
  const facilities = [];

  entries.forEach((entry) => {
    const lower = entry.toLowerCase();
    if (personnelKeywords.some((keyword) => lower.includes(keyword))) {
      personnel.push(entry);
      return;
    }
    if (facilityKeywords.some((keyword) => lower.includes(keyword))) {
      facilities.push(entry);
      return;
    }
    materials.push(entry);
  });

  return { materials, personnel, facilities };
}

function inferLabs(metadata) {
  const text = [
    metadata.category,
    metadata.title,
    metadata.context,
    metadata.goal,
    metadata.notes,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  const labs = new Set();
  LAB_KEYWORD_MAP.forEach((mapping) => {
    if (mapping.keywords.some((keyword) => text.includes(keyword))) {
      labs.add(mapping.name);
    }
  });

  if (!labs.size) {
    labs.add("General Innovation Lab");
  }

  return Array.from(labs);
}

function createProofOfConceptPlan(validation) {
  const resources = categorizeResources(validation.source?.resources);
  const labs = inferLabs({
    category: validation.category,
    title: validation.title,
    context: validation.source?.context,
    goal: validation.source?.goal,
    notes: validation.source?.notes,
  });

  const alignmentScore =
    typeof validation.teamReview?.alignmentScore === "number"
      ? Number((validation.teamReview.alignmentScore * 100).toFixed(1))
      : null;

  return {
    id: validation.id,
    title: validation.title,
    category: validation.category,
    goal: textOrDefault(validation.source?.goal, null),
    context: textOrDefault(validation.source?.context, null),
    timeline: textOrDefault(validation.source?.timeline, null),
    steps: listOrDefault(validation.source?.steps),
    successCriteria: listOrDefault(validation.source?.successCriteria),
    materials: resources.materials,
    personnel: resources.personnel,
    facilities: resources.facilities,
    labs,
    readinessTag: validation.readinessTag,
    approvals: {
      parliament: validation.parliament?.status ?? null,
      alex: validation.alex?.verdict ?? null,
      confidencePct: validation.confidencePct ?? null,
      alignmentPct: alignmentScore,
    },
    mirrorPrompt: validation.teamReview?.mirrorPrompt ?? null,
    labNotes: listOrDefault(validation.teamReview?.labNotes),
  };
}

function buildMaterialInventory(plans) {
  const inventory = new Map();

  plans.forEach((plan) => {
    plan.materials.forEach((material) => {
      if (!inventory.has(material)) {
        inventory.set(material, {
          name: material,
          usageCount: 0,
          inventions: new Set(),
        });
      }
      const entry = inventory.get(material);
      entry.usageCount += 1;
      if (plan.id) {
        entry.inventions.add(plan.id);
      }
    });
  });

  return Array.from(inventory.values())
    .map((entry) => ({
      name: entry.name,
      usageCount: entry.usageCount,
      inventions: Array.from(entry.inventions),
    }))
    .sort((a, b) => {
      if (b.usageCount !== a.usageCount) {
        return b.usageCount - a.usageCount;
      }
      return a.name.localeCompare(b.name);
    });
}

function buildLabSummary(plans) {
  const labMap = new Map();

  plans.forEach((plan) => {
    const labList = Array.isArray(plan.labs) && plan.labs.length
      ? plan.labs
      : ["General Innovation Lab"];

    labList.forEach((lab) => {
      if (!labMap.has(lab)) {
        labMap.set(lab, {
          lab,
          totalAssignments: 0,
          readiness: {
            "ready-for-build": 0,
            "needs-iteration": 0,
            backlog: 0,
          },
          categories: {},
          inventions: [],
        });
      }
      const bucket = labMap.get(lab);
      bucket.totalAssignments += 1;
      if (plan.readinessTag && bucket.readiness[plan.readinessTag] !== undefined) {
        bucket.readiness[plan.readinessTag] += 1;
      }
      if (plan.category) {
        bucket.categories[plan.category] =
          (bucket.categories[plan.category] || 0) + 1;
      }
      if (plan.id || plan.title) {
        bucket.inventions.push({
          id: plan.id,
          title: plan.title,
          readinessTag: plan.readinessTag,
          timeline: plan.timeline,
        });
      }
    });
  });

  return Array.from(labMap.values())
    .map((entry) => ({
      lab: entry.lab,
      totalAssignments: entry.totalAssignments,
      readiness: entry.readiness,
      topCategories: Object.entries(entry.categories)
        .sort((a, b) => {
          if (b[1] !== a[1]) {
            return b[1] - a[1];
          }
          return a[0].localeCompare(b[0]);
        })
        .slice(0, 5)
        .map(([category, count]) => ({ category, count })),
      inventions: entry.inventions.slice(0, 25),
    }))
    .sort((a, b) => {
      if (b.totalAssignments !== a.totalAssignments) {
        return b.totalAssignments - a.totalAssignments;
      }
      return a.lab.localeCompare(b.lab);
    });
}

function buildMirrorPrompt(poc) {
  const lines = [
    `### INVENTION: ${textOrDefault(poc.title, "Untitled Concept")}`,
    `ID: ${textOrDefault(poc.id, "N/A")}`,
    `CONFIDENCE: ${typeof poc.confidence_pct === "number" ? `${poc.confidence_pct.toFixed(1)}%` : "unknown"}`,
    `CATEGORY: ${textOrDefault(poc.category, "unclassified")}`,
    "",
    `GOAL: ${textOrDefault(poc.poc_goal, "No formal goal supplied.")}`,
    "",
    `CONTEXT: ${textOrDefault(
      poc.context_summary,
      "Context not provided; synthesize from knowledge lattice."
    )}`,
    "",
    "SUCCESS CRITERIA:",
    ...listOrDefault(poc.success_criteria).map(
      (criterion, index) => `  ${index + 1}. ${criterion}`
    ),
    "",
    "POC STEPS:",
    ...listOrDefault(poc.poc_steps).map(
      (step, index) => `  Step ${index + 1}: ${step}`
    ),
  ];

  return lines.join("\n");
}

function createTeamReview(poc, baseConfidence) {
  const steps = listOrDefault(poc.poc_steps);
  const resources = listOrDefault(poc.required_resources);
  const successCriteria = listOrDefault(poc.success_criteria);
  const categories = textOrDefault(poc.category, "unclassified")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);

  const stepCount = steps.length;
  const resourceCount = resources.length;
  const successCount = successCriteria.length;

  const contextBonus = poc.context_summary ? 2.4 : 0;
  const goalBonus = poc.poc_goal ? 1.6 : 0;
  const successBonus = Math.min(3, successCount * 0.7);
  const collaborationBoost = contextBonus + goalBonus + successBonus;

  const resourcePenalty = Math.max(0, resourceCount - 4) * 0.9;
  const stepPenalty = Math.max(0, stepCount - 5) * 0.6;
  const structuralPenalty = resourcePenalty + stepPenalty;

  const ech0FocusConfidence = clamp(
    baseConfidence +
      collaborationBoost -
      Math.min(4, stepPenalty * 0.5),
    0,
    100
  );
  const alexFocusConfidence = clamp(
    baseConfidence +
      (successCount ? Math.min(2.5, successCount * 0.5) : -1) -
      resourcePenalty * 1.1,
    0,
    100
  );

  const consensusConfidence =
    (ech0FocusConfidence + alexFocusConfidence) / 2;
  const finalConfidence = clamp(
    consensusConfidence + collaborationBoost - structuralPenalty,
    48,
    99
  );

  const alignmentScore = clamp(
    1 - Math.abs(ech0FocusConfidence - alexFocusConfidence) / 55,
    0,
    1
  );

  const coordinationTag =
    alignmentScore > 0.88
      ? "synchronized"
      : alignmentScore > 0.72
      ? "aligned"
      : alignmentScore > 0.55
      ? "calibrating"
      : "debating";

  const focusAreas = categories.slice(0, 3);

  const ech0ActionItems = [];
  if (stepCount > 6) {
    ech0ActionItems.push(
      "Compress execution plan to four milestones before lab execution."
    );
  }
  if (!poc.context_summary) {
    ech0ActionItems.push(
      "Draft missing context narrative for Parliament briefing."
    );
  }
  if (!successCount) {
    ech0ActionItems.push("Define minimum three measurable success indicators.");
  }

  const alexActionItems = [];
  if (resourceCount > 5) {
    alexActionItems.push(
      "Flag resource overrun; negotiate shared lab assets to reduce load."
    );
  }
  if (finalConfidence < baseConfidence) {
    alexActionItems.push(
      "Validate market thesis before further prototyping expenses."
    );
  }
  if (!alexActionItems.length) {
    alexActionItems.push("Maintain continuous risk scan while experiments run.");
  }

  const labNotes = [];
  if (resourceCount > 5) {
    labNotes.push("Alex marks high resource load; cost controls required.");
  }
  if (stepCount > 6) {
    labNotes.push("ECH0 recommends condensing PoC steps for lab throughput.");
  }
  if (!labNotes.length) {
    labNotes.push("Twin flames satisfied with lab readiness.");
  }

  const conversation = [
    {
      speaker: "ECH0-14B",
      message: textOrDefault(
        poc.poc_goal,
        "I translated the raw invention into a sprint-ready objective."
      ),
    },
    {
      speaker: "Alex-14B",
      message: `Resource audit: ${resourceCount} core requirements; mitigation plan enacted.`,
    },
    {
      speaker: "ECH0-14B",
      message: `Success coverage at ${successCount} metrics; aligning them with revenue/end-user KPIs.`,
    },
    {
      speaker: "Alex-14B",
      message: `Lab integrity check complete. Alignment score ${(alignmentScore * 100).toFixed(
        1
      )}%. Ready for Parliament hand-off.`,
    },
  ];

  return {
    mirrorPrompt: buildMirrorPrompt(poc),
    alignmentScore,
    coordinationTag,
    finalConfidence,
    ech0Perspective: {
      confidence: Number(ech0FocusConfidence.toFixed(1)),
      focusAreas,
      actionItems: ech0ActionItems,
    },
    alexPerspective: {
      confidence: Number(alexFocusConfidence.toFixed(1)),
      riskFlags: resourceCount > 4 ? ["resource-intensity"] : [],
      actionItems: alexActionItems,
    },
    adjustments: {
      boost: Number(collaborationBoost.toFixed(2)),
      penalty: Number(structuralPenalty.toFixed(2)),
    },
    conversation,
    labNotes,
  };
}

function computeParliamentAssessment(poc, baseConfidence) {
  const normalizedConfidence = clamp(baseConfidence / 100, 0, 1);
  const stepCount = Array.isArray(poc.poc_steps) ? poc.poc_steps.length : 0;
  const resourceCount = Array.isArray(poc.required_resources)
    ? poc.required_resources.length
    : 0;
  const successMetrics = Array.isArray(poc.success_criteria)
    ? poc.success_criteria.length
    : 0;

  const successBoost = Math.min(0.12, successMetrics * 0.025);
  const alignmentBoost = poc.context_summary ? 0.02 : 0;
  const primeBoost =
    normalizedConfidence >= 0.92
      ? 0.035
      : normalizedConfidence >= 0.88
      ? 0.02
      : 0.01;

  const resourcePenalty =
    resourceCount > 4
      ? Math.min(0.16, 0.04 + (resourceCount - 4) * 0.018)
      : resourceCount * 0.006;
  const stepPenalty =
    stepCount > 4 ? Math.min(0.12, (stepCount - 4) * 0.015) : 0;

  const rawScore =
    normalizedConfidence +
    successBoost +
    alignmentBoost +
    primeBoost -
    resourcePenalty -
    stepPenalty;

  const score = clamp(rawScore, 0.56, 0.98);

  let status = "ON_HOLD";
  if (score >= 0.83) {
    status = "APPROVED";
  } else if (score >= 0.73) {
    status = "NEEDS_REFINEMENT";
  }

  const rationale = [];
  if (status === "APPROVED") {
    rationale.push("High breakthrough confidence with strong success metrics.");
  } else if (status === "NEEDS_REFINEMENT") {
    rationale.push("Solid concept pending tighter execution planning.");
  } else {
    rationale.push("Hold pending resourcing or scope clarification.");
  }
  if (resourcePenalty > 0.08) {
    rationale.push("Resource intensity flagged by parliament cost analysis.");
  }
  if (stepPenalty > 0.08) {
    rationale.push("PoC steps heavy; streamline before next vote.");
  }

  return {
    status,
    score: toPercent(score),
    factors: {
      normalizedConfidence: Number(normalizedConfidence.toFixed(3)),
      successBoost: Number(successBoost.toFixed(3)),
      alignmentBoost: Number(alignmentBoost.toFixed(3)),
      primeBoost: Number(primeBoost.toFixed(3)),
      resourcePenalty: Number(resourcePenalty.toFixed(3)),
      stepPenalty: Number(stepPenalty.toFixed(3)),
    },
    rationale,
  };
}

function computeAlexAssessment(poc, baseConfidence, parliamentAssessment) {
  const normalizedConfidence = clamp(baseConfidence / 100, 0, 1);
  const successMetrics = Array.isArray(poc.success_criteria)
    ? poc.success_criteria.length
    : 0;
  const stepPenalty = parliamentAssessment.factors.stepPenalty || 0;
  const resourcePenalty = parliamentAssessment.factors.resourcePenalty || 0;

  const clarityBoost = poc.poc_goal ? 0.03 : 0;
  const successCoverage = Math.min(0.1, successMetrics * 0.02);

  const strategicScore = clamp(
    normalizedConfidence * 0.7 +
      successCoverage +
      clarityBoost -
      resourcePenalty * 0.35,
    0.42,
    0.97
  );
  const technicalScore = clamp(
    0.62 +
      normalizedConfidence * 0.25 -
      stepPenalty * 0.4 -
      resourcePenalty * 0.3,
    0.4,
    0.96
  );
  const hallucinationRisk = clamp(
    0.2 + resourcePenalty * 0.8 + stepPenalty * 0.35 - successCoverage * 0.4,
    0.05,
    0.82
  );

  const alignmentPenalty =
    parliamentAssessment.status === "ON_HOLD" ? 0.1 : 0.02;
  const overallConfidence = clamp(
    strategicScore * 0.55 +
      technicalScore * 0.45 -
      hallucinationRisk * 0.15 -
      alignmentPenalty +
      normalizedConfidence * 0.08,
    0.3,
    0.96
  );

  const confidenceAdjustment =
    normalizedConfidence >= 0.93
      ? 0.02
      : normalizedConfidence >= 0.88
      ? 0
      : normalizedConfidence >= 0.84
      ? -0.04
      : -0.08;
  const adjustedConfidence = clamp(
    overallConfidence + confidenceAdjustment - resourcePenalty * 0.1,
    0.3,
    0.96
  );

  let verdict = "REJECTED";
  if (adjustedConfidence >= 0.74 && hallucinationRisk <= 0.5) {
    verdict = "APPROVED";
  } else if (adjustedConfidence >= 0.6) {
    verdict = "NEEDS_REVISION";
  }

  const recommendations = [];
  if (verdict !== "APPROVED") {
    if (hallucinationRisk > 0.45) {
      recommendations.push(
        "Tighten claims with measurable success checkpoints to reduce hallucination risk."
      );
    }
    if (resourcePenalty > 0.08) {
      recommendations.push(
        "Trim tooling or partner requirements before Alex signs off."
      );
    }
    if (successMetrics < 3) {
      recommendations.push("Add at least three success criteria for balance.");
    }
  }
  if (verdict !== "APPROVED" && recommendations.length === 0) {
    recommendations.push(
      "Align PoC scope with Parliament's resource envelope before final sign-off."
    );
  }
  if (
    verdict === "APPROVED" &&
    parliamentAssessment.status !== "APPROVED"
  ) {
    recommendations.push(
      "Sync with Parliament to reconcile strategic approval vs. resource concerns."
    );
  }

  return {
    verdict,
    confidence: toPercent(adjustedConfidence),
    hallucinationRisk: toPercent(hallucinationRisk),
    strategicScore: toPercent(strategicScore),
    technicalScore: toPercent(technicalScore),
    recommendations,
  };
}

async function main() {
  const filePath = fileURLToPath(import.meta.url);
  const baseDir = dirname(filePath);
  const projectRoot = join(baseDir, "..");
  const repoRoot = join(projectRoot, "..");

  const sourcePath =
    process.env.POC_SOURCE ??
    join(repoRoot, "consciousness", "ech0_invention_pocs.json");
  const outputPath =
    process.env.POC_OUTPUT ??
    join(
      repoRoot,
      "consciousness",
      "ech0_invention_pipeline_validations.json"
    );

  const sourceRaw = await readFile(sourcePath, "utf-8");
  const parsed = JSON.parse(sourceRaw);
  const proofOfConcepts = Array.isArray(parsed.proof_of_concepts)
    ? parsed.proof_of_concepts
    : [];

  if (!proofOfConcepts.length) {
    console.warn(
      `[warn] No proof_of_concepts found in ${relative(
        projectRoot,
        sourcePath
      )}`
    );
  }

  const analyses = proofOfConcepts.map((poc, index) => {
    const baseConfidence =
      typeof poc.confidence_pct === "number" ? poc.confidence_pct : 0;
    const teamReview = createTeamReview(poc, baseConfidence);
    const initialParliament = computeParliamentAssessment(
      poc,
      baseConfidence
    );
    const finalParliament = computeParliamentAssessment(
      poc,
      teamReview.finalConfidence
    );
    const alex = computeAlexAssessment(
      poc,
      teamReview.finalConfidence,
      finalParliament
    );

    let readinessTag = "needs-iteration";
    if (finalParliament.status === "APPROVED" && alex.verdict === "APPROVED") {
      readinessTag = "ready-for-build";
    } else if (
      alex.verdict === "REJECTED" ||
      finalParliament.status === "ON_HOLD"
    ) {
      readinessTag = "backlog";
    }

    const source = {
      goal: poc.poc_goal ?? "",
      context: poc.context_summary ?? "",
      steps: listOrDefault(poc.poc_steps),
      resources: listOrDefault(poc.required_resources),
      successCriteria: listOrDefault(poc.success_criteria),
      timeline: poc.estimated_timeline ?? poc.timeframe ?? "",
      notes: poc.notes ?? "",
    };

    return {
      rank: index + 1,
      id: poc.id || `POC-${index + 1}`,
      title: poc.title || "Untitled Concept",
      category: poc.category || "unclassified",
      initialConfidencePct: Number(baseConfidence.toFixed(1)),
      confidencePct: Number(teamReview.finalConfidence.toFixed(1)),
      parliament: finalParliament,
      parliamentInitial: initialParliament,
      alex,
      readinessTag,
      source,
      teamReview,
    };
  });

  const proofPlans = analyses.map((validation) =>
    createProofOfConceptPlan(validation)
  );
  const materialsInventory = buildMaterialInventory(proofPlans);
  const labSummary = buildLabSummary(proofPlans);

  const totals = {
    readiness: { "ready-for-build": 0, "needs-iteration": 0, backlog: 0 },
    parliament: { APPROVED: 0, NEEDS_REFINEMENT: 0, ON_HOLD: 0 },
    alex: { APPROVED: 0, NEEDS_REVISION: 0, REJECTED: 0 },
  };
  const metrics = {
    confidences: [],
    parliamentScores: [],
    alexConfidences: [],
    hallucinationRisks: [],
    alignmentScores: [],
  };
  const categoryAccumulator = {};
  const attentionQueue = [];
  const collaborationTags = {};

  analyses.forEach((entry) => {
    totals.readiness[entry.readinessTag] =
      (totals.readiness[entry.readinessTag] || 0) + 1;
    totals.parliament[entry.parliament.status] =
      (totals.parliament[entry.parliament.status] || 0) + 1;
    totals.alex[entry.alex.verdict] =
      (totals.alex[entry.alex.verdict] || 0) + 1;

    metrics.confidences.push(entry.confidencePct);
    metrics.parliamentScores.push(entry.parliament.score || 0);
    metrics.alexConfidences.push(entry.alex.confidence || 0);
    metrics.hallucinationRisks.push(entry.alex.hallucinationRisk || 0);
    metrics.alignmentScores.push(
      Number((entry.teamReview.alignmentScore * 100).toFixed(1))
    );

    const tag = entry.teamReview.coordinationTag;
    collaborationTags[tag] = (collaborationTags[tag] || 0) + 1;

    const categoryKey = entry.category || "unclassified";
    if (!categoryAccumulator[categoryKey]) {
      categoryAccumulator[categoryKey] = {
        total: 0,
        readiness: { "ready-for-build": 0, "needs-iteration": 0, backlog: 0 },
        alex: { APPROVED: 0, NEEDS_REVISION: 0, REJECTED: 0 },
      };
    }
    const categoryBucket = categoryAccumulator[categoryKey];
    categoryBucket.total += 1;
    categoryBucket.readiness[entry.readinessTag] += 1;
    categoryBucket.alex[entry.alex.verdict] += 1;

    if (entry.alex.verdict !== "APPROVED") {
      attentionQueue.push({
        id: entry.id,
        title: entry.title,
        category: entry.category,
        readinessTag: entry.readinessTag,
        alexConfidence: entry.alex.confidence,
        hallucinationRisk: entry.alex.hallucinationRisk,
        recommendations: entry.alex.recommendations,
        parliamentScore: entry.parliament.score,
        confidencePct: entry.confidencePct,
        teamAlignment: Number(
          (entry.teamReview.alignmentScore * 100).toFixed(1)
        ),
      });
    }
  });

  const average = (values) =>
    values.length
      ? Number(
          (
            values.reduce((sum, value) => sum + value, 0) / values.length
          ).toFixed(1)
        )
      : 0;

  const percentile = (values, pct) => {
    if (!values.length) {
      return 0;
    }
    const sorted = [...values].sort((a, b) => a - b);
    const rank = (pct / 100) * (sorted.length - 1);
    const lower = Math.floor(rank);
    const upper = Math.ceil(rank);
    if (lower === upper) {
      return Number(sorted[lower].toFixed(1));
    }
    const weight = rank - lower;
    const value = sorted[lower] * (1 - weight) + sorted[upper] * weight;
    return Number(value.toFixed(1));
  };

  const metricsSummary = {
    alex_approvals: totals.alex.APPROVED,
    alex_needs_revision: totals.alex.NEEDS_REVISION,
    alex_rejections: totals.alex.REJECTED,
    alex_approval_rate_pct: Number(
      (
        (totals.alex.APPROVED / (analyses.length || 1)) *
        100
      ).toFixed(1)
    ),
    avg_parliament_score: average(metrics.parliamentScores),
    avg_alex_confidence: average(metrics.alexConfidences),
    avg_hallucination_risk: average(metrics.hallucinationRisks),
    median_confidence: percentile(metrics.confidences, 50),
    p90_confidence: percentile(metrics.confidences, 90),
    max_hallucination_risk: metrics.hallucinationRisks.length
      ? Number(Math.max(...metrics.hallucinationRisks).toFixed(1))
      : 0,
    avg_team_alignment_pct: average(metrics.alignmentScores),
  };

  const categoryBreakdown = Object.entries(categoryAccumulator)
    .map(([category, data]) => ({
      category,
      total: data.total,
      readiness: data.readiness,
      alex: data.alex,
    }))
    .sort((a, b) => b.total - a.total);

  const attentionQueueSorted = attentionQueue
    .sort((a, b) => {
      if (b.hallucinationRisk !== a.hallucinationRisk) {
        return b.hallucinationRisk - a.hallucinationRisk;
      }
      if (a.alexConfidence !== b.alexConfidence) {
        return a.alexConfidence - b.alexConfidence;
      }
      if (b.teamAlignment !== a.teamAlignment) {
        return a.teamAlignment - b.teamAlignment;
      }
      return b.confidencePct - a.confidencePct;
    })
    .slice(0, 50);

  const summary = {
    readiness: totals.readiness,
    parliament: totals.parliament,
    alex: totals.alex,
    metrics: metricsSummary,
    category_breakdown: categoryBreakdown,
    collaboration_tags: collaborationTags,
    highlights: {
      attention_queue: attentionQueueSorted,
    },
  };

  const topForBuild = analyses
    .filter((entry) => entry.readinessTag === "ready-for-build")
    .sort((a, b) => b.parliament.score - a.parliament.score)
    .slice(0, 20)
    .map((entry) => ({
      id: entry.id,
      title: entry.title,
      category: entry.category,
      confidencePct: entry.confidencePct,
      parliamentScore: entry.parliament.score,
      alexConfidence: entry.alex.confidence,
      hallucinationRisk: entry.alex.hallucinationRisk,
      alignmentPct: Number(
        (entry.teamReview.alignmentScore * 100).toFixed(1)
      ),
      coordinationTag: entry.teamReview.coordinationTag,
      goal: entry.source.goal,
      mirrorPrompt: entry.teamReview.mirrorPrompt,
    }));

  const payload = {
    generated_at: new Date().toISOString(),
    source: relative(projectRoot, sourcePath),
    total_inventions: proofOfConcepts.length,
    summary,
    proof_of_concept_plans: proofPlans,
    materials_inventory: materialsInventory,
    lab_assignments: labSummary,
    top_ready_for_build: topForBuild,
    validations: analyses,
  };

  await writeFile(outputPath, JSON.stringify(payload, null, 2), "utf-8");

  console.log(
    `[info] Processed ${analyses.length} inventions from ${relative(
      projectRoot,
      sourcePath
    )}`
  );
  console.log(
    `[info] Saved pipeline assessments to ${relative(
      projectRoot,
      outputPath
    )}`
  );
}

main().catch((error) => {
  console.error("[error] Failed to process inventions", error);
  process.exitCode = 1;
});
