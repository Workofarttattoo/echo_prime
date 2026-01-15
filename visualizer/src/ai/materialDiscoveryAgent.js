import { weightedRandomChoice, randomBetween } from "../utils/random.js";
import { seedMaterials } from "../data/materialLibrary.js";

function computeNoveltyScore(material, history) {
  if (history.length === 0) {
    return 1;
  }
  const averageStability =
    history.reduce((sum, entry) => sum + entry.material.stability, 0) /
    history.length;
  return Math.max(0, material.stability - averageStability) + 0.5;
}

function computePerformanceScore(material, feedback) {
  const densityScore = 1 / (1 + Math.abs(material.descriptors.density - 5));
  const entropyScore = 1 - material.descriptors.entropy * 0.5;
  const feedbackScore = feedback.length
    ? feedback.reduce((sum, entry) => sum + entry.score, 0) / feedback.length
    : 0.6;
  return (densityScore + entropyScore + feedbackScore) / 3;
}

function mixComposition(base, adjustments) {
  const composition = { ...base };
  Object.keys(adjustments).forEach((key) => {
    composition[key] = Math.max(
      0,
      (composition[key] || 0) + adjustments[key]
    );
  });
  const total = Object.values(composition).reduce(
    (sum, value) => sum + value,
    0
  );
  Object.keys(composition).forEach((key) => {
    composition[key] /= total;
  });
  return composition;
}

function mutateMaterial(baseMaterial) {
  const dopant = {
    compositionDelta: { b: 0.01, n: 0.01 },
    stabilityDelta: randomBetween(-0.05, 0.04),
  };

  return {
    id: `${baseMaterial.id}-variant-${Math.random()
      .toString(36)
      .slice(2, 6)}`,
    name: `${baseMaterial.name} Variant`,
    baseLattice: baseMaterial.baseLattice,
    composition: mixComposition(baseMaterial.composition, dopant.compositionDelta),
    stability: Math.min(1, Math.max(0, baseMaterial.stability + dopant.stabilityDelta)),
    descriptors: {
      density: baseMaterial.descriptors.density + randomBetween(-0.4, 0.4),
      hardness: baseMaterial.descriptors.hardness + randomBetween(-0.3, 0.25),
      entropy: Math.max(0, baseMaterial.descriptors.entropy + randomBetween(-0.1, 0.1)),
    },
  };
}

export class MaterialDiscoveryAgent {
  constructor(config = {}) {
    this.explorationBias = config.explorationBias || 0.4;
    this.exploitBias = config.exploitBias || 0.6;
    this.noveltyThreshold = config.noveltyThreshold || 0.5;
    this.discoveryHistory = [];
    this.feedback = [];
  }

  proposeCandidate() {
    const weightedOptions = seedMaterials.map((material) => {
      const novelty = computeNoveltyScore(material, this.discoveryHistory);
      const performance = computePerformanceScore(material, this.feedback);
      const explorationWeight = this.explorationBias * novelty;
      const exploitationWeight = this.exploitBias * performance;
      const weight = explorationWeight + exploitationWeight;
      return { item: material, weight };
    });

    const baseSelection = weightedRandomChoice(weightedOptions);
    const shouldMutate = computeNoveltyScore(baseSelection, this.discoveryHistory) <
      this.noveltyThreshold;

    const candidate = shouldMutate
      ? mutateMaterial(baseSelection)
      : baseSelection;

    const conditions = {
      temperature: randomBetween(450, 1800),
      pressure: randomBetween(1.2, 15),
      dopantRatio: randomBetween(0, 0.32),
      annealingTime: randomBetween(20, 480),
    };

    return { material: candidate, conditions };
  }

  registerFeedback(result) {
    const score = result.performance.compositeScore;
    this.feedback.push({ score, timestamp: Date.now() });
    if (this.feedback.length > 250) {
      this.feedback.shift();
    }
    this.discoveryHistory.push(result.material);
    if (this.discoveryHistory.length > 250) {
      this.discoveryHistory.shift();
    }
  }
}
