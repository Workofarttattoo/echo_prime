import { normalize } from "../utils/random.js";

function aggregateByScale(atoms, scale) {
  if (scale === "atomic") {
    return {
      coordinationNumber: estimateCoordination(atoms),
      averageCharge: average(atoms.map((atom) => atom.partialCharge)),
      anisotropy: estimateAnisotropy(atoms),
    };
  }

  if (scale === "mesoscopic") {
    return {
      grainSize: estimateGrainSize(atoms),
      defectDensity: estimateDefectDensity(atoms),
      orientationEntropy: estimateOrientationEntropy(atoms),
    };
  }

  return {
    elasticModulus: estimateElasticModulus(atoms),
    thermalConductivity: estimateThermalConductivity(atoms),
    fractureToughness: estimateFractureToughness(atoms),
  };
}

function average(values) {
  if (values.length === 0) {
    return 0;
  }
  const total = values.reduce((sum, value) => sum + value, 0);
  return total / values.length;
}

function estimateCoordination(atoms) {
  const neighborCounts = atoms.map((atom) => Math.floor(Math.random() * 6) + 6);
  return average(neighborCounts);
}

function estimateAnisotropy(atoms) {
  return normalize([
    Math.abs(average(atoms.map((atom) => atom.position[0]))),
    Math.abs(average(atoms.map((atom) => atom.position[1]))),
    Math.abs(average(atoms.map((atom) => atom.position[2]))),
  ]).reduce((sum, value) => sum + value, 0);
}

function estimateGrainSize(atoms) {
  return average(atoms.map((atom) => Math.abs(atom.partialCharge))) * 35 + 20;
}

function estimateDefectDensity(atoms) {
  return 0.05 + Math.abs(average(atoms.map((atom) => atom.partialCharge))) * 2;
}

function estimateOrientationEntropy(atoms) {
  return Math.min(1, 0.2 + Math.random() * 0.6);
}

function estimateElasticModulus(atoms) {
  return 120 + average(atoms.map((atom) => atom.partialCharge ** 2)) * 600;
}

function estimateThermalConductivity(atoms) {
  return 2 + average(atoms.map((atom) => Math.abs(atom.partialCharge))) * 10;
}

function estimateFractureToughness(atoms) {
  return 15 + average(atoms.map((atom) => atom.partialCharge ** 2)) * 50;
}

export class MultiScaleModeler {
  constructor(config = {}) {
    this.scales = config.scales || [
      { id: "atomic", emphasis: 0.6 },
      { id: "mesoscopic", emphasis: 0.3 },
      { id: "continuum", emphasis: 0.1 },
    ];
    this.couplingStrength = config.couplingStrength || 0.4;
  }

  modelStructure(structure, conditions) {
    const results = this.scales.map((scale) => {
      const metrics = aggregateByScale(structure.atoms, scale.id);
      return {
        id: scale.id,
        metrics,
        emphasis: scale.emphasis,
      };
    });

    return {
      structureId: structure.materialId,
      conditions,
      scales: results,
      coupledScore: this.calculateCoupledScore(results),
    };
  }

  calculateCoupledScore(results) {
    const weighted = results.map(
      (result) =>
        result.emphasis *
        average(Object.values(result.metrics).map((value) => Number(value)))
    );
    const normalized = normalize(weighted);
    return normalized.reduce((sum, value) => sum + value, 0) * this.couplingStrength;
  }
}
