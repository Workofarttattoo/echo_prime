export const defaultConfig = {
  port: Number.parseInt(process.env.PORT || "4100", 10),
  tickIntervalMs: Number.parseInt(process.env.TICK_INTERVAL_MS || "2500", 10),
  workflow: {
    maxConcurrent: Number.parseInt(
      process.env.MAX_CONCURRENT_WORKFLOWS || "3",
      10
    ),
    stages: [
      "ingesting-data",
      "predicting-crystal-structure",
      "running-simulations",
      "optimizing-experiment",
      "awaiting-validation",
    ],
  },
  discovery: {
    explorationBias: Number.parseFloat(
      process.env.DISCOVERY_EXPLORATION_BIAS || "0.35"
    ),
    exploitBias: Number.parseFloat(
      process.env.DISCOVERY_EXPLOIT_BIAS || "0.55"
    ),
    noveltyThreshold: Number.parseFloat(
      process.env.DISCOVERY_NOVELTY_THRESHOLD || "0.6"
    ),
  },
  predictor: {
    latticeSpacing: Number.parseFloat(
      process.env.PREDICTOR_LATTICE_SPACING || "1.8"
    ),
    thermalExpansionCoeff: Number.parseFloat(
      process.env.PREDICTOR_EXPANSION_COEFF || "0.0045"
    ),
  },
  modeling: {
    scales: [
      { id: "atomic", emphasis: 0.6 },
      { id: "mesoscopic", emphasis: 0.25 },
      { id: "continuum", emphasis: 0.15 },
    ],
    couplingStrength: Number.parseFloat(
      process.env.MODELING_COUPLING_STRENGTH || "0.42"
    ),
  },
  optimization: {
    maxIterations: Number.parseInt(
      process.env.OPT_MAX_ITERATIONS || "80",
      10
    ),
    learningRate: Number.parseFloat(
      process.env.OPT_LEARNING_RATE || "0.12"
    ),
    explorationWeight: Number.parseFloat(
      process.env.OPT_EXPLORATION_WEIGHT || "0.3"
    ),
    exploitationWeight: Number.parseFloat(
      process.env.OPT_EXPLOITATION_WEIGHT || "0.7"
    ),
  },
};
