import { jitter, normalize } from "../utils/random.js";

const defaultParameterRanges = {
  temperature: { min: 150, max: 2500 },
  pressure: { min: 0.5, max: 25 },
  dopantRatio: { min: 0, max: 0.4 },
  annealingTime: { min: 10, max: 720 },
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function scaleValue(value, { min, max }) {
  return clamp(value, min, max);
}

export class ExperimentOptimizer {
  constructor(config = {}) {
    const ranges = config.parameterRanges || defaultParameterRanges;
    this.ranges = ranges;
    this.learningRate = config.learningRate || 0.1;
    this.explorationWeight = config.explorationWeight || 0.3;
    this.exploitationWeight = config.exploitationWeight || 0.7;
    this.maxIterations = config.maxIterations || 50;
    this.history = [];
  }

  scoreResult(result) {
    const metrics = [
      result.performance.metrics.energyEfficiency,
      result.performance.metrics.structuralIntegrity,
      1 - result.performance.metrics.predictionError,
    ];
    const normalized = normalize(metrics);
    return (
      normalized[0] * 0.35 + normalized[1] * 0.45 + normalized[2] * 0.2
    );
  }

  registerResult(result) {
    const score = this.scoreResult(result);
    this.history.push({ result, score });
    if (this.history.length > this.maxIterations) {
      this.history.shift();
    }
  }

  proposeNextParameters(currentParameters) {
    if (this.history.length === 0) {
      return this.randomizeInitialParameters();
    }

    const best = [...this.history].sort((a, b) => b.score - a.score)[0];

    const explorationVector = this.randomizeInitialParameters();
    const exploitationVector = best.result.parameters;

    const nextParameters = {};
    Object.keys(this.ranges).forEach((key) => {
      const explorationComponent =
        explorationVector[key] * this.explorationWeight;
      const exploitationComponent =
        exploitationVector[key] * this.exploitationWeight;
      const currentComponent =
        currentParameters[key] * (1 - this.learningRate);
      const combined =
        currentComponent +
        this.learningRate * (explorationComponent + exploitationComponent);
      nextParameters[key] = scaleValue(combined, this.ranges[key]);
    });

    return nextParameters;
  }

  randomizeInitialParameters() {
    const parameters = {};
    Object.keys(this.ranges).forEach((key) => {
      const { min, max } = this.ranges[key];
      const value = min + Math.random() * (max - min);
      parameters[key] = scaleValue(jitter(value, (max - min) * 0.05), {
        min,
        max,
      });
    });
    return parameters;
  }
}
