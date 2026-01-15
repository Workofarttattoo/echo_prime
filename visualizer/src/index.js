import { EventEmitter } from "node:events";
import { randomBetween } from "./utils/random.js";
import { defaultConfig } from "../config/default.js";
import { MaterialDiscoveryAgent } from "./ai/materialDiscoveryAgent.js";
import { CrystalStructurePredictor } from "./prediction/crystalStructurePredictor.js";
import { MultiScaleModeler } from "./modeling/multiScaleModeler.js";
import { ExperimentOptimizer } from "./optimization/experimentOptimizer.js";
import { WorkflowManager } from "./workflows/workflowManager.js";
import { SystemState } from "./state/systemState.js";

function buildWorkflowDefinition(material, parameters) {
  return {
    id: `workflow-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    material,
    parameters,
    createdAt: Date.now(),
  };
}

function derivePerformance(structure, modelingResult) {
  const structuralIntegrity =
    modelingResult.coupledScore * randomBetween(0.8, 1.1);
  const energyEfficiency =
    0.5 +
    (1 - structure.metadata.predictedDensity / 20) * 0.45;
  const predictionError = Math.max(
    0,
    1 - modelingResult.coupledScore * randomBetween(0.8, 1.2)
  );

  return {
    compositeScore:
      structuralIntegrity * 0.45 +
      energyEfficiency * 0.35 +
      (1 - predictionError) * 0.2,
    metrics: {
      structuralIntegrity,
      energyEfficiency,
      predictionError,
    },
  };
}

export class VisualizationOrchestrator {
  constructor(config = {}) {
    this.config = {
      ...defaultConfig,
      ...config,
      workflow: { ...defaultConfig.workflow, ...config.workflow },
      discovery: { ...defaultConfig.discovery, ...config.discovery },
      predictor: { ...defaultConfig.predictor, ...config.predictor },
      modeling: { ...defaultConfig.modeling, ...config.modeling },
      optimization: {
        ...defaultConfig.optimization,
        ...config.optimization,
      },
    };
    this.events = new EventEmitter();
    this.discoveryAgent = new MaterialDiscoveryAgent(this.config.discovery);
    this.predictor = new CrystalStructurePredictor(this.config.predictor);
    this.modeler = new MultiScaleModeler(this.config.modeling);
    this.optimizer = new ExperimentOptimizer(this.config.optimization);
    this.workflowManager = new WorkflowManager({
      maxConcurrent: this.config.workflow.maxConcurrent,
      stages: this.config.workflow.stages,
    });
    this.state = new SystemState();
    this.interval = null;
  }

  start() {
    if (this.interval) {
      return;
    }
    this.interval = setInterval(
      () => this.tick(),
      this.config.tickIntervalMs
    );
    this.tick();
  }

  stop() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  onState(listener) {
    this.events.on("state", listener);
  }

  offState(listener) {
    this.events.off("state", listener);
  }

  tick() {
    const candidate = this.discoveryAgent.proposeCandidate();
    const workflowDefinition = buildWorkflowDefinition(
      candidate.material,
      candidate.conditions
    );
    this.workflowManager.scheduleWorkflow(workflowDefinition);

    const predictedStructure = this.predictor.predictStructure(
      candidate.material,
      candidate.conditions
    );
    const modelingResult = this.modeler.modelStructure(
      predictedStructure,
      candidate.conditions
    );
    const performance = derivePerformance(
      predictedStructure,
      modelingResult
    );

    this.optimizer.registerResult({
      material: candidate.material,
      parameters: candidate.conditions,
      structure: predictedStructure,
      modeling: modelingResult,
      performance,
    });

    const nextParameters = this.optimizer.proposeNextParameters(
      candidate.conditions
    );
    this.discoveryAgent.registerFeedback({
      material: candidate.material,
      performance,
    });

    const workflows = this.workflowManager.updateWorkflows(Date.now(), {
      [workflowDefinition.id]: {
        metrics: performance.metrics,
      },
    });
    this.workflowManager.ensureCapacity();

    const snapshot = this.state.update({
      workflows,
      predictions: [
        {
          materialId: predictedStructure.materialId,
          baseLattice: predictedStructure.baseLattice,
          metadata: predictedStructure.metadata,
          atomCount: predictedStructure.atoms.length,
        },
      ],
      modeling: [modelingResult],
      optimization: [
        {
          suggestedParameters: nextParameters,
          previousParameters: candidate.conditions,
        },
      ],
      discovery: [
        {
          material: candidate.material,
          performance,
        },
      ],
      structure: predictedStructure,
    });

    this.events.emit("state", snapshot);
  }
}

export function createVisualizationSystem(config) {
  const orchestrator = new VisualizationOrchestrator(config);
  orchestrator.start();
  return orchestrator;
}
