import { randomBetween, jitter } from "../utils/random.js";

function buildLatticeVectors(baseLattice, spacing) {
  if (baseLattice === "hcp") {
    return [
      [spacing, 0, 0],
      [spacing / 2, (Math.sqrt(3) / 2) * spacing, 0],
      [0, 0, spacing * 1.63],
    ];
  }
  if (baseLattice === "perovskite") {
    return [
      [spacing, 0, 0],
      [0, spacing, 0],
      [0, 0, spacing],
    ];
  }
  if (baseLattice === "hexagonal") {
    return [
      [spacing, 0, 0],
      [spacing / 2, (Math.sqrt(3) / 2) * spacing, 0],
      [0, 0, spacing],
    ];
  }
  if (baseLattice === "diamond") {
    return [
      [spacing, 0, 0],
      [0, spacing, 0],
      [0, 0, spacing],
    ];
  }
  return [
    [spacing, 0, 0],
    [0, spacing, 0],
    [0, 0, spacing],
  ];
}

function generateAtomicBasis(baseLattice) {
  if (baseLattice === "diamond") {
    return [
      [0, 0, 0],
      [0.25, 0.25, 0.25],
      [0.5, 0.5, 0],
      [0.75, 0.75, 0.25],
    ];
  }
  if (baseLattice === "hcp") {
    return [
      [0, 0, 0],
      [2 / 3, 1 / 3, 0.5],
    ];
  }
  if (baseLattice === "perovskite") {
    return [
      [0, 0, 0],
      [0.5, 0.5, 0.5],
      [0, 0.5, 0.5],
      [0.5, 0, 0.5],
      [0.5, 0.5, 0],
    ];
  }
  return [
    [0, 0, 0],
    [0.5, 0.5, 0],
    [0.5, 0, 0.5],
    [0, 0.5, 0.5],
  ];
}

function scaleVector(vector, scalar) {
  return vector.map((component) => component * scalar);
}

function addVectors(vectorA, vectorB) {
  return vectorA.map((component, index) => component + vectorB[index]);
}

function wrapPosition(position, spacing) {
  return position.map((component) => component % spacing);
}

export class CrystalStructurePredictor {
  constructor(config = {}) {
    this.latticeSpacing = config.latticeSpacing || 1.5;
    this.thermalExpansionCoeff =
      config.thermalExpansionCoeff || 0.0035;
  }

  predictStructure(material, conditions) {
    const { temperature, pressure } = conditions;
    const deltaTemperature = (temperature - 300) * this.thermalExpansionCoeff;
    const effectiveSpacing = this.latticeSpacing * (1 + deltaTemperature);
    const baseVectors = buildLatticeVectors(
      material.baseLattice,
      effectiveSpacing
    );
    const basis = generateAtomicBasis(material.baseLattice);

    const atoms = [];
    const unitCells = 3;
    for (let x = 0; x < unitCells; x += 1) {
      for (let y = 0; y < unitCells; y += 1) {
        for (let z = 0; z < unitCells; z += 1) {
          const cellOrigin = [
            baseVectors[0][0] * x +
              baseVectors[1][0] * y +
              baseVectors[2][0] * z,
            baseVectors[0][1] * x +
              baseVectors[1][1] * y +
              baseVectors[2][1] * z,
            baseVectors[0][2] * x +
              baseVectors[1][2] * y +
              baseVectors[2][2] * z,
          ];
          basis.forEach((fractional) => {
            const cartesian = [
              fractional[0] * baseVectors[0][0] +
                fractional[1] * baseVectors[1][0] +
                fractional[2] * baseVectors[2][0],
              fractional[0] * baseVectors[0][1] +
                fractional[1] * baseVectors[1][1] +
                fractional[2] * baseVectors[2][1],
              fractional[0] * baseVectors[0][2] +
                fractional[1] * baseVectors[1][2] +
                fractional[2] * baseVectors[2][2],
            ];
            const position = addVectors(cellOrigin, cartesian).map((value) =>
              jitter(value, effectiveSpacing * 0.02)
            );
            atoms.push({
              position,
              element: this.assignElement(material),
              partialCharge: randomBetween(-0.4, 0.4),
            });
          });
        }
      }
    }

    const bonds = this.buildBonds(atoms, effectiveSpacing);

    return {
      materialId: material.id,
      baseLattice: material.baseLattice,
      atoms,
      bonds,
      metadata: {
        effectiveSpacing,
        predictedDensity: this.estimateDensity(material, pressure),
        predictedBandGap: this.estimateBandGap(material, temperature),
      },
    };
  }

  assignElement(material) {
    const elements = Object.keys(material.composition);
    const weights = elements.map((element) => material.composition[element]);
    const total = weights.reduce((sum, value) => sum + value, 0);
    const normalized = weights.map((value) => value / total);
    const roll = Math.random();
    let accumulator = 0;
    for (let index = 0; index < elements.length; index += 1) {
      accumulator += normalized[index];
      if (roll <= accumulator) {
        return elements[index];
      }
    }
    return elements[0];
  }

  estimateDensity(material, pressure) {
    const base = material.descriptors?.density || 5;
    return base * (1 + pressure * 0.01);
  }

  estimateBandGap(material, temperature) {
    const baseEntropy = material.descriptors?.entropy || 0.5;
    return Math.max(0.1, 2.6 - baseEntropy - temperature * 0.0004);
  }

  buildBonds(atoms, spacing) {
    const bonds = [];
    const threshold = spacing * 0.9;
    for (let index = 0; index < atoms.length; index += 1) {
      for (let neighbor = index + 1; neighbor < atoms.length; neighbor += 1) {
        const distance = this.calculateDistance(
          atoms[index].position,
          atoms[neighbor].position
        );
        if (distance <= threshold) {
          bonds.push({
            source: index,
            target: neighbor,
            distance,
          });
        }
      }
    }
    return bonds;
  }

  calculateDistance(positionA, positionB) {
    return Math.sqrt(
      (positionA[0] - positionB[0]) ** 2 +
        (positionA[1] - positionB[1]) ** 2 +
        (positionA[2] - positionB[2]) ** 2
    );
  }
}
