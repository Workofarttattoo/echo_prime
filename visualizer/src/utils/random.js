const random = Math.random;

export function randomBetween(min, max) {
  return min + (max - min) * random();
}

export function randomChoice(list) {
  if (!Array.isArray(list) || list.length === 0) {
    throw new Error("randomChoice expects a non-empty array");
  }
  const index = Math.floor(random() * list.length);
  return list[index];
}

export function jitter(value, magnitude = 0.1) {
  return value + randomBetween(-magnitude, magnitude);
}

export function normalize(values) {
  const sum = values.reduce((total, value) => total + value, 0);
  if (sum === 0) {
    return values.map(() => 0);
  }
  return values.map((value) => value / sum);
}

export function weightedRandomChoice(weightedItems) {
  const weights = weightedItems.map((item) => item.weight);
  const normalized = normalize(weights);
  const roll = random();
  let accumulator = 0;
  for (let index = 0; index < weightedItems.length; index += 1) {
    accumulator += normalized[index];
    if (roll <= accumulator) {
      return weightedItems[index].item;
    }
  }
  return weightedItems[weightedItems.length - 1].item;
}
