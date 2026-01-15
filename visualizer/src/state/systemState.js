export class SystemState {
  constructor() {
    this.current = {
      workflows: [],
      predictions: [],
      modeling: [],
      optimization: [],
      discovery: [],
      timestamp: Date.now(),
    };
  }

  update(partial) {
    this.current = {
      ...this.current,
      ...partial,
      timestamp: Date.now(),
    };
    return this.current;
  }

  snapshot() {
    return this.current;
  }
}
