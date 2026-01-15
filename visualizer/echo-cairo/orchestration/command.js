export class Command {
  constructor({ update = null, goto = null, resume = null, interrupt = null } = {}) {
    this.update = update;
    this.goto = goto;
    this.resume = resume;
    this.interrupt = interrupt;
  }

  static update(update) {
    return new Command({ update });
  }

  static goto(goto, update = null) {
    return new Command({ goto, update });
  }

  static interrupt(interrupt, options = {}) {
    return new Command({ interrupt, ...options });
  }

  static resume(resume, options = {}) {
    return new Command({ resume, ...options });
  }
}

export function isCommand(value) {
  return value instanceof Command;
}
