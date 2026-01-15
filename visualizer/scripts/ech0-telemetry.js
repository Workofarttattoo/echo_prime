import { randomUUID } from "node:crypto";

export class TelemetryTracker {
  constructor(options = {}) {
    this.enabled = options.enabled ?? true;
    this.events = [];
    this.activeSpans = new Map();
    this.counters = {
      promptTokens: 0,
      completionTokens: 0,
      successes: 0,
      failures: 0,
    };
  }

  startSpan(name, metadata = {}) {
    if (!this.enabled) {
      return { end: () => null };
    }
    const id = randomUUID();
    const span = {
      id,
      name,
      metadata,
      start: Date.now(),
    };
    this.activeSpans.set(id, span);
    return {
      id,
      end: (extra = {}) => this.finishSpan(id, extra),
    };
  }

  finishSpan(id, extra = {}) {
    if (!this.enabled) {
      return null;
    }
    const span = this.activeSpans.get(id);
    if (!span) {
      return null;
    }
    const end = Date.now();
    const entry = {
      ...span,
      end,
      durationMs: end - span.start,
      extra,
      recordedAt: new Date(end).toISOString(),
    };
    this.activeSpans.delete(id);
    this.events.push({ type: "span", ...entry });
    return entry;
  }

  recordTokenUsage({ prompt = 0, completion = 0 } = {}) {
    if (!this.enabled) {
      return null;
    }
    this.counters.promptTokens += prompt;
    this.counters.completionTokens += completion;
    const entry = {
      type: "token-usage",
      prompt,
      completion,
      total: prompt + completion,
      recordedAt: new Date().toISOString(),
    };
    this.events.push(entry);
    return entry;
  }

  recordOutcome(success) {
    if (!this.enabled) {
      return null;
    }
    if (success) {
      this.counters.successes += 1;
    } else {
      this.counters.failures += 1;
    }
    const entry = {
      type: "outcome",
      success,
      recordedAt: new Date().toISOString(),
    };
    this.events.push(entry);
    return entry;
  }

  recordEvent(name, payload = {}) {
    if (!this.enabled) {
      return null;
    }
    const entry = {
      type: "event",
      name,
      payload,
      recordedAt: new Date().toISOString(),
    };
    this.events.push(entry);
    return entry;
  }

  summary() {
    return {
      enabled: this.enabled,
      counters: { ...this.counters },
      recentEvents: this.events.slice(-5),
    };
  }
}
