import https from "node:https";
import { URL } from "node:url";

const BASE_URL = "https://www.courtlistener.com/api/rest/v3";

export const COURT_LISTENER_ENDPOINTS = new Set([
  "dockets",
  "docket-entries",
  "opinions",
  "recap-documents",
]);

function buildHeaders(apiKey, userAgent) {
  return {
    Authorization: `Token ${apiKey}`,
    Accept: "application/json",
    "User-Agent": userAgent,
  };
}

function requestJson(targetUrl, headers) {
  return new Promise((resolve, reject) => {
    const urlObject = new URL(targetUrl);

    const request = https.request(
      urlObject,
      {
        method: "GET",
        headers,
      },
      (response) => {
        let body = "";

        response.setEncoding("utf8");
        response.on("data", (chunk) => {
          body += chunk;
        });

        response.on("end", () => {
          const status = response.statusCode || 0;

          if (status < 200 || status >= 300) {
            const message = body ? body.slice(0, 240) : "No response body";
            reject(
              new Error(
                `CourtListener request failed with status ${status}: ${message}`
              )
            );
            return;
          }

          if (!body) {
            resolve({});
            return;
          }

          try {
            resolve(JSON.parse(body));
          } catch (error) {
            reject(
              new Error(`Unable to parse CourtListener response: ${error.message}`)
            );
          }
        });
      }
    );

    request.on("error", (error) => {
      reject(error);
    });

    request.end();
  });
}

export class CourtListenerClient {
  constructor({ apiKey, userAgent = "gavl-courtlistener-ingest/0.1.0" } = {}) {
    if (!apiKey) {
      throw new Error("CourtListener API key is required");
    }

    this.apiKey = apiKey;
    this.userAgent = userAgent;
    this.headers = buildHeaders(this.apiKey, this.userAgent);
  }

  buildUrl(endpoint, params = {}) {
    const url = new URL(`${BASE_URL}/${endpoint}/`);

    Object.entries(params).forEach(([key, value]) => {
      if (value === undefined || value === null || value === "") {
        return;
      }

      url.searchParams.set(key, String(value));
    });

    return url.toString();
  }

  async *paginate(endpoint, { params = {}, maxPages } = {}) {
    if (!COURT_LISTENER_ENDPOINTS.has(endpoint)) {
      const allowed = Array.from(COURT_LISTENER_ENDPOINTS).join(", ");
      throw new Error(
        `Unsupported CourtListener endpoint "${endpoint}". Supported endpoints: ${allowed}`
      );
    }

    let iterations = 0;
    let nextUrl = this.buildUrl(endpoint, params);

    while (nextUrl) {
      if (maxPages && iterations >= maxPages) {
        break;
      }

      const payload = await requestJson(nextUrl, this.headers);
      yield payload;

      nextUrl = payload.next || null;
      iterations += 1;
    }
  }

  async collect(endpoint, options = {}) {
    const records = [];

    for await (const page of this.paginate(endpoint, options)) {
      const pageResults = Array.isArray(page.results) ? page.results : [];
      records.push(...pageResults);
    }

    return records;
  }
}
