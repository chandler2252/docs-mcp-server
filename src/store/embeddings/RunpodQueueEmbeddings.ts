import type { Embeddings } from "@langchain/core/embeddings";

type RunpodRunsyncResponse =
  | { id: string; status: "COMPLETED"; output: any }
  | { id: string; status: "FAILED"; error?: any; output?: any }
  | { id: string; status: "IN_QUEUE" | "IN_PROGRESS"; output?: any };

export class RunpodQueueEmbeddings implements Embeddings {
  private readonly endpointId: string;
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly dimensions?: number;
  private readonly timeoutMs: number;

  constructor(opts: {
    endpointId: string;
    apiKey: string;
    model: string;
    baseUrl?: string; // default https://api.runpod.ai/v2
      dimensions?: number; // e.g. 2560
    timeoutMs?: number; // default 300000
  }) {
    this.endpointId = opts.endpointId;
    this.apiKey = opts.apiKey;
    this.baseUrl = (opts.baseUrl ?? "https://api.runpod.ai/v2").replace(/\/+$/, "");
    this.model = opts.model;
    this.dimensions = opts.dimensions;
    this.timeoutMs = opts.timeoutMs ?? 300000;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return this.embed(texts);
  }

  async embedQuery(text: string): Promise<number[]> {
    const [v] = await this.embed([text]);
    return v;
  }

  private async embed(texts: string[]): Promise<number[][]> {
    const url = `${this.baseUrl}/${this.endpointId}/runsync`;

    const payload = {
      input: {
        model: this.model,
        texts,
        // We *ask* the worker for this dimension, but we don't trust it;
        // we will truncate to this.dimensions locally as well.
        ...(this.dimensions ? { dimensions: this.dimensions } : {}),
      },
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          accept: "application/json",
          "content-type": "application/json",
          authorization: this.apiKey,
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`RunPod /runsync failed: HTTP ${res.status} ${res.statusText}\n${text}`);
      }

      const json = (await res.json()) as RunpodRunsyncResponse;

      if (json.status !== "COMPLETED") {
        throw new Error(`RunPod job not completed: ${json.status} (id=${json.id})`);
      }

      const out = json.output;

      // Expect either { embeddings: number[][] } or [ { embeddings: number[][] } ]
      let embeddings = Array.isArray(out) ? out?.[0]?.embeddings : out?.embeddings;

      if (!Array.isArray(embeddings) || !Array.isArray(embeddings[0])) {
        throw new Error(`Unexpected RunPod output shape: ${JSON.stringify(out).slice(0, 500)}`);
      }

      let vectors = embeddings as number[][];

      // Hard truncate to requested dimensions if the remote model returned more.
      if (this.dimensions && vectors.length > 0 && vectors[0].length > this.dimensions) {
        vectors = vectors.map((v) => v.slice(0, this.dimensions));
      }

      return vectors;
    } finally {
      clearTimeout(timer);
    }
  }
}
