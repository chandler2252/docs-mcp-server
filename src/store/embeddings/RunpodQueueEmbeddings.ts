import type { Embeddings } from "@langchain/core/embeddings";

type RunpodRunsyncResponse =
    | { id: string; status: "COMPLETED"; output: any }
    | { id: string; status: "FAILED"; error?: any; output?: any }
    | { id: string; status: "IN_QUEUE" | "IN_PROGRESS"; output?: any };

function envInt(name: string, fallback: number): number {
    const v = process.env[name];
    if (!v) return fallback;
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

function hrMs(): number {
    return Number(process.hrtime.bigint() / 1_000_000n);
}

/**
 * RunPod /runsync payload limits:
 * - /run: 10 MB
 * - /runsync: 20 MB
 * We keep a safety margin to avoid edge cases with escaping, headers, etc.
 */
const DEFAULT_RUNSYNC_LIMIT_BYTES = 20 * 1024 * 1024; // 20MB
const DEFAULT_SAFETY_BYTES = 512 * 1024; // 512KB safety margin

export class RunpodQueueEmbeddings implements Embeddings {
    private readonly endpointId: string;
    private readonly apiKey: string;
    private readonly baseUrl: string;
    private readonly model: string;
    private readonly dimensions?: number;
    private readonly timeoutMs: number;

    private readonly maxTextsPerRequest: number;
    private readonly maxPayloadBytes: number;

    constructor(opts: {
        endpointId: string;
        apiKey: string;
        model: string;
        baseUrl?: string; // default https://api.runpod.ai/v2
        dimensions?: number; // e.g. 2560
        timeoutMs?: number; // default 300000

        /**
         * Optional hard caps. If omitted we read env:
         * - DOCS_MCP_EMBEDDING_BATCH_SIZE (default 128)
         * - RUNPOD_MAX_PAYLOAD_BYTES (default 20MB)
         * - RUNPOD_PAYLOAD_SAFETY_BYTES (default 512KB)
         */
        maxTextsPerRequest?: number;
        maxPayloadBytes?: number;
    }) {
        this.endpointId = opts.endpointId;
        this.apiKey = opts.apiKey;
        this.baseUrl = (opts.baseUrl ?? "https://api.runpod.ai/v2").replace(/\/+$/, "");
        this.model = opts.model;
        this.dimensions = opts.dimensions;
        this.timeoutMs = opts.timeoutMs ?? 300000;

        const safety = envInt("RUNPOD_PAYLOAD_SAFETY_BYTES", DEFAULT_SAFETY_BYTES);
        const limit = envInt("RUNPOD_MAX_PAYLOAD_BYTES", DEFAULT_RUNSYNC_LIMIT_BYTES);
        this.maxPayloadBytes = (opts.maxPayloadBytes ?? limit) - safety;

        // “Chunks per batch” control (you asked for 512)
        const envBatch = envInt("DOCS_MCP_EMBEDDING_BATCH_SIZE", 128);
        this.maxTextsPerRequest = opts.maxTextsPerRequest ?? envBatch;
    }

    async embedDocuments(texts: string[]): Promise<number[][]> {
        return this.embed(texts);
    }

    async embedQuery(text: string): Promise<number[]> {
        const [v] = await this.embed([text]);
        return v;
    }

    private buildPayload(texts: string[]) {
        return {
            input: {
                model: this.model,
                texts,
                ...(this.dimensions ? { dimensions: this.dimensions } : {}),
                // Optional passthrough knobs for your handler (safe if ignored)
                ...(process.env.RUNPOD_HANDLER_BATCH_SIZE
                    ? { batch_size: Number(process.env.RUNPOD_HANDLER_BATCH_SIZE) }
                    : {}),
                ...(process.env.RUNPOD_HANDLER_MAX_LENGTH
                    ? { max_length: Number(process.env.RUNPOD_HANDLER_MAX_LENGTH) }
                    : {}),
            },
        };
    }

    private payloadBytes(payload: any): number {
        return Buffer.byteLength(JSON.stringify(payload), "utf8");
    }

    private splitByBudget(allTexts: string[]): string[][] {
        const batches: string[][] = [];
        let cur: string[] = [];

        for (const t of allTexts) {
            // Enforce max count cap first
            const wouldExceedCount = cur.length + 1 > this.maxTextsPerRequest;

            // Enforce payload cap via actual JSON size (safe, a bit more CPU)
            const candidate = [...cur, t];
            const bytes = this.payloadBytes(this.buildPayload(candidate));
            const wouldExceedBytes = bytes > this.maxPayloadBytes;

            if (cur.length === 0) {
                // Single text too large: still send it; server may reject, but we need a clear error
                cur.push(t);
                batches.push(cur);
                cur = [];
                continue;
            }

            if (wouldExceedCount || wouldExceedBytes) {
                batches.push(cur);
                cur = [t];
            } else {
                cur.push(t);
            }
        }

        if (cur.length) batches.push(cur);
        return batches;
    }

    private async embed(texts: string[]): Promise<number[][]> {
        const url = `${this.baseUrl}/${this.endpointId}/runsync`;
        const batches = this.splitByBudget(texts);
        const out: number[][] = [];

        for (let i = 0; i < batches.length; i++) {
            const batchTexts = batches[i];
            const payload = this.buildPayload(batchTexts);
            const body = JSON.stringify(payload);
            const bodyBytes = Buffer.byteLength(body, "utf8");
            const totalChars = batchTexts.reduce((acc, s) => acc + s.length, 0);

            // embed_start log
            console.log(
                JSON.stringify({
                    stage: "embed_start",
                    provider: "runpod",
                    batch_index: i,
                    batch_count: batches.length,
                    texts_count: batchTexts.length,
                    total_chars: totalChars,
                    payload_bytes: bodyBytes,
                    url,
                }),
            );

            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), this.timeoutMs);

            const t0 = hrMs();
            try {
                const res = await fetch(url, {
                    method: "POST",
                    headers: {
                        accept: "application/json",
                        "content-type": "application/json",
                        // RunPod: raw API key in `authorization` header for /runsync
                        authorization: this.apiKey,
                    },
                    body,
                    signal: controller.signal,
                });

                const t1 = hrMs();

                if (!res.ok) {
                    const text = await res.text().catch(() => "");
                    console.log(
                        JSON.stringify({
                            stage: "embed_end",
                            provider: "runpod",
                            ok: false,
                            http_status: res.status,
                            http_status_text: res.statusText,
                            ms: t1 - t0,
                            payload_bytes: bodyBytes,
                            response_snippet: text.slice(0, 800),
                        }),
                    );
                    throw new Error(
                        `RunPod /runsync failed: HTTP ${res.status} ${res.statusText}\n${text}`,
                    );
                }

                const json = (await res.json()) as RunpodRunsyncResponse;

                if (json.status !== "COMPLETED") {
                    throw new Error(`RunPod job not completed: ${json.status} (id=${json.id})`);
                }

                const outObj = json.output;
                const embeddings = Array.isArray(outObj)
                    ? outObj?.[0]?.embeddings
                    : outObj?.embeddings;

                const metrics = Array.isArray(outObj) ? outObj?.[0]?.metrics : outObj?.metrics;

                if (!Array.isArray(embeddings) || !Array.isArray(embeddings[0])) {
                    throw new Error(
                        `Unexpected RunPod output shape: ${JSON.stringify(outObj).slice(0, 500)}`,
                    );
                }

                // embed_end log including RunPod request id + handler metrics if present
                console.log(
                    JSON.stringify({
                        stage: "embed_end",
                        provider: "runpod",
                        ok: true,
                        runpod_id: json.id,
                        ms: t1 - t0,
                        texts_count: batchTexts.length,
                        payload_bytes: bodyBytes,
                        handler_metrics: metrics ?? null,
                    }),
                );

                out.push(...(embeddings as number[][]));
            } finally {
                clearTimeout(timer);
            }
        }

        return out;
    }
}
