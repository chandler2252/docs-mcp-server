/**
 * Worker command - Starts external pipeline worker (HTTP API).
 */

import type { Command } from "commander";
import { Option } from "commander";
import { startAppServer } from "../../app";
import { PipelineFactory, type PipelineOptions } from "../../pipeline";
import { createLocalDocumentManagement } from "../../store";
import { TelemetryEvent, telemetry } from "../../telemetry";
import { DEFAULT_HOST, DEFAULT_MAX_CONCURRENCY } from "../../utils/config";
import { logger } from "../../utils/logger";
import { registerGlobalServices } from "../main";
import {
    createAppServerConfig,
    ensurePlaywrightBrowsersInstalled,
    getEventBus,
    resolveEmbeddingContext,
    validateHost,
    validatePort,
} from "../utils";

function parsePositiveInt(name: string, v: string, min = 1, max = 1024): number {
    const n = Number(v);
    if (!Number.isInteger(n) || n < min || n > max) {
        throw new Error(`${name} must be an integer between ${min} and ${max}`);
    }
    return n;
}

export function createWorkerCommand(program: Command): Command {
    return program
        .command("worker")
        .description("Start external pipeline worker (HTTP API)")
        .addOption(
            new Option("--port <number>", "Port for worker API")
                .env("DOCS_MCP_PORT")
                .env("PORT")
                .default("8080")
                .argParser((v: string) => {
                    const n = Number(v);
                    if (!Number.isInteger(n) || n < 1 || n > 65535) {
                        throw new Error("Port must be an integer between 1 and 65535");
                    }
                    return String(n);
                }),
        )
        .addOption(
            new Option("--host <host>", "Host to bind the worker API to")
                .env("DOCS_MCP_HOST")
                .env("HOST")
                .default(DEFAULT_HOST)
                .argParser(validateHost),
        )
        .addOption(
            new Option(
                "--concurrency <number>",
                "Max concurrent pipeline work items per worker process",
            )
                .env("DOCS_MCP_WORKER_CONCURRENCY")
                .default(String(DEFAULT_MAX_CONCURRENCY))
                .argParser((v: string) => String(parsePositiveInt("concurrency", v, 1, 4096))),
        )
        .addOption(
            new Option(
                "--embedding-model <model>",
                "Embedding model configuration (e.g., 'openai:text-embedding-3-small')",
            ).env("DOCS_MCP_EMBEDDING_MODEL"),
        )
        .option("--resume", "Resume interrupted jobs on startup", true)
        .option("--no-resume", "Do not resume jobs on startup")
        .action(
            async (
                cmdOptions: {
                    port: string;
                    host: string;
                    concurrency: string;
                    embeddingModel?: string;
                    resume: boolean;
                },
                command?: Command,
            ) => {
                const port = validatePort(cmdOptions.port);
                const host = validateHost(cmdOptions.host);
                const concurrency = parsePositiveInt("concurrency", cmdOptions.concurrency, 1, 4096);

                await telemetry.track(TelemetryEvent.CLI_COMMAND, {
                    command: "worker",
                    port: String(port),
                    host,
                    concurrency,
                    resume: cmdOptions.resume,
                });

                try {
                    ensurePlaywrightBrowsersInstalled();
                    const embeddingConfig = resolveEmbeddingContext(cmdOptions.embeddingModel);

                    const globalOptions = program.opts();
                    const eventBus = getEventBus(command);

                    const docService = await createLocalDocumentManagement(
                        globalOptions.storePath,
                        eventBus,
                        embeddingConfig,
                    );

                    const pipelineOptions: PipelineOptions = {
                        recoverJobs: cmdOptions.resume,
                        concurrency,
                    };

                    logger.info(
                        `🧵 Worker pipeline concurrency = ${concurrency} (resume=${cmdOptions.resume})`,
                    );

                    const pipeline = await PipelineFactory.createPipeline(
                        docService,
                        eventBus,
                        pipelineOptions,
                    );

                    const config = createAppServerConfig({
                        enableWebInterface: false,
                        enableMcpServer: false,
                        enableApiServer: true,
                        enableWorker: true,
                        port,
                        host,
                        startupContext: {
                            cliCommand: "worker",
                        },
                    });

                    const appServer = await startAppServer(docService, pipeline, eventBus, config);

                    registerGlobalServices({
                        appServer,
                        docService,
                    });

                    await new Promise(() => { });
                } catch (error) {
                    logger.error(`❌ Failed to start external pipeline worker: ${error}`);
                    process.exit(1);
                }
            },
        );
}
