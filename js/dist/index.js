"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/index.ts
var index_exports = {};
__export(index_exports, {
  FinetuneConfigSchema: () => FinetuneConfigSchema,
  Langtune: () => Langtune
});
module.exports = __toCommonJS(index_exports);

// src/client.ts
var import_execa = require("execa");

// src/types.ts
var import_zod = require("zod");
var FinetuneConfigSchema = import_zod.z.object({
  model: import_zod.z.string(),
  trainFile: import_zod.z.string(),
  preset: import_zod.z.string().optional().default("small"),
  epochs: import_zod.z.number().optional().default(3),
  batchSize: import_zod.z.number().optional().default(4),
  learningRate: import_zod.z.number().optional().default(2e-4),
  loraRank: import_zod.z.number().optional().default(16),
  outputDir: import_zod.z.string().optional().default("./output"),
  useTriton: import_zod.z.boolean().optional().default(false),
  useLisa: import_zod.z.boolean().optional().default(false)
});

// src/client.ts
var Langtune = class {
  constructor(options = {}) {
    this.cliPath = options.cliPath || "langtune";
    this.apiKey = options.apiKey;
  }
  getEnv() {
    return {
      ...process.env,
      LANGTUNE_API_KEY: this.apiKey || process.env.LANGTUNE_API_KEY
    };
  }
  /**
   * Run a fine-tuning job using the local CLI
   */
  async finetune(config) {
    const validated = FinetuneConfigSchema.parse(config);
    const args = [
      "train",
      "--model",
      validated.model,
      "--train-file",
      validated.trainFile,
      "--preset",
      validated.preset,
      "--epochs",
      validated.epochs.toString(),
      "--batch-size",
      validated.batchSize.toString(),
      "--learning-rate",
      validated.learningRate.toString(),
      "--lora-rank",
      validated.loraRank.toString(),
      "--output-dir",
      validated.outputDir
    ];
    if (validated.useTriton) {
      args.push("--use-triton");
    }
    if (validated.useLisa) {
      args.push("--use-lisa");
    }
    try {
      await (0, import_execa.execa)(this.cliPath, args, {
        stdio: "inherit",
        env: this.getEnv()
      });
    } catch (error) {
      throw new Error(`Langtune fine-tuning failed: ${error}`);
    }
  }
  /**
   * Generate text using a model
   */
  async generate(modelPath, options) {
    const args = [
      "generate",
      "--model",
      modelPath,
      "--prompt",
      options.prompt
    ];
    if (options.maxTokens) {
      args.push("--max-tokens", options.maxTokens.toString());
    }
    if (options.temperature) {
      args.push("--temperature", options.temperature.toString());
    }
    try {
      const { stdout } = await (0, import_execa.execa)(this.cliPath, args, {
        env: this.getEnv()
      });
      return stdout;
    } catch (error) {
      throw new Error(`Langtune generation failed: ${error}`);
    }
  }
};
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  FinetuneConfigSchema,
  Langtune
});
//# sourceMappingURL=index.js.map