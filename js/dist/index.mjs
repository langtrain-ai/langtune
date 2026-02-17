// src/client.ts
import { execa } from "execa";

// src/types.ts
import { z } from "zod";
var FinetuneConfigSchema = z.object({
  model: z.string(),
  trainFile: z.string(),
  preset: z.string().optional().default("small"),
  epochs: z.number().optional().default(3),
  batchSize: z.number().optional().default(4),
  learningRate: z.number().optional().default(2e-4),
  loraRank: z.number().optional().default(16),
  outputDir: z.string().optional().default("./output"),
  useTriton: z.boolean().optional().default(false),
  useLisa: z.boolean().optional().default(false)
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
      await execa(this.cliPath, args, {
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
      const { stdout } = await execa(this.cliPath, args, {
        env: this.getEnv()
      });
      return stdout;
    } catch (error) {
      throw new Error(`Langtune generation failed: ${error}`);
    }
  }
};
export {
  FinetuneConfigSchema,
  Langtune
};
//# sourceMappingURL=index.mjs.map