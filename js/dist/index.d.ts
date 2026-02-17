import { z } from 'zod';

declare const FinetuneConfigSchema: z.ZodObject<{
    model: z.ZodString;
    trainFile: z.ZodString;
    preset: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    epochs: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    batchSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    learningRate: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    loraRank: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    outputDir: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    useTriton: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    useLisa: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
}, "strip", z.ZodTypeAny, {
    model: string;
    trainFile: string;
    preset: string;
    epochs: number;
    batchSize: number;
    learningRate: number;
    loraRank: number;
    outputDir: string;
    useTriton: boolean;
    useLisa: boolean;
}, {
    model: string;
    trainFile: string;
    preset?: string | undefined;
    epochs?: number | undefined;
    batchSize?: number | undefined;
    learningRate?: number | undefined;
    loraRank?: number | undefined;
    outputDir?: string | undefined;
    useTriton?: boolean | undefined;
    useLisa?: boolean | undefined;
}>;
type FinetuneConfig = z.infer<typeof FinetuneConfigSchema>;
interface GenerationOptions {
    prompt: string;
    maxTokens?: number;
    temperature?: number;
}

declare class Langtune {
    private cliPath;
    private apiKey?;
    constructor(options?: {
        cliPath?: string;
        apiKey?: string;
    });
    private getEnv;
    /**
     * Run a fine-tuning job using the local CLI
     */
    finetune(config: FinetuneConfig): Promise<void>;
    /**
     * Generate text using a model
     */
    generate(modelPath: string, options: GenerationOptions): Promise<string>;
}

export { type FinetuneConfig, FinetuneConfigSchema, type GenerationOptions, Langtune };
