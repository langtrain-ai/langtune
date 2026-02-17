import { z } from 'zod';

export const FinetuneConfigSchema = z.object({
    model: z.string(),
    trainFile: z.string(),
    preset: z.string().optional().default('small'),
    epochs: z.number().optional().default(3),
    batchSize: z.number().optional().default(4),
    learningRate: z.number().optional().default(2e-4),
    loraRank: z.number().optional().default(16),
    outputDir: z.string().optional().default('./output'),
    useTriton: z.boolean().optional().default(false),
    useLisa: z.boolean().optional().default(false),
});

export type FinetuneConfig = z.infer<typeof FinetuneConfigSchema>;

export interface GenerationOptions {
    prompt: string;
    maxTokens?: number;
    temperature?: number;
}
