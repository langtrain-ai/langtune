import { execa } from 'execa';
import { FinetuneConfig, FinetuneConfigSchema, GenerationOptions } from './types';

export class Langtune {
    private cliPath: string;
    private apiKey?: string;

    constructor(options: { cliPath?: string, apiKey?: string } = {}) {
        this.cliPath = options.cliPath || 'langtune';
        this.apiKey = options.apiKey;
    }

    private getEnv() {
        return {
            ...process.env,
            LANGTUNE_API_KEY: this.apiKey || process.env.LANGTUNE_API_KEY
        };
    }

    /**
     * Run a fine-tuning job using the local CLI
     */
    async finetune(config: FinetuneConfig): Promise<void> {
        const validated = FinetuneConfigSchema.parse(config);

        // Construct CLI arguments
        const args = [
            'train',
            '--model', validated.model,
            '--train-file', validated.trainFile,
            '--preset', validated.preset,
            '--epochs', validated.epochs.toString(),
            '--batch-size', validated.batchSize.toString(),
            '--learning-rate', validated.learningRate.toString(),
            '--lora-rank', validated.loraRank.toString(),
            '--output-dir', validated.outputDir,
        ];

        if (validated.useTriton) {
            args.push('--use-triton');
        }

        if (validated.useLisa) {
            args.push('--use-lisa');
        }

        try {
            await execa(this.cliPath, args, {
                stdio: 'inherit',
                env: this.getEnv()
            });
        } catch (error) {
            throw new Error(`Langtune fine-tuning failed: ${error}`);
        }
    }

    /**
     * Generate text using a model
     */
    async generate(modelPath: string, options: GenerationOptions): Promise<string> {
        const args = [
            'generate',
            '--model', modelPath,
            '--prompt', options.prompt,
        ];

        if (options.maxTokens) {
            args.push('--max-tokens', options.maxTokens.toString());
        }

        if (options.temperature) {
            args.push('--temperature', options.temperature.toString());
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
}
