
import { Injectable, Logger } from '@nestjs/common';
import { spawn } from 'child_process';
import * as path from 'path';
import { OptimizeRequestDto, OptimizeResponseDto } from './tuning.dto';

@Injectable()
export class TuningService {
    private readonly logger = new Logger(TuningService.name);
    // Assuming the python script is relative to the backend project root or a fixed path
    private readonly pythonScriptPath = path.resolve(__dirname, '../../../../ai-training/scripts/optimize_cli.py');

    async optimizeMap(request: OptimizeRequestDto): Promise<OptimizeResponseDto> {
        return new Promise((resolve, reject) => {
            // Use 'python' command
            const pythonProcess = spawn('python', [this.pythonScriptPath]);

            let resultData = '';
            let errorData = '';

            pythonProcess.stdout.on('data', (data) => {
                resultData += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                errorData += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code !== 0) {
                    this.logger.error(`Python script exited with code ${code}: ${errorData}`);
                    reject(new Error(`Optimization failed: ${errorData}`));
                    return;
                }

                try {
                    const response: OptimizeResponseDto = JSON.parse(resultData);
                    if (response.status === 'error') {
                        reject(new Error(`Optimizer error: ${JSON.stringify(response)}`));
                    } else {
                        resolve(response);
                    }
                } catch (e) {
                    this.logger.error(`Failed to parse python output: ${resultData}`);
                    reject(new Error('Failed to parse optimization results'));
                }
            });

            // Send input data
            pythonProcess.stdin.write(JSON.stringify(request));
            pythonProcess.stdin.end();
        });
    }
}
