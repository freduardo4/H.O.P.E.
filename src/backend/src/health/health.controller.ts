import { Controller, Get } from '@nestjs/common';

export interface HealthCheckResponse {
    status: 'ok' | 'degraded' | 'error';
    timestamp: string;
    uptime: number;
    version: string;
}

@Controller('health')
export class HealthController {
    private readonly startTime = Date.now();
    private readonly version = process.env.npm_package_version || '1.0.0';

    @Get()
    check(): HealthCheckResponse {
        return {
            status: 'ok',
            timestamp: new Date().toISOString(),
            uptime: Math.floor((Date.now() - this.startTime) / 1000),
            version: this.version,
        };
    }

    @Get('ready')
    readiness(): { ready: boolean } {
        return { ready: true };
    }

    @Get('live')
    liveness(): { alive: boolean } {
        return { alive: true };
    }
}
