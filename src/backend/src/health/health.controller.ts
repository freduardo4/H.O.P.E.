import { Controller, Get } from '@nestjs/common';
import { DataSource } from 'typeorm';
import { InjectDataSource } from '@nestjs/typeorm';

export interface HealthCheckResponse {
    status: 'ok' | 'degraded' | 'error';
    timestamp: string;
    uptime: number;
    version: string;
    database: {
        status: 'connected' | 'disconnected';
    };
}

@Controller('health')
export class HealthController {
    private readonly startTime = Date.now();
    private readonly version = process.env.npm_package_version || '1.0.0';

    constructor(
        @InjectDataSource() private dataSource: DataSource,
    ) { }

    @Get()
    check(): HealthCheckResponse {
        const isDbConnected = this.dataSource.isInitialized;

        return {
            status: isDbConnected ? 'ok' : 'degraded',
            timestamp: new Date().toISOString(),
            uptime: Math.floor((Date.now() - this.startTime) / 1000),
            version: this.version,
            database: {
                status: isDbConnected ? 'connected' : 'disconnected',
            },
        };
    }

    @Get('ready')
    readiness(): { ready: boolean } {
        return { ready: this.dataSource.isInitialized };
    }

    @Get('live')
    liveness(): { alive: boolean } {
        return { alive: true };
    }
}
