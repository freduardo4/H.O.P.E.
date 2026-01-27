import { DataSource } from 'typeorm';
export interface HealthCheckResponse {
    status: 'ok' | 'degraded' | 'error';
    timestamp: string;
    uptime: number;
    version: string;
    database: {
        status: 'connected' | 'disconnected';
    };
}
export declare class HealthController {
    private dataSource;
    private readonly startTime;
    private readonly version;
    constructor(dataSource: DataSource);
    check(): HealthCheckResponse;
    readiness(): {
        ready: boolean;
    };
    liveness(): {
        alive: boolean;
    };
}
