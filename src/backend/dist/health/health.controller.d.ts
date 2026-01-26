export interface HealthCheckResponse {
    status: 'ok' | 'degraded' | 'error';
    timestamp: string;
    uptime: number;
    version: string;
}
export declare class HealthController {
    private readonly startTime;
    private readonly version;
    check(): HealthCheckResponse;
    readiness(): {
        ready: boolean;
    };
    liveness(): {
        alive: boolean;
    };
}
