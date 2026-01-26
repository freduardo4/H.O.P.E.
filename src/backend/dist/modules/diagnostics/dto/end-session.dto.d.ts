export declare class EndSessionDto {
    notes?: string;
    dtcCodes?: string[];
    performanceMetrics?: {
        maxRpm?: number;
        maxSpeed?: number;
        maxBoost?: number;
        avgLoad?: number;
    };
    ecuSnapshot?: Record<string, any>;
}
