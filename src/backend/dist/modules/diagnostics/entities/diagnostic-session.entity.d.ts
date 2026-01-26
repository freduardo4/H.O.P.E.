export declare enum SessionStatus {
    ACTIVE = "active",
    COMPLETED = "completed",
    CANCELLED = "cancelled"
}
export declare enum SessionType {
    DIAGNOSTIC = "diagnostic",
    PERFORMANCE = "performance",
    TUNE = "tune",
    MAINTENANCE = "maintenance"
}
export declare class DiagnosticSession {
    id: string;
    tenantId: string;
    vehicleId: string;
    technicianId: string;
    type: SessionType;
    status: SessionStatus;
    startTime: Date;
    endTime: Date;
    mileageAtSession: number;
    notes: string;
    ecuSnapshot: Record<string, any>;
    dtcCodes: string[];
    performanceMetrics: {
        maxRpm?: number;
        maxSpeed?: number;
        maxBoost?: number;
        avgLoad?: number;
    };
    createdAt: Date;
    updatedAt: Date;
    get duration(): number | null;
}
