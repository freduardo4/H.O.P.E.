export declare enum ReportType {
    DIAGNOSTIC = "diagnostic",
    TUNING = "tuning",
    COMPARISON = "comparison",
    MAINTENANCE = "maintenance"
}
export declare enum ReportStatus {
    PENDING = "pending",
    GENERATING = "generating",
    COMPLETED = "completed",
    FAILED = "failed"
}
export declare class Report {
    id: string;
    tenantId: string;
    vehicleId: string;
    customerId: string;
    diagnosticSessionId: string;
    reportType: ReportType;
    status: ReportStatus;
    title: string;
    description: string;
    s3Key: string;
    s3Bucket: string;
    fileSize: number;
    data: {
        dtcCodes?: string[];
        obd2Readings?: any[];
        anomalies?: any[];
        beforePower?: number;
        afterPower?: number;
        beforeTorque?: number;
        afterTorque?: number;
        fuelConsumptionImprovement?: number;
        modifications?: string[];
        [key: string]: any;
    };
    generatedBy: string;
    generatedAt: Date;
    errorMessage: string;
    createdAt: Date;
    updatedAt: Date;
}
