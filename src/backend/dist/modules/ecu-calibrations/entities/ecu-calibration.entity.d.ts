export declare enum CalibrationType {
    STOCK = "stock",
    STAGE_1 = "stage1",
    STAGE_2 = "stage2",
    STAGE_3 = "stage3",
    CUSTOM = "custom"
}
export declare enum FileFormat {
    BIN = "bin",
    HEX = "hex",
    S19 = "s19",
    UNKNOWN = "unknown"
}
export declare class ECUCalibration {
    id: string;
    tenantId: string;
    vehicleId: string;
    customerId: string;
    fileName: string;
    s3Key: string;
    s3Bucket: string;
    fileSize: number;
    fileFormat: FileFormat;
    calibrationType: CalibrationType;
    checksum: string;
    version: number;
    previousVersionId: string;
    ecuType: string;
    ecuSoftwareVersion: string;
    notes: string;
    metadata: {
        enginePowerStock?: number;
        enginePowerTuned?: number;
        torqueStock?: number;
        torqueTuned?: number;
        fuelConsumptionImprovement?: number;
        [key: string]: any;
    };
    isActive: boolean;
    uploadedBy: string;
    createdAt: Date;
    updatedAt: Date;
}
