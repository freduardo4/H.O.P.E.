import { CalibrationType, FileFormat } from '../entities/ecu-calibration.entity';
export declare class CreateECUCalibrationDto {
    vehicleId: string;
    customerId?: string;
    fileName: string;
    fileSize: number;
    fileFormat?: FileFormat;
    calibrationType: CalibrationType;
    checksum: string;
    previousVersionId?: string;
    ecuType?: string;
    ecuSoftwareVersion?: string;
    notes?: string;
    metadata?: {
        enginePowerStock?: number;
        enginePowerTuned?: number;
        torqueStock?: number;
        torqueTuned?: number;
        fuelConsumptionImprovement?: number;
        [key: string]: any;
    };
}
