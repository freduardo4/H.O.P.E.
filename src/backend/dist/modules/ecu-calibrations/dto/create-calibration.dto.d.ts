import { CalibrationStatus, CalibrationProtocol } from '../entities/ecu-calibration.entity';
export declare class MapDataDto {
    name: string;
    address: string;
    originalValues: number[];
    modifiedValues: number[];
    unit: string;
    description?: string;
}
export declare class ModificationDto {
    type: string;
    parameter: string;
    originalValue: string;
    newValue: string;
    timestamp: string;
    technicianId: string;
}
export declare class CreateCalibrationDto {
    vehicleId: string;
    name: string;
    description?: string;
    version: string;
    parentVersionId?: string;
    status?: CalibrationStatus;
    protocol?: CalibrationProtocol;
    ecuPartNumber?: string;
    ecuSoftwareNumber?: string;
    ecuHardwareNumber?: string;
    originalChecksum?: string;
    modifiedChecksum?: string;
    fileSize?: number;
    s3Key?: string;
    originalS3Key?: string;
    mapData?: MapDataDto[];
    modifications?: ModificationDto[];
    notes?: string;
}
