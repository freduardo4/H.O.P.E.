import { IsString, IsEnum, IsOptional, IsNumber, IsObject, IsUUID } from 'class-validator';
import { CalibrationType, FileFormat } from '../entities/ecu-calibration.entity';

export class CreateECUCalibrationDto {
    @IsUUID()
    vehicleId: string;

    @IsUUID()
    @IsOptional()
    customerId?: string;

    @IsString()
    fileName: string;

    @IsNumber()
    fileSize: number;

    @IsEnum(FileFormat)
    @IsOptional()
    fileFormat?: FileFormat;

    @IsEnum(CalibrationType)
    calibrationType: CalibrationType;

    @IsString()
    checksum: string;

    @IsString()
    @IsOptional()
    previousVersionId?: string;

    @IsString()
    @IsOptional()
    ecuType?: string;

    @IsString()
    @IsOptional()
    ecuSoftwareVersion?: string;

    @IsString()
    @IsOptional()
    notes?: string;

    @IsObject()
    @IsOptional()
    metadata?: {
        enginePowerStock?: number;
        enginePowerTuned?: number;
        torqueStock?: number;
        torqueTuned?: number;
        fuelConsumptionImprovement?: number;
        [key: string]: any;
    };
}
