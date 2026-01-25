import {
    IsString,
    IsNumber,
    IsOptional,
    IsEnum,
    IsUUID,
    Length,
    Min,
    Max,
} from 'class-validator';
import { FuelType, TransmissionType } from '../entities/vehicle.entity';

export class CreateVehicleDto {
    @IsUUID()
    @IsOptional()
    customerId?: string;

    @IsString()
    @IsOptional()
    @Length(17, 17, { message: 'VIN must be exactly 17 characters' })
    vin?: string;

    @IsString()
    make: string;

    @IsString()
    model: string;

    @IsNumber()
    @Min(1900)
    @Max(2100)
    year: number;

    @IsString()
    @IsOptional()
    variant?: string;

    @IsString()
    @IsOptional()
    engineCode?: string;

    @IsNumber()
    @IsOptional()
    engineDisplacement?: number;

    @IsNumber()
    @IsOptional()
    enginePower?: number;

    @IsEnum(FuelType)
    @IsOptional()
    fuelType?: FuelType;

    @IsEnum(TransmissionType)
    @IsOptional()
    transmission?: TransmissionType;

    @IsString()
    @IsOptional()
    licensePlate?: string;

    @IsNumber()
    @IsOptional()
    @Min(0)
    mileage?: number;

    @IsString()
    @IsOptional()
    ecuType?: string;

    @IsString()
    @IsOptional()
    ecuSoftwareVersion?: string;

    @IsString()
    @IsOptional()
    notes?: string;
}
