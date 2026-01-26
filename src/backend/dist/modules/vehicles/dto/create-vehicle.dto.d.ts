import { FuelType, TransmissionType } from '../entities/vehicle.entity';
export declare class CreateVehicleDto {
    customerId?: string;
    vin?: string;
    make: string;
    model: string;
    year: number;
    variant?: string;
    engineCode?: string;
    engineDisplacement?: number;
    enginePower?: number;
    fuelType?: FuelType;
    transmission?: TransmissionType;
    licensePlate?: string;
    mileage?: number;
    ecuType?: string;
    ecuSoftwareVersion?: string;
    notes?: string;
}
