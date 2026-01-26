import { Repository } from 'typeorm';
import { Vehicle } from './entities/vehicle.entity';
import { CreateVehicleDto, UpdateVehicleDto } from './dto';
export interface VehicleSearchOptions {
    tenantId: string;
    customerId?: string;
    make?: string;
    model?: string;
    year?: number;
    vin?: string;
    licensePlate?: string;
    page?: number;
    limit?: number;
}
export interface PaginatedVehicles {
    items: Vehicle[];
    total: number;
    page: number;
    limit: number;
    totalPages: number;
}
export declare class VehiclesService {
    private readonly vehicleRepository;
    constructor(vehicleRepository: Repository<Vehicle>);
    create(tenantId: string, dto: CreateVehicleDto): Promise<Vehicle>;
    findAll(options: VehicleSearchOptions): Promise<PaginatedVehicles>;
    findOne(tenantId: string, id: string): Promise<Vehicle>;
    findByVin(tenantId: string, vin: string): Promise<Vehicle | null>;
    update(tenantId: string, id: string, dto: UpdateVehicleDto): Promise<Vehicle>;
    updateMileage(tenantId: string, id: string, mileage: number): Promise<Vehicle>;
    remove(tenantId: string, id: string): Promise<void>;
    hardDelete(tenantId: string, id: string): Promise<void>;
    getVehicleStats(tenantId: string): Promise<{
        total: number;
        byMake: {
            make: string;
            count: number;
        }[];
        byYear: {
            year: number;
            count: number;
        }[];
        byFuelType: {
            fuelType: string;
            count: number;
        }[];
    }>;
}
