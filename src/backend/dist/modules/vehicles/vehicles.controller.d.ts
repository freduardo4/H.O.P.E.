import { VehiclesService, PaginatedVehicles } from './vehicles.service';
import { CreateVehicleDto, UpdateVehicleDto } from './dto';
import { Vehicle } from './entities/vehicle.entity';
import { User } from '../auth/entities/user.entity';
export declare class VehiclesController {
    private readonly vehiclesService;
    constructor(vehiclesService: VehiclesService);
    create(user: User, dto: CreateVehicleDto): Promise<Vehicle>;
    findAll(user: User, customerId?: string, make?: string, model?: string, year?: number, vin?: string, licensePlate?: string, page?: number, limit?: number): Promise<PaginatedVehicles>;
    getStats(user: User): Promise<{
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
    findOne(user: User, id: string): Promise<Vehicle>;
    findByVin(user: User, vin: string): Promise<Vehicle | null>;
    update(user: User, id: string, dto: UpdateVehicleDto): Promise<Vehicle>;
    updateMileage(user: User, id: string, mileage: number): Promise<Vehicle>;
    remove(user: User, id: string): Promise<{
        message: string;
    }>;
}
