import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
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

@Injectable()
export class VehiclesService {
    constructor(
        @InjectRepository(Vehicle)
        private readonly vehicleRepository: Repository<Vehicle>,
    ) {}

    async create(tenantId: string, dto: CreateVehicleDto): Promise<Vehicle> {
        const vehicle = this.vehicleRepository.create({
            ...dto,
            tenantId,
        });

        return this.vehicleRepository.save(vehicle);
    }

    async findAll(options: VehicleSearchOptions): Promise<PaginatedVehicles> {
        const { tenantId, customerId, make, model, year, vin, licensePlate, page = 1, limit = 20 } = options;

        const queryBuilder = this.vehicleRepository
            .createQueryBuilder('vehicle')
            .where('vehicle.tenantId = :tenantId', { tenantId })
            .andWhere('vehicle.isActive = :isActive', { isActive: true });

        if (customerId) {
            queryBuilder.andWhere('vehicle.customerId = :customerId', { customerId });
        }

        if (make) {
            queryBuilder.andWhere('LOWER(vehicle.make) LIKE LOWER(:make)', { make: `%${make}%` });
        }

        if (model) {
            queryBuilder.andWhere('LOWER(vehicle.model) LIKE LOWER(:model)', { model: `%${model}%` });
        }

        if (year) {
            queryBuilder.andWhere('vehicle.year = :year', { year });
        }

        if (vin) {
            queryBuilder.andWhere('vehicle.vin = :vin', { vin });
        }

        if (licensePlate) {
            queryBuilder.andWhere('LOWER(vehicle.licensePlate) LIKE LOWER(:licensePlate)', {
                licensePlate: `%${licensePlate}%`,
            });
        }

        const total = await queryBuilder.getCount();
        const items = await queryBuilder
            .orderBy('vehicle.updatedAt', 'DESC')
            .skip((page - 1) * limit)
            .take(limit)
            .getMany();

        return {
            items,
            total,
            page,
            limit,
            totalPages: Math.ceil(total / limit),
        };
    }

    async findOne(tenantId: string, id: string): Promise<Vehicle> {
        const vehicle = await this.vehicleRepository.findOne({
            where: { id, tenantId, isActive: true },
        });

        if (!vehicle) {
            throw new NotFoundException(`Vehicle with ID ${id} not found`);
        }

        return vehicle;
    }

    async findByVin(tenantId: string, vin: string): Promise<Vehicle | null> {
        return this.vehicleRepository.findOne({
            where: { vin, tenantId, isActive: true },
        });
    }

    async update(tenantId: string, id: string, dto: UpdateVehicleDto): Promise<Vehicle> {
        const vehicle = await this.findOne(tenantId, id);

        Object.assign(vehicle, dto);

        return this.vehicleRepository.save(vehicle);
    }

    async updateMileage(tenantId: string, id: string, mileage: number): Promise<Vehicle> {
        const vehicle = await this.findOne(tenantId, id);

        vehicle.mileage = mileage;

        return this.vehicleRepository.save(vehicle);
    }

    async remove(tenantId: string, id: string): Promise<void> {
        const vehicle = await this.findOne(tenantId, id);

        // Soft delete - mark as inactive
        vehicle.isActive = false;
        await this.vehicleRepository.save(vehicle);
    }

    async hardDelete(tenantId: string, id: string): Promise<void> {
        const vehicle = await this.findOne(tenantId, id);
        await this.vehicleRepository.remove(vehicle);
    }

    async getVehicleStats(tenantId: string): Promise<{
        total: number;
        byMake: { make: string; count: number }[];
        byYear: { year: number; count: number }[];
        byFuelType: { fuelType: string; count: number }[];
    }> {
        const total = await this.vehicleRepository.count({
            where: { tenantId, isActive: true },
        });

        const byMake = await this.vehicleRepository
            .createQueryBuilder('vehicle')
            .select('vehicle.make', 'make')
            .addSelect('COUNT(*)', 'count')
            .where('vehicle.tenantId = :tenantId', { tenantId })
            .andWhere('vehicle.isActive = :isActive', { isActive: true })
            .groupBy('vehicle.make')
            .orderBy('count', 'DESC')
            .getRawMany();

        const byYear = await this.vehicleRepository
            .createQueryBuilder('vehicle')
            .select('vehicle.year', 'year')
            .addSelect('COUNT(*)', 'count')
            .where('vehicle.tenantId = :tenantId', { tenantId })
            .andWhere('vehicle.isActive = :isActive', { isActive: true })
            .groupBy('vehicle.year')
            .orderBy('vehicle.year', 'DESC')
            .getRawMany();

        const byFuelType = await this.vehicleRepository
            .createQueryBuilder('vehicle')
            .select('vehicle.fuelType', 'fuelType')
            .addSelect('COUNT(*)', 'count')
            .where('vehicle.tenantId = :tenantId', { tenantId })
            .andWhere('vehicle.isActive = :isActive', { isActive: true })
            .andWhere('vehicle.fuelType IS NOT NULL')
            .groupBy('vehicle.fuelType')
            .orderBy('count', 'DESC')
            .getRawMany();

        return { total, byMake, byYear, byFuelType };
    }
}
