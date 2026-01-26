"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.VehiclesService = void 0;
const common_1 = require("@nestjs/common");
const typeorm_1 = require("@nestjs/typeorm");
const typeorm_2 = require("typeorm");
const vehicle_entity_1 = require("./entities/vehicle.entity");
let VehiclesService = class VehiclesService {
    constructor(vehicleRepository) {
        this.vehicleRepository = vehicleRepository;
    }
    async create(tenantId, dto) {
        const vehicle = this.vehicleRepository.create({
            ...dto,
            tenantId,
        });
        return this.vehicleRepository.save(vehicle);
    }
    async findAll(options) {
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
    async findOne(tenantId, id) {
        const vehicle = await this.vehicleRepository.findOne({
            where: { id, tenantId, isActive: true },
        });
        if (!vehicle) {
            throw new common_1.NotFoundException(`Vehicle with ID ${id} not found`);
        }
        return vehicle;
    }
    async findByVin(tenantId, vin) {
        return this.vehicleRepository.findOne({
            where: { vin, tenantId, isActive: true },
        });
    }
    async update(tenantId, id, dto) {
        const vehicle = await this.findOne(tenantId, id);
        Object.assign(vehicle, dto);
        return this.vehicleRepository.save(vehicle);
    }
    async updateMileage(tenantId, id, mileage) {
        const vehicle = await this.findOne(tenantId, id);
        vehicle.mileage = mileage;
        return this.vehicleRepository.save(vehicle);
    }
    async remove(tenantId, id) {
        const vehicle = await this.findOne(tenantId, id);
        vehicle.isActive = false;
        await this.vehicleRepository.save(vehicle);
    }
    async hardDelete(tenantId, id) {
        const vehicle = await this.findOne(tenantId, id);
        await this.vehicleRepository.remove(vehicle);
    }
    async getVehicleStats(tenantId) {
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
};
exports.VehiclesService = VehiclesService;
exports.VehiclesService = VehiclesService = __decorate([
    (0, common_1.Injectable)(),
    __param(0, (0, typeorm_1.InjectRepository)(vehicle_entity_1.Vehicle)),
    __metadata("design:paramtypes", [typeorm_2.Repository])
], VehiclesService);
//# sourceMappingURL=vehicles.service.js.map