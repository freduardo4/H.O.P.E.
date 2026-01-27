"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const typeorm_1 = require("@nestjs/typeorm");
const common_1 = require("@nestjs/common");
const vehicles_service_1 = require("./vehicles.service");
const vehicle_entity_1 = require("./entities/vehicle.entity");
describe('VehiclesService', () => {
    let service;
    let repository;
    const mockVehicle = {
        id: 'vehicle-uuid',
        tenantId: 'tenant-uuid',
        customerId: 'customer-uuid',
        vin: 'WDB9062331Y123456',
        make: 'Mercedes-Benz',
        model: 'E-Class',
        year: 2022,
        variant: 'E350',
        engineCode: 'M276',
        engineDisplacement: 3500,
        enginePower: 302,
        fuelType: vehicle_entity_1.FuelType.GASOLINE,
        transmission: vehicle_entity_1.TransmissionType.AUTOMATIC,
        licensePlate: 'ABC-1234',
        mileage: 25000,
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    const mockQueryBuilder = {
        where: jest.fn().mockReturnThis(),
        andWhere: jest.fn().mockReturnThis(),
        orderBy: jest.fn().mockReturnThis(),
        skip: jest.fn().mockReturnThis(),
        take: jest.fn().mockReturnThis(),
        select: jest.fn().mockReturnThis(),
        addSelect: jest.fn().mockReturnThis(),
        groupBy: jest.fn().mockReturnThis(),
        getCount: jest.fn(),
        getMany: jest.fn(),
        getRawMany: jest.fn(),
    };
    beforeEach(async () => {
        const mockRepository = {
            findOne: jest.fn(),
            find: jest.fn(),
            create: jest.fn(),
            save: jest.fn(),
            remove: jest.fn(),
            count: jest.fn(),
            createQueryBuilder: jest.fn(() => mockQueryBuilder),
        };
        const module = await testing_1.Test.createTestingModule({
            providers: [
                vehicles_service_1.VehiclesService,
                {
                    provide: (0, typeorm_1.getRepositoryToken)(vehicle_entity_1.Vehicle),
                    useValue: mockRepository,
                },
            ],
        }).compile();
        service = module.get(vehicles_service_1.VehiclesService);
        repository = module.get((0, typeorm_1.getRepositoryToken)(vehicle_entity_1.Vehicle));
        jest.clearAllMocks();
    });
    describe('create', () => {
        it('should create a new vehicle', async () => {
            const dto = {
                make: 'Mercedes-Benz',
                model: 'E-Class',
                year: 2022,
                vin: 'WDB9062331Y123456',
            };
            repository.create.mockReturnValue(mockVehicle);
            repository.save.mockResolvedValue(mockVehicle);
            const result = await service.create('tenant-uuid', dto);
            expect(repository.create).toHaveBeenCalledWith({
                ...dto,
                tenantId: 'tenant-uuid',
            });
            expect(result.make).toBe('Mercedes-Benz');
            expect(result.model).toBe('E-Class');
        });
        it('should create vehicle with all optional fields', async () => {
            const dto = {
                make: 'BMW',
                model: '3 Series',
                year: 2023,
                customerId: 'customer-uuid',
                vin: 'WBA12345678901234',
                fuelType: vehicle_entity_1.FuelType.DIESEL,
                transmission: vehicle_entity_1.TransmissionType.AUTOMATIC,
                mileage: 10000,
            };
            const fullVehicle = { ...mockVehicle, ...dto };
            repository.create.mockReturnValue(fullVehicle);
            repository.save.mockResolvedValue(fullVehicle);
            const result = await service.create('tenant-uuid', dto);
            expect(result.fuelType).toBe(vehicle_entity_1.FuelType.DIESEL);
            expect(result.transmission).toBe(vehicle_entity_1.TransmissionType.AUTOMATIC);
        });
    });
    describe('findAll', () => {
        it('should return paginated vehicles', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockVehicle]);
            const result = await service.findAll({
                tenantId: 'tenant-uuid',
                page: 1,
                limit: 20,
            });
            expect(result.items).toHaveLength(1);
            expect(result.total).toBe(1);
            expect(result.page).toBe(1);
            expect(result.limit).toBe(20);
            expect(result.totalPages).toBe(1);
        });
        it('should filter by make', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockVehicle]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                make: 'Mercedes',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('LOWER(vehicle.make) LIKE LOWER(:make)', { make: '%Mercedes%' });
        });
        it('should filter by model', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockVehicle]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                model: 'E-Class',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('LOWER(vehicle.model) LIKE LOWER(:model)', { model: '%E-Class%' });
        });
        it('should filter by year', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockVehicle]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                year: 2022,
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('vehicle.year = :year', { year: 2022 });
        });
        it('should filter by VIN', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockVehicle]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                vin: 'WDB9062331Y123456',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('vehicle.vin = :vin', { vin: 'WDB9062331Y123456' });
        });
        it('should handle pagination correctly', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(100);
            mockQueryBuilder.getMany.mockResolvedValue([mockVehicle]);
            const result = await service.findAll({
                tenantId: 'tenant-uuid',
                page: 3,
                limit: 10,
            });
            expect(mockQueryBuilder.skip).toHaveBeenCalledWith(20);
            expect(mockQueryBuilder.take).toHaveBeenCalledWith(10);
            expect(result.totalPages).toBe(10);
        });
    });
    describe('findOne', () => {
        it('should return a vehicle by id', async () => {
            repository.findOne.mockResolvedValue(mockVehicle);
            const result = await service.findOne('tenant-uuid', 'vehicle-uuid');
            expect(repository.findOne).toHaveBeenCalledWith({
                where: { id: 'vehicle-uuid', tenantId: 'tenant-uuid', isActive: true },
            });
            expect(result.id).toBe('vehicle-uuid');
        });
        it('should throw NotFoundException if vehicle not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.findOne('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
            await expect(service.findOne('tenant-uuid', 'non-existent-uuid')).rejects.toThrow('Vehicle with ID non-existent-uuid not found');
        });
    });
    describe('findByVin', () => {
        it('should return a vehicle by VIN', async () => {
            repository.findOne.mockResolvedValue(mockVehicle);
            const result = await service.findByVin('tenant-uuid', 'WDB9062331Y123456');
            expect(repository.findOne).toHaveBeenCalledWith({
                where: { vin: 'WDB9062331Y123456', tenantId: 'tenant-uuid', isActive: true },
            });
            expect(result?.vin).toBe('WDB9062331Y123456');
        });
        it('should return null if VIN not found', async () => {
            repository.findOne.mockResolvedValue(null);
            const result = await service.findByVin('tenant-uuid', 'INVALID_VIN');
            expect(result).toBeNull();
        });
    });
    describe('update', () => {
        it('should update a vehicle', async () => {
            const updatedVehicle = { ...mockVehicle, mileage: 30000 };
            repository.findOne.mockResolvedValue(mockVehicle);
            repository.save.mockResolvedValue(updatedVehicle);
            const result = await service.update('tenant-uuid', 'vehicle-uuid', {
                mileage: 30000,
            });
            expect(result.mileage).toBe(30000);
        });
        it('should throw NotFoundException if vehicle not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.update('tenant-uuid', 'non-existent-uuid', { mileage: 30000 })).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('updateMileage', () => {
        it('should update vehicle mileage', async () => {
            const updatedVehicle = { ...mockVehicle, mileage: 35000 };
            repository.findOne.mockResolvedValue(mockVehicle);
            repository.save.mockResolvedValue(updatedVehicle);
            const result = await service.updateMileage('tenant-uuid', 'vehicle-uuid', 35000);
            expect(result.mileage).toBe(35000);
        });
    });
    describe('remove', () => {
        it('should soft delete a vehicle', async () => {
            const deletedVehicle = { ...mockVehicle, isActive: false };
            repository.findOne.mockResolvedValue(mockVehicle);
            repository.save.mockResolvedValue(deletedVehicle);
            await service.remove('tenant-uuid', 'vehicle-uuid');
            expect(repository.save).toHaveBeenCalledWith(expect.objectContaining({ isActive: false }));
        });
        it('should throw NotFoundException if vehicle not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.remove('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('hardDelete', () => {
        it('should permanently delete a vehicle', async () => {
            repository.findOne.mockResolvedValue(mockVehicle);
            repository.remove.mockResolvedValue(mockVehicle);
            await service.hardDelete('tenant-uuid', 'vehicle-uuid');
            expect(repository.remove).toHaveBeenCalledWith(mockVehicle);
        });
    });
    describe('getVehicleStats', () => {
        it('should return vehicle statistics', async () => {
            repository.count.mockResolvedValue(50);
            mockQueryBuilder.getRawMany
                .mockResolvedValueOnce([
                { make: 'Mercedes-Benz', count: '20' },
                { make: 'BMW', count: '15' },
                { make: 'Audi', count: '15' },
            ])
                .mockResolvedValueOnce([
                { year: 2023, count: '25' },
                { year: 2022, count: '15' },
                { year: 2021, count: '10' },
            ])
                .mockResolvedValueOnce([
                { fuelType: vehicle_entity_1.FuelType.GASOLINE, count: '30' },
                { fuelType: vehicle_entity_1.FuelType.DIESEL, count: '15' },
                { fuelType: vehicle_entity_1.FuelType.ELECTRIC, count: '5' },
            ]);
            const result = await service.getVehicleStats('tenant-uuid');
            expect(result.total).toBe(50);
            expect(result.byMake).toHaveLength(3);
            expect(result.byYear).toHaveLength(3);
            expect(result.byFuelType).toHaveLength(3);
        });
    });
});
//# sourceMappingURL=vehicles.service.spec.js.map