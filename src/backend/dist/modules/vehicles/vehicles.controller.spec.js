"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const vehicles_controller_1 = require("./vehicles.controller");
const vehicles_service_1 = require("./vehicles.service");
const vehicle_entity_1 = require("./entities/vehicle.entity");
const user_entity_1 = require("../auth/entities/user.entity");
describe('VehiclesController', () => {
    let controller;
    let service;
    const mockUser = {
        id: 'user-uuid',
        email: 'test@example.com',
        firstName: 'Test',
        lastName: 'User',
        role: user_entity_1.UserRole.TECHNICIAN,
        tenantId: 'tenant-uuid',
        isActive: true,
    };
    const mockVehicle = {
        id: 'vehicle-uuid',
        tenantId: 'tenant-uuid',
        customerId: 'customer-uuid',
        vin: '1HGBH41JXMN109186',
        make: 'Volkswagen',
        model: 'Golf GTI',
        year: 2022,
        variant: 'Mk8',
        engineCode: 'DNUE',
        engineDisplacement: 1984,
        enginePower: 245,
        fuelType: vehicle_entity_1.FuelType.GASOLINE,
        transmission: vehicle_entity_1.TransmissionType.DSG,
        licensePlate: 'ABC-123',
        mileage: 25000,
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    beforeEach(async () => {
        const mockService = {
            create: jest.fn(),
            findAll: jest.fn(),
            findOne: jest.fn(),
            findByVin: jest.fn(),
            update: jest.fn(),
            updateMileage: jest.fn(),
            remove: jest.fn(),
            getVehicleStats: jest.fn(),
        };
        const module = await testing_1.Test.createTestingModule({
            controllers: [vehicles_controller_1.VehiclesController],
            providers: [
                {
                    provide: vehicles_service_1.VehiclesService,
                    useValue: mockService,
                },
            ],
        }).compile();
        controller = module.get(vehicles_controller_1.VehiclesController);
        service = module.get(vehicles_service_1.VehiclesService);
    });
    describe('create', () => {
        it('should create a new vehicle', async () => {
            const dto = {
                customerId: 'customer-uuid',
                vin: '1HGBH41JXMN109186',
                make: 'Volkswagen',
                model: 'Golf GTI',
                year: 2022,
            };
            service.create.mockResolvedValue(mockVehicle);
            const result = await controller.create(mockUser, dto);
            expect(service.create).toHaveBeenCalledWith('tenant-uuid', dto);
            expect(result.make).toBe('Volkswagen');
            expect(result.model).toBe('Golf GTI');
        });
    });
    describe('findAll', () => {
        it('should return paginated vehicles', async () => {
            const paginatedResult = {
                items: [mockVehicle],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAll.mockResolvedValue(paginatedResult);
            const result = await controller.findAll(mockUser);
            expect(service.findAll).toHaveBeenCalledWith({
                tenantId: 'tenant-uuid',
                customerId: undefined,
                make: undefined,
                model: undefined,
                year: undefined,
                vin: undefined,
                licensePlate: undefined,
                page: 1,
                limit: 20,
            });
            expect(result.items).toHaveLength(1);
            expect(result.total).toBe(1);
        });
        it('should filter by make', async () => {
            const paginatedResult = {
                items: [mockVehicle],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAll.mockResolvedValue(paginatedResult);
            await controller.findAll(mockUser, undefined, 'Volkswagen');
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ make: 'Volkswagen' }));
        });
    });
    describe('findOne', () => {
        it('should return a single vehicle by id', async () => {
            service.findOne.mockResolvedValue(mockVehicle);
            const result = await controller.findOne(mockUser, 'vehicle-uuid');
            expect(service.findOne).toHaveBeenCalledWith('tenant-uuid', 'vehicle-uuid');
            expect(result.id).toBe('vehicle-uuid');
        });
    });
    describe('findByVin', () => {
        it('should return a vehicle by VIN', async () => {
            service.findByVin.mockResolvedValue(mockVehicle);
            const result = await controller.findByVin(mockUser, '1HGBH41JXMN109186');
            expect(service.findByVin).toHaveBeenCalledWith('tenant-uuid', '1HGBH41JXMN109186');
            expect(result?.vin).toBe('1HGBH41JXMN109186');
        });
    });
    describe('update', () => {
        it('should update a vehicle', async () => {
            const updatedVehicle = { ...mockVehicle, mileage: 30000 };
            service.update.mockResolvedValue(updatedVehicle);
            const result = await controller.update(mockUser, 'vehicle-uuid', { mileage: 30000 });
            expect(service.update).toHaveBeenCalledWith('tenant-uuid', 'vehicle-uuid', { mileage: 30000 });
            expect(result.mileage).toBe(30000);
        });
    });
    describe('updateMileage', () => {
        it('should update vehicle mileage', async () => {
            const updatedVehicle = { ...mockVehicle, mileage: 35000 };
            service.updateMileage.mockResolvedValue(updatedVehicle);
            const result = await controller.updateMileage(mockUser, 'vehicle-uuid', 35000);
            expect(service.updateMileage).toHaveBeenCalledWith('tenant-uuid', 'vehicle-uuid', 35000);
            expect(result.mileage).toBe(35000);
        });
    });
    describe('remove', () => {
        it('should soft delete a vehicle', async () => {
            service.remove.mockResolvedValue(undefined);
            const result = await controller.remove(mockUser, 'vehicle-uuid');
            expect(service.remove).toHaveBeenCalledWith('tenant-uuid', 'vehicle-uuid');
            expect(result.message).toBe('Vehicle deleted successfully');
        });
    });
    describe('getStats', () => {
        it('should return vehicle statistics', async () => {
            const stats = {
                total: 10,
                byMake: [{ make: 'Volkswagen', count: 5 }],
                byYear: [{ year: 2022, count: 3 }],
                byFuelType: [{ fuelType: 'gasoline', count: 8 }],
            };
            service.getVehicleStats.mockResolvedValue(stats);
            const result = await controller.getStats(mockUser);
            expect(service.getVehicleStats).toHaveBeenCalledWith('tenant-uuid');
            expect(result.total).toBe(10);
        });
    });
});
//# sourceMappingURL=vehicles.controller.spec.js.map