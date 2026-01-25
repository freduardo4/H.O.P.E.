import { Test, TestingModule } from '@nestjs/testing';
import { VehiclesController } from './vehicles.controller';
import { VehiclesService } from './vehicles.service';
import { CreateVehicleDto } from './dto';
import { FuelType, TransmissionType } from './entities/vehicle.entity';
import { UserRole } from '../auth/entities/user.entity';

describe('VehiclesController', () => {
    let controller: VehiclesController;
    let service: jest.Mocked<VehiclesService>;

    const mockUser = {
        id: 'user-uuid',
        email: 'test@example.com',
        firstName: 'Test',
        lastName: 'User',
        role: UserRole.TECHNICIAN,
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
        fuelType: FuelType.GASOLINE,
        transmission: TransmissionType.DSG,
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

        const module: TestingModule = await Test.createTestingModule({
            controllers: [VehiclesController],
            providers: [
                {
                    provide: VehiclesService,
                    useValue: mockService,
                },
            ],
        }).compile();

        controller = module.get<VehiclesController>(VehiclesController);
        service = module.get(VehiclesService);
    });

    describe('create', () => {
        it('should create a new vehicle', async () => {
            const dto: CreateVehicleDto = {
                customerId: 'customer-uuid',
                vin: '1HGBH41JXMN109186',
                make: 'Volkswagen',
                model: 'Golf GTI',
                year: 2022,
            };

            service.create.mockResolvedValue(mockVehicle as any);

            const result = await controller.create(mockUser as any, dto);

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

            service.findAll.mockResolvedValue(paginatedResult as any);

            const result = await controller.findAll(mockUser as any);

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

            service.findAll.mockResolvedValue(paginatedResult as any);

            await controller.findAll(
                mockUser as any,
                undefined,
                'Volkswagen',
            );

            expect(service.findAll).toHaveBeenCalledWith(
                expect.objectContaining({ make: 'Volkswagen' }),
            );
        });
    });

    describe('findOne', () => {
        it('should return a single vehicle by id', async () => {
            service.findOne.mockResolvedValue(mockVehicle as any);

            const result = await controller.findOne(mockUser as any, 'vehicle-uuid');

            expect(service.findOne).toHaveBeenCalledWith('tenant-uuid', 'vehicle-uuid');
            expect(result.id).toBe('vehicle-uuid');
        });
    });

    describe('findByVin', () => {
        it('should return a vehicle by VIN', async () => {
            service.findByVin.mockResolvedValue(mockVehicle as any);

            const result = await controller.findByVin(mockUser as any, '1HGBH41JXMN109186');

            expect(service.findByVin).toHaveBeenCalledWith('tenant-uuid', '1HGBH41JXMN109186');
            expect(result?.vin).toBe('1HGBH41JXMN109186');
        });
    });

    describe('update', () => {
        it('should update a vehicle', async () => {
            const updatedVehicle = { ...mockVehicle, mileage: 30000 };
            service.update.mockResolvedValue(updatedVehicle as any);

            const result = await controller.update(
                mockUser as any,
                'vehicle-uuid',
                { mileage: 30000 },
            );

            expect(service.update).toHaveBeenCalledWith('tenant-uuid', 'vehicle-uuid', { mileage: 30000 });
            expect(result.mileage).toBe(30000);
        });
    });

    describe('updateMileage', () => {
        it('should update vehicle mileage', async () => {
            const updatedVehicle = { ...mockVehicle, mileage: 35000 };
            service.updateMileage.mockResolvedValue(updatedVehicle as any);

            const result = await controller.updateMileage(mockUser as any, 'vehicle-uuid', 35000);

            expect(service.updateMileage).toHaveBeenCalledWith('tenant-uuid', 'vehicle-uuid', 35000);
            expect(result.mileage).toBe(35000);
        });
    });

    describe('remove', () => {
        it('should soft delete a vehicle', async () => {
            service.remove.mockResolvedValue(undefined);

            const result = await controller.remove(mockUser as any, 'vehicle-uuid');

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

            const result = await controller.getStats(mockUser as any);

            expect(service.getVehicleStats).toHaveBeenCalledWith('tenant-uuid');
            expect(result.total).toBe(10);
        });
    });
});
