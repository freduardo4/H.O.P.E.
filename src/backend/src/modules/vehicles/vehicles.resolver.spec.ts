import { Test, TestingModule } from '@nestjs/testing';
import { VehiclesResolver, PaginatedVehicles } from './vehicles.resolver';
import { VehiclesService } from './vehicles.service';
import { Vehicle } from './entities/vehicle.entity';
import { User, UserRole } from '../auth/entities/user.entity';

describe('VehiclesResolver', () => {
    let resolver: VehiclesResolver;
    let service: jest.Mocked<VehiclesService>;

    const mockUser: Partial<User> = {
        id: 'user-1',
        tenantId: 'tenant-1',
        role: UserRole.TECHNICIAN,
    };

    const mockVehicle: Partial<Vehicle> = {
        id: 'vehicle-1',
        vin: '1234567890ABCDEFG',
        make: 'Toyota',
        model: 'Supra',
        year: 2020,
    };

    const mockPaginatedResult: PaginatedVehicles = {
        items: [mockVehicle as Vehicle],
        total: 1,
        page: 1,
        limit: 10,
        totalPages: 1,
    };

    beforeEach(async () => {
        const mockVehiclesService = {
            findAll: jest.fn(),
            findOne: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                VehiclesResolver,
                {
                    provide: VehiclesService,
                    useValue: mockVehiclesService,
                },
            ],
        }).compile();

        resolver = module.get<VehiclesResolver>(VehiclesResolver);
        service = module.get(VehiclesService);
    });

    it('should be defined', () => {
        expect(resolver).toBeDefined();
    });

    describe('findAll', () => {
        it('should return paginated vehicles', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);
            const result = await resolver.findAll(mockUser as User, { make: 'Toyota' });
            expect(result).toEqual(mockPaginatedResult);
            expect(service.findAll).toHaveBeenCalledWith({
                tenantId: 'tenant-1',
                make: 'Toyota',
            });
        });
    });

    describe('findOne', () => {
        it('should return a single vehicle', async () => {
            service.findOne.mockResolvedValue(mockVehicle as Vehicle);
            const result = await resolver.findOne(mockUser as User, 'vehicle-1');
            expect(result).toEqual(mockVehicle);
            expect(service.findOne).toHaveBeenCalledWith('tenant-1', 'vehicle-1');
        });
    });
});
