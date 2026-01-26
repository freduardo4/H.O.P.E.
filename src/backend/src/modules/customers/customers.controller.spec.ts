import { Test, TestingModule } from '@nestjs/testing';
import { CustomersController } from './customers.controller';
import { CustomersService } from './customers.service';
import { CreateCustomerDto, UpdateCustomerDto } from './dto';
import { CustomerType } from './entities/customer.entity';
import { UserRole } from '../auth/entities/user.entity';

describe('CustomersController', () => {
    let controller: CustomersController;
    let service: jest.Mocked<CustomersService>;

    const mockUser = {
        id: 'user-uuid',
        email: 'test@example.com',
        firstName: 'Test',
        lastName: 'User',
        role: UserRole.TECHNICIAN,
        tenantId: 'tenant-uuid',
        isActive: true,
    };

    const mockCustomer = {
        id: 'customer-uuid',
        tenantId: 'tenant-uuid',
        firstName: 'John',
        lastName: 'Doe',
        email: 'john.doe@example.com',
        phone: '+1234567890',
        type: CustomerType.INDIVIDUAL,
        companyName: null,
        taxId: null,
        street: '123 Main St',
        city: 'New York',
        state: 'NY',
        postalCode: '10001',
        country: 'USA',
        notes: 'VIP customer',
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    beforeEach(async () => {
        const mockService = {
            create: jest.fn(),
            findAll: jest.fn(),
            findOne: jest.fn(),
            findByEmail: jest.fn(),
            update: jest.fn(),
            remove: jest.fn(),
            getStats: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            controllers: [CustomersController],
            providers: [
                {
                    provide: CustomersService,
                    useValue: mockService,
                },
            ],
        }).compile();

        controller = module.get<CustomersController>(CustomersController);
        service = module.get(CustomersService);
    });

    describe('create', () => {
        it('should create a new customer', async () => {
            const dto: CreateCustomerDto = {
                firstName: 'John',
                lastName: 'Doe',
                email: 'john.doe@example.com',
                phone: '+1234567890',
                type: CustomerType.INDIVIDUAL,
            };

            service.create.mockResolvedValue(mockCustomer as any);

            const result = await controller.create(mockUser as any, dto);

            expect(service.create).toHaveBeenCalledWith('tenant-uuid', dto);
            expect(result.firstName).toBe('John');
            expect(result.lastName).toBe('Doe');
            expect(result.email).toBe('john.doe@example.com');
        });

        it('should create a business customer', async () => {
            const dto: CreateCustomerDto = {
                firstName: 'Jane',
                lastName: 'Smith',
                email: 'contact@acmecorp.com',
                type: CustomerType.BUSINESS,
                companyName: 'Acme Corp',
                taxId: 'US123456789',
            };

            const businessCustomer = {
                ...mockCustomer,
                ...dto,
                id: 'business-customer-uuid',
            };

            service.create.mockResolvedValue(businessCustomer as any);

            const result = await controller.create(mockUser as any, dto);

            expect(service.create).toHaveBeenCalledWith('tenant-uuid', dto);
            expect(result.type).toBe(CustomerType.BUSINESS);
            expect(result.companyName).toBe('Acme Corp');
        });
    });

    describe('findAll', () => {
        it('should return paginated customers', async () => {
            const paginatedResult = {
                items: [mockCustomer],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };

            service.findAll.mockResolvedValue(paginatedResult as any);

            const result = await controller.findAll(mockUser as any);

            expect(service.findAll).toHaveBeenCalledWith({
                tenantId: 'tenant-uuid',
                type: undefined,
                search: undefined,
                city: undefined,
                page: 1,
                limit: 20,
            });
            expect(result.items).toHaveLength(1);
            expect(result.total).toBe(1);
        });

        it('should filter by customer type', async () => {
            const paginatedResult = {
                items: [mockCustomer],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };

            service.findAll.mockResolvedValue(paginatedResult as any);

            await controller.findAll(mockUser as any, CustomerType.INDIVIDUAL);

            expect(service.findAll).toHaveBeenCalledWith(
                expect.objectContaining({ type: CustomerType.INDIVIDUAL }),
            );
        });

        it('should filter by search term', async () => {
            const paginatedResult = {
                items: [mockCustomer],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };

            service.findAll.mockResolvedValue(paginatedResult as any);

            await controller.findAll(
                mockUser as any,
                undefined,
                'John',
            );

            expect(service.findAll).toHaveBeenCalledWith(
                expect.objectContaining({ search: 'John' }),
            );
        });

        it('should filter by city', async () => {
            const paginatedResult = {
                items: [mockCustomer],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };

            service.findAll.mockResolvedValue(paginatedResult as any);

            await controller.findAll(
                mockUser as any,
                undefined,
                undefined,
                'New York',
            );

            expect(service.findAll).toHaveBeenCalledWith(
                expect.objectContaining({ city: 'New York' }),
            );
        });

        it('should handle pagination', async () => {
            const paginatedResult = {
                items: [mockCustomer],
                total: 50,
                page: 2,
                limit: 10,
                totalPages: 5,
            };

            service.findAll.mockResolvedValue(paginatedResult as any);

            await controller.findAll(
                mockUser as any,
                undefined,
                undefined,
                undefined,
                2,
                10,
            );

            expect(service.findAll).toHaveBeenCalledWith(
                expect.objectContaining({ page: 2, limit: 10 }),
            );
        });
    });

    describe('findOne', () => {
        it('should return a single customer by id', async () => {
            service.findOne.mockResolvedValue(mockCustomer as any);

            const result = await controller.findOne(mockUser as any, 'customer-uuid');

            expect(service.findOne).toHaveBeenCalledWith('tenant-uuid', 'customer-uuid');
            expect(result.id).toBe('customer-uuid');
            expect(result.email).toBe('john.doe@example.com');
        });
    });

    describe('findByEmail', () => {
        it('should return a customer by email', async () => {
            service.findByEmail.mockResolvedValue(mockCustomer as any);

            const result = await controller.findByEmail(mockUser as any, 'john.doe@example.com');

            expect(service.findByEmail).toHaveBeenCalledWith('tenant-uuid', 'john.doe@example.com');
            expect(result?.email).toBe('john.doe@example.com');
        });

        it('should return null when customer not found', async () => {
            service.findByEmail.mockResolvedValue(null);

            const result = await controller.findByEmail(mockUser as any, 'unknown@example.com');

            expect(service.findByEmail).toHaveBeenCalledWith('tenant-uuid', 'unknown@example.com');
            expect(result).toBeNull();
        });
    });

    describe('update', () => {
        it('should update a customer', async () => {
            const dto: UpdateCustomerDto = {
                phone: '+0987654321',
                city: 'Los Angeles',
            };

            const updatedCustomer = {
                ...mockCustomer,
                phone: '+0987654321',
                city: 'Los Angeles',
            };

            service.update.mockResolvedValue(updatedCustomer as any);

            const result = await controller.update(mockUser as any, 'customer-uuid', dto);

            expect(service.update).toHaveBeenCalledWith('tenant-uuid', 'customer-uuid', dto);
            expect(result.phone).toBe('+0987654321');
            expect(result.city).toBe('Los Angeles');
        });

        it('should update customer email', async () => {
            const dto: UpdateCustomerDto = {
                email: 'new.email@example.com',
            };

            const updatedCustomer = {
                ...mockCustomer,
                email: 'new.email@example.com',
            };

            service.update.mockResolvedValue(updatedCustomer as any);

            const result = await controller.update(mockUser as any, 'customer-uuid', dto);

            expect(result.email).toBe('new.email@example.com');
        });
    });

    describe('remove', () => {
        it('should soft delete a customer', async () => {
            service.remove.mockResolvedValue(undefined);

            const result = await controller.remove(mockUser as any, 'customer-uuid');

            expect(service.remove).toHaveBeenCalledWith('tenant-uuid', 'customer-uuid');
            expect(result.message).toBe('Customer deleted successfully');
        });
    });

    describe('getStats', () => {
        it('should return customer statistics', async () => {
            const stats = {
                total: 100,
                byType: [
                    { type: CustomerType.INDIVIDUAL, count: 70 },
                    { type: CustomerType.BUSINESS, count: 25 },
                    { type: CustomerType.FLEET, count: 5 },
                ],
                byCity: [
                    { city: 'New York', count: 30 },
                    { city: 'Los Angeles', count: 20 },
                ],
                recentCustomers: [mockCustomer],
            };

            service.getStats.mockResolvedValue(stats as any);

            const result = await controller.getStats(mockUser as any);

            expect(service.getStats).toHaveBeenCalledWith('tenant-uuid');
            expect(result.total).toBe(100);
            expect(result.byType).toHaveLength(3);
            expect(result.byCity).toHaveLength(2);
        });
    });
});
