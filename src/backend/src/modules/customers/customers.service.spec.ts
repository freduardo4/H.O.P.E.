import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { ConflictException, NotFoundException } from '@nestjs/common';
import { CustomersService } from './customers.service';
import { Customer, CustomerType } from './entities/customer.entity';
import { CreateCustomerDto } from './dto';

describe('CustomersService', () => {
    let service: CustomersService;
    let repository: jest.Mocked<Repository<Customer>>;

    const mockCustomer: Partial<Customer> = {
        id: 'customer-uuid',
        tenantId: 'tenant-uuid',
        firstName: 'John',
        lastName: 'Doe',
        email: 'john.doe@example.com',
        phone: '+1234567890',
        type: CustomerType.INDIVIDUAL,
        companyName: null,
        taxId: null,
        address: '123 Main St',
        city: 'New York',
        state: 'NY',
        postalCode: '10001',
        country: 'USA',
        notes: 'VIP customer',
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    const mockQueryBuilder = {
        where: jest.fn().mockReturnThis(),
        andWhere: jest.fn().mockReturnThis(),
        orderBy: jest.fn().mockReturnThis(),
        addOrderBy: jest.fn().mockReturnThis(),
        skip: jest.fn().mockReturnThis(),
        take: jest.fn().mockReturnThis(),
        select: jest.fn().mockReturnThis(),
        addSelect: jest.fn().mockReturnThis(),
        groupBy: jest.fn().mockReturnThis(),
        limit: jest.fn().mockReturnThis(),
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
            count: jest.fn(),
            createQueryBuilder: jest.fn(() => mockQueryBuilder),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                CustomersService,
                {
                    provide: getRepositoryToken(Customer),
                    useValue: mockRepository,
                },
            ],
        }).compile();

        service = module.get<CustomersService>(CustomersService);
        repository = module.get(getRepositoryToken(Customer));

        // Reset mock implementations
        jest.clearAllMocks();
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

            repository.findOne.mockResolvedValue(null);
            repository.create.mockReturnValue(mockCustomer as Customer);
            repository.save.mockResolvedValue(mockCustomer as Customer);

            const result = await service.create('tenant-uuid', dto);

            expect(repository.findOne).toHaveBeenCalledWith({
                where: { email: dto.email, tenantId: 'tenant-uuid' },
            });
            expect(repository.create).toHaveBeenCalledWith({
                ...dto,
                tenantId: 'tenant-uuid',
            });
            expect(result.email).toBe('john.doe@example.com');
        });

        it('should throw ConflictException if email already exists', async () => {
            const dto: CreateCustomerDto = {
                firstName: 'John',
                lastName: 'Doe',
                email: 'john.doe@example.com',
                type: CustomerType.INDIVIDUAL,
            };

            repository.findOne.mockResolvedValue(mockCustomer as Customer);

            await expect(service.create('tenant-uuid', dto)).rejects.toThrow(
                ConflictException,
            );
            await expect(service.create('tenant-uuid', dto)).rejects.toThrow(
                'Customer with email john.doe@example.com already exists',
            );
        });

        it('should create a business customer with company details', async () => {
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
                id: 'business-uuid',
            };

            repository.findOne.mockResolvedValue(null);
            repository.create.mockReturnValue(businessCustomer as Customer);
            repository.save.mockResolvedValue(businessCustomer as Customer);

            const result = await service.create('tenant-uuid', dto);

            expect(result.type).toBe(CustomerType.BUSINESS);
            expect(result.companyName).toBe('Acme Corp');
        });
    });

    describe('findAll', () => {
        it('should return paginated customers', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockCustomer as Customer]);

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

        it('should filter by customer type', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockCustomer as Customer]);

            await service.findAll({
                tenantId: 'tenant-uuid',
                type: CustomerType.INDIVIDUAL,
            });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                'customer.type = :type',
                { type: CustomerType.INDIVIDUAL },
            );
        });

        it('should filter by search term', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockCustomer as Customer]);

            await service.findAll({
                tenantId: 'tenant-uuid',
                search: 'John',
            });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                expect.stringContaining('LOWER(customer.firstName)'),
                { search: '%John%' },
            );
        });

        it('should filter by city', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockCustomer as Customer]);

            await service.findAll({
                tenantId: 'tenant-uuid',
                city: 'New York',
            });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                'LOWER(customer.city) LIKE LOWER(:city)',
                { city: '%New York%' },
            );
        });

        it('should handle pagination correctly', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(50);
            mockQueryBuilder.getMany.mockResolvedValue([mockCustomer as Customer]);

            const result = await service.findAll({
                tenantId: 'tenant-uuid',
                page: 3,
                limit: 10,
            });

            expect(mockQueryBuilder.skip).toHaveBeenCalledWith(20); // (3-1) * 10
            expect(mockQueryBuilder.take).toHaveBeenCalledWith(10);
            expect(result.totalPages).toBe(5);
        });
    });

    describe('findOne', () => {
        it('should return a customer by id', async () => {
            repository.findOne.mockResolvedValue(mockCustomer as Customer);

            const result = await service.findOne('tenant-uuid', 'customer-uuid');

            expect(repository.findOne).toHaveBeenCalledWith({
                where: { id: 'customer-uuid', tenantId: 'tenant-uuid', isActive: true },
            });
            expect(result.id).toBe('customer-uuid');
        });

        it('should throw NotFoundException if customer not found', async () => {
            repository.findOne.mockResolvedValue(null);

            await expect(
                service.findOne('tenant-uuid', 'non-existent-uuid'),
            ).rejects.toThrow(NotFoundException);
            await expect(
                service.findOne('tenant-uuid', 'non-existent-uuid'),
            ).rejects.toThrow('Customer with ID non-existent-uuid not found');
        });

        it('should not return customers from other tenants', async () => {
            repository.findOne.mockResolvedValue(null);

            await expect(
                service.findOne('other-tenant-uuid', 'customer-uuid'),
            ).rejects.toThrow(NotFoundException);
        });
    });

    describe('findByEmail', () => {
        it('should return a customer by email', async () => {
            repository.findOne.mockResolvedValue(mockCustomer as Customer);

            const result = await service.findByEmail('tenant-uuid', 'john.doe@example.com');

            expect(repository.findOne).toHaveBeenCalledWith({
                where: { email: 'john.doe@example.com', tenantId: 'tenant-uuid', isActive: true },
            });
            expect(result?.email).toBe('john.doe@example.com');
        });

        it('should return null if customer not found', async () => {
            repository.findOne.mockResolvedValue(null);

            const result = await service.findByEmail('tenant-uuid', 'unknown@example.com');

            expect(result).toBeNull();
        });
    });

    describe('update', () => {
        it('should update a customer', async () => {
            const updatedCustomer = {
                ...mockCustomer,
                phone: '+0987654321',
            };

            repository.findOne.mockResolvedValue(mockCustomer as Customer);
            repository.save.mockResolvedValue(updatedCustomer as Customer);

            const result = await service.update('tenant-uuid', 'customer-uuid', {
                phone: '+0987654321',
            });

            expect(result.phone).toBe('+0987654321');
        });

        it('should allow updating email if new email is unique', async () => {
            const updatedCustomer = {
                ...mockCustomer,
                email: 'new.email@example.com',
            };

            repository.findOne
                .mockResolvedValueOnce(mockCustomer as Customer) // findOne for the customer
                .mockResolvedValueOnce(null); // findOne for email check
            repository.save.mockResolvedValue(updatedCustomer as Customer);

            const result = await service.update('tenant-uuid', 'customer-uuid', {
                email: 'new.email@example.com',
            });

            expect(result.email).toBe('new.email@example.com');
        });

        it('should throw ConflictException if new email already exists', async () => {
            const existingCustomer = {
                ...mockCustomer,
                id: 'another-customer-uuid',
                email: 'existing@example.com',
            };

            repository.findOne
                .mockResolvedValueOnce(mockCustomer as Customer) // findOne for the customer
                .mockResolvedValueOnce(existingCustomer as Customer); // findOne for email check

            await expect(
                service.update('tenant-uuid', 'customer-uuid', {
                    email: 'existing@example.com',
                }),
            ).rejects.toThrow(ConflictException);
        });

        it('should allow keeping the same email', async () => {
            repository.findOne.mockResolvedValue(mockCustomer as Customer);
            repository.save.mockResolvedValue(mockCustomer as Customer);

            const result = await service.update('tenant-uuid', 'customer-uuid', {
                email: 'john.doe@example.com', // same email
                phone: '+0987654321',
            });

            // Should only call findOne once (not check for duplicate)
            expect(result.email).toBe('john.doe@example.com');
        });

        it('should throw NotFoundException if customer not found', async () => {
            repository.findOne.mockResolvedValue(null);

            await expect(
                service.update('tenant-uuid', 'non-existent-uuid', { phone: '+123' }),
            ).rejects.toThrow(NotFoundException);
        });
    });

    describe('remove', () => {
        it('should soft delete a customer', async () => {
            const deletedCustomer = {
                ...mockCustomer,
                isActive: false,
            };

            repository.findOne.mockResolvedValue(mockCustomer as Customer);
            repository.save.mockResolvedValue(deletedCustomer as Customer);

            await service.remove('tenant-uuid', 'customer-uuid');

            expect(repository.save).toHaveBeenCalledWith(
                expect.objectContaining({ isActive: false }),
            );
        });

        it('should throw NotFoundException if customer not found', async () => {
            repository.findOne.mockResolvedValue(null);

            await expect(
                service.remove('tenant-uuid', 'non-existent-uuid'),
            ).rejects.toThrow(NotFoundException);
        });
    });

    describe('getStats', () => {
        it('should return customer statistics', async () => {
            repository.count.mockResolvedValue(100);
            mockQueryBuilder.getRawMany
                .mockResolvedValueOnce([
                    { type: CustomerType.INDIVIDUAL, count: '70' },
                    { type: CustomerType.BUSINESS, count: '25' },
                    { type: CustomerType.FLEET, count: '5' },
                ])
                .mockResolvedValueOnce([
                    { city: 'New York', count: '30' },
                    { city: 'Los Angeles', count: '20' },
                ]);
            repository.find.mockResolvedValue([mockCustomer as Customer]);

            const result = await service.getStats('tenant-uuid');

            expect(result.total).toBe(100);
            expect(result.byType).toHaveLength(3);
            expect(result.byCity).toHaveLength(2);
            expect(result.recentCustomers).toHaveLength(1);
        });
    });
});
