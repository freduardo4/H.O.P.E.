"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const typeorm_1 = require("@nestjs/typeorm");
const common_1 = require("@nestjs/common");
const customers_service_1 = require("./customers.service");
const customer_entity_1 = require("./entities/customer.entity");
describe('CustomersService', () => {
    let service;
    let repository;
    const mockCustomer = {
        id: 'customer-uuid',
        tenantId: 'tenant-uuid',
        firstName: 'John',
        lastName: 'Doe',
        email: 'john.doe@example.com',
        phone: '+1234567890',
        type: customer_entity_1.CustomerType.INDIVIDUAL,
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
        const module = await testing_1.Test.createTestingModule({
            providers: [
                customers_service_1.CustomersService,
                {
                    provide: (0, typeorm_1.getRepositoryToken)(customer_entity_1.Customer),
                    useValue: mockRepository,
                },
            ],
        }).compile();
        service = module.get(customers_service_1.CustomersService);
        repository = module.get((0, typeorm_1.getRepositoryToken)(customer_entity_1.Customer));
        jest.clearAllMocks();
    });
    describe('create', () => {
        it('should create a new customer', async () => {
            const dto = {
                firstName: 'John',
                lastName: 'Doe',
                email: 'john.doe@example.com',
                phone: '+1234567890',
                type: customer_entity_1.CustomerType.INDIVIDUAL,
            };
            repository.findOne.mockResolvedValue(null);
            repository.create.mockReturnValue(mockCustomer);
            repository.save.mockResolvedValue(mockCustomer);
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
            const dto = {
                firstName: 'John',
                lastName: 'Doe',
                email: 'john.doe@example.com',
                type: customer_entity_1.CustomerType.INDIVIDUAL,
            };
            repository.findOne.mockResolvedValue(mockCustomer);
            await expect(service.create('tenant-uuid', dto)).rejects.toThrow(common_1.ConflictException);
            await expect(service.create('tenant-uuid', dto)).rejects.toThrow('Customer with email john.doe@example.com already exists');
        });
        it('should create a business customer with company details', async () => {
            const dto = {
                firstName: 'Jane',
                lastName: 'Smith',
                email: 'contact@acmecorp.com',
                type: customer_entity_1.CustomerType.BUSINESS,
                companyName: 'Acme Corp',
                taxId: 'US123456789',
            };
            const businessCustomer = {
                ...mockCustomer,
                ...dto,
                id: 'business-uuid',
            };
            repository.findOne.mockResolvedValue(null);
            repository.create.mockReturnValue(businessCustomer);
            repository.save.mockResolvedValue(businessCustomer);
            const result = await service.create('tenant-uuid', dto);
            expect(result.type).toBe(customer_entity_1.CustomerType.BUSINESS);
            expect(result.companyName).toBe('Acme Corp');
        });
    });
    describe('findAll', () => {
        it('should return paginated customers', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockCustomer]);
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
            mockQueryBuilder.getMany.mockResolvedValue([mockCustomer]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                type: customer_entity_1.CustomerType.INDIVIDUAL,
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('customer.type = :type', { type: customer_entity_1.CustomerType.INDIVIDUAL });
        });
        it('should filter by search term', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockCustomer]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                search: 'John',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(expect.stringContaining('LOWER(customer.firstName)'), { search: '%John%' });
        });
        it('should filter by city', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockCustomer]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                city: 'New York',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('LOWER(customer.city) LIKE LOWER(:city)', { city: '%New York%' });
        });
        it('should handle pagination correctly', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(50);
            mockQueryBuilder.getMany.mockResolvedValue([mockCustomer]);
            const result = await service.findAll({
                tenantId: 'tenant-uuid',
                page: 3,
                limit: 10,
            });
            expect(mockQueryBuilder.skip).toHaveBeenCalledWith(20);
            expect(mockQueryBuilder.take).toHaveBeenCalledWith(10);
            expect(result.totalPages).toBe(5);
        });
    });
    describe('findOne', () => {
        it('should return a customer by id', async () => {
            repository.findOne.mockResolvedValue(mockCustomer);
            const result = await service.findOne('tenant-uuid', 'customer-uuid');
            expect(repository.findOne).toHaveBeenCalledWith({
                where: { id: 'customer-uuid', tenantId: 'tenant-uuid', isActive: true },
            });
            expect(result.id).toBe('customer-uuid');
        });
        it('should throw NotFoundException if customer not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.findOne('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
            await expect(service.findOne('tenant-uuid', 'non-existent-uuid')).rejects.toThrow('Customer with ID non-existent-uuid not found');
        });
        it('should not return customers from other tenants', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.findOne('other-tenant-uuid', 'customer-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('findByEmail', () => {
        it('should return a customer by email', async () => {
            repository.findOne.mockResolvedValue(mockCustomer);
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
            repository.findOne.mockResolvedValue(mockCustomer);
            repository.save.mockResolvedValue(updatedCustomer);
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
                .mockResolvedValueOnce(mockCustomer)
                .mockResolvedValueOnce(null);
            repository.save.mockResolvedValue(updatedCustomer);
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
                .mockResolvedValueOnce(mockCustomer)
                .mockResolvedValueOnce(existingCustomer);
            await expect(service.update('tenant-uuid', 'customer-uuid', {
                email: 'existing@example.com',
            })).rejects.toThrow(common_1.ConflictException);
        });
        it('should allow keeping the same email', async () => {
            repository.findOne.mockResolvedValue(mockCustomer);
            repository.save.mockResolvedValue(mockCustomer);
            const result = await service.update('tenant-uuid', 'customer-uuid', {
                email: 'john.doe@example.com',
                phone: '+0987654321',
            });
            expect(result.email).toBe('john.doe@example.com');
        });
        it('should throw NotFoundException if customer not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.update('tenant-uuid', 'non-existent-uuid', { phone: '+123' })).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('remove', () => {
        it('should soft delete a customer', async () => {
            const deletedCustomer = {
                ...mockCustomer,
                isActive: false,
            };
            repository.findOne.mockResolvedValue(mockCustomer);
            repository.save.mockResolvedValue(deletedCustomer);
            await service.remove('tenant-uuid', 'customer-uuid');
            expect(repository.save).toHaveBeenCalledWith(expect.objectContaining({ isActive: false }));
        });
        it('should throw NotFoundException if customer not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.remove('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('getStats', () => {
        it('should return customer statistics', async () => {
            repository.count.mockResolvedValue(100);
            mockQueryBuilder.getRawMany
                .mockResolvedValueOnce([
                { type: customer_entity_1.CustomerType.INDIVIDUAL, count: '70' },
                { type: customer_entity_1.CustomerType.BUSINESS, count: '25' },
                { type: customer_entity_1.CustomerType.FLEET, count: '5' },
            ])
                .mockResolvedValueOnce([
                { city: 'New York', count: '30' },
                { city: 'Los Angeles', count: '20' },
            ]);
            repository.find.mockResolvedValue([mockCustomer]);
            const result = await service.getStats('tenant-uuid');
            expect(result.total).toBe(100);
            expect(result.byType).toHaveLength(3);
            expect(result.byCity).toHaveLength(2);
            expect(result.recentCustomers).toHaveLength(1);
        });
    });
});
//# sourceMappingURL=customers.service.spec.js.map