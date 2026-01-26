import { Injectable, NotFoundException, ConflictException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Customer, CustomerType } from './entities/customer.entity';
import { CreateCustomerDto, UpdateCustomerDto } from './dto';

export interface CustomerSearchOptions {
    tenantId: string;
    type?: CustomerType;
    search?: string;
    city?: string;
    page?: number;
    limit?: number;
}

export interface PaginatedCustomers {
    items: Customer[];
    total: number;
    page: number;
    limit: number;
    totalPages: number;
}

@Injectable()
export class CustomersService {
    constructor(
        @InjectRepository(Customer)
        private readonly customerRepository: Repository<Customer>,
    ) {}

    async create(tenantId: string, dto: CreateCustomerDto): Promise<Customer> {
        const existingCustomer = await this.customerRepository.findOne({
            where: { email: dto.email, tenantId },
        });

        if (existingCustomer) {
            throw new ConflictException(`Customer with email ${dto.email} already exists`);
        }

        const customer = this.customerRepository.create({
            ...dto,
            tenantId,
        });

        return this.customerRepository.save(customer);
    }

    async findAll(options: CustomerSearchOptions): Promise<PaginatedCustomers> {
        const { tenantId, type, search, city, page = 1, limit = 20 } = options;

        const queryBuilder = this.customerRepository
            .createQueryBuilder('customer')
            .where('customer.tenantId = :tenantId', { tenantId })
            .andWhere('customer.isActive = :isActive', { isActive: true });

        if (type) {
            queryBuilder.andWhere('customer.type = :type', { type });
        }

        if (search) {
            queryBuilder.andWhere(
                '(LOWER(customer.firstName) LIKE LOWER(:search) OR LOWER(customer.lastName) LIKE LOWER(:search) OR LOWER(customer.email) LIKE LOWER(:search) OR LOWER(customer.companyName) LIKE LOWER(:search))',
                { search: `%${search}%` },
            );
        }

        if (city) {
            queryBuilder.andWhere('LOWER(customer.city) LIKE LOWER(:city)', { city: `%${city}%` });
        }

        const total = await queryBuilder.getCount();
        const items = await queryBuilder
            .orderBy('customer.lastName', 'ASC')
            .addOrderBy('customer.firstName', 'ASC')
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

    async findOne(tenantId: string, id: string): Promise<Customer> {
        const customer = await this.customerRepository.findOne({
            where: { id, tenantId, isActive: true },
        });

        if (!customer) {
            throw new NotFoundException(`Customer with ID ${id} not found`);
        }

        return customer;
    }

    async findByEmail(tenantId: string, email: string): Promise<Customer | null> {
        return this.customerRepository.findOne({
            where: { email, tenantId, isActive: true },
        });
    }

    async update(tenantId: string, id: string, dto: UpdateCustomerDto): Promise<Customer> {
        const customer = await this.findOne(tenantId, id);

        if (dto.email && dto.email !== customer.email) {
            const existingCustomer = await this.customerRepository.findOne({
                where: { email: dto.email, tenantId },
            });

            if (existingCustomer && existingCustomer.id !== id) {
                throw new ConflictException(`Customer with email ${dto.email} already exists`);
            }
        }

        Object.assign(customer, dto);

        return this.customerRepository.save(customer);
    }

    async remove(tenantId: string, id: string): Promise<void> {
        const customer = await this.findOne(tenantId, id);

        customer.isActive = false;
        await this.customerRepository.save(customer);
    }

    async getStats(tenantId: string): Promise<{
        total: number;
        byType: { type: string; count: number }[];
        byCity: { city: string; count: number }[];
        recentCustomers: Customer[];
    }> {
        const total = await this.customerRepository.count({
            where: { tenantId, isActive: true },
        });

        const byType = await this.customerRepository
            .createQueryBuilder('customer')
            .select('customer.type', 'type')
            .addSelect('COUNT(*)', 'count')
            .where('customer.tenantId = :tenantId', { tenantId })
            .andWhere('customer.isActive = :isActive', { isActive: true })
            .groupBy('customer.type')
            .getRawMany();

        const byCity = await this.customerRepository
            .createQueryBuilder('customer')
            .select('customer.city', 'city')
            .addSelect('COUNT(*)', 'count')
            .where('customer.tenantId = :tenantId', { tenantId })
            .andWhere('customer.isActive = :isActive', { isActive: true })
            .andWhere('customer.city IS NOT NULL')
            .groupBy('customer.city')
            .orderBy('count', 'DESC')
            .limit(10)
            .getRawMany();

        const recentCustomers = await this.customerRepository.find({
            where: { tenantId, isActive: true },
            order: { createdAt: 'DESC' },
            take: 5,
        });

        return { total, byType, byCity, recentCustomers };
    }
}
