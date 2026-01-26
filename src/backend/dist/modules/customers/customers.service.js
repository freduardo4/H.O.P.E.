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
exports.CustomersService = void 0;
const common_1 = require("@nestjs/common");
const typeorm_1 = require("@nestjs/typeorm");
const typeorm_2 = require("typeorm");
const customer_entity_1 = require("./entities/customer.entity");
let CustomersService = class CustomersService {
    constructor(customerRepository) {
        this.customerRepository = customerRepository;
    }
    async create(tenantId, dto) {
        const existingCustomer = await this.customerRepository.findOne({
            where: { email: dto.email, tenantId },
        });
        if (existingCustomer) {
            throw new common_1.ConflictException(`Customer with email ${dto.email} already exists`);
        }
        const customer = this.customerRepository.create({
            ...dto,
            tenantId,
        });
        return this.customerRepository.save(customer);
    }
    async findAll(options) {
        const { tenantId, type, search, city, page = 1, limit = 20 } = options;
        const queryBuilder = this.customerRepository
            .createQueryBuilder('customer')
            .where('customer.tenantId = :tenantId', { tenantId })
            .andWhere('customer.isActive = :isActive', { isActive: true });
        if (type) {
            queryBuilder.andWhere('customer.type = :type', { type });
        }
        if (search) {
            queryBuilder.andWhere('(LOWER(customer.firstName) LIKE LOWER(:search) OR LOWER(customer.lastName) LIKE LOWER(:search) OR LOWER(customer.email) LIKE LOWER(:search) OR LOWER(customer.companyName) LIKE LOWER(:search))', { search: `%${search}%` });
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
    async findOne(tenantId, id) {
        const customer = await this.customerRepository.findOne({
            where: { id, tenantId, isActive: true },
        });
        if (!customer) {
            throw new common_1.NotFoundException(`Customer with ID ${id} not found`);
        }
        return customer;
    }
    async findByEmail(tenantId, email) {
        return this.customerRepository.findOne({
            where: { email, tenantId, isActive: true },
        });
    }
    async update(tenantId, id, dto) {
        const customer = await this.findOne(tenantId, id);
        if (dto.email && dto.email !== customer.email) {
            const existingCustomer = await this.customerRepository.findOne({
                where: { email: dto.email, tenantId },
            });
            if (existingCustomer && existingCustomer.id !== id) {
                throw new common_1.ConflictException(`Customer with email ${dto.email} already exists`);
            }
        }
        Object.assign(customer, dto);
        return this.customerRepository.save(customer);
    }
    async remove(tenantId, id) {
        const customer = await this.findOne(tenantId, id);
        customer.isActive = false;
        await this.customerRepository.save(customer);
    }
    async getStats(tenantId) {
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
};
exports.CustomersService = CustomersService;
exports.CustomersService = CustomersService = __decorate([
    (0, common_1.Injectable)(),
    __param(0, (0, typeorm_1.InjectRepository)(customer_entity_1.Customer)),
    __metadata("design:paramtypes", [typeorm_2.Repository])
], CustomersService);
//# sourceMappingURL=customers.service.js.map