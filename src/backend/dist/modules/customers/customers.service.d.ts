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
export declare class CustomersService {
    private readonly customerRepository;
    constructor(customerRepository: Repository<Customer>);
    create(tenantId: string, dto: CreateCustomerDto): Promise<Customer>;
    findAll(options: CustomerSearchOptions): Promise<PaginatedCustomers>;
    findOne(tenantId: string, id: string): Promise<Customer>;
    findByEmail(tenantId: string, email: string): Promise<Customer | null>;
    update(tenantId: string, id: string, dto: UpdateCustomerDto): Promise<Customer>;
    remove(tenantId: string, id: string): Promise<void>;
    getStats(tenantId: string): Promise<{
        total: number;
        byType: {
            type: string;
            count: number;
        }[];
        byCity: {
            city: string;
            count: number;
        }[];
        recentCustomers: Customer[];
    }>;
}
