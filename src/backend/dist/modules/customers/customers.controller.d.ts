import { CustomersService, PaginatedCustomers } from './customers.service';
import { CreateCustomerDto, UpdateCustomerDto } from './dto';
import { Customer, CustomerType } from './entities/customer.entity';
import { User } from '../auth/entities/user.entity';
export declare class CustomersController {
    private readonly customersService;
    constructor(customersService: CustomersService);
    create(user: User, dto: CreateCustomerDto): Promise<Customer>;
    findAll(user: User, type?: CustomerType, search?: string, city?: string, page?: number, limit?: number): Promise<PaginatedCustomers>;
    getStats(user: User): Promise<{
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
    findOne(user: User, id: string): Promise<Customer>;
    findByEmail(user: User, email: string): Promise<Customer | null>;
    update(user: User, id: string, dto: UpdateCustomerDto): Promise<Customer>;
    remove(user: User, id: string): Promise<{
        message: string;
    }>;
}
