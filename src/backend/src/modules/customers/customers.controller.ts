import {
    Controller,
    Get,
    Post,
    Body,
    Patch,
    Param,
    Delete,
    Query,
    UseGuards,
    ParseUUIDPipe,
} from '@nestjs/common';
import { CustomersService, PaginatedCustomers } from './customers.service';
import { CreateCustomerDto, UpdateCustomerDto } from './dto';
import { Customer, CustomerType } from './entities/customer.entity';
import { JwtAuthGuard, CurrentUser, Roles, RolesGuard } from '../auth';
import { UserRole, User } from '../auth/entities/user.entity';

@Controller('customers')
@UseGuards(JwtAuthGuard, RolesGuard)
export class CustomersController {
    constructor(private readonly customersService: CustomersService) {}

    @Post()
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async create(
        @CurrentUser() user: User,
        @Body() dto: CreateCustomerDto,
    ): Promise<Customer> {
        return this.customersService.create(user.tenantId, dto);
    }

    @Get()
    async findAll(
        @CurrentUser() user: User,
        @Query('type') type?: CustomerType,
        @Query('search') search?: string,
        @Query('city') city?: string,
        @Query('page') page = 1,
        @Query('limit') limit = 20,
    ): Promise<PaginatedCustomers> {
        return this.customersService.findAll({
            tenantId: user.tenantId,
            type,
            search,
            city,
            page: Number(page),
            limit: Number(limit),
        });
    }

    @Get('stats')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER)
    async getStats(@CurrentUser() user: User) {
        return this.customersService.getStats(user.tenantId);
    }

    @Get(':id')
    async findOne(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<Customer> {
        return this.customersService.findOne(user.tenantId, id);
    }

    @Get('email/:email')
    async findByEmail(
        @CurrentUser() user: User,
        @Param('email') email: string,
    ): Promise<Customer | null> {
        return this.customersService.findByEmail(user.tenantId, email);
    }

    @Patch(':id')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async update(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
        @Body() dto: UpdateCustomerDto,
    ): Promise<Customer> {
        return this.customersService.update(user.tenantId, id, dto);
    }

    @Delete(':id')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER)
    async remove(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<{ message: string }> {
        await this.customersService.remove(user.tenantId, id);
        return { message: 'Customer deleted successfully' };
    }
}
