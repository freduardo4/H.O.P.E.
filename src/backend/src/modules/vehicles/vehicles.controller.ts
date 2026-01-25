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
import { VehiclesService, PaginatedVehicles } from './vehicles.service';
import { CreateVehicleDto, UpdateVehicleDto } from './dto';
import { Vehicle } from './entities/vehicle.entity';
import { JwtAuthGuard, CurrentUser, Roles, RolesGuard } from '../auth';
import { UserRole, User } from '../auth/entities/user.entity';

@Controller('vehicles')
@UseGuards(JwtAuthGuard, RolesGuard)
export class VehiclesController {
    constructor(private readonly vehiclesService: VehiclesService) {}

    @Post()
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async create(
        @CurrentUser() user: User,
        @Body() dto: CreateVehicleDto,
    ): Promise<Vehicle> {
        return this.vehiclesService.create(user.tenantId, dto);
    }

    @Get()
    async findAll(
        @CurrentUser() user: User,
        @Query('customerId') customerId?: string,
        @Query('make') make?: string,
        @Query('model') model?: string,
        @Query('year') year?: number,
        @Query('vin') vin?: string,
        @Query('licensePlate') licensePlate?: string,
        @Query('page') page = 1,
        @Query('limit') limit = 20,
    ): Promise<PaginatedVehicles> {
        return this.vehiclesService.findAll({
            tenantId: user.tenantId,
            customerId,
            make,
            model,
            year,
            vin,
            licensePlate,
            page: Number(page),
            limit: Number(limit),
        });
    }

    @Get('stats')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER)
    async getStats(@CurrentUser() user: User) {
        return this.vehiclesService.getVehicleStats(user.tenantId);
    }

    @Get(':id')
    async findOne(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<Vehicle> {
        return this.vehiclesService.findOne(user.tenantId, id);
    }

    @Get('vin/:vin')
    async findByVin(
        @CurrentUser() user: User,
        @Param('vin') vin: string,
    ): Promise<Vehicle | null> {
        return this.vehiclesService.findByVin(user.tenantId, vin);
    }

    @Patch(':id')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async update(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
        @Body() dto: UpdateVehicleDto,
    ): Promise<Vehicle> {
        return this.vehiclesService.update(user.tenantId, id, dto);
    }

    @Patch(':id/mileage')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async updateMileage(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
        @Body('mileage') mileage: number,
    ): Promise<Vehicle> {
        return this.vehiclesService.updateMileage(user.tenantId, id, mileage);
    }

    @Delete(':id')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER)
    async remove(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<{ message: string }> {
        await this.vehiclesService.remove(user.tenantId, id);
        return { message: 'Vehicle deleted successfully' };
    }
}
