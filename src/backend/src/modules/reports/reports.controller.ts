import {
    Controller,
    Get,
    Post,
    Body,
    Param,
    Delete,
    Query,
    UseGuards,
    ParseUUIDPipe,
} from '@nestjs/common';
import { ReportsService, PaginatedReports } from './reports.service';
import { CreateReportDto } from './dto';
import { Report, ReportType, ReportStatus } from './entities/report.entity';
import { JwtAuthGuard, CurrentUser, Roles, RolesGuard } from '../auth';
import { UserRole, User } from '../auth/entities/user.entity';

@Controller('reports')
@UseGuards(JwtAuthGuard, RolesGuard)
export class ReportsController {
    constructor(private readonly reportsService: ReportsService) { }

    @Post()
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async create(
        @CurrentUser() user: User,
        @Body() dto: CreateReportDto,
    ): Promise<Report> {
        return this.reportsService.create(user.tenantId, dto, user.id);
    }

    @Get()
    async findAll(
        @CurrentUser() user: User,
        @Query('vehicleId') vehicleId?: string,
        @Query('customerId') customerId?: string,
        @Query('reportType') reportType?: ReportType,
        @Query('status') status?: ReportStatus,
        @Query('page') page = 1,
        @Query('limit') limit = 20,
    ): Promise<PaginatedReports> {
        return this.reportsService.findAll({
            tenantId: user.tenantId,
            vehicleId,
            customerId,
            reportType,
            status,
            page: Number(page),
            limit: Number(limit),
        });
    }

    @Get(':id')
    async findOne(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<Report> {
        return this.reportsService.findOne(user.tenantId, id);
    }

    @Get(':id/download-url')
    async getDownloadUrl(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<{ url: string; expiresIn: number }> {
        const url = await this.reportsService.getDownloadUrl(user.tenantId, id);
        return { url, expiresIn: 3600 };
    }

    @Post(':id/regenerate')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async regenerate(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<Report> {
        return this.reportsService.regenerate(user.tenantId, id, user.id);
    }

    @Delete(':id')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER)
    async remove(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<{ message: string }> {
        await this.reportsService.remove(user.tenantId, id);
        return { message: 'Report deleted successfully' };
    }
}
