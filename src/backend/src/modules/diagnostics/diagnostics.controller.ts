import {
    Controller,
    Get,
    Post,
    Body,
    Param,
    Query,
    UseGuards,
    ParseUUIDPipe,
} from '@nestjs/common';
import { DiagnosticsService, PaginatedSessions, SessionAnalytics } from './diagnostics.service';
import { CreateSessionDto, LogReadingDto, LogReadingsBatchDto, EndSessionDto } from './dto';
import { DiagnosticSession, SessionStatus, SessionType } from './entities/diagnostic-session.entity';
import { OBD2Reading } from './entities/obd2-reading.entity';
import { JwtAuthGuard, CurrentUser, Roles, RolesGuard } from '../auth';
import { UserRole, User } from '../auth/entities/user.entity';

@Controller('diagnostics')
@UseGuards(JwtAuthGuard, RolesGuard)
export class DiagnosticsController {
    constructor(private readonly diagnosticsService: DiagnosticsService) {}

    @Post('sessions')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async createSession(
        @CurrentUser() user: User,
        @Body() dto: CreateSessionDto,
    ): Promise<DiagnosticSession> {
        return this.diagnosticsService.createSession(user.tenantId, user.id, dto);
    }

    @Get('sessions')
    async findAllSessions(
        @CurrentUser() user: User,
        @Query('vehicleId') vehicleId?: string,
        @Query('technicianId') technicianId?: string,
        @Query('type') type?: SessionType,
        @Query('status') status?: SessionStatus,
        @Query('startDate') startDate?: string,
        @Query('endDate') endDate?: string,
        @Query('page') page = 1,
        @Query('limit') limit = 20,
    ): Promise<PaginatedSessions> {
        return this.diagnosticsService.findAllSessions({
            tenantId: user.tenantId,
            vehicleId,
            technicianId,
            type,
            status,
            startDate: startDate ? new Date(startDate) : undefined,
            endDate: endDate ? new Date(endDate) : undefined,
            page: Number(page),
            limit: Number(limit),
        });
    }

    @Get('sessions/:id')
    async findSession(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<DiagnosticSession> {
        return this.diagnosticsService.findSession(user.tenantId, id);
    }

    @Post('sessions/:id/end')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async endSession(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
        @Body() dto: EndSessionDto,
    ): Promise<DiagnosticSession> {
        return this.diagnosticsService.endSession(user.tenantId, id, dto);
    }

    @Post('sessions/:id/cancel')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async cancelSession(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<DiagnosticSession> {
        return this.diagnosticsService.cancelSession(user.tenantId, id);
    }

    @Post('readings')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async logReading(@Body() dto: LogReadingDto): Promise<OBD2Reading> {
        return this.diagnosticsService.logReading(dto);
    }

    @Post('readings/batch')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async logReadingsBatch(@Body() dto: LogReadingsBatchDto): Promise<OBD2Reading[]> {
        return this.diagnosticsService.logReadingsBatch(dto.readings);
    }

    @Get('sessions/:id/readings')
    async getSessionReadings(
        @Param('id', ParseUUIDPipe) sessionId: string,
        @Query('pid') pid?: string,
        @Query('startTime') startTime?: string,
        @Query('endTime') endTime?: string,
        @Query('limit') limit?: number,
    ): Promise<OBD2Reading[]> {
        return this.diagnosticsService.getSessionReadings(sessionId, {
            pid,
            startTime: startTime ? new Date(startTime) : undefined,
            endTime: endTime ? new Date(endTime) : undefined,
            limit: limit ? Number(limit) : undefined,
        });
    }

    @Get('sessions/:id/readings/latest')
    async getLatestReadings(
        @Param('id', ParseUUIDPipe) sessionId: string,
    ): Promise<Record<string, OBD2Reading>> {
        const latestMap = await this.diagnosticsService.getLatestReadings(sessionId);
        return Object.fromEntries(latestMap);
    }

    @Get('analytics')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER)
    async getAnalytics(
        @CurrentUser() user: User,
        @Query('startDate') startDate: string,
        @Query('endDate') endDate: string,
    ): Promise<SessionAnalytics> {
        return this.diagnosticsService.getSessionAnalytics(
            user.tenantId,
            new Date(startDate),
            new Date(endDate),
        );
    }
}
