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
    UseInterceptors,
    UploadedFile,
    BadRequestException,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ECUCalibrationsService, PaginatedCalibrations } from './ecu-calibrations.service';
import { CreateECUCalibrationDto, UpdateECUCalibrationDto } from './dto';
import { ECUCalibration, CalibrationType } from './entities/ecu-calibration.entity';
import { JwtAuthGuard, CurrentUser, Roles, RolesGuard } from '../auth';
import { UserRole, User } from '../auth/entities/user.entity';

@Controller('ecu-calibrations')
@UseGuards(JwtAuthGuard, RolesGuard)
export class ECUCalibrationsController {
    constructor(private readonly calibrationsService: ECUCalibrationsService) { }

    @Post('upload')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    @UseInterceptors(FileInterceptor('file'))
    async uploadFile(
        @CurrentUser() user: User,
        @UploadedFile() file: Express.Multer.File,
        @Body() dto: CreateECUCalibrationDto,
    ): Promise<ECUCalibration> {
        if (!file) {
            throw new BadRequestException('No file uploaded');
        }

        return this.calibrationsService.uploadFile({
            tenantId: user.tenantId,
            dto,
            fileBuffer: file.buffer,
            uploadedBy: user.id,
        });
    }

    @Get()
    async findAll(
        @CurrentUser() user: User,
        @Query('vehicleId') vehicleId?: string,
        @Query('customerId') customerId?: string,
        @Query('calibrationType') calibrationType?: CalibrationType,
        @Query('page') page = 1,
        @Query('limit') limit = 20,
    ): Promise<PaginatedCalibrations> {
        return this.calibrationsService.findAll({
            tenantId: user.tenantId,
            vehicleId,
            customerId,
            calibrationType,
            page: Number(page),
            limit: Number(limit),
        });
    }

    @Get(':id')
    async findOne(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<ECUCalibration> {
        return this.calibrationsService.findOne(user.tenantId, id);
    }

    @Get(':id/download-url')
    async getDownloadUrl(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<{ url: string; expiresIn: number }> {
        const url = await this.calibrationsService.getDownloadUrl(user.tenantId, id);
        return { url, expiresIn: 3600 };
    }

    @Get('vehicle/:vehicleId/history')
    async getVersionHistory(
        @CurrentUser() user: User,
        @Param('vehicleId', ParseUUIDPipe) vehicleId: string,
    ): Promise<ECUCalibration[]> {
        return this.calibrationsService.getVersionHistory(user.tenantId, vehicleId);
    }

    @Patch(':id')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER, UserRole.TECHNICIAN)
    async update(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
        @Body() dto: UpdateECUCalibrationDto,
    ): Promise<ECUCalibration> {
        return this.calibrationsService.update(user.tenantId, id, dto);
    }

    @Delete(':id')
    @Roles(UserRole.ADMIN, UserRole.SHOP_OWNER)
    async remove(
        @CurrentUser() user: User,
        @Param('id', ParseUUIDPipe) id: string,
    ): Promise<{ message: string }> {
        await this.calibrationsService.remove(user.tenantId, id);
        return { message: 'ECU Calibration deleted successfully' };
    }
}
