import { Test, TestingModule } from '@nestjs/testing';
import { BadRequestException } from '@nestjs/common';
import { ECUCalibrationsController } from './ecu-calibrations.controller';
import { ECUCalibrationsService, PaginatedCalibrations } from './ecu-calibrations.service';
import { ECUCalibration, CalibrationType, FileFormat } from './entities/ecu-calibration.entity';
import { User, UserRole } from '../auth/entities/user.entity';

describe('ECUCalibrationsController', () => {
    let controller: ECUCalibrationsController;
    let service: jest.Mocked<ECUCalibrationsService>;

    const mockUser: Partial<User> = {
        id: 'user-uuid',
        tenantId: 'tenant-uuid',
        email: 'test@example.com',
        role: UserRole.TECHNICIAN,
    };

    const mockCalibration: Partial<ECUCalibration> = {
        id: 'calibration-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        customerId: 'customer-uuid',
        calibrationType: CalibrationType.STOCK,
        version: 1,
        fileName: 'calibration.bin',
        fileFormat: FileFormat.BIN,
        s3Key: 'tenant-uuid/ecu-calibrations/vehicle-uuid/calibration.bin',
        s3Bucket: 'hope-ecu-calibrations',
        fileSize: 1024000,
        checksum: 'abc123def456',
        notes: 'Stock ECU calibration',
        uploadedBy: 'user-uuid',
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    const mockPaginatedResult: PaginatedCalibrations = {
        data: [mockCalibration as ECUCalibration],
        total: 1,
        page: 1,
        limit: 20,
        totalPages: 1,
    };

    beforeEach(async () => {
        const mockService = {
            uploadFile: jest.fn(),
            findAll: jest.fn(),
            findOne: jest.fn(),
            getDownloadUrl: jest.fn(),
            getVersionHistory: jest.fn(),
            update: jest.fn(),
            remove: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            controllers: [ECUCalibrationsController],
            providers: [
                {
                    provide: ECUCalibrationsService,
                    useValue: mockService,
                },
            ],
        }).compile();

        controller = module.get<ECUCalibrationsController>(ECUCalibrationsController);
        service = module.get(ECUCalibrationsService);

        jest.clearAllMocks();
    });

    describe('uploadFile', () => {
        it('should upload a calibration file', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                calibrationType: CalibrationType.STOCK,
                fileName: 'calibration.bin',
                fileSize: 1024000,
                checksum: 'abc123def456',
            };

            const mockFile = {
                buffer: Buffer.from('calibration data'),
                originalname: 'calibration.bin',
                mimetype: 'application/octet-stream',
                size: 1024000,
            } as Express.Multer.File;

            service.uploadFile.mockResolvedValue(mockCalibration as ECUCalibration);

            const result = await controller.uploadFile(mockUser as User, mockFile, dto);

            expect(service.uploadFile).toHaveBeenCalledWith({
                tenantId: 'tenant-uuid',
                dto,
                fileBuffer: mockFile.buffer,
                uploadedBy: 'user-uuid',
            });
            expect(result.id).toBe('calibration-uuid');
        });

        it('should throw BadRequestException when no file is uploaded', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                calibrationType: CalibrationType.STOCK,
                fileName: 'calibration.bin',
                fileSize: 1024000,
                checksum: 'abc123def456',
            };

            await expect(
                controller.uploadFile(mockUser as User, undefined as any, dto),
            ).rejects.toThrow(BadRequestException);
            await expect(
                controller.uploadFile(mockUser as User, undefined as any, dto),
            ).rejects.toThrow('No file uploaded');
        });
    });

    describe('findAll', () => {
        it('should return paginated calibrations', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);

            const result = await controller.findAll(mockUser as User);

            expect(service.findAll).toHaveBeenCalledWith({
                tenantId: 'tenant-uuid',
                vehicleId: undefined,
                customerId: undefined,
                calibrationType: undefined,
                page: 1,
                limit: 20,
            });
            expect(result.data).toHaveLength(1);
            expect(result.total).toBe(1);
        });

        it('should filter by vehicleId', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);

            await controller.findAll(mockUser as User, 'vehicle-uuid');

            expect(service.findAll).toHaveBeenCalledWith(
                expect.objectContaining({ vehicleId: 'vehicle-uuid' }),
            );
        });

        it('should filter by calibration type', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);

            await controller.findAll(
                mockUser as User,
                undefined,
                undefined,
                CalibrationType.STAGE_1,
            );

            expect(service.findAll).toHaveBeenCalledWith(
                expect.objectContaining({ calibrationType: CalibrationType.STAGE_1 }),
            );
        });

        it('should pass pagination parameters', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);

            await controller.findAll(
                mockUser as User,
                undefined,
                undefined,
                undefined,
                3,
                50,
            );

            expect(service.findAll).toHaveBeenCalledWith(
                expect.objectContaining({ page: 3, limit: 50 }),
            );
        });
    });

    describe('findOne', () => {
        it('should return a single calibration', async () => {
            service.findOne.mockResolvedValue(mockCalibration as ECUCalibration);

            const result = await controller.findOne(mockUser as User, 'calibration-uuid');

            expect(service.findOne).toHaveBeenCalledWith('tenant-uuid', 'calibration-uuid');
            expect(result.id).toBe('calibration-uuid');
        });
    });

    describe('getDownloadUrl', () => {
        it('should return download URL with expiry', async () => {
            const downloadUrl = 'https://s3.amazonaws.com/bucket/calibration.bin?signed=true';
            service.getDownloadUrl.mockResolvedValue(downloadUrl);

            const result = await controller.getDownloadUrl(mockUser as User, 'calibration-uuid');

            expect(service.getDownloadUrl).toHaveBeenCalledWith('tenant-uuid', 'calibration-uuid');
            expect(result.url).toBe(downloadUrl);
            expect(result.expiresIn).toBe(3600);
        });
    });

    describe('getVersionHistory', () => {
        it('should return version history for a vehicle', async () => {
            const historyItems = [
                { ...mockCalibration, version: 1 },
                { ...mockCalibration, id: 'calibration-2', version: 2 },
            ];
            service.getVersionHistory.mockResolvedValue(historyItems as ECUCalibration[]);

            const result = await controller.getVersionHistory(mockUser as User, 'vehicle-uuid');

            expect(service.getVersionHistory).toHaveBeenCalledWith('tenant-uuid', 'vehicle-uuid');
            expect(result).toHaveLength(2);
        });
    });

    describe('update', () => {
        it('should update a calibration', async () => {
            const updateDto = { notes: 'Updated notes' };
            const updatedCalibration = { ...mockCalibration, ...updateDto };

            service.update.mockResolvedValue(updatedCalibration as ECUCalibration);

            const result = await controller.update(
                mockUser as User,
                'calibration-uuid',
                updateDto,
            );

            expect(service.update).toHaveBeenCalledWith(
                'tenant-uuid',
                'calibration-uuid',
                updateDto,
            );
            expect(result.notes).toBe('Updated notes');
        });
    });

    describe('remove', () => {
        it('should delete a calibration and return success message', async () => {
            service.remove.mockResolvedValue(undefined);

            const result = await controller.remove(mockUser as User, 'calibration-uuid');

            expect(service.remove).toHaveBeenCalledWith('tenant-uuid', 'calibration-uuid');
            expect(result.message).toBe('ECU Calibration deleted successfully');
        });
    });
});
