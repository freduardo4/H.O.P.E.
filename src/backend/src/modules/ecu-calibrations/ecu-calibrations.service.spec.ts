import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { NotFoundException, BadRequestException } from '@nestjs/common';
import { ECUCalibrationsService } from './ecu-calibrations.service';
import { ECUCalibration, CalibrationType, FileFormat } from './entities/ecu-calibration.entity';
import { CreateECUCalibrationDto, UpdateECUCalibrationDto } from './dto';

// Mock AWS SDK
const mockS3Send = jest.fn();
jest.mock('@aws-sdk/client-s3', () => ({
    S3Client: jest.fn().mockImplementation(() => ({
        send: mockS3Send,
    })),
    PutObjectCommand: jest.fn(),
    GetObjectCommand: jest.fn(),
    DeleteObjectCommand: jest.fn(),
}));

jest.mock('@aws-sdk/s3-request-presigner', () => ({
    getSignedUrl: jest.fn().mockResolvedValue('https://s3.amazonaws.com/signed-url'),
}));

describe('ECUCalibrationsService', () => {
    let service: ECUCalibrationsService;
    let repository: jest.Mocked<Repository<ECUCalibration>>;

    const mockCalibration: Partial<ECUCalibration> = {
        id: 'calibration-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        customerId: 'customer-uuid',
        fileName: 'stock_tune.bin',
        s3Key: 'tenant-uuid/ecu-calibrations/vehicle-uuid/12345-stock_tune.bin',
        s3Bucket: 'hope-ecu-calibrations',
        fileSize: 2048,
        fileFormat: FileFormat.BIN,
        calibrationType: CalibrationType.STOCK,
        checksum: 'abc123def456',
        version: 1,
        ecuType: 'Bosch ME7',
        isActive: true,
        uploadedBy: 'user-uuid',
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    const mockStage1Calibration: Partial<ECUCalibration> = {
        ...mockCalibration,
        id: 'stage1-calibration-uuid',
        fileName: 'stage1_tune.bin',
        calibrationType: CalibrationType.STAGE_1,
        version: 2,
        previousVersionId: 'calibration-uuid',
        metadata: {
            enginePowerStock: 300,
            enginePowerTuned: 350,
            torqueStock: 400,
            torqueTuned: 480,
        },
    };

    const mockQueryBuilder = {
        where: jest.fn().mockReturnThis(),
        andWhere: jest.fn().mockReturnThis(),
        orderBy: jest.fn().mockReturnThis(),
        skip: jest.fn().mockReturnThis(),
        take: jest.fn().mockReturnThis(),
        getManyAndCount: jest.fn(),
    };

    beforeEach(async () => {
        const mockRepository = {
            findOne: jest.fn(),
            find: jest.fn(),
            create: jest.fn(),
            save: jest.fn(),
            createQueryBuilder: jest.fn(() => mockQueryBuilder),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                ECUCalibrationsService,
                {
                    provide: getRepositoryToken(ECUCalibration),
                    useValue: mockRepository,
                },
            ],
        }).compile();

        service = module.get<ECUCalibrationsService>(ECUCalibrationsService);
        repository = module.get(getRepositoryToken(ECUCalibration));

        jest.clearAllMocks();
        mockS3Send.mockResolvedValue({});
    });

    describe('uploadFile', () => {
        it('should upload a file and create calibration record', async () => {
            const dto: CreateECUCalibrationDto = {
                vehicleId: 'vehicle-uuid',
                fileName: 'stock_tune.bin',
                fileSize: 2048,
                calibrationType: CalibrationType.STOCK,
                checksum: 'abc123def456',
            };

            repository.create.mockReturnValue(mockCalibration as ECUCalibration);
            repository.save.mockResolvedValue(mockCalibration as ECUCalibration);

            const result = await service.uploadFile({
                tenantId: 'tenant-uuid',
                dto,
                fileBuffer: Buffer.from('test data'),
                uploadedBy: 'user-uuid',
            });

            expect(mockS3Send).toHaveBeenCalled();
            expect(repository.create).toHaveBeenCalledWith(
                expect.objectContaining({
                    ...dto,
                    tenantId: 'tenant-uuid',
                    s3Bucket: 'hope-ecu-calibrations',
                    version: 1,
                    uploadedBy: 'user-uuid',
                }),
            );
            expect(result.fileName).toBe('stock_tune.bin');
        });

        it('should increment version when previousVersionId is provided', async () => {
            const dto: CreateECUCalibrationDto = {
                vehicleId: 'vehicle-uuid',
                fileName: 'stage1_tune.bin',
                fileSize: 2048,
                calibrationType: CalibrationType.STAGE_1,
                checksum: 'def456ghi789',
                previousVersionId: 'calibration-uuid',
            };

            repository.findOne.mockResolvedValue(mockCalibration as ECUCalibration);
            repository.create.mockReturnValue(mockStage1Calibration as ECUCalibration);
            repository.save.mockResolvedValue(mockStage1Calibration as ECUCalibration);

            const result = await service.uploadFile({
                tenantId: 'tenant-uuid',
                dto,
                fileBuffer: Buffer.from('test data'),
                uploadedBy: 'user-uuid',
            });

            expect(repository.findOne).toHaveBeenCalledWith({
                where: { id: 'calibration-uuid', tenantId: 'tenant-uuid' },
            });
            expect(result.version).toBe(2);
        });

        it('should throw BadRequestException on S3 upload failure', async () => {
            const dto: CreateECUCalibrationDto = {
                vehicleId: 'vehicle-uuid',
                fileName: 'stock_tune.bin',
                fileSize: 2048,
                calibrationType: CalibrationType.STOCK,
                checksum: 'abc123def456',
            };

            mockS3Send.mockRejectedValue(new Error('S3 upload failed'));

            await expect(
                service.uploadFile({
                    tenantId: 'tenant-uuid',
                    dto,
                    fileBuffer: Buffer.from('test data'),
                    uploadedBy: 'user-uuid',
                }),
            ).rejects.toThrow(BadRequestException);
        });
    });

    describe('findAll', () => {
        it('should return paginated calibrations', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([
                [mockCalibration as ECUCalibration],
                1,
            ]);

            const result = await service.findAll({
                tenantId: 'tenant-uuid',
                page: 1,
                limit: 20,
            });

            expect(result.data).toHaveLength(1);
            expect(result.total).toBe(1);
            expect(result.page).toBe(1);
            expect(result.limit).toBe(20);
            expect(result.totalPages).toBe(1);
        });

        it('should filter active calibrations only', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([
                [mockCalibration as ECUCalibration],
                1,
            ]);

            await service.findAll({
                tenantId: 'tenant-uuid',
            });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                'calibration.isActive = :isActive',
                { isActive: true },
            );
        });

        it('should filter by vehicleId', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([
                [mockCalibration as ECUCalibration],
                1,
            ]);

            await service.findAll({
                tenantId: 'tenant-uuid',
                vehicleId: 'vehicle-uuid',
            });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                'calibration.vehicleId = :vehicleId',
                { vehicleId: 'vehicle-uuid' },
            );
        });

        it('should filter by calibrationType', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([
                [mockStage1Calibration as ECUCalibration],
                1,
            ]);

            await service.findAll({
                tenantId: 'tenant-uuid',
                calibrationType: 'stage1',
            });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                'calibration.calibrationType = :calibrationType',
                { calibrationType: 'stage1' },
            );
        });

        it('should filter by customerId', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([
                [mockCalibration as ECUCalibration],
                1,
            ]);

            await service.findAll({
                tenantId: 'tenant-uuid',
                customerId: 'customer-uuid',
            });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                'calibration.customerId = :customerId',
                { customerId: 'customer-uuid' },
            );
        });

        it('should handle pagination correctly', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([
                [mockCalibration as ECUCalibration],
                50,
            ]);

            const result = await service.findAll({
                tenantId: 'tenant-uuid',
                page: 3,
                limit: 10,
            });

            expect(mockQueryBuilder.skip).toHaveBeenCalledWith(20);
            expect(mockQueryBuilder.take).toHaveBeenCalledWith(10);
            expect(result.totalPages).toBe(5);
        });
    });

    describe('findOne', () => {
        it('should return a calibration by id', async () => {
            repository.findOne.mockResolvedValue(mockCalibration as ECUCalibration);

            const result = await service.findOne('tenant-uuid', 'calibration-uuid');

            expect(repository.findOne).toHaveBeenCalledWith({
                where: { id: 'calibration-uuid', tenantId: 'tenant-uuid' },
            });
            expect(result.id).toBe('calibration-uuid');
        });

        it('should throw NotFoundException if calibration not found', async () => {
            repository.findOne.mockResolvedValue(null);

            await expect(
                service.findOne('tenant-uuid', 'non-existent-uuid'),
            ).rejects.toThrow(NotFoundException);
            await expect(
                service.findOne('tenant-uuid', 'non-existent-uuid'),
            ).rejects.toThrow('ECU Calibration with ID non-existent-uuid not found');
        });
    });

    describe('getDownloadUrl', () => {
        it('should return signed URL for calibration file', async () => {
            repository.findOne.mockResolvedValue(mockCalibration as ECUCalibration);

            const result = await service.getDownloadUrl('tenant-uuid', 'calibration-uuid');

            expect(result).toBe('https://s3.amazonaws.com/signed-url');
        });

        it('should throw NotFoundException if calibration not found', async () => {
            repository.findOne.mockResolvedValue(null);

            await expect(
                service.getDownloadUrl('tenant-uuid', 'non-existent-uuid'),
            ).rejects.toThrow(NotFoundException);
        });

        it('should accept custom expiration time', async () => {
            repository.findOne.mockResolvedValue(mockCalibration as ECUCalibration);

            const result = await service.getDownloadUrl('tenant-uuid', 'calibration-uuid', 7200);

            expect(result).toBe('https://s3.amazonaws.com/signed-url');
        });
    });

    describe('update', () => {
        it('should update calibration metadata', async () => {
            const updateDto: UpdateECUCalibrationDto = {
                notes: 'Updated notes',
                ecuType: 'Bosch ME7.5',
            };

            const updatedCalibration = {
                ...mockCalibration,
                ...updateDto,
            };

            repository.findOne.mockResolvedValue(mockCalibration as ECUCalibration);
            repository.save.mockResolvedValue(updatedCalibration as ECUCalibration);

            const result = await service.update('tenant-uuid', 'calibration-uuid', updateDto);

            expect(result.notes).toBe('Updated notes');
            expect(result.ecuType).toBe('Bosch ME7.5');
        });

        it('should throw NotFoundException if calibration not found', async () => {
            repository.findOne.mockResolvedValue(null);

            await expect(
                service.update('tenant-uuid', 'non-existent-uuid', { notes: 'test' }),
            ).rejects.toThrow(NotFoundException);
        });
    });

    describe('remove', () => {
        it('should soft delete calibration by marking as inactive', async () => {
            repository.findOne.mockResolvedValue(mockCalibration as ECUCalibration);
            repository.save.mockResolvedValue({
                ...mockCalibration,
                isActive: false,
            } as ECUCalibration);

            await service.remove('tenant-uuid', 'calibration-uuid');

            expect(repository.save).toHaveBeenCalledWith(
                expect.objectContaining({ isActive: false }),
            );
        });

        it('should throw NotFoundException if calibration not found', async () => {
            repository.findOne.mockResolvedValue(null);

            await expect(
                service.remove('tenant-uuid', 'non-existent-uuid'),
            ).rejects.toThrow(NotFoundException);
        });
    });

    describe('getVersionHistory', () => {
        it('should return version history for a vehicle', async () => {
            const calibrations = [
                mockStage1Calibration as ECUCalibration,
                mockCalibration as ECUCalibration,
            ];
            repository.find.mockResolvedValue(calibrations);

            const result = await service.getVersionHistory('tenant-uuid', 'vehicle-uuid');

            expect(repository.find).toHaveBeenCalledWith({
                where: { tenantId: 'tenant-uuid', vehicleId: 'vehicle-uuid', isActive: true },
                order: { version: 'DESC', createdAt: 'DESC' },
            });
            expect(result).toHaveLength(2);
            expect(result[0].version).toBe(2);
        });

        it('should return empty array if no calibrations found', async () => {
            repository.find.mockResolvedValue([]);

            const result = await service.getVersionHistory('tenant-uuid', 'new-vehicle-uuid');

            expect(result).toHaveLength(0);
        });
    });
});
