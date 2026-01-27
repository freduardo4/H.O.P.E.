"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const typeorm_1 = require("@nestjs/typeorm");
const common_1 = require("@nestjs/common");
const ecu_calibrations_service_1 = require("./ecu-calibrations.service");
const ecu_calibration_entity_1 = require("./entities/ecu-calibration.entity");
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
    let service;
    let repository;
    const mockCalibration = {
        id: 'calibration-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        customerId: 'customer-uuid',
        fileName: 'stock_tune.bin',
        s3Key: 'tenant-uuid/ecu-calibrations/vehicle-uuid/12345-stock_tune.bin',
        s3Bucket: 'hope-ecu-calibrations',
        fileSize: 2048,
        fileFormat: ecu_calibration_entity_1.FileFormat.BIN,
        calibrationType: ecu_calibration_entity_1.CalibrationType.STOCK,
        checksum: 'abc123def456',
        version: 1,
        ecuType: 'Bosch ME7',
        isActive: true,
        uploadedBy: 'user-uuid',
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    const mockStage1Calibration = {
        ...mockCalibration,
        id: 'stage1-calibration-uuid',
        fileName: 'stage1_tune.bin',
        calibrationType: ecu_calibration_entity_1.CalibrationType.STAGE_1,
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
        const module = await testing_1.Test.createTestingModule({
            providers: [
                ecu_calibrations_service_1.ECUCalibrationsService,
                {
                    provide: (0, typeorm_1.getRepositoryToken)(ecu_calibration_entity_1.ECUCalibration),
                    useValue: mockRepository,
                },
            ],
        }).compile();
        service = module.get(ecu_calibrations_service_1.ECUCalibrationsService);
        repository = module.get((0, typeorm_1.getRepositoryToken)(ecu_calibration_entity_1.ECUCalibration));
        jest.clearAllMocks();
        mockS3Send.mockResolvedValue({});
    });
    describe('uploadFile', () => {
        it('should upload a file and create calibration record', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                fileName: 'stock_tune.bin',
                fileSize: 2048,
                calibrationType: ecu_calibration_entity_1.CalibrationType.STOCK,
                checksum: 'abc123def456',
            };
            repository.create.mockReturnValue(mockCalibration);
            repository.save.mockResolvedValue(mockCalibration);
            const result = await service.uploadFile({
                tenantId: 'tenant-uuid',
                dto,
                fileBuffer: Buffer.from('test data'),
                uploadedBy: 'user-uuid',
            });
            expect(mockS3Send).toHaveBeenCalled();
            expect(repository.create).toHaveBeenCalledWith(expect.objectContaining({
                ...dto,
                tenantId: 'tenant-uuid',
                s3Bucket: 'hope-ecu-calibrations',
                version: 1,
                uploadedBy: 'user-uuid',
            }));
            expect(result.fileName).toBe('stock_tune.bin');
        });
        it('should increment version when previousVersionId is provided', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                fileName: 'stage1_tune.bin',
                fileSize: 2048,
                calibrationType: ecu_calibration_entity_1.CalibrationType.STAGE_1,
                checksum: 'def456ghi789',
                previousVersionId: 'calibration-uuid',
            };
            repository.findOne.mockResolvedValue(mockCalibration);
            repository.create.mockReturnValue(mockStage1Calibration);
            repository.save.mockResolvedValue(mockStage1Calibration);
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
            const dto = {
                vehicleId: 'vehicle-uuid',
                fileName: 'stock_tune.bin',
                fileSize: 2048,
                calibrationType: ecu_calibration_entity_1.CalibrationType.STOCK,
                checksum: 'abc123def456',
            };
            mockS3Send.mockRejectedValue(new Error('S3 upload failed'));
            await expect(service.uploadFile({
                tenantId: 'tenant-uuid',
                dto,
                fileBuffer: Buffer.from('test data'),
                uploadedBy: 'user-uuid',
            })).rejects.toThrow(common_1.BadRequestException);
        });
    });
    describe('findAll', () => {
        it('should return paginated calibrations', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([
                [mockCalibration],
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
                [mockCalibration],
                1,
            ]);
            await service.findAll({
                tenantId: 'tenant-uuid',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('calibration.isActive = :isActive', { isActive: true });
        });
        it('should filter by vehicleId', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([
                [mockCalibration],
                1,
            ]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                vehicleId: 'vehicle-uuid',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('calibration.vehicleId = :vehicleId', { vehicleId: 'vehicle-uuid' });
        });
        it('should filter by calibrationType', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([
                [mockStage1Calibration],
                1,
            ]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                calibrationType: 'stage1',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('calibration.calibrationType = :calibrationType', { calibrationType: 'stage1' });
        });
        it('should filter by customerId', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([
                [mockCalibration],
                1,
            ]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                customerId: 'customer-uuid',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('calibration.customerId = :customerId', { customerId: 'customer-uuid' });
        });
        it('should handle pagination correctly', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([
                [mockCalibration],
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
            repository.findOne.mockResolvedValue(mockCalibration);
            const result = await service.findOne('tenant-uuid', 'calibration-uuid');
            expect(repository.findOne).toHaveBeenCalledWith({
                where: { id: 'calibration-uuid', tenantId: 'tenant-uuid' },
            });
            expect(result.id).toBe('calibration-uuid');
        });
        it('should throw NotFoundException if calibration not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.findOne('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
            await expect(service.findOne('tenant-uuid', 'non-existent-uuid')).rejects.toThrow('ECU Calibration with ID non-existent-uuid not found');
        });
    });
    describe('getDownloadUrl', () => {
        it('should return signed URL for calibration file', async () => {
            repository.findOne.mockResolvedValue(mockCalibration);
            const result = await service.getDownloadUrl('tenant-uuid', 'calibration-uuid');
            expect(result).toBe('https://s3.amazonaws.com/signed-url');
        });
        it('should throw NotFoundException if calibration not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.getDownloadUrl('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
        it('should accept custom expiration time', async () => {
            repository.findOne.mockResolvedValue(mockCalibration);
            const result = await service.getDownloadUrl('tenant-uuid', 'calibration-uuid', 7200);
            expect(result).toBe('https://s3.amazonaws.com/signed-url');
        });
    });
    describe('update', () => {
        it('should update calibration metadata', async () => {
            const updateDto = {
                notes: 'Updated notes',
                ecuType: 'Bosch ME7.5',
            };
            const updatedCalibration = {
                ...mockCalibration,
                ...updateDto,
            };
            repository.findOne.mockResolvedValue(mockCalibration);
            repository.save.mockResolvedValue(updatedCalibration);
            const result = await service.update('tenant-uuid', 'calibration-uuid', updateDto);
            expect(result.notes).toBe('Updated notes');
            expect(result.ecuType).toBe('Bosch ME7.5');
        });
        it('should throw NotFoundException if calibration not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.update('tenant-uuid', 'non-existent-uuid', { notes: 'test' })).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('remove', () => {
        it('should soft delete calibration by marking as inactive', async () => {
            repository.findOne.mockResolvedValue(mockCalibration);
            repository.save.mockResolvedValue({
                ...mockCalibration,
                isActive: false,
            });
            await service.remove('tenant-uuid', 'calibration-uuid');
            expect(repository.save).toHaveBeenCalledWith(expect.objectContaining({ isActive: false }));
        });
        it('should throw NotFoundException if calibration not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.remove('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('getVersionHistory', () => {
        it('should return version history for a vehicle', async () => {
            const calibrations = [
                mockStage1Calibration,
                mockCalibration,
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
//# sourceMappingURL=ecu-calibrations.service.spec.js.map