"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const common_1 = require("@nestjs/common");
const ecu_calibrations_controller_1 = require("./ecu-calibrations.controller");
const ecu_calibrations_service_1 = require("./ecu-calibrations.service");
const ecu_calibration_entity_1 = require("./entities/ecu-calibration.entity");
const user_entity_1 = require("../auth/entities/user.entity");
describe('ECUCalibrationsController', () => {
    let controller;
    let service;
    const mockUser = {
        id: 'user-uuid',
        tenantId: 'tenant-uuid',
        email: 'test@example.com',
        role: user_entity_1.UserRole.TECHNICIAN,
    };
    const mockCalibration = {
        id: 'calibration-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        customerId: 'customer-uuid',
        calibrationType: ecu_calibration_entity_1.CalibrationType.STOCK,
        version: 1,
        fileName: 'calibration.bin',
        fileFormat: ecu_calibration_entity_1.FileFormat.BIN,
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
    const mockPaginatedResult = {
        data: [mockCalibration],
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
        const module = await testing_1.Test.createTestingModule({
            controllers: [ecu_calibrations_controller_1.ECUCalibrationsController],
            providers: [
                {
                    provide: ecu_calibrations_service_1.ECUCalibrationsService,
                    useValue: mockService,
                },
            ],
        }).compile();
        controller = module.get(ecu_calibrations_controller_1.ECUCalibrationsController);
        service = module.get(ecu_calibrations_service_1.ECUCalibrationsService);
        jest.clearAllMocks();
    });
    describe('uploadFile', () => {
        it('should upload a calibration file', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                calibrationType: ecu_calibration_entity_1.CalibrationType.STOCK,
                fileName: 'calibration.bin',
                fileSize: 1024000,
                checksum: 'abc123def456',
            };
            const mockFile = {
                buffer: Buffer.from('calibration data'),
                originalname: 'calibration.bin',
                mimetype: 'application/octet-stream',
                size: 1024000,
            };
            service.uploadFile.mockResolvedValue(mockCalibration);
            const result = await controller.uploadFile(mockUser, mockFile, dto);
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
                calibrationType: ecu_calibration_entity_1.CalibrationType.STOCK,
                fileName: 'calibration.bin',
                fileSize: 1024000,
                checksum: 'abc123def456',
            };
            await expect(controller.uploadFile(mockUser, undefined, dto)).rejects.toThrow(common_1.BadRequestException);
            await expect(controller.uploadFile(mockUser, undefined, dto)).rejects.toThrow('No file uploaded');
        });
    });
    describe('findAll', () => {
        it('should return paginated calibrations', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);
            const result = await controller.findAll(mockUser);
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
            await controller.findAll(mockUser, 'vehicle-uuid');
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ vehicleId: 'vehicle-uuid' }));
        });
        it('should filter by calibration type', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);
            await controller.findAll(mockUser, undefined, undefined, ecu_calibration_entity_1.CalibrationType.STAGE_1);
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ calibrationType: ecu_calibration_entity_1.CalibrationType.STAGE_1 }));
        });
        it('should pass pagination parameters', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);
            await controller.findAll(mockUser, undefined, undefined, undefined, 3, 50);
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ page: 3, limit: 50 }));
        });
    });
    describe('findOne', () => {
        it('should return a single calibration', async () => {
            service.findOne.mockResolvedValue(mockCalibration);
            const result = await controller.findOne(mockUser, 'calibration-uuid');
            expect(service.findOne).toHaveBeenCalledWith('tenant-uuid', 'calibration-uuid');
            expect(result.id).toBe('calibration-uuid');
        });
    });
    describe('getDownloadUrl', () => {
        it('should return download URL with expiry', async () => {
            const downloadUrl = 'https://s3.amazonaws.com/bucket/calibration.bin?signed=true';
            service.getDownloadUrl.mockResolvedValue(downloadUrl);
            const result = await controller.getDownloadUrl(mockUser, 'calibration-uuid');
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
            service.getVersionHistory.mockResolvedValue(historyItems);
            const result = await controller.getVersionHistory(mockUser, 'vehicle-uuid');
            expect(service.getVersionHistory).toHaveBeenCalledWith('tenant-uuid', 'vehicle-uuid');
            expect(result).toHaveLength(2);
        });
    });
    describe('update', () => {
        it('should update a calibration', async () => {
            const updateDto = { notes: 'Updated notes' };
            const updatedCalibration = { ...mockCalibration, ...updateDto };
            service.update.mockResolvedValue(updatedCalibration);
            const result = await controller.update(mockUser, 'calibration-uuid', updateDto);
            expect(service.update).toHaveBeenCalledWith('tenant-uuid', 'calibration-uuid', updateDto);
            expect(result.notes).toBe('Updated notes');
        });
    });
    describe('remove', () => {
        it('should delete a calibration and return success message', async () => {
            service.remove.mockResolvedValue(undefined);
            const result = await controller.remove(mockUser, 'calibration-uuid');
            expect(service.remove).toHaveBeenCalledWith('tenant-uuid', 'calibration-uuid');
            expect(result.message).toBe('ECU Calibration deleted successfully');
        });
    });
});
//# sourceMappingURL=ecu-calibrations.controller.spec.js.map