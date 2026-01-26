"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const ecu_calibrations_controller_1 = require("./ecu-calibrations.controller");
const ecu_calibrations_service_1 = require("./ecu-calibrations.service");
const ecu_calibration_entity_1 = require("./entities/ecu-calibration.entity");
const user_entity_1 = require("../auth/entities/user.entity");
describe('ECUCalibrationsController', () => {
    let controller;
    let service;
    const mockUser = {
        id: 'user-uuid',
        email: 'test@example.com',
        firstName: 'Test',
        lastName: 'User',
        role: user_entity_1.UserRole.TECHNICIAN,
        tenantId: 'tenant-uuid',
        isActive: true,
    };
    const mockCalibration = {
        id: 'calibration-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        name: 'Stage 1 Tune',
        description: 'Performance tune with +40hp',
        version: '1.0.0',
        parentVersionId: null,
        status: ecu_calibration_entity_1.CalibrationStatus.DRAFT,
        protocol: ecu_calibration_entity_1.CalibrationProtocol.UDS,
        ecuPartNumber: '06K906016AC',
        ecuSoftwareNumber: 'SW123456',
        ecuHardwareNumber: 'HW789012',
        originalChecksum: 'ABC123',
        modifiedChecksum: 'DEF456',
        fileSize: 4096,
        s3Key: 'calibrations/calibration-uuid/modified.bin',
        originalS3Key: 'calibrations/calibration-uuid/original.bin',
        mapData: [
            {
                name: 'Boost Pressure',
                address: '0x4000',
                originalValues: [1.2, 1.3, 1.4],
                modifiedValues: [1.4, 1.5, 1.6],
                unit: 'bar',
            },
        ],
        modifications: [
            {
                type: 'map_modification',
                parameter: 'Boost Pressure',
                originalValue: '1.2 bar',
                newValue: '1.4 bar',
                timestamp: '2024-01-15T10:00:00Z',
                technicianId: 'user-uuid',
            },
        ],
        notes: 'Verified on dyno',
        createdById: 'user-uuid',
        approvedById: null,
        approvedAt: null,
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    beforeEach(async () => {
        const mockService = {
            create: jest.fn(),
            findAll: jest.fn(),
            findOne: jest.fn(),
            findVersionHistory: jest.fn(),
            update: jest.fn(),
            updateStatus: jest.fn(),
            createVersion: jest.fn(),
            remove: jest.fn(),
            getStats: jest.fn(),
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
    });
    describe('create', () => {
        it('should create a new calibration', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                name: 'Stage 1 Tune',
                version: '1.0.0',
                protocol: ecu_calibration_entity_1.CalibrationProtocol.UDS,
            };
            service.create.mockResolvedValue(mockCalibration);
            const result = await controller.create(mockUser, dto);
            expect(service.create).toHaveBeenCalledWith('tenant-uuid', 'user-uuid', dto);
            expect(result.name).toBe('Stage 1 Tune');
            expect(result.status).toBe(ecu_calibration_entity_1.CalibrationStatus.DRAFT);
        });
    });
    describe('findAll', () => {
        it('should return paginated calibrations', async () => {
            const paginatedResult = {
                items: [mockCalibration],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAll.mockResolvedValue(paginatedResult);
            const result = await controller.findAll(mockUser);
            expect(service.findAll).toHaveBeenCalledWith({
                tenantId: 'tenant-uuid',
                vehicleId: undefined,
                status: undefined,
                protocol: undefined,
                page: 1,
                limit: 20,
            });
            expect(result.items).toHaveLength(1);
        });
        it('should filter by vehicle ID', async () => {
            const paginatedResult = {
                items: [mockCalibration],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAll.mockResolvedValue(paginatedResult);
            await controller.findAll(mockUser, 'vehicle-uuid');
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ vehicleId: 'vehicle-uuid' }));
        });
        it('should filter by status', async () => {
            const paginatedResult = {
                items: [mockCalibration],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAll.mockResolvedValue(paginatedResult);
            await controller.findAll(mockUser, undefined, ecu_calibration_entity_1.CalibrationStatus.PRODUCTION);
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ status: ecu_calibration_entity_1.CalibrationStatus.PRODUCTION }));
        });
        it('should filter by protocol', async () => {
            const paginatedResult = {
                items: [mockCalibration],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAll.mockResolvedValue(paginatedResult);
            await controller.findAll(mockUser, undefined, undefined, 'uds');
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ protocol: 'uds' }));
        });
    });
    describe('findOne', () => {
        it('should return a single calibration by id', async () => {
            service.findOne.mockResolvedValue(mockCalibration);
            const result = await controller.findOne(mockUser, 'calibration-uuid');
            expect(service.findOne).toHaveBeenCalledWith('tenant-uuid', 'calibration-uuid');
            expect(result.id).toBe('calibration-uuid');
        });
    });
    describe('getVersionHistory', () => {
        it('should return version history for a vehicle', async () => {
            const versions = [
                mockCalibration,
                { ...mockCalibration, id: 'v2-uuid', version: '2.0.0', parentVersionId: 'calibration-uuid' },
            ];
            service.findVersionHistory.mockResolvedValue(versions);
            const result = await controller.getVersionHistory(mockUser, 'vehicle-uuid');
            expect(service.findVersionHistory).toHaveBeenCalledWith('tenant-uuid', 'vehicle-uuid');
            expect(result).toHaveLength(2);
        });
    });
    describe('update', () => {
        it('should update a calibration', async () => {
            const updatedCalibration = {
                ...mockCalibration,
                description: 'Updated description',
            };
            service.update.mockResolvedValue(updatedCalibration);
            const result = await controller.update(mockUser, 'calibration-uuid', {
                description: 'Updated description',
            });
            expect(service.update).toHaveBeenCalledWith('tenant-uuid', 'calibration-uuid', {
                description: 'Updated description',
            });
            expect(result.description).toBe('Updated description');
        });
    });
    describe('updateStatus', () => {
        it('should update calibration status', async () => {
            const approvedCalibration = {
                ...mockCalibration,
                status: ecu_calibration_entity_1.CalibrationStatus.APPROVED,
                approvedById: 'user-uuid',
                approvedAt: new Date(),
            };
            service.updateStatus.mockResolvedValue(approvedCalibration);
            const result = await controller.updateStatus(mockUser, 'calibration-uuid', ecu_calibration_entity_1.CalibrationStatus.APPROVED);
            expect(service.updateStatus).toHaveBeenCalledWith('tenant-uuid', 'calibration-uuid', ecu_calibration_entity_1.CalibrationStatus.APPROVED, 'user-uuid');
            expect(result.status).toBe(ecu_calibration_entity_1.CalibrationStatus.APPROVED);
        });
        it('should update to production status', async () => {
            const productionCalibration = {
                ...mockCalibration,
                status: ecu_calibration_entity_1.CalibrationStatus.PRODUCTION,
            };
            service.updateStatus.mockResolvedValue(productionCalibration);
            const result = await controller.updateStatus(mockUser, 'calibration-uuid', ecu_calibration_entity_1.CalibrationStatus.PRODUCTION);
            expect(result.status).toBe(ecu_calibration_entity_1.CalibrationStatus.PRODUCTION);
        });
    });
    describe('createVersion', () => {
        it('should create a new version from existing calibration', async () => {
            const newVersion = {
                ...mockCalibration,
                id: 'new-version-uuid',
                version: '2.0.0',
                parentVersionId: 'calibration-uuid',
                status: ecu_calibration_entity_1.CalibrationStatus.DRAFT,
            };
            service.createVersion.mockResolvedValue(newVersion);
            const result = await controller.createVersion(mockUser, 'calibration-uuid', { version: '2.0.0', description: 'Stage 2 improvements' });
            expect(service.createVersion).toHaveBeenCalledWith('tenant-uuid', 'calibration-uuid', 'user-uuid', { version: '2.0.0', description: 'Stage 2 improvements' });
            expect(result.parentVersionId).toBe('calibration-uuid');
            expect(result.status).toBe(ecu_calibration_entity_1.CalibrationStatus.DRAFT);
        });
    });
    describe('remove', () => {
        it('should soft delete a calibration', async () => {
            service.remove.mockResolvedValue(undefined);
            const result = await controller.remove(mockUser, 'calibration-uuid');
            expect(service.remove).toHaveBeenCalledWith('tenant-uuid', 'calibration-uuid');
            expect(result.message).toBe('Calibration deleted successfully');
        });
    });
    describe('getStats', () => {
        it('should return calibration statistics', async () => {
            const stats = {
                total: 25,
                byStatus: [
                    { status: ecu_calibration_entity_1.CalibrationStatus.DRAFT, count: 5 },
                    { status: ecu_calibration_entity_1.CalibrationStatus.APPROVED, count: 10 },
                    { status: ecu_calibration_entity_1.CalibrationStatus.PRODUCTION, count: 10 },
                ],
                byProtocol: [
                    { protocol: ecu_calibration_entity_1.CalibrationProtocol.UDS, count: 15 },
                    { protocol: ecu_calibration_entity_1.CalibrationProtocol.KWP2000, count: 10 },
                ],
            };
            service.getStats.mockResolvedValue(stats);
            const result = await controller.getStats(mockUser);
            expect(service.getStats).toHaveBeenCalledWith('tenant-uuid');
            expect(result.total).toBe(25);
            expect(result.byStatus).toHaveLength(3);
            expect(result.byProtocol).toHaveLength(2);
        });
    });
});
//# sourceMappingURL=ecu-calibrations.controller.spec.js.map