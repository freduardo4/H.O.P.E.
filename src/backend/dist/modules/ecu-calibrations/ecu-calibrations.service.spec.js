"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const typeorm_1 = require("@nestjs/typeorm");
const common_1 = require("@nestjs/common");
const ecu_calibrations_service_1 = require("./ecu-calibrations.service");
const ecu_calibration_entity_1 = require("./entities/ecu-calibration.entity");
describe('ECUCalibrationsService', () => {
    let service;
    let repository;
    const mockCalibration = {
        id: 'calibration-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        name: 'Stage 1 Tune',
        description: 'Performance tune',
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
        modifications: [],
        notes: 'Verified on dyno',
        createdById: 'user-uuid',
        approvedById: null,
        approvedAt: null,
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    const mockQueryBuilder = {
        where: jest.fn().mockReturnThis(),
        andWhere: jest.fn().mockReturnThis(),
        orderBy: jest.fn().mockReturnThis(),
        skip: jest.fn().mockReturnThis(),
        take: jest.fn().mockReturnThis(),
        select: jest.fn().mockReturnThis(),
        addSelect: jest.fn().mockReturnThis(),
        groupBy: jest.fn().mockReturnThis(),
        getCount: jest.fn(),
        getMany: jest.fn(),
        getRawMany: jest.fn(),
    };
    beforeEach(async () => {
        const mockRepository = {
            findOne: jest.fn(),
            find: jest.fn(),
            create: jest.fn(),
            save: jest.fn(),
            count: jest.fn(),
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
    });
    describe('create', () => {
        it('should create a new calibration', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                name: 'Stage 1 Tune',
                version: '1.0.0',
                protocol: ecu_calibration_entity_1.CalibrationProtocol.UDS,
            };
            repository.create.mockReturnValue(mockCalibration);
            repository.save.mockResolvedValue(mockCalibration);
            const result = await service.create('tenant-uuid', 'user-uuid', dto);
            expect(repository.create).toHaveBeenCalledWith({
                ...dto,
                tenantId: 'tenant-uuid',
                createdById: 'user-uuid',
            });
            expect(result.name).toBe('Stage 1 Tune');
            expect(result.status).toBe(ecu_calibration_entity_1.CalibrationStatus.DRAFT);
        });
        it('should create calibration with map data', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                name: 'Stage 2 Tune',
                version: '1.0.0',
                mapData: [
                    {
                        name: 'Boost',
                        address: '0x4000',
                        originalValues: [1.2],
                        modifiedValues: [1.5],
                        unit: 'bar',
                    },
                ],
            };
            repository.create.mockReturnValue({ ...mockCalibration, ...dto });
            repository.save.mockResolvedValue({ ...mockCalibration, ...dto });
            const result = await service.create('tenant-uuid', 'user-uuid', dto);
            expect(result.mapData).toHaveLength(1);
        });
    });
    describe('findAll', () => {
        it('should return paginated calibrations', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockCalibration]);
            const result = await service.findAll({
                tenantId: 'tenant-uuid',
                page: 1,
                limit: 20,
            });
            expect(result.items).toHaveLength(1);
            expect(result.total).toBe(1);
            expect(result.page).toBe(1);
            expect(result.limit).toBe(20);
            expect(result.totalPages).toBe(1);
        });
        it('should filter by vehicle ID', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockCalibration]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                vehicleId: 'vehicle-uuid',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('calibration.vehicleId = :vehicleId', { vehicleId: 'vehicle-uuid' });
        });
        it('should filter by status', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockCalibration]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                status: ecu_calibration_entity_1.CalibrationStatus.PRODUCTION,
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('calibration.status = :status', { status: ecu_calibration_entity_1.CalibrationStatus.PRODUCTION });
        });
        it('should filter by protocol', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockCalibration]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                protocol: 'uds',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('calibration.protocol = :protocol', { protocol: 'uds' });
        });
        it('should handle pagination correctly', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(100);
            mockQueryBuilder.getMany.mockResolvedValue([mockCalibration]);
            const result = await service.findAll({
                tenantId: 'tenant-uuid',
                page: 5,
                limit: 10,
            });
            expect(mockQueryBuilder.skip).toHaveBeenCalledWith(40);
            expect(mockQueryBuilder.take).toHaveBeenCalledWith(10);
            expect(result.totalPages).toBe(10);
        });
    });
    describe('findOne', () => {
        it('should return a calibration by id', async () => {
            repository.findOne.mockResolvedValue(mockCalibration);
            const result = await service.findOne('tenant-uuid', 'calibration-uuid');
            expect(repository.findOne).toHaveBeenCalledWith({
                where: { id: 'calibration-uuid', tenantId: 'tenant-uuid', isActive: true },
            });
            expect(result.id).toBe('calibration-uuid');
        });
        it('should throw NotFoundException if calibration not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.findOne('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
        it('should not return calibrations from other tenants', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.findOne('other-tenant-uuid', 'calibration-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('findVersionHistory', () => {
        it('should return version history for a vehicle', async () => {
            const versions = [
                mockCalibration,
                { ...mockCalibration, id: 'v2-uuid', version: '2.0.0', parentVersionId: 'calibration-uuid' },
            ];
            repository.find.mockResolvedValue(versions);
            const result = await service.findVersionHistory('tenant-uuid', 'vehicle-uuid');
            expect(repository.find).toHaveBeenCalledWith({
                where: { tenantId: 'tenant-uuid', vehicleId: 'vehicle-uuid', isActive: true },
                order: { createdAt: 'DESC' },
            });
            expect(result).toHaveLength(2);
        });
        it('should return empty array if no versions found', async () => {
            repository.find.mockResolvedValue([]);
            const result = await service.findVersionHistory('tenant-uuid', 'vehicle-uuid');
            expect(result).toHaveLength(0);
        });
    });
    describe('update', () => {
        it('should update a calibration', async () => {
            const updatedCalibration = {
                ...mockCalibration,
                description: 'Updated description',
            };
            repository.findOne.mockResolvedValue(mockCalibration);
            repository.save.mockResolvedValue(updatedCalibration);
            const result = await service.update('tenant-uuid', 'calibration-uuid', {
                description: 'Updated description',
            });
            expect(result.description).toBe('Updated description');
        });
        it('should throw NotFoundException if calibration not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.update('tenant-uuid', 'non-existent-uuid', { description: 'New' })).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('updateStatus', () => {
        it('should update calibration status to approved', async () => {
            const approvedCalibration = {
                ...mockCalibration,
                status: ecu_calibration_entity_1.CalibrationStatus.APPROVED,
                approvedById: 'approver-uuid',
                approvedAt: expect.any(Date),
            };
            repository.findOne.mockResolvedValue({ ...mockCalibration });
            repository.save.mockResolvedValue(approvedCalibration);
            const result = await service.updateStatus('tenant-uuid', 'calibration-uuid', ecu_calibration_entity_1.CalibrationStatus.APPROVED, 'approver-uuid');
            expect(result.status).toBe(ecu_calibration_entity_1.CalibrationStatus.APPROVED);
            expect(result.approvedById).toBe('approver-uuid');
        });
        it('should update to production without approval fields', async () => {
            const productionCalibration = {
                ...mockCalibration,
                status: ecu_calibration_entity_1.CalibrationStatus.PRODUCTION,
            };
            repository.findOne.mockResolvedValue({ ...mockCalibration });
            repository.save.mockResolvedValue(productionCalibration);
            const result = await service.updateStatus('tenant-uuid', 'calibration-uuid', ecu_calibration_entity_1.CalibrationStatus.PRODUCTION, 'user-uuid');
            expect(result.status).toBe(ecu_calibration_entity_1.CalibrationStatus.PRODUCTION);
        });
        it('should update to testing status', async () => {
            const testingCalibration = {
                ...mockCalibration,
                status: ecu_calibration_entity_1.CalibrationStatus.TESTING,
            };
            repository.findOne.mockResolvedValue({ ...mockCalibration });
            repository.save.mockResolvedValue(testingCalibration);
            const result = await service.updateStatus('tenant-uuid', 'calibration-uuid', ecu_calibration_entity_1.CalibrationStatus.TESTING, 'user-uuid');
            expect(result.status).toBe(ecu_calibration_entity_1.CalibrationStatus.TESTING);
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
                createdById: 'new-user-uuid',
                approvedById: undefined,
                approvedAt: undefined,
            };
            repository.findOne.mockResolvedValue(mockCalibration);
            repository.create.mockReturnValue(newVersion);
            repository.save.mockResolvedValue(newVersion);
            const result = await service.createVersion('tenant-uuid', 'calibration-uuid', 'new-user-uuid', { version: '2.0.0' });
            expect(repository.create).toHaveBeenCalledWith(expect.objectContaining({
                parentVersionId: 'calibration-uuid',
                status: ecu_calibration_entity_1.CalibrationStatus.DRAFT,
                createdById: 'new-user-uuid',
                id: undefined,
            }));
            expect(result.parentVersionId).toBe('calibration-uuid');
            expect(result.status).toBe(ecu_calibration_entity_1.CalibrationStatus.DRAFT);
        });
        it('should inherit map data from parent', async () => {
            const newVersion = {
                ...mockCalibration,
                id: 'new-version-uuid',
                version: '2.0.0',
                parentVersionId: 'calibration-uuid',
            };
            repository.findOne.mockResolvedValue(mockCalibration);
            repository.create.mockReturnValue(newVersion);
            repository.save.mockResolvedValue(newVersion);
            await service.createVersion('tenant-uuid', 'calibration-uuid', 'user-uuid', {});
            expect(repository.create).toHaveBeenCalledWith(expect.objectContaining({
                mapData: mockCalibration.mapData,
            }));
        });
        it('should throw NotFoundException if parent not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.createVersion('tenant-uuid', 'non-existent-uuid', 'user-uuid', {})).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('remove', () => {
        it('should soft delete a calibration', async () => {
            const deletedCalibration = {
                ...mockCalibration,
                isActive: false,
            };
            repository.findOne.mockResolvedValue({ ...mockCalibration });
            repository.save.mockResolvedValue(deletedCalibration);
            await service.remove('tenant-uuid', 'calibration-uuid');
            expect(repository.save).toHaveBeenCalledWith(expect.objectContaining({ isActive: false }));
        });
        it('should throw NotFoundException if calibration not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.remove('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('getStats', () => {
        it('should return calibration statistics', async () => {
            repository.count.mockResolvedValue(25);
            mockQueryBuilder.getRawMany
                .mockResolvedValueOnce([
                { status: ecu_calibration_entity_1.CalibrationStatus.DRAFT, count: '5' },
                { status: ecu_calibration_entity_1.CalibrationStatus.PRODUCTION, count: '20' },
            ])
                .mockResolvedValueOnce([
                { protocol: ecu_calibration_entity_1.CalibrationProtocol.UDS, count: '15' },
                { protocol: ecu_calibration_entity_1.CalibrationProtocol.KWP2000, count: '10' },
            ]);
            const result = await service.getStats('tenant-uuid');
            expect(result.total).toBe(25);
            expect(result.byStatus).toHaveLength(2);
            expect(result.byProtocol).toHaveLength(2);
        });
    });
});
//# sourceMappingURL=ecu-calibrations.service.spec.js.map