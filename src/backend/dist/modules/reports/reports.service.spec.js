"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const typeorm_1 = require("@nestjs/typeorm");
const common_1 = require("@nestjs/common");
const reports_service_1 = require("./reports.service");
const report_entity_1 = require("./entities/report.entity");
describe('ReportsService', () => {
    let service;
    let repository;
    const mockReport = {
        id: 'report-uuid',
        tenantId: 'tenant-uuid',
        sessionId: 'session-uuid',
        vehicleId: 'vehicle-uuid',
        customerId: 'customer-uuid',
        title: 'Diagnostic Report - 2024-01-15',
        description: 'Full vehicle diagnostic',
        type: report_entity_1.ReportType.DIAGNOSTIC,
        status: report_entity_1.ReportStatus.DRAFT,
        format: report_entity_1.ReportFormat.PDF,
        summary: {
            totalReadings: 150,
            dtcsFound: 2,
            anomaliesDetected: 1,
            recommendations: ['Replace air filter'],
        },
        vehicleInfo: {
            make: 'Volkswagen',
            model: 'Golf GTI',
            year: 2022,
        },
        diagnosticData: {
            dtcs: [{ code: 'P0300', description: 'Misfire', severity: 'warning' }],
        },
        createdById: 'user-uuid',
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
                reports_service_1.ReportsService,
                {
                    provide: (0, typeorm_1.getRepositoryToken)(report_entity_1.Report),
                    useValue: mockRepository,
                },
            ],
        }).compile();
        service = module.get(reports_service_1.ReportsService);
        repository = module.get((0, typeorm_1.getRepositoryToken)(report_entity_1.Report));
        jest.clearAllMocks();
    });
    describe('create', () => {
        it('should create a new report', async () => {
            const dto = {
                title: 'Diagnostic Report',
                type: report_entity_1.ReportType.DIAGNOSTIC,
                vehicleId: 'vehicle-uuid',
                sessionId: 'session-uuid',
            };
            repository.create.mockReturnValue(mockReport);
            repository.save.mockResolvedValue(mockReport);
            const result = await service.create('tenant-uuid', 'user-uuid', dto);
            expect(repository.create).toHaveBeenCalledWith({
                ...dto,
                tenantId: 'tenant-uuid',
                createdById: 'user-uuid',
            });
            expect(result.title).toBe('Diagnostic Report - 2024-01-15');
        });
        it('should create a tuning report', async () => {
            const dto = {
                title: 'Tuning Report - Stage 2',
                type: report_entity_1.ReportType.TUNING,
                vehicleId: 'vehicle-uuid',
            };
            const tuningReport = {
                ...mockReport,
                type: report_entity_1.ReportType.TUNING,
                title: 'Tuning Report - Stage 2',
            };
            repository.create.mockReturnValue(tuningReport);
            repository.save.mockResolvedValue(tuningReport);
            const result = await service.create('tenant-uuid', 'user-uuid', dto);
            expect(result.type).toBe(report_entity_1.ReportType.TUNING);
        });
    });
    describe('findAll', () => {
        it('should return paginated reports', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockReport]);
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
            mockQueryBuilder.getMany.mockResolvedValue([mockReport]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                vehicleId: 'vehicle-uuid',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('report.vehicleId = :vehicleId', { vehicleId: 'vehicle-uuid' });
        });
        it('should filter by customer ID', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockReport]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                customerId: 'customer-uuid',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('report.customerId = :customerId', { customerId: 'customer-uuid' });
        });
        it('should filter by session ID', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockReport]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                sessionId: 'session-uuid',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('report.sessionId = :sessionId', { sessionId: 'session-uuid' });
        });
        it('should filter by report type', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockReport]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                type: report_entity_1.ReportType.DIAGNOSTIC,
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('report.type = :type', { type: report_entity_1.ReportType.DIAGNOSTIC });
        });
        it('should filter by status', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(1);
            mockQueryBuilder.getMany.mockResolvedValue([mockReport]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                status: report_entity_1.ReportStatus.APPROVED,
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('report.status = :status', { status: report_entity_1.ReportStatus.APPROVED });
        });
        it('should handle pagination correctly', async () => {
            mockQueryBuilder.getCount.mockResolvedValue(100);
            mockQueryBuilder.getMany.mockResolvedValue([mockReport]);
            const result = await service.findAll({
                tenantId: 'tenant-uuid',
                page: 3,
                limit: 10,
            });
            expect(mockQueryBuilder.skip).toHaveBeenCalledWith(20);
            expect(mockQueryBuilder.take).toHaveBeenCalledWith(10);
            expect(result.totalPages).toBe(10);
        });
    });
    describe('findOne', () => {
        it('should return a report by id', async () => {
            repository.findOne.mockResolvedValue(mockReport);
            const result = await service.findOne('tenant-uuid', 'report-uuid');
            expect(repository.findOne).toHaveBeenCalledWith({
                where: { id: 'report-uuid', tenantId: 'tenant-uuid', isActive: true },
            });
            expect(result.id).toBe('report-uuid');
        });
        it('should throw NotFoundException if report not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.findOne('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('update', () => {
        it('should update a report', async () => {
            const updatedReport = {
                ...mockReport,
                title: 'Updated Title',
            };
            repository.findOne.mockResolvedValue(mockReport);
            repository.save.mockResolvedValue(updatedReport);
            const result = await service.update('tenant-uuid', 'report-uuid', {
                title: 'Updated Title',
            });
            expect(result.title).toBe('Updated Title');
        });
        it('should throw NotFoundException if report not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.update('tenant-uuid', 'non-existent-uuid', { title: 'New' })).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('updateStatus', () => {
        it('should update report status to approved', async () => {
            const approvedReport = {
                ...mockReport,
                status: report_entity_1.ReportStatus.APPROVED,
                approvedById: 'approver-uuid',
                approvedAt: expect.any(Date),
            };
            repository.findOne.mockResolvedValue({ ...mockReport });
            repository.save.mockResolvedValue(approvedReport);
            const result = await service.updateStatus('tenant-uuid', 'report-uuid', report_entity_1.ReportStatus.APPROVED, 'approver-uuid');
            expect(result.status).toBe(report_entity_1.ReportStatus.APPROVED);
            expect(result.approvedById).toBe('approver-uuid');
        });
        it('should update status without approval fields for non-approved status', async () => {
            const pendingReport = {
                ...mockReport,
                status: report_entity_1.ReportStatus.PENDING_REVIEW,
            };
            repository.findOne.mockResolvedValue({ ...mockReport });
            repository.save.mockResolvedValue(pendingReport);
            const result = await service.updateStatus('tenant-uuid', 'report-uuid', report_entity_1.ReportStatus.PENDING_REVIEW, 'user-uuid');
            expect(result.status).toBe(report_entity_1.ReportStatus.PENDING_REVIEW);
        });
    });
    describe('markAsSent', () => {
        it('should mark report as sent', async () => {
            const sentReport = {
                ...mockReport,
                status: report_entity_1.ReportStatus.SENT,
                sentAt: expect.any(Date),
                sentTo: 'customer@example.com',
            };
            repository.findOne.mockResolvedValue({ ...mockReport });
            repository.save.mockResolvedValue(sentReport);
            const result = await service.markAsSent('tenant-uuid', 'report-uuid', 'customer@example.com');
            expect(result.status).toBe(report_entity_1.ReportStatus.SENT);
            expect(result.sentTo).toBe('customer@example.com');
        });
    });
    describe('generateFromSession', () => {
        it('should generate report from session', async () => {
            const generatedReport = {
                ...mockReport,
                title: expect.stringContaining('Diagnostic Report'),
                type: report_entity_1.ReportType.DIAGNOSTIC,
                status: report_entity_1.ReportStatus.DRAFT,
            };
            repository.create.mockReturnValue(generatedReport);
            repository.save.mockResolvedValue(generatedReport);
            const vehicleInfo = {
                make: 'Volkswagen',
                model: 'Golf GTI',
                year: 2022,
            };
            const result = await service.generateFromSession('tenant-uuid', 'session-uuid', 'user-uuid', vehicleInfo);
            expect(repository.create).toHaveBeenCalledWith(expect.objectContaining({
                tenantId: 'tenant-uuid',
                sessionId: 'session-uuid',
                createdById: 'user-uuid',
                type: report_entity_1.ReportType.DIAGNOSTIC,
                status: report_entity_1.ReportStatus.DRAFT,
                vehicleInfo,
            }));
            expect(result.type).toBe(report_entity_1.ReportType.DIAGNOSTIC);
        });
        it('should generate report without vehicle info', async () => {
            const generatedReport = {
                ...mockReport,
                vehicleInfo: undefined,
            };
            repository.create.mockReturnValue(generatedReport);
            repository.save.mockResolvedValue(generatedReport);
            await service.generateFromSession('tenant-uuid', 'session-uuid', 'user-uuid');
            expect(repository.create).toHaveBeenCalledWith(expect.objectContaining({
                vehicleInfo: undefined,
            }));
        });
    });
    describe('remove', () => {
        it('should soft delete a report', async () => {
            const deletedReport = {
                ...mockReport,
                isActive: false,
            };
            repository.findOne.mockResolvedValue({ ...mockReport });
            repository.save.mockResolvedValue(deletedReport);
            await service.remove('tenant-uuid', 'report-uuid');
            expect(repository.save).toHaveBeenCalledWith(expect.objectContaining({ isActive: false }));
        });
        it('should throw NotFoundException if report not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.remove('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('getStats', () => {
        it('should return report statistics', async () => {
            repository.count.mockResolvedValue(50);
            mockQueryBuilder.getRawMany
                .mockResolvedValueOnce([
                { type: report_entity_1.ReportType.DIAGNOSTIC, count: '30' },
                { type: report_entity_1.ReportType.TUNING, count: '20' },
            ])
                .mockResolvedValueOnce([
                { status: report_entity_1.ReportStatus.DRAFT, count: '10' },
                { status: report_entity_1.ReportStatus.APPROVED, count: '40' },
            ]);
            repository.find.mockResolvedValue([mockReport]);
            const result = await service.getStats('tenant-uuid');
            expect(result.total).toBe(50);
            expect(result.byType).toHaveLength(2);
            expect(result.byStatus).toHaveLength(2);
            expect(result.recentReports).toHaveLength(1);
        });
    });
});
//# sourceMappingURL=reports.service.spec.js.map